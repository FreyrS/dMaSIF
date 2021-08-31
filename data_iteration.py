import torch
import numpy as np
from helper import *
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler
from sklearn.metrics import roc_auc_score
from pathlib import Path
import math
from tqdm import tqdm
from geometry_processing import save_vtk
from helper import numpy, diagonal_ranges
import time


def process_single(protein_pair, chain_idx=1):
    """Turn the PyG data object into a dict."""

    P = {}
    with_mesh = "face_p1" in protein_pair.keys
    preprocessed = "gen_xyz_p1" in protein_pair.keys

    if chain_idx == 1:
        # Ground truth labels are available on mesh vertices:
        P["mesh_labels"] = protein_pair.y_p1 if with_mesh else None

        # N.B.: The DataLoader should use the optional argument
        #       "follow_batch=['xyz_p1', 'xyz_p2']", as described on the PyG tutorial.
        P["mesh_batch"] = protein_pair.xyz_p1_batch if with_mesh else None

        # Surface information:
        P["mesh_xyz"] = protein_pair.xyz_p1 if with_mesh else None
        P["mesh_triangles"] = protein_pair.face_p1 if with_mesh else None

        # Atom information:
        P["atoms"] = protein_pair.atom_coords_p1
        P["batch_atoms"] = protein_pair.atom_coords_p1_batch

        # Chemical features: atom coordinates and types.
        P["atom_xyz"] = protein_pair.atom_coords_p1
        P["atomtypes"] = protein_pair.atom_types_p1

        P["xyz"] = protein_pair.gen_xyz_p1 if preprocessed else None
        P["normals"] = protein_pair.gen_normals_p1 if preprocessed else None
        P["batch"] = protein_pair.gen_batch_p1 if preprocessed else None
        P["labels"] = protein_pair.gen_labels_p1 if preprocessed else None

    elif chain_idx == 2:
        # Ground truth labels are available on mesh vertices:
        P["mesh_labels"] = protein_pair.y_p2 if with_mesh else None

        # N.B.: The DataLoader should use the optional argument
        #       "follow_batch=['xyz_p1', 'xyz_p2']", as described on the PyG tutorial.
        P["mesh_batch"] = protein_pair.xyz_p2_batch if with_mesh else None

        # Surface information:
        P["mesh_xyz"] = protein_pair.xyz_p2 if with_mesh else None
        P["mesh_triangles"] = protein_pair.face_p2 if with_mesh else None

        # Atom information:
        P["atoms"] = protein_pair.atom_coords_p2
        P["batch_atoms"] = protein_pair.atom_coords_p2_batch

        # Chemical features: atom coordinates and types.
        P["atom_xyz"] = protein_pair.atom_coords_p2
        P["atomtypes"] = protein_pair.atom_types_p2

        P["xyz"] = protein_pair.gen_xyz_p2 if preprocessed else None
        P["normals"] = protein_pair.gen_normals_p2 if preprocessed else None
        P["batch"] = protein_pair.gen_batch_p2 if preprocessed else None
        P["labels"] = protein_pair.gen_labels_p2 if preprocessed else None

    return P


def save_protein_batch_single(protein_pair_id, P, save_path, pdb_idx):

    protein_pair_id = protein_pair_id.split("_")
    pdb_id = protein_pair_id[0] + "_" + protein_pair_id[pdb_idx]

    batch = P["batch"]

    xyz = P["xyz"]

    inputs = P["input_features"]

    embedding = P["embedding_1"] if pdb_idx == 1 else P["embedding_2"]
    emb_id = 1 if pdb_idx == 1 else 2

    predictions = torch.sigmoid(P["iface_preds"]) if "iface_preds" in P.keys() else 0.0*embedding[:,0].view(-1, 1)

    labels = P["labels"].view(-1, 1) if P["labels"] is not None else 0.0 * predictions

    coloring = torch.cat([inputs, embedding, predictions, labels], axis=1)

    save_vtk(str(save_path / pdb_id) + f"_pred_emb{emb_id}", xyz, values=coloring)
    np.save(str(save_path / pdb_id) + "_predcoords", numpy(xyz))
    np.save(str(save_path / pdb_id) + f"_predfeatures_emb{emb_id}", numpy(coloring))


def project_iface_labels(P, threshold=2.0):

    queries = P["xyz"]
    batch_queries = P["batch"]
    source = P["mesh_xyz"]
    batch_source = P["mesh_batch"]
    labels = P["mesh_labels"]
    x_i = LazyTensor(queries[:, None, :])  # (N, 1, D)
    y_j = LazyTensor(source[None, :, :])  # (1, M, D)

    D_ij = ((x_i - y_j) ** 2).sum(-1)  # (N, M)
    D_ij.ranges = diagonal_ranges(batch_queries, batch_source)
    nn_i = D_ij.argmin(dim=1).view(-1)  # (N,)
    nn_dist_i = (
        D_ij.min(dim=1).view(-1, 1) < threshold
    ).float()  # If chain is not connected because of missing densities MaSIF cut out a part of the protein

    query_labels = labels[nn_i] * nn_dist_i

    P["labels"] = query_labels


def process(args, protein_pair, net):
    P1 = process_single(protein_pair, chain_idx=1)
    if not "gen_xyz_p1" in protein_pair.keys:
        net.preprocess_surface(P1)
        #if P1["mesh_labels"] is not None:
        #    project_iface_labels(P1)
    P2 = None
    if not args.single_protein:
        P2 = process_single(protein_pair, chain_idx=2)
        if not "gen_xyz_p2" in protein_pair.keys:
            net.preprocess_surface(P2)
            #if P2["mesh_labels"] is not None:
            #    project_iface_labels(P2)

    return P1, P2


def generate_matchinglabels(args, P1, P2):
    if args.random_rotation:
        P1["xyz"] = torch.matmul(P1["rand_rot"].T, P1["xyz"].T).T + P1["atom_center"]
        P2["xyz"] = torch.matmul(P2["rand_rot"].T, P2["xyz"].T).T + P2["atom_center"]
    xyz1_i = LazyTensor(P1["xyz"][:, None, :].contiguous())
    xyz2_j = LazyTensor(P2["xyz"][None, :, :].contiguous())

    xyz_dists = ((xyz1_i - xyz2_j) ** 2).sum(-1).sqrt()
    xyz_dists = (1.0 - xyz_dists).step()

    p1_iface_labels = (xyz_dists.sum(1) > 1.0).float().view(-1)
    p2_iface_labels = (xyz_dists.sum(0) > 1.0).float().view(-1)

    P1["labels"] = p1_iface_labels
    P2["labels"] = p2_iface_labels


def compute_loss(args, P1, P2, n_points_sample=16):

    if args.search:
        pos_xyz1 = P1["xyz"][P1["labels"] == 1]
        pos_xyz2 = P2["xyz"][P2["labels"] == 1]
        pos_descs1 = P1["embedding_1"][P1["labels"] == 1]
        pos_descs2 = P2["embedding_2"][P2["labels"] == 1]

        pos_xyz_dists = (
            ((pos_xyz1[:, None, :] - pos_xyz2[None, :, :]) ** 2).sum(-1).sqrt()
        )
        pos_desc_dists = torch.matmul(pos_descs1, pos_descs2.T)

        pos_preds = pos_desc_dists[pos_xyz_dists < 1.0]
        pos_labels = torch.ones_like(pos_preds)

        n_desc_sample = 100
        sample_desc2 = torch.randperm(len(P2["embedding_2"]))[:n_desc_sample]
        sample_desc2 = P2["embedding_2"][sample_desc2]
        neg_preds = torch.matmul(pos_descs1, sample_desc2.T).view(-1)
        neg_labels = torch.zeros_like(neg_preds)

        # For symmetry
        pos_descs1_2 = P1["embedding_2"][P1["labels"] == 1]
        pos_descs2_2 = P2["embedding_1"][P2["labels"] == 1]

        pos_desc_dists2 = torch.matmul(pos_descs2_2, pos_descs1_2.T)
        pos_preds2 = pos_desc_dists2[pos_xyz_dists.T < 1.0]
        pos_preds = torch.cat([pos_preds, pos_preds2], dim=0)
        pos_labels = torch.ones_like(pos_preds)

        sample_desc1_2 = torch.randperm(len(P1["embedding_2"]))[:n_desc_sample]
        sample_desc1_2 = P1["embedding_2"][sample_desc1_2]
        neg_preds_2 = torch.matmul(pos_descs2_2, sample_desc1_2.T).view(-1)

        neg_preds = torch.cat([neg_preds, neg_preds_2], dim=0)
        neg_labels = torch.zeros_like(neg_preds)

    else:
        pos_preds = P1["iface_preds"][P1["labels"] == 1]
        pos_labels = P1["labels"][P1["labels"] == 1]
        neg_preds = P1["iface_preds"][P1["labels"] == 0]
        neg_labels = P1["labels"][P1["labels"] == 0]

    n_points_sample = len(pos_labels)
    pos_indices = torch.randperm(len(pos_labels))[:n_points_sample]
    neg_indices = torch.randperm(len(neg_labels))[:n_points_sample]

    pos_preds = pos_preds[pos_indices]
    pos_labels = pos_labels[pos_indices]
    neg_preds = neg_preds[neg_indices]
    neg_labels = neg_labels[neg_indices]

    preds_concat = torch.cat([pos_preds, neg_preds])
    labels_concat = torch.cat([pos_labels, neg_labels])

    loss = F.binary_cross_entropy_with_logits(preds_concat, labels_concat)

    return loss, preds_concat, labels_concat


def extract_single(P_batch, number):
    P = {}  # First and second proteins
    batch = P_batch["batch"] == number
    batch_atoms = P_batch["batch_atoms"] == number

    with_mesh = P_batch["labels"] is not None
    # Ground truth labels are available on mesh vertices:
    P["labels"] = P_batch["labels"][batch] if with_mesh else None

    P["batch"] = P_batch["batch"][batch]

    # Surface information:
    P["xyz"] = P_batch["xyz"][batch]
    P["normals"] = P_batch["normals"][batch]

    # Atom information:
    P["atoms"] = P_batch["atoms"][batch_atoms]
    P["batch_atoms"] = P_batch["batch_atoms"][batch_atoms]

    # Chemical features: atom coordinates and types.
    P["atom_xyz"] = P_batch["atom_xyz"][batch_atoms]
    P["atomtypes"] = P_batch["atomtypes"][batch_atoms]

    return P


def iterate(
    net,
    dataset,
    optimizer,
    args,
    test=False,
    save_path=None,
    pdb_ids=None,
    summary_writer=None,
    epoch_number=None,
):
    """Goes through one epoch of the dataset, returns information for Tensorboard."""

    if test:
        net.eval()
        torch.set_grad_enabled(False)
    else:
        net.train()
        torch.set_grad_enabled(True)

    # Statistics and fancy graphs to summarize the epoch:
    info = []
    total_processed_pairs = 0
    # Loop over one epoch:
    for it, protein_pair in enumerate(
        tqdm(dataset)
    ):  # , desc="Test " if test else "Train")):
        protein_batch_size = protein_pair.atom_coords_p1_batch[-1].item() + 1
        if save_path is not None:
            batch_ids = pdb_ids[
                total_processed_pairs : total_processed_pairs + protein_batch_size
            ]
            total_processed_pairs += protein_batch_size

        protein_pair.to(args.device)

        if not test:
            optimizer.zero_grad()

        # Generate the surface:
        torch.cuda.synchronize()
        surface_time = time.time()
        P1_batch, P2_batch = process(args, protein_pair, net)
        torch.cuda.synchronize()
        surface_time = time.time() - surface_time

        for protein_it in range(protein_batch_size):
            torch.cuda.synchronize()
            iteration_time = time.time()

            P1 = extract_single(P1_batch, protein_it)
            P2 = None if args.single_protein else extract_single(P2_batch, protein_it)


            if args.random_rotation:
                P1["rand_rot"] = protein_pair.rand_rot1.view(-1, 3, 3)[0]
                P1["atom_center"] = protein_pair.atom_center1.view(-1, 1, 3)[0]
                P1["xyz"] = P1["xyz"] - P1["atom_center"]
                P1["xyz"] = (
                    torch.matmul(P1["rand_rot"], P1["xyz"].T).T
                ).contiguous()
                P1["normals"] = (
                    torch.matmul(P1["rand_rot"], P1["normals"].T).T
                ).contiguous()
                if not args.single_protein:
                    P2["rand_rot"] = protein_pair.rand_rot2.view(-1, 3, 3)[0]
                    P2["atom_center"] = protein_pair.atom_center2.view(-1, 1, 3)[0]
                    P2["xyz"] = P2["xyz"] - P2["atom_center"]
                    P2["xyz"] = (
                        torch.matmul(P2["rand_rot"], P2["xyz"].T).T
                    ).contiguous()
                    P2["normals"] = (
                        torch.matmul(P2["rand_rot"], P2["normals"].T).T
                    ).contiguous()
            else:
                P1["rand_rot"] = torch.eye(3, device=P1["xyz"].device)
                P1["atom_center"] = torch.zeros((1, 3), device=P1["xyz"].device)
                if not args.single_protein:
                    P2["rand_rot"] = torch.eye(3, device=P2["xyz"].device)
                    P2["atom_center"] = torch.zeros((1, 3), device=P2["xyz"].device)
                    
            torch.cuda.synchronize()
            prediction_time = time.time()
            outputs = net(P1, P2)
            torch.cuda.synchronize()
            prediction_time = time.time() - prediction_time

            P1 = outputs["P1"]
            P2 = outputs["P2"]

            if args.search:
                generate_matchinglabels(args, P1, P2)

            if P1["labels"] is not None:
                loss, sampled_preds, sampled_labels = compute_loss(args, P1, P2)
            else:
                loss = torch.tensor(0.0)
                sampled_preds = None
                sampled_labels = None

            # Compute the gradient, update the model weights:
            if not test:
                torch.cuda.synchronize()
                back_time = time.time()
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                back_time = time.time() - back_time

            if it == protein_it == 0 and not test:
                for para_it, parameter in enumerate(net.atomnet.parameters()):
                    if parameter.requires_grad:
                        summary_writer.add_histogram(
                            f"Gradients/Atomnet/para_{para_it}_{parameter.shape}",
                            parameter.grad.view(-1),
                            epoch_number,
                        )
                for para_it, parameter in enumerate(net.conv.parameters()):
                    if parameter.requires_grad:
                        summary_writer.add_histogram(
                            f"Gradients/Conv/para_{para_it}_{parameter.shape}",
                            parameter.grad.view(-1),
                            epoch_number,
                        )

                for d, features in enumerate(P1["input_features"].T):
                    summary_writer.add_histogram(f"Input features/{d}", features)

            if save_path is not None:
                save_protein_batch_single(
                    batch_ids[protein_it], P1, save_path, pdb_idx=1
                )
                if not args.single_protein:
                    save_protein_batch_single(
                        batch_ids[protein_it], P2, save_path, pdb_idx=2
                    )

            try:
                if sampled_labels is not None:
                    roc_auc = roc_auc_score(
                        np.rint(numpy(sampled_labels.view(-1))),
                        numpy(sampled_preds.view(-1)),
                    )
                else:
                    roc_auc = 0.0
            except Exception as e:
                print("Problem with computing roc-auc")
                print(e)
                continue

            R_values = outputs["R_values"]

            info.append(
                dict(
                    {
                        "Loss": loss.item(),
                        "ROC-AUC": roc_auc,
                        "conv_time": outputs["conv_time"],
                        "memory_usage": outputs["memory_usage"],
                    },
                    # Merge the "R_values" dict into "info", with a prefix:
                    **{"R_values/" + k: v for k, v in R_values.items()},
                )
            )
            torch.cuda.synchronize()
            iteration_time = time.time() - iteration_time

    # Turn a list of dicts into a dict of lists:
    newdict = {}
    for k, v in [(key, d[key]) for d in info for key in d]:
        if k not in newdict:
            newdict[k] = [v]
        else:
            newdict[k].append(v)
    info = newdict

    # Final post-processing:
    return info

def iterate_surface_precompute(dataset, net, args):
    processed_dataset = []
    for it, protein_pair in enumerate(tqdm(dataset)):
        protein_pair.to(args.device)
        P1, P2 = process(args, protein_pair, net)
        if args.random_rotation:
            P1["rand_rot"] = protein_pair.rand_rot1
            P1["atom_center"] = protein_pair.atom_center1
            P1["xyz"] = (
                torch.matmul(P1["rand_rot"].T, P1["xyz"].T).T + P1["atom_center"]
            )
            P1["normals"] = torch.matmul(P1["rand_rot"].T, P1["normals"].T).T
            if not args.single_protein:
                P2["rand_rot"] = protein_pair.rand_rot2
                P2["atom_center"] = protein_pair.atom_center2
                P2["xyz"] = (
                    torch.matmul(P2["rand_rot"].T, P2["xyz"].T).T + P2["atom_center"]
                )
                P2["normals"] = torch.matmul(P2["rand_rot"].T, P2["normals"].T).T
        protein_pair = protein_pair.to_data_list()[0]
        protein_pair.gen_xyz_p1 = P1["xyz"]
        protein_pair.gen_normals_p1 = P1["normals"]
        protein_pair.gen_batch_p1 = P1["batch"]
        protein_pair.gen_labels_p1 = P1["labels"]
        protein_pair.gen_xyz_p2 = P2["xyz"]
        protein_pair.gen_normals_p2 = P2["normals"]
        protein_pair.gen_batch_p2 = P2["batch"]
        protein_pair.gen_labels_p2 = P2["labels"]
        processed_dataset.append(protein_pair.to("cpu"))
    return processed_dataset
