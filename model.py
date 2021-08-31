import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler
from pykeops.torch import LazyTensor

from geometry_processing import (
    curvatures,
    mesh_normals_areas,
    tangent_vectors,
    atoms_to_points_normals,
)
from helper import soft_dimension, diagonal_ranges
from benchmark_models import DGCNN_seg, PointNet2_seg, dMaSIFConv_seg


def knn_atoms(x, y, x_batch, y_batch, k):
    N, D = x.shape
    x_i = LazyTensor(x[:, None, :])
    y_j = LazyTensor(y[None, :, :])

    pairwise_distance_ij = ((x_i - y_j) ** 2).sum(-1)
    pairwise_distance_ij.ranges = diagonal_ranges(x_batch, y_batch)

    # N.B.: KeOps doesn't yet support backprop through Kmin reductions...
    # dists, idx = pairwise_distance_ij.Kmin_argKmin(K=k,axis=1)
    # So we have to re-compute the values ourselves:
    idx = pairwise_distance_ij.argKmin(K=k, axis=1)  # (N, K)
    x_ik = y[idx.view(-1)].view(N, k, D)
    dists = ((x[:, None, :] - x_ik) ** 2).sum(-1)

    return idx, dists


def get_atom_features(x, y, x_batch, y_batch, y_atomtype, k=16):

    idx, dists = knn_atoms(x, y, x_batch, y_batch, k=k)  # (num_points, k)
    num_points, _ = idx.size()

    idx = idx.view(-1)
    dists = 1 / dists.view(-1, 1)
    _, num_dims = y_atomtype.size()

    feature = y_atomtype[idx, :]
    feature = torch.cat([feature, dists], dim=1)
    feature = feature.view(num_points, k, num_dims + 1)

    return feature


class Atom_embedding(nn.Module):
    def __init__(self, args):
        super(Atom_embedding, self).__init__()
        self.D = args.atom_dims
        self.k = 16
        self.conv1 = nn.Linear(self.D + 1, self.D)
        self.conv2 = nn.Linear(self.D, self.D)
        self.conv3 = nn.Linear(2 * self.D, self.D)
        self.bn1 = nn.BatchNorm1d(self.D)
        self.bn2 = nn.BatchNorm1d(self.D)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y, y_atomtypes, x_batch, y_batch):
        fx = get_atom_features(x, y, x_batch, y_batch, y_atomtypes, k=self.k)
        fx = self.conv1(fx)
        fx = fx.view(-1, self.D)
        fx = self.bn1(self.relu(fx))
        fx = fx.view(-1, self.k, self.D)
        fx1 = fx.sum(dim=1, keepdim=False)

        fx = self.conv2(fx)
        fx = fx.view(-1, self.D)
        fx = self.bn2(self.relu(fx))
        fx = fx.view(-1, self.k, self.D)
        fx2 = fx.sum(dim=1, keepdim=False)
        fx = torch.cat((fx1, fx2), dim=-1)
        fx = self.conv3(fx)

        return fx


class AtomNet(nn.Module):
    def __init__(self, args):
        super(AtomNet, self).__init__()
        self.args = args

        self.transform_types = nn.Sequential(
            nn.Linear(args.atom_dims, args.atom_dims),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(args.atom_dims, args.atom_dims),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(args.atom_dims, args.atom_dims),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.embed = Atom_embedding(args)

    def forward(self, xyz, atom_xyz, atomtypes, batch, atom_batch):
        # Run a DGCNN on the available information:
        atomtypes = self.transform_types(atomtypes)
        return self.embed(xyz, atom_xyz, atomtypes, batch, atom_batch)

class Atom_embedding_MP(nn.Module):
    def __init__(self, args):
        super(Atom_embedding_MP, self).__init__()
        self.D = args.atom_dims
        self.k = 16
        self.n_layers = 3
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * self.D + 1, 2 * self.D + 1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(2 * self.D + 1, self.D),
                )
                for i in range(self.n_layers)
            ]
        )
        self.norm = nn.ModuleList(
            [nn.GroupNorm(2, self.D) for i in range(self.n_layers)]
        )
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y, y_atomtypes, x_batch, y_batch):
        idx, dists = knn_atoms(x, y, x_batch, y_batch, k=self.k)  # N, 9, 7
        num_points = x.shape[0]
        num_dims = y_atomtypes.shape[-1]

        point_emb = torch.ones_like(x[:, 0])[:, None].repeat(1, num_dims)
        for i in range(self.n_layers):
            features = y_atomtypes[idx.reshape(-1), :]
            features = torch.cat([features, dists.reshape(-1, 1)], dim=1)
            features = features.view(num_points, self.k, num_dims + 1)
            features = torch.cat(
                [point_emb[:, None, :].repeat(1, self.k, 1), features], dim=-1
            )  # N, 8, 13

            messages = self.mlp[i](features)  # N,8,6
            messages = messages.sum(1)  # N,6
            point_emb = point_emb + self.relu(self.norm[i](messages))

        return point_emb

class Atom_Atom_embedding_MP(nn.Module):
    def __init__(self, args):
        super(Atom_Atom_embedding_MP, self).__init__()
        self.D = args.atom_dims
        self.k = 17
        self.n_layers = 3

        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * self.D + 1, 2 * self.D + 1),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(2 * self.D + 1, self.D),
                )
                for i in range(self.n_layers)
            ]
        )

        self.norm = nn.ModuleList(
            [nn.GroupNorm(2, self.D) for i in range(self.n_layers)]
        )
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, y, y_atomtypes, x_batch, y_batch):
        idx, dists = knn_atoms(x, y, x_batch, y_batch, k=self.k)  # N, 9, 7
        idx = idx[:, 1:]  # Remove self
        dists = dists[:, 1:]
        k = self.k - 1
        num_points = y_atomtypes.shape[0]

        out = y_atomtypes
        for i in range(self.n_layers):
            _, num_dims = out.size()
            features = out[idx.reshape(-1), :]
            features = torch.cat([features, dists.reshape(-1, 1)], dim=1)
            features = features.view(num_points, k, num_dims + 1)
            features = torch.cat(
                [out[:, None, :].repeat(1, k, 1), features], dim=-1
            )  # N, 8, 13

            messages = self.mlp[i](features)  # N,8,6
            messages = messages.sum(1)  # N,6
            out = out + self.relu(self.norm[i](messages))

        return out

class AtomNet_MP(nn.Module):
    def __init__(self, args):
        super(AtomNet_MP, self).__init__()
        self.args = args

        self.transform_types = nn.Sequential(
            nn.Linear(args.atom_dims, args.atom_dims),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(args.atom_dims, args.atom_dims),
        )

        self.embed = Atom_embedding_MP(args)
        self.atom_atom = Atom_Atom_embedding_MP(args)

    def forward(self, xyz, atom_xyz, atomtypes, batch, atom_batch):
        # Run a DGCNN on the available information:
        atomtypes = self.transform_types(atomtypes)
        atomtypes = self.atom_atom(
            atom_xyz, atom_xyz, atomtypes, atom_batch, atom_batch
        )
        atomtypes = self.embed(xyz, atom_xyz, atomtypes, batch, atom_batch)
        return atomtypes


def combine_pair(P1, P2):
    P1P2 = {}
    for key in P1:
        v1 = P1[key]
        v2 = P2[key]
        if v1 is None:
            continue

        if key == "batch" or key == "batch_atoms":
            v1v2 = torch.cat([v1, v2 + v1[-1] + 1], dim=0)
        elif key == "triangles":
            # v1v2 = torch.cat([v1,v2],dim=1)
            continue
        else:
            v1v2 = torch.cat([v1, v2], dim=0)
        P1P2[key] = v1v2

    return P1P2


def split_pair(P1P2):
    batch_size = P1P2["batch_atoms"][-1] + 1
    p1_indices = P1P2["batch"] < batch_size // 2
    p2_indices = P1P2["batch"] >= batch_size // 2

    p1_atom_indices = P1P2["batch_atoms"] < batch_size // 2
    p2_atom_indices = P1P2["batch_atoms"] >= batch_size // 2

    P1 = {}
    P2 = {}
    for key in P1P2:
        v1v2 = P1P2[key]

        if (key == "rand_rot") or (key == "atom_center"):
            n = v1v2.shape[0] // 2
            P1[key] = v1v2[:n].view(-1, 3)
            P2[key] = v1v2[n:].view(-1, 3)
        elif "atom" in key:
            P1[key] = v1v2[p1_atom_indices]
            P2[key] = v1v2[p2_atom_indices]
        elif key == "triangles":
            continue
            # P1[key] = v1v2[:,p1_atom_indices]
            # P2[key] = v1v2[:,p2_atom_indices]
        else:
            P1[key] = v1v2[p1_indices]
            P2[key] = v1v2[p2_indices]

    P2["batch"] = P2["batch"] - batch_size + 1
    P2["batch_atoms"] = P2["batch_atoms"] - batch_size + 1

    return P1, P2



def project_iface_labels(P, threshold=2.0):

    queries = P["xyz"]
    batch_queries = P["batch"]
    source = P["mesh_xyz"]
    batch_source = P["mesh_batch"]
    labels = P["mesh_labels"]
    x_i = LazyTensor(queries[:, None, :])  # (N, 1, D)
    y_j = LazyTensor(source[None, :, :])  # (1, M, D)

    D_ij = ((x_i - y_j) ** 2).sum(-1).sqrt()  # (N, M)
    D_ij.ranges = diagonal_ranges(batch_queries, batch_source)
    nn_i = D_ij.argmin(dim=1).view(-1)  # (N,)
    nn_dist_i = (
        D_ij.min(dim=1).view(-1, 1) < threshold
    ).float()  # If chain is not connected because of missing densities MaSIF cut out a part of the protein
    query_labels = labels[nn_i] * nn_dist_i
    P["labels"] = query_labels

class dMaSIF(nn.Module):
    def __init__(self, args):
        super(dMaSIF, self).__init__()
        # Additional geometric features: mean and Gauss curvatures computed at different scales.
        self.curvature_scales = args.curvature_scales
        self.args = args

        I = args.in_channels
        O = args.orientation_units
        E = args.emb_dims
        H = args.post_units

        # Computes chemical features
        self.atomnet = AtomNet_MP(args)
        self.dropout = nn.Dropout(args.dropout)

        if args.embedding_layer == "dMaSIF":
            # Post-processing, without batch norm:
            self.orientation_scores = nn.Sequential(
                nn.Linear(I, O),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(O, 1),
            )

            # Segmentation network:
            self.conv = dMaSIFConv_seg(
                args,
                in_channels=I,
                out_channels=E,
                n_layers=args.n_layers,
                radius=args.radius,
            )

            # Asymmetric embedding
            if args.search:
                self.orientation_scores2 = nn.Sequential(
                    nn.Linear(I, O),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(O, 1),
                )

                self.conv2 = dMaSIFConv_seg(
                    args,
                    in_channels=I,
                    out_channels=E,
                    n_layers=args.n_layers,
                    radius=args.radius,
                )

        elif args.embedding_layer == "DGCNN":
            self.conv = DGCNN_seg(I + 3, E,self.args.n_layers,self.args.k)
            if args.search:
                self.conv2 = DGCNN_seg(I + 3, E,self.args.n_layers,self.args.k)

        elif args.embedding_layer == "PointNet++":
            self.conv = PointNet2_seg(args, I, E)
            if args.search:
                self.conv2 = PointNet2_seg(args, I, E)

        if args.site:
            # Post-processing, without batch norm:
            self.net_out = nn.Sequential(
                nn.Linear(E, H),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(H, H),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(H, 1),
            )

    def features(self, P, i=1):
        """Estimates geometric and chemical features from a protein surface or a cloud of atoms."""
        if (
            not self.args.use_mesh and "xyz" not in P
        ):  # Compute the pseudo-surface directly from the atoms
            # (Note that we use the fact that dicts are "passed by reference" here)
            P["xyz"], P["normals"], P["batch"] = atoms_to_points_normals(
                P["atoms"],
                P["batch_atoms"],
                atomtypes=P["atomtypes"],
                resolution=self.args.resolution,
                sup_sampling=self.args.sup_sampling,
            )

        # Estimate the curvatures using the triangles or the estimated normals:
        P_curvatures = curvatures(
            P["xyz"],
            triangles=P["triangles"] if self.args.use_mesh else None,
            normals=None if self.args.use_mesh else P["normals"],
            scales=self.curvature_scales,
            batch=P["batch"],
        )

        # Compute chemical features on-the-fly:
        chemfeats = self.atomnet(
            P["xyz"], P["atom_xyz"], P["atomtypes"], P["batch"], P["batch_atoms"]
        )

        if self.args.no_chem:
            chemfeats = 0.0 * chemfeats
        if self.args.no_geom:
            P_curvatures = 0.0 * P_curvatures

        # Concatenate our features:
        return torch.cat([P_curvatures, chemfeats], dim=1).contiguous()

    def embed(self, P):
        """Embeds all points of a protein in a high-dimensional vector space."""

        features = self.dropout(self.features(P))
        P["input_features"] = features

        torch.cuda.synchronize(device=features.device)
        torch.cuda.reset_max_memory_allocated(device=P["atoms"].device)
        begin = time.time()

        # Ours:
        if self.args.embedding_layer == "dMaSIF":
            self.conv.load_mesh(
                P["xyz"],
                triangles=P["triangles"] if self.args.use_mesh else None,
                normals=None if self.args.use_mesh else P["normals"],
                weights=self.orientation_scores(features),
                batch=P["batch"],
            )
            P["embedding_1"] = self.conv(features)
            if self.args.search:
                self.conv2.load_mesh(
                    P["xyz"],
                    triangles=P["triangles"] if self.args.use_mesh else None,
                    normals=None if self.args.use_mesh else P["normals"],
                    weights=self.orientation_scores2(features),
                    batch=P["batch"],
                )
                P["embedding_2"] = self.conv2(features)

        # First baseline:
        elif self.args.embedding_layer == "DGCNN":
            features = torch.cat([features, P["xyz"]], dim=-1).contiguous()
            P["embedding_1"] = self.conv(P["xyz"], features, P["batch"])
            if self.args.search:
                P["embedding_2"] = self.conv2(
                    P["xyz"], features, P["batch"]
                )

        # Second baseline
        elif self.args.embedding_layer == "PointNet++":
            P["embedding_1"] = self.conv(P["xyz"], features, P["batch"])
            if self.args.search:
                P["embedding_2"] = self.conv2(P["xyz"], features, P["batch"])

        torch.cuda.synchronize(device=features.device)
        end = time.time()
        memory_usage = torch.cuda.max_memory_allocated(device=P["atoms"].device)
        conv_time = end - begin

        return conv_time, memory_usage

    def preprocess_surface(self, P):
        P["xyz"], P["normals"], P["batch"] = atoms_to_points_normals(
            P["atoms"],
            P["batch_atoms"],
            atomtypes=P["atomtypes"],
            resolution=self.args.resolution,
            sup_sampling=self.args.sup_sampling,
            distance=self.args.distance,
        )
        if P['mesh_labels'] is not None:
            project_iface_labels(P)

    def forward(self, P1, P2=None):
        # Compute embeddings of the point clouds:
        if P2 is not None:
            P1P2 = combine_pair(P1, P2)
        else:
            P1P2 = P1

        conv_time, memory_usage = self.embed(P1P2)

        # Monitor the approximate rank of our representations:
        R_values = {}
        R_values["input"] = soft_dimension(P1P2["input_features"])
        R_values["conv"] = soft_dimension(P1P2["embedding_1"])

        if self.args.site:
            P1P2["iface_preds"] = self.net_out(P1P2["embedding_1"])

        if P2 is not None:
            P1, P2 = split_pair(P1P2)
        else:
            P1 = P1P2

        return {
            "P1": P1,
            "P2": P2,
            "R_values": R_values,
            "conv_time": conv_time,
            "memory_usage": memory_usage,
        }
