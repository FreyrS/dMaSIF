import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_curve, roc_auc_score


masif_preds = Path("masif_preds/")
timings = Path("timings/")
raw_data = Path("surface_data/raw/protein_surfaces/01-benchmark_surfaces_npy")

experiment_names = [
    "TangentConv_site_1layer_5A_epoch49",
    "TangentConv_site_1layer_9A_epoch49",
    "TangentConv_site_1layer_15A_epoch49",
    "TangentConv_site_3layer_15A_epoch17",
    "TangentConv_site_3layer_5A_epoch49",
    "TangentConv_site_3layer_9A_epoch46",
    "PointNet_site_3layer_9A_epoch37",
    "PointNet_site_3layer_5A_epoch46",
    "DGCNN_site_1layer_k100_epoch32",
    "PointNet_site_1layer_5A_epoch30",
    "PointNet_site_1layer_9A_epoch30",
    "DGCNN_site_1layer_k40_epoch46",
    "DGCNN_site_3layer_k40_epoch33",
]

experiment_names = [
    'Rebuttal_TangentConv_site_1L_8dim_9A_gc_subsamp20_dist05_epoch42',
    'Rebuttal_TangentConv_site_1L_8dim_9A_gc_subsamp20_dist20_epoch49',
    'Rebuttal_TangentConv_site_1L_8dim_9A_gc_subsamp20_dist105_epoch44',
    'Rebuttal_TangentConv_site_1L_8dim_9A_gc_subsamp20_var01_epoch43',
    'Rebuttal_TangentConv_site_1L_8dim_9A_gc_subsamp20_var02_epoch49',
    'Rebuttal_TangentConv_site_1L_8dim_9A_gc_subsamp20_var005_epoch37'
]

for experiment_name in experiment_names:
    print(experiment_name)
    datafolder = Path(f"preds/{experiment_name}")
    pdb_list = [p.stem[:-5] for p in datafolder.glob("*pred.vtk")]

    n_meshpoints = []
    n_predpoints = []
    meshpoints_mindists = []
    predpoints_mindists = []
    for pdb_id in tqdm(pdb_list):
        predpoints = np.load(datafolder / (pdb_id + "_predcoords.npy"))
        meshpoints = np.load(datafolder / (pdb_id + "_meshpoints.npy"))
        n_meshpoints.append(meshpoints.shape[0])
        n_predpoints.append(predpoints.shape[0])

        pdists = cdist(meshpoints, predpoints)
        meshpoints_mindists.append(pdists.min(1))
        predpoints_mindists.append(pdists.min(0))

    all_meshpoints_mindists = np.concatenate(meshpoints_mindists)
    all_predpoints_mindists = np.concatenate(predpoints_mindists)

    meshpoint_percentile = np.percentile(all_meshpoints_mindists, 99)
    predpoint_percentile = np.percentile(all_predpoints_mindists, 99)

    meshpoints_masks = []
    predpoints_masks = []
    for pdb_id in tqdm(pdb_list):
        predpoints = np.load(datafolder / (pdb_id + "_predcoords.npy"))
        meshpoints = np.load(datafolder / (pdb_id + "_meshpoints.npy"))

        pdists = cdist(meshpoints, predpoints)
        meshpoints_masks.append(pdists.min(1) < meshpoint_percentile)
        predpoints_masks.append(pdists.min(0) < predpoint_percentile)

    predpoints_preds = []
    predpoints_labels = []
    npoints = []
    for i, pdb_id in enumerate(tqdm(pdb_list)):
        predpoints_features = np.load(datafolder / (pdb_id + "_predfeatures.npy"))
        predpoints_features = predpoints_features[predpoints_masks[i]]

        predpoints_preds.append(predpoints_features[:, -2])
        predpoints_labels.append(predpoints_features[:, -1])
        npoints.append(predpoints_features.shape[0])

    predpoints_labels = np.concatenate(predpoints_labels)
    predpoints_preds = np.concatenate(predpoints_preds)
    rocauc = roc_auc_score(predpoints_labels.reshape(-1), predpoints_preds.reshape(-1))
    print("ROC-AUC", rocauc)

    np.save(timings / f"{experiment_name}_predpoints_preds", predpoints_preds)
    np.save(timings / f"{experiment_name}_predpoints_labels", predpoints_labels)
    np.save(timings / f"{experiment_name}_npoints", npoints)
