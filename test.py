# # Standard imports:
# import numpy as np
# import torch
# from torch.utils.data import random_split
# from torch_geometric.loader import DataLoader
# from torch_geometric.transforms import Compose
# from pathlib import Path

# # Custom data loader and model:
# from data import ProteinPairsSurfaces, PairData, CenterPairAtoms
# from data import RandomRotationPairAtoms, NormalizeChemFeatures, iface_valid_filter
# from helper import *
# from Arguments import parser

# # args
# random_rotation = True
# batch_size = 8
# search = True
# radius = 12.

# # We load the train and test datasets.
# # Random transforms, to ensure that no network/baseline overfits on pose parameters:
# transformations = (
#     Compose([NormalizeChemFeatures(), CenterPairAtoms(), RandomRotationPairAtoms()])
#     if random_rotation
#     else Compose([NormalizeChemFeatures()])
# )

# # PyTorch geometric expects an explicit list of "batched variables":
# batch_vars = ["xyz_p1", "xyz_p2", "atom_coords_p1", "atom_coords_p2"]
# # Load the train dataset:
# train_dataset = ProteinPairsSurfaces(
#     "surface_data", ppi=search, train=True, transform=transformations
# )
# # train_dataset = [data for data in train_dataset if iface_valid_filter(data)]
# # train_loader = DataLoader(
# #     train_dataset, batch_size=1, follow_batch=batch_vars, shuffle=True
# # )
# print("Preprocessing training dataset")

# Testing PyKeops installation
import pykeops

# Changing verbose and mode
pykeops.verbose = True
pykeops.build_type = 'Debug'

# Clean up the already compiled files
pykeops.clean_pykeops()

# Test Numpy integration
pykeops.test_numpy_bindings()