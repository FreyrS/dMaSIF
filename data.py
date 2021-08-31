import torch
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.transforms import Compose
import numpy as np
from scipy.spatial.transform import Rotation
import math
import urllib.request
import tarfile
from pathlib import Path
import requests
from data_preprocessing.convert_pdb2npy import convert_pdbs
from data_preprocessing.convert_ply2npy import convert_plys

tensor = torch.FloatTensor
inttensor = torch.LongTensor


def numpy(x):
    return x.detach().cpu().numpy()


def iface_valid_filter(protein_pair):
    labels1 = protein_pair.y_p1.reshape(-1)
    labels2 = protein_pair.y_p2.reshape(-1)
    valid1 = (
        (torch.sum(labels1) < 0.75 * len(labels1))
        and (torch.sum(labels1) > 30)
        and (torch.sum(labels1) > 0.01 * labels2.shape[0])
    )
    valid2 = (
        (torch.sum(labels2) < 0.75 * len(labels2))
        and (torch.sum(labels2) > 30)
        and (torch.sum(labels2) > 0.01 * labels1.shape[0])
    )

    return valid1 and valid2


class RandomRotationPairAtoms(object):
    r"""Randomly rotate a protein"""

    def __call__(self, data):
        R1 = tensor(Rotation.random().as_matrix())
        R2 = tensor(Rotation.random().as_matrix())

        data.atom_coords_p1 = torch.matmul(R1, data.atom_coords_p1.T).T
        data.xyz_p1 = torch.matmul(R1, data.xyz_p1.T).T
        data.normals_p1 = torch.matmul(R1, data.normals_p1.T).T

        data.atom_coords_p2 = torch.matmul(R2, data.atom_coords_p2.T).T
        data.xyz_p2 = torch.matmul(R2, data.xyz_p2.T).T
        data.normals_p2 = torch.matmul(R2, data.normals_p2.T).T

        data.rand_rot1 = R1
        data.rand_rot2 = R2
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class CenterPairAtoms(object):
    r"""Centers a protein"""

    def __call__(self, data):
        atom_center1 = data.atom_coords_p1.mean(dim=-2, keepdim=True)
        atom_center2 = data.atom_coords_p2.mean(dim=-2, keepdim=True)

        data.atom_coords_p1 = data.atom_coords_p1 - atom_center1
        data.atom_coords_p2 = data.atom_coords_p2 - atom_center2

        data.xyz_p1 = data.xyz_p1 - atom_center1
        data.xyz_p2 = data.xyz_p2 - atom_center2

        data.atom_center1 = atom_center1
        data.atom_center2 = atom_center2
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class NormalizeChemFeatures(object):
    r"""Centers a protein"""

    def __call__(self, data):
        pb_upper = 3.0
        pb_lower = -3.0

        chem_p1 = data.chemical_features_p1
        chem_p2 = data.chemical_features_p2

        pb_p1 = chem_p1[:, 0]
        pb_p2 = chem_p2[:, 0]
        hb_p1 = chem_p1[:, 1]
        hb_p2 = chem_p2[:, 1]
        hp_p1 = chem_p1[:, 2]
        hp_p2 = chem_p2[:, 2]

        # Normalize PB
        pb_p1 = torch.clamp(pb_p1, pb_lower, pb_upper)
        pb_p1 = (pb_p1 - pb_lower) / (pb_upper - pb_lower)
        pb_p1 = 2 * pb_p1 - 1

        pb_p2 = torch.clamp(pb_p2, pb_lower, pb_upper)
        pb_p2 = (pb_p2 - pb_lower) / (pb_upper - pb_lower)
        pb_p2 = 2 * pb_p2 - 1

        # Normalize HP
        hp_p1 = hp_p1 / 4.5
        hp_p2 = hp_p2 / 4.5

        data.chemical_features_p1 = torch.stack([pb_p1, hb_p1, hp_p1]).T
        data.chemical_features_p2 = torch.stack([pb_p2, hb_p2, hp_p2]).T

        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


def load_protein_npy(pdb_id, data_dir, center=False, single_pdb=False):
    """Loads a protein surface mesh and its features"""

    # Load the data, and read the connectivity information:
    triangles = (
        None
        if single_pdb
        else inttensor(np.load(data_dir / (pdb_id + "_triangles.npy"))).T
    )
    # Normalize the point cloud, as specified by the user:
    points = None if single_pdb else tensor(np.load(data_dir / (pdb_id + "_xyz.npy")))
    center_location = None if single_pdb else torch.mean(points, axis=0, keepdims=True)

    atom_coords = tensor(np.load(data_dir / (pdb_id + "_atomxyz.npy")))
    atom_types = tensor(np.load(data_dir / (pdb_id + "_atomtypes.npy")))

    if center:
        points = points - center_location
        atom_coords = atom_coords - center_location

    # Interface labels
    iface_labels = (
        None
        if single_pdb
        else tensor(np.load(data_dir / (pdb_id + "_iface_labels.npy")).reshape((-1, 1)))
    )

    # Features
    chemical_features = (
        None if single_pdb else tensor(np.load(data_dir / (pdb_id + "_features.npy")))
    )

    # Normals
    normals = (
        None if single_pdb else tensor(np.load(data_dir / (pdb_id + "_normals.npy")))
    )

    protein_data = Data(
        xyz=points,
        face=triangles,
        chemical_features=chemical_features,
        y=iface_labels,
        normals=normals,
        center_location=center_location,
        num_nodes=None if single_pdb else points.shape[0],
        atom_coords=atom_coords,
        atom_types=atom_types,
    )
    return protein_data


class PairData(Data):
    def __init__(
        self,
        xyz_p1=None,
        xyz_p2=None,
        face_p1=None,
        face_p2=None,
        chemical_features_p1=None,
        chemical_features_p2=None,
        y_p1=None,
        y_p2=None,
        normals_p1=None,
        normals_p2=None,
        center_location_p1=None,
        center_location_p2=None,
        atom_coords_p1=None,
        atom_coords_p2=None,
        atom_types_p1=None,
        atom_types_p2=None,
        atom_center1=None,
        atom_center2=None,
        rand_rot1=None,
        rand_rot2=None,
    ):
        super().__init__()
        self.xyz_p1 = xyz_p1
        self.xyz_p2 = xyz_p2
        self.face_p1 = face_p1
        self.face_p2 = face_p2

        self.chemical_features_p1 = chemical_features_p1
        self.chemical_features_p2 = chemical_features_p2
        self.y_p1 = y_p1
        self.y_p2 = y_p2
        self.normals_p1 = normals_p1
        self.normals_p2 = normals_p2
        self.center_location_p1 = center_location_p1
        self.center_location_p2 = center_location_p2
        self.atom_coords_p1 = atom_coords_p1
        self.atom_coords_p2 = atom_coords_p2
        self.atom_types_p1 = atom_types_p1
        self.atom_types_p2 = atom_types_p2
        self.atom_center1 = atom_center1
        self.atom_center2 = atom_center2
        self.rand_rot1 = rand_rot1
        self.rand_rot2 = rand_rot2

    def __inc__(self, key, value):
        if key == "face_p1":
            return self.xyz_p1.size(0)
        if key == "face_p2":
            return self.xyz_p2.size(0)
        else:
            return super(PairData, self).__inc__(key, value)

    def __cat_dim__(self, key, value):
        if ("index" in key) or ("face" in key):
            return 1
        else:
            return 0


def load_protein_pair(pdb_id, data_dir,single_pdb=False):
    """Loads a protein surface mesh and its features"""
    pspl = pdb_id.split("_")
    p1_id = pspl[0] + "_" + pspl[1]
    p2_id = pspl[0] + "_" + pspl[2]

    p1 = load_protein_npy(p1_id, data_dir, center=False,single_pdb=single_pdb)
    p2 = load_protein_npy(p2_id, data_dir, center=False,single_pdb=single_pdb)
    # pdist = ((p1['xyz'][:,None,:]-p2['xyz'][None,:,:])**2).sum(-1).sqrt()
    # pdist = pdist<2.0
    # y_p1 = (pdist.sum(1)>0).to(torch.float).reshape(-1,1)
    # y_p2 = (pdist.sum(0)>0).to(torch.float).reshape(-1,1)
    y_p1 = p1["y"]
    y_p2 = p2["y"]

    protein_pair_data = PairData(
        xyz_p1=p1["xyz"],
        xyz_p2=p2["xyz"],
        face_p1=p1["face"],
        face_p2=p2["face"],
        chemical_features_p1=p1["chemical_features"],
        chemical_features_p2=p2["chemical_features"],
        y_p1=y_p1,
        y_p2=y_p2,
        normals_p1=p1["normals"],
        normals_p2=p2["normals"],
        center_location_p1=p1["center_location"],
        center_location_p2=p2["center_location"],
        atom_coords_p1=p1["atom_coords"],
        atom_coords_p2=p2["atom_coords"],
        atom_types_p1=p1["atom_types"],
        atom_types_p2=p2["atom_types"],
    )
    return protein_pair_data


class ProteinPairsSurfaces(InMemoryDataset):
    url = ""

    def __init__(self, root, ppi=False, train=True, transform=None, pre_transform=None):
        self.ppi = ppi
        super(ProteinPairsSurfaces, self).__init__(root, transform, pre_transform)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return "masif_site_masif_search_pdbs_and_ply_files.tar.gz"

    @property
    def processed_file_names(self):
        if not self.ppi:
            file_names = [
                "training_pairs_data.pt",
                "testing_pairs_data.pt",
                "training_pairs_data_ids.npy",
                "testing_pairs_data_ids.npy",
            ]
        else:
            file_names = [
                "training_pairs_data_ppi.pt",
                "testing_pairs_data_ppi.pt",
                "training_pairs_data_ids_ppi.npy",
                "testing_pairs_data_ids_ppi.npy",
            ]
        return file_names

    def download(self):
        url = 'https://zenodo.org/record/2625420/files/masif_site_masif_search_pdbs_and_ply_files.tar.gz'
        target_path = self.raw_paths[0]
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(target_path, 'wb') as f:
                f.write(response.raw.read())
                
        #raise RuntimeError(
        #    "Dataset not found. Please download {} from {} and move it to {}".format(
        #        self.raw_file_names, self.url, self.raw_dir
        #    )
        #)

    def process(self):
        pdb_dir = Path(self.root) / "raw" / "01-benchmark_pdbs"
        surf_dir = Path(self.root) / "raw" / "01-benchmark_surfaces"
        protein_dir = Path(self.root) / "raw" / "01-benchmark_surfaces_npy"
        lists_dir = Path('./lists')

        # Untar surface files
        if not (pdb_dir.exists() and surf_dir.exists()):
            tar = tarfile.open(self.raw_paths[0])
            tar.extractall(self.raw_dir)
            tar.close()

        if not protein_dir.exists():
            protein_dir.mkdir(parents=False, exist_ok=False)
            convert_plys(surf_dir,protein_dir)
            convert_pdbs(pdb_dir,protein_dir)

        with open(lists_dir / "training.txt") as f_tr, open(
            lists_dir / "testing.txt"
        ) as f_ts:
            training_list = sorted(f_tr.read().splitlines())
            testing_list = sorted(f_ts.read().splitlines())

        with open(lists_dir / "training_ppi.txt") as f_tr, open(
            lists_dir / "testing_ppi.txt"
        ) as f_ts:
            training_pairs_list = sorted(f_tr.read().splitlines())
            testing_pairs_list = sorted(f_ts.read().splitlines())
            pairs_list = sorted(training_pairs_list + testing_pairs_list)

        if not self.ppi:
            training_pairs_list = []
            for p in pairs_list:
                pspl = p.split("_")
                p1 = pspl[0] + "_" + pspl[1]
                p2 = pspl[0] + "_" + pspl[2]

                if p1 in training_list:
                    training_pairs_list.append(p)
                if p2 in training_list:
                    training_pairs_list.append(pspl[0] + "_" + pspl[2] + "_" + pspl[1])

            testing_pairs_list = []
            for p in pairs_list:
                pspl = p.split("_")
                p1 = pspl[0] + "_" + pspl[1]
                p2 = pspl[0] + "_" + pspl[2]
                if p1 in testing_list:
                    testing_pairs_list.append(p)
                if p2 in testing_list:
                    testing_pairs_list.append(pspl[0] + "_" + pspl[2] + "_" + pspl[1])

        # # Read data into huge `Data` list.
        training_pairs_data = []
        training_pairs_data_ids = []
        for p in training_pairs_list:
            try:
                protein_pair = load_protein_pair(p, protein_dir)
            except FileNotFoundError:
                continue
            training_pairs_data.append(protein_pair)
            training_pairs_data_ids.append(p)

        testing_pairs_data = []
        testing_pairs_data_ids = []
        for p in testing_pairs_list:
            try:
                protein_pair = load_protein_pair(p, protein_dir)
            except FileNotFoundError:
                continue
            testing_pairs_data.append(protein_pair)
            testing_pairs_data_ids.append(p)

        if self.pre_filter is not None:
            training_pairs_data = [
                data for data in training_pairs_data if self.pre_filter(data)
            ]
            testing_pairs_data = [
                data for data in testing_pairs_data if self.pre_filter(data)
            ]

        if self.pre_transform is not None:
            training_pairs_data = [
                self.pre_transform(data) for data in training_pairs_data
            ]
            testing_pairs_data = [
                self.pre_transform(data) for data in testing_pairs_data
            ]

        training_pairs_data, training_pairs_slices = self.collate(training_pairs_data)
        torch.save(
            (training_pairs_data, training_pairs_slices), self.processed_paths[0]
        )
        np.save(self.processed_paths[2], training_pairs_data_ids)
        testing_pairs_data, testing_pairs_slices = self.collate(testing_pairs_data)
        torch.save((testing_pairs_data, testing_pairs_slices), self.processed_paths[1])
        np.save(self.processed_paths[3], testing_pairs_data_ids)
