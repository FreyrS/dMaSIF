import numpy as np
from pathlib import Path
from tqdm import tqdm
from plyfile import PlyData, PlyElement


def load_surface_np(fname, center):
    """Loads a .ply mesh to return a point cloud and connectivity."""

    # Load the data, and read the connectivity information:
    plydata = PlyData.read(str(fname))
    triangles = np.vstack(plydata["face"].data["vertex_indices"])

    # Normalize the point cloud, as specified by the user:
    points = np.vstack([[v[0], v[1], v[2]] for v in plydata["vertex"]])
    if center:
        points = points - np.mean(points, axis=0, keepdims=True)

    nx = plydata["vertex"]["nx"]
    ny = plydata["vertex"]["ny"]
    nz = plydata["vertex"]["nz"]
    normals = np.stack([nx, ny, nz]).T

    # Interface labels
    iface_labels = plydata["vertex"]["iface"]

    # Features
    charge = plydata["vertex"]["charge"]
    hbond = plydata["vertex"]["hbond"]
    hphob = plydata["vertex"]["hphob"]
    features = np.stack([charge, hbond, hphob]).T

    return {
        "xyz": points,
        "triangles": triangles,
        "features": features,
        "iface_labels": iface_labels,
        "normals": normals,
    }


def convert_plys(ply_dir, npy_dir):
    print("Converting PLYs")
    for p in tqdm(ply_dir.glob("*.ply")):
        protein = load_surface_np(p, center=False)
        np.save(npy_dir / (p.stem + "_xyz.npy"), protein["xyz"])
        np.save(npy_dir / (p.stem + "_triangles.npy"), protein["triangles"])
        np.save(npy_dir / (p.stem + "_features.npy"), protein["features"])
        np.save(npy_dir / (p.stem + "_iface_labels.npy"), protein["iface_labels"])
        np.save(npy_dir / (p.stem + "_normals.npy"), protein["normals"])

