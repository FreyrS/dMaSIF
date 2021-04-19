# Arne Schneuing 2021 LPDI EPFL

import os
import numpy as np
from pymol import cmd, stored
from pymol.cgo import *


def bwr_gradient(vals):
    """ Blue-white-red gradient """
    max = np.max(vals)
    min = np.min(vals)

    # normalize values
    vals -= (max + min) / 2
    if min >= max:
        vals *= 0
    else:
        vals *= 2 / (max - min)

    blue_vals = np.copy(vals)
    blue_vals[vals >= 0] = 0
    blue_vals = abs(blue_vals)
    red_vals = vals
    red_vals[vals < 0] = 0
    r = 1.0 - blue_vals
    g = 1.0 - (blue_vals + red_vals)
    b = 1.0 - red_vals
    return np.stack(([COLOR] * len(vals), r, g, b))


class MyVTK:
    def __init__(self, filename, n_feat = 26):
        with open(filename, "r") as f:
            lines = f.readlines()

        li = 0

        # Read header
        while True:
            if lines[li].startswith('POINTS'):
                self.len = int(lines[li].split(" ")[1])
                li += 1
                break

            # skip unimportant lines
            li += 1

        # Read vertex coordinates
        self.vertices = np.zeros((self.len, 3))
        for i in range(self.len):
            self.vertices[i, :] = [float(x) for x in lines[li].split(" ")]
            li += 1

        # Skip indices
        assert lines[li].startswith("VERTICES")
        li += 3

        # Read features
        self.features = np.zeros((self.len, n_feat))
        self.feat_names = [""] * n_feat

        for i in range(n_feat):
            assert li + 2 < len(lines)

            self.feat_names[i] = lines[li].split(" ")[1]
            li += 2
            self.features[:, i] = [float(x) for x in lines[li].split(" ")]
            li += 1

    def __len__(self):
        return self.len

    def get_vertices(self):
        return self.vertices

    def get_features(self):
        return self.features

    def get_feature_names(self):
        return self.feat_names


def send2pymol(verts, feat, feat_names, basename, dotSize=0.3):
    # Draw vertices
    group_names = ""
    for j in range(feat.shape[1]):
        colors = bwr_gradient(feat[:, j])
        spheres = np.stack(([SPHERE] * len(verts), verts[:, 0], verts[:, 1], 
                            verts[:, 2], [dotSize] * len(verts)))
        obj = np.empty(colors.size + spheres.size)
        obj[0::9] = colors[0, :]
        obj[1::9] = colors[1, :]
        obj[2::9] = colors[2, :]
        obj[3::9] = colors[3, :]
        obj[4::9] = spheres[0, :]
        obj[5::9] = spheres[1, :]
        obj[6::9] = spheres[2, :]
        obj[7::9] = spheres[3, :]
        obj[8::9] = spheres[4, :]

        name = feat_names[j] + "_" + basename
        cmd.load_cgo(obj, name)

        if j > 0:
            group_names += " "
        group_names += name

    # group the resulting objects
    cmd.group(basename, group_names)


def load_vtk(filename):
    basename = os.path.splitext(os.path.basename(filename))[0]

    vtk = MyVTK(filename)
    verts = vtk.get_vertices()
    feat = vtk.get_features()
    feat_names = vtk.get_feature_names()

    send2pymol(verts, feat, feat_names, basename)


def load_pred(directory, pdb_id):
    verts = np.load(os.path.join(directory, pdb_id + '_predcoords.npy'))
    feat = np.load(os.path.join(directory, pdb_id + '_predfeatures.npy'))
    feat_names = [f"feat{x}" for x in range(len(verts))]

    send2pymol(verts, feat, feat_names, pdb_id + '_pred')


# ------------------------------------------------------------------------------
cmd.extend("loadvtk", load_vtk)
cmd.extend("loadpred", load_pred)
