import argparse

parser = argparse.ArgumentParser(description="Network parameters")

# Main parameters
parser.add_argument(
    "--experiment_name", type=str, help="Name of experiment", required=True
)
parser.add_argument(
    "--use_mesh", type=bool, default=False, help="Use precomputed surfaces"
)
parser.add_argument(
    "--embedding_layer",
    type=str,
    default="dMaSIF",
    choices=["dMaSIF", "DGCNN", "PointNet++"],
    help="Which convolutional embedding layer to use",
)
parser.add_argument("--profile", type=bool, default=False, help="Profile code")

# Geometric parameters
parser.add_argument(
    "--curvature_scales",
    type=list,
    default=[1.0, 2.0, 3.0, 5.0, 10.0],
    help="Scales at which we compute the geometric features (mean and Gauss curvatures)",
)
parser.add_argument(
    "--resolution",
    type=float,
    default=1.0,
    help="Resolution of the generated point cloud",
)
parser.add_argument(
    "--distance",
    type=float,
    default=1.05,
    help="Distance parameter in surface generation",
)
parser.add_argument(
    "--variance",
    type=float,
    default=0.1,
    help="Variance parameter in surface generation",
)
parser.add_argument(
    "--sup_sampling", type=int, default=20, help="Sup-sampling ratio around atoms"
)

# Hyper-parameters for the embedding
parser.add_argument(
    "--atom_dims",
    type=int,
    default=6,
    help="Number of atom types and dimension of resulting chemical features",
)
parser.add_argument(
    "--emb_dims",
    type=int,
    default=8,
    help="Number of input features (+ 3 xyz coordinates for DGCNNs)",
)
parser.add_argument(
    "--in_channels",
    type=int,
    default=16,
    help="Number of embedding dimensions",
)
parser.add_argument(
    "--orientation_units",
    type=int,
    default=16,
    help="Number of hidden units for the orientation score MLP",
)
parser.add_argument(
    "--unet_hidden_channels",
    type=int,
    default=8,
    help="Number of hidden units for TangentConv UNet",
)
parser.add_argument(
    "--post_units",
    type=int,
    default=8,
    help="Number of hidden units for the post-processing MLP",
)
parser.add_argument(
    "--n_layers", type=int, default=1, help="Number of convolutional layers"
)
parser.add_argument(
    "--radius", type=float, default=9.0, help="Radius to use for the convolution"
)
parser.add_argument(
    "--k",
    type=int,
    default=40,
    help="Number of nearset neighbours for DGCNN and PointNet++",
)
parser.add_argument(
    "--dropout",
    type=float,
    default=0.0,
    help="Amount of Dropout for the input features",
)

# Training
parser.add_argument(
    "--n_epochs", type=int, default=50, help="Number of training epochs"
)
parser.add_argument(
    "--batch_size", type=int, default=1, help="Number of proteins in a batch"
)
parser.add_argument(
    "--device", type=str, default="cuda:0", help="Which gpu/cpu to train on"
)
parser.add_argument(
    "--restart_training",
    type=str,
    default="",
    help="Which model to restart the training from",
)
parser.add_argument(
    "--n_rocauc_samples",
    type=int,
    default=100,
    help="Number of samples for the Matching ROC-AUC",
)
parser.add_argument(
    "--validation_fraction",
    type=float,
    default=0.1,
    help="Fraction of training dataset to use for validation",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument(
    "--random_rotation",
    type=bool,
    default=False,
    help="Move proteins to center and add random rotation",
)
parser.add_argument(
    "--single_protein",
    type=bool,
    default=False,
    help="Use single protein in a pair or both",
)
parser.add_argument("--site", type=bool, default=False, help="Predict interaction site")
parser.add_argument(
    "--search",
    type=bool,
    default=False,
    help="Predict matching between two partners",
)
parser.add_argument(
    "--no_chem", type=bool, default=False, help="Predict without chemical information"
)
parser.add_argument(
    "--no_geom", type=bool, default=False, help="Predict without curvature information"
)
parser.add_argument(
    "--single_pdb",
    type=str,
    default="",
    help="Which structure to do inference on",
)
parser.add_argument(
    "--pdb_list",
    type=str,
    default="",
    help="Which structures to do inference on",
)
