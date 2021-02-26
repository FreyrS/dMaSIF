import torch
from typing import Optional
from pykeops.torch import LazyTensor
from torch_geometric.nn import EdgeConv, Reshape

from torch_cluster import knn

from math import ceil
from torch_geometric.nn.inits import reset

from torch.nn import ELU, Conv1d
from torch.nn import Sequential as S, Linear as L, BatchNorm1d as BN


def ranges_slices(batch):
    """Helper function for the diagonal ranges function."""
    Ns = batch.bincount()
    indices = Ns.cumsum(0)
    ranges = torch.cat((0 * indices[:1], indices))
    ranges = (
        torch.stack((ranges[:-1], ranges[1:])).t().int().contiguous().to(batch.device)
    )
    slices = (1 + torch.arange(len(Ns))).int().to(batch.device)

    return ranges, slices


def diagonal_ranges(batch_x=None, batch_y=None):
    """Encodes the block-diagonal structure associated to a batch vector."""

    if batch_x is None and batch_y is None:
        return None

    ranges_x, slices_x = ranges_slices(batch_x)
    ranges_y, slices_y = ranges_slices(batch_y)

    return ranges_x, slices_x, ranges_y, ranges_y, slices_y, ranges_x


@torch.jit.ignore
def keops_knn(
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
    batch_x: Optional[torch.Tensor] = None,
    batch_y: Optional[torch.Tensor] = None,
    cosine: bool = False,
) -> torch.Tensor:
    r"""Straightforward modification of PyTorch_geometric's knn method."""

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

    y_i = LazyTensor(y[:, None, :])
    x_j = LazyTensor(x[None, :, :])

    if cosine:
        D_ij = -(y_i | x_j)
    else:
        D_ij = ((y_i - x_j) ** 2).sum(-1)

    D_ij.ranges = diagonal_ranges(batch_y, batch_x)
    idy = D_ij.argKmin(k, dim=1)  # (N, K)

    rows = torch.arange(k * len(y), device=idy.device) // k

    return torch.stack([rows, idy.view(-1)], dim=0)


knns = {"torch": knn, "keops": keops_knn}


@torch.jit.ignore
def knn_graph(
    x: torch.Tensor,
    k: int,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
    flow: str = "source_to_target",
    cosine: bool = False,
    target: Optional[torch.Tensor] = None,
    batch_target: Optional[torch.Tensor] = None,
    backend: str = "torch",
) -> torch.Tensor:
    r"""Straightforward modification of PyTorch_geometric's knn_graph method to allow for source/targets."""

    assert flow in ["source_to_target", "target_to_source"]
    if target is None:
        target = x
    if batch_target is None:
        batch_target = batch

    row, col = knns[backend](
        x, target, k if loop else k + 1, batch, batch_target, cosine=cosine
    )
    row, col = (col, row) if flow == "source_to_target" else (row, col)
    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0)


class MyDynamicEdgeConv(EdgeConv):
    r"""Straightforward modification of PyTorch_geometric's DynamicEdgeConv layer."""

    def __init__(self, nn, k, aggr="max", **kwargs):
        super(MyDynamicEdgeConv, self).__init__(nn=nn, aggr=aggr, **kwargs)
        self.k = k

    def forward(self, x, batch=None):
        """"""
        edge_index = knn_graph(
            x, self.k, batch, loop=False, flow=self.flow, backend="keops"
        )
        return super(MyDynamicEdgeConv, self).forward(x, edge_index)

    def __repr__(self):
        return "{}(nn={}, k={})".format(self.__class__.__name__, self.nn, self.k)


class MyXConv(torch.nn.Module):
    def __init__(
        self,
        in_channels=None,
        out_channels=None,
        dim=None,
        kernel_size=None,
        hidden_channels=None,
        dilation=1,
        bias=True,
        backend="torch",
    ):
        super(MyXConv, self).__init__()

        self.in_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels // 4
        if hidden_channels == 0:
            hidden_channels = 1

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.backend = backend

        C_in, C_delta, C_out = in_channels, hidden_channels, out_channels
        D, K = dim, kernel_size

        self.mlp1 = S(
            L(dim, C_delta),
            ELU(),
            BN(C_delta),
            L(C_delta, C_delta),
            ELU(),
            BN(C_delta),
            Reshape(-1, K, C_delta),
        )

        self.mlp2 = S(
            L(D * K, K ** 2),
            ELU(),
            BN(K ** 2),
            Reshape(-1, K, K),
            Conv1d(K, K ** 2, K, groups=K),
            ELU(),
            BN(K ** 2),
            Reshape(-1, K, K),
            Conv1d(K, K ** 2, K, groups=K),
            BN(K ** 2),
            Reshape(-1, K, K),
        )

        C_in = C_in + C_delta
        depth_multiplier = int(ceil(C_out / C_in))
        self.conv = S(
            Conv1d(C_in, C_in * depth_multiplier, K, groups=C_in),
            Reshape(-1, C_in * depth_multiplier),
            L(C_in * depth_multiplier, C_out, bias=bias),
        )

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp1)
        reset(self.mlp2)
        reset(self.conv)

    def forward(self, x, source, batch_source, target, batch_target):
        """"""
        # Load data shapes:
        # pos = pos.unsqueeze(-1) if pos.dim() == 1 else pos
        (Nin, _), (N, D), K = source.size(), target.size(), self.kernel_size

        # Compute K-nn:
        row, col = knn_graph(
            source,
            K * self.dilation,
            batch_source,
            loop=True,
            flow="target_to_source",
            target=target,
            batch_target=batch_target,
            backend=self.backend,
        )
        # row is a vector of size N*K*dilation that indexes "target"
        # col is a vector of size N*K*dilation that indexes "source"

        # If needed, sup-sample the K-NN graph:
        if self.dilation > 1:
            dil = self.dilation
            index = torch.randint(
                K * dil,
                (N, K),
                dtype=torch.long,
                layout=torch.strided,
                device=row.device,
            )
            arange = torch.arange(N, dtype=torch.long, device=row.device)
            arange = arange * (K * dil)
            index = (index + arange.view(-1, 1)).view(-1)  # (N*K,)
            row, col = row[index], col[index]

        # assert row.max() < N
        # assert col.max() < Nin

        # Line 1: local difference vector:
        pos = source[col] - target[row]  # (N * K, D)

        # Line 2: compute F_delta
        x_star = self.mlp1(pos.view(N * K, D))

        # Line 3: concatenate the features and reshape:
        if x is not None:
            x = x.unsqueeze(-1) if x.dim() == 1 else x
            x = x[col].view(N, K, self.in_channels)
            x_star = torch.cat([x_star, x], dim=-1)
        x_star = x_star.transpose(1, 2).contiguous()
        x_star = x_star.view(N, self.in_channels + self.hidden_channels, K, 1)

        # Line 4: Compute the transformation matrix:
        transform_matrix = self.mlp2(pos.view(N, K * D))
        transform_matrix = transform_matrix.view(N, 1, K, K)

        # Line 5: Apply it to the neighborhood:
        x_transformed = torch.matmul(transform_matrix, x_star)
        x_transformed = x_transformed.view(N, -1, K)  # (N, I+H, K)

        # Line 6: Apply the convolution filter:
        out = self.conv(x_transformed)  # (N, Cout)

        return out

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )
