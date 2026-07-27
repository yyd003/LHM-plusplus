"""Run small GPU operations through the pinned compiled extensions."""

from __future__ import annotations

import torch
import torch_scatter
from pytorch3d.ops import knn_points

import spconv.pytorch as spconv


def probe_torch_scatter() -> None:
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    index = torch.tensor([0, 1], device="cuda")
    output = torch_scatter.scatter_add(source, index, dim=0)
    torch.cuda.synchronize()
    assert torch.equal(output, source)
    print(f"torch-scatter {torch_scatter.__version__}: PASS")


def probe_pytorch3d() -> None:
    points_a = torch.randn(2, 128, 3, device="cuda")
    points_b = torch.randn(2, 256, 3, device="cuda")
    result = knn_points(points_a, points_b, K=4)
    torch.cuda.synchronize()
    assert result.dists.shape == (2, 128, 4)
    print("PyTorch3D CUDA KNN: PASS")


def probe_spconv() -> None:
    features = torch.randn(32, 8, device="cuda")
    indices = torch.zeros((32, 4), device="cuda", dtype=torch.int32)
    linear_index = torch.arange(32, device="cuda", dtype=torch.int32)
    indices[:, 1] = linear_index % 8
    indices[:, 2] = (linear_index // 8) % 4
    sparse_tensor = spconv.SparseConvTensor(features, indices, [8, 4, 1], 1)
    layer = spconv.SubMConv3d(8, 16, 3, padding=1, bias=False).cuda()
    with torch.inference_mode():
        output = layer(sparse_tensor)
    torch.cuda.synchronize()
    assert output.features.shape == (32, 16)
    print("spconv CUDA SubMConv3d: PASS")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the extension probe")

    probe_torch_scatter()
    probe_pytorch3d()
    probe_spconv()


if __name__ == "__main__":
    main()
