"""Run small GPU operations through the pinned compiled extensions."""

from __future__ import annotations

import torch
import torch_scatter
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from flash_attn import flash_attn_varlen_qkvpacked_func
from gsplat.rendering import rasterization
from pytorch3d.ops import knn_points
from simple_knn._C import distCUDA2

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


def probe_diff_gaussian_rasterization() -> None:
    settings = GaussianRasterizationSettings(
        image_height=32,
        image_width=32,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=torch.zeros(3, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4, device="cuda"),
        projmatrix=torch.eye(4, device="cuda"),
        sh_degree=0,
        campos=torch.zeros(3, device="cuda"),
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )
    rasterizer = GaussianRasterizer(settings)
    means = torch.tensor([[0.0, 0.0, 0.5]], device="cuda")
    image, radii, _ = rasterizer(
        means,
        torch.zeros_like(means),
        torch.tensor([[0.9]], device="cuda"),
        colors_precomp=torch.tensor([[1.0, 0.2, 0.1]], device="cuda"),
        scales=torch.tensor([[0.15, 0.15, 0.15]], device="cuda"),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
    )
    torch.cuda.synchronize()
    assert image.shape == (3, 32, 32)
    assert radii.max() > 0
    assert torch.isfinite(image).all()
    print("diff-gaussian-rasterization CUDA: PASS")


def probe_gsplat() -> None:
    render, alpha, metadata = rasterization(
        torch.tensor([[0.0, 0.0, 2.0]], device="cuda"),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
        torch.tensor([[0.1, 0.1, 0.1]], device="cuda"),
        torch.tensor([0.9], device="cuda"),
        torch.tensor([[1.0, 0.2, 0.1]], device="cuda"),
        torch.eye(4, device="cuda").unsqueeze(0),
        torch.tensor(
            [[[32.0, 0.0, 16.0], [0.0, 32.0, 16.0], [0.0, 0.0, 1.0]]],
            device="cuda",
        ),
        32,
        32,
        render_mode="RGB",
    )
    torch.cuda.synchronize()
    assert render.shape == (1, 32, 32, 3)
    assert alpha.shape == (1, 32, 32, 1)
    assert metadata["radii"].max() > 0
    assert torch.isfinite(render).all()
    print("gsplat CUDA rasterization: PASS")


def probe_simple_knn() -> None:
    distances = distCUDA2(torch.randn(2048, 3, device="cuda"))
    torch.cuda.synchronize()
    assert distances.shape == (2048,)
    assert torch.isfinite(distances).all()
    assert (distances >= 0).all()
    print("simple-knn CUDA distCUDA2: PASS")


def probe_flash_attention() -> None:
    lengths = [512, 768]
    total = sum(lengths)
    cumulative_lengths = torch.tensor(
        [0, lengths[0], total], device="cuda", dtype=torch.int32
    )
    for dtype in (torch.float16, torch.bfloat16):
        qkv = torch.randn(total, 3, 16, 64, device="cuda", dtype=dtype)
        with torch.inference_mode():
            output = flash_attn_varlen_qkvpacked_func(
                qkv,
                cumulative_lengths,
                max(lengths),
                dropout_p=0.0,
            )
        torch.cuda.synchronize()
        assert output.shape == (total, 16, 64)
        assert torch.isfinite(output).all()
        print(f"FlashAttention varlen {dtype}: PASS")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the extension probe")

    probe_torch_scatter()
    probe_pytorch3d()
    probe_spconv()
    probe_diff_gaussian_rasterization()
    probe_gsplat()
    probe_simple_knn()
    probe_flash_attention()


if __name__ == "__main__":
    main()
