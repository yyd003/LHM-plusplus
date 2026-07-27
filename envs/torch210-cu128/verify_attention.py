"""Probe the attention kernels required by the PyTorch 2.10 LHM++ branch."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from xformers.ops import memory_efficient_attention


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def probe_xformers(dtype: torch.dtype) -> None:
    query = torch.randn(1, 1024, 16, 64, device="cuda", dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    with torch.inference_mode():
        output = memory_efficient_attention(query, key, value)
    synchronize()
    print(f"xFormers {dtype}: PASS {tuple(output.shape)}")


def probe_flash_sdpa(dtype: torch.dtype) -> None:
    query = torch.randn(1, 16, 1024, 64, device="cuda", dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    with torch.inference_mode(), sdpa_kernel(
        backends=[SDPBackend.FLASH_ATTENTION]
    ):
        output = F.scaled_dot_product_attention(query, key, value)
    synchronize()
    print(f"PyTorch Flash SDPA {dtype}: PASS {tuple(output.shape)}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Blackwell attention probe")

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Compute capability: {torch.cuda.get_device_capability()}")

    for dtype in (torch.float16, torch.bfloat16):
        probe_xformers(dtype)
        probe_flash_sdpa(dtype)


if __name__ == "__main__":
    main()
