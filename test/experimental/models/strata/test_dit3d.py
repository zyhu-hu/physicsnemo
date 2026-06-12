# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import pytest
import torch

from physicsnemo.core.module import Module
from physicsnemo.experimental.models.strata import DiT3D, PixelDiT
from physicsnemo.experimental.models.strata.layers import Natten3DSelfAttention
from physicsnemo.experimental.models.strata.pixel import PixelDiTBlock
from test.common import validate_checkpoint
from test.conftest import requires_module

_DATA = Path(__file__).parent / "data"


def _make_pos(b: int, h: int, w: int) -> torch.Tensor:
    """Deterministic (B, 2, H, W) latitude / longitude grid in radians."""
    lat = torch.linspace(-1.0, 1.0, h).reshape(1, h, 1).expand(b, h, w)
    lon = torch.linspace(0.0, 1.5, w).reshape(1, 1, w).expand(b, h, w)
    return torch.stack([lat, lon], dim=1).contiguous()


def _seed_params(model: torch.nn.Module, seed: int) -> torch.nn.Module:
    """Fill all parameters with reproducible random values (MOD-008b pattern).

    DiT3D zero-initializes its output head, so a freshly constructed model
    produces all-zero outputs; seeding the parameters gives meaningful,
    reproducible forward outputs for non-regression and checkpoint tests.
    """
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=gen, dtype=param.dtype))
    return model


# --------------------------------------------------------------------------- #
# Non-regression fixtures. Each builder returns (model, args). Goldens store
# {args, state_dict, y}; the test loads state_dict + args and compares y, so it
# is independent of init/RNG changes across PyTorch versions. All use the CPU-
# reproducible full-attention path (attn_kernel=-1).
# --------------------------------------------------------------------------- #
def _build_dit3d_axial():
    model = DiT3D(
        in_channels=4,
        input_shape=(4, 8, 8),
        patch_size=(1, 2, 2),
        embed_dim=32,
        num_heads=4,
        num_layers=3,
        attn_kernel=-1,
        do_alt_depthwise_attn=True,
        gated_attention=True,
        rope_mode="axial",
    )
    gen = torch.Generator().manual_seed(11)
    return _seed_params(model, seed=10), (torch.randn(2, 4, 4, 8, 8, generator=gen),)


def _build_dit3d_stereo():
    model = DiT3D(
        in_channels=3,
        input_shape=(6, 8, 8),
        patch_size=(1, 2, 2),
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        attn_kernel=-1,
        qk_norm=True,
        rope_mode="stereographic",
    )
    gen = torch.Generator().manual_seed(21)
    return _seed_params(model, seed=20), (
        torch.randn(2, 3, 6, 8, 8, generator=gen),
        _make_pos(2, 8, 8),
    )


def _build_pixeldit_pixelproj():
    model = PixelDiT(
        semantic_config=dict(
            in_channels=4,
            input_shape=(4, 8, 8),
            patch_size=(1, 2, 2),
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            attn_kernel=-1,
        ),
        embed_dim_pixel=16,
        num_layers_pixel=2,
        num_heads_pixel=2,
        attn_kernel_pixel=-1,
        adaln_mode="pixel_proj",
    )
    gen = torch.Generator().manual_seed(31)
    return _seed_params(model, seed=30), (torch.randn(2, 4, 4, 8, 8, generator=gen),)


def _build_pixeldit_bilinear():
    model = PixelDiT(
        semantic_config=dict(
            in_channels=4,
            input_shape=(4, 8, 8),
            patch_size=(1, 2, 2),
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            attn_kernel=-1,
        ),
        embed_dim_pixel=16,
        num_layers_pixel=3,
        num_heads_pixel=2,
        attn_kernel_pixel=-1,
        adaln_mode="bilinear_dw",
        first_block_only_adaln=True,
    )
    gen = torch.Generator().manual_seed(41)
    return _seed_params(model, seed=40), (torch.randn(2, 4, 4, 8, 8, generator=gen),)


# name -> (builder, golden path). Drives the non-regression test and the golden
# generator (data/_generate_dit3d_goldens.py).
_FIXTURE_REGISTRY = [
    ("dit3d_axial", _build_dit3d_axial, _DATA / "dit3d_axial.pth"),
    ("dit3d_stereo", _build_dit3d_stereo, _DATA / "dit3d_stereo.pth"),
    ("pixeldit_pixelproj", _build_pixeldit_pixelproj, _DATA / "pixeldit_pixelproj.pth"),
    ("pixeldit_bilinear", _build_pixeldit_bilinear, _DATA / "pixeldit_bilinear.pth"),
]


# --------------------------------------------------------------------------- #
# Constructor / attribute tests (MOD-008a)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "config", ["default", "custom"], ids=["with_defaults", "with_custom_args"]
)
def test_dit3d_constructor(config):
    """DiT3D constructor and public attributes."""
    if config == "default":
        model = DiT3D(in_channels=4)
        assert model.out_channels == 4  # defaults to in_channels
        assert model.embed_dim == 768
        assert model.num_heads == 8
        assert model.num_layers == 12
        assert model.patch_size == (1, 1, 1)
        assert model.rope_mode == "none"
        assert model.input_shape == (16, 64, 64)
    else:
        model = DiT3D(
            in_channels=3,
            out_channels=5,
            input_shape=(4, 8, 8),
            patch_size=(1, 2, 2),
            embed_dim=64,
            num_heads=8,
            num_layers=4,
            attn_kernel=-1,
            rope_mode="stereographic",
        )
        assert model.out_channels == 5
        assert model.embed_dim == 64
        assert model.num_layers == 4
        assert model.patch_size == (1, 2, 2)
        assert model.rope_mode == "stereographic"

    assert isinstance(model, Module), "DiT3D should inherit physicsnemo.Module"
    assert hasattr(model, "meta")
    assert len(model.blocks) == model.num_layers
    assert (model.rope is None) == (model.rope_mode == "none")


@pytest.mark.parametrize(
    "config", ["default", "custom"], ids=["with_defaults", "with_custom_args"]
)
def test_pixeldit_constructor(config):
    """PixelDiT constructor and public attributes."""
    semantic_config = dict(
        in_channels=4,
        input_shape=(4, 8, 8),
        patch_size=(1, 2, 2),
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        attn_kernel=-1,
    )
    if config == "default":
        model = PixelDiT(semantic_config=semantic_config)
        assert model.embed_dim_pixel == 128
        assert model.num_layers_pixel == 4
        assert model.adaln_mode == "pixel_proj"
        assert model.first_block_only_adaln is False
        # All blocks inject conditioning when not first-block-only.
        assert all(isinstance(b, PixelDiTBlock) for b in model.pixel_blocks)
    else:
        model = PixelDiT(
            semantic_config=semantic_config,
            embed_dim_pixel=16,
            num_layers_pixel=3,
            num_heads_pixel=2,
            attn_kernel_pixel=-1,
            adaln_mode="bilinear_dw",
            first_block_only_adaln=True,
        )
        assert model.embed_dim_pixel == 16
        assert model.num_layers_pixel == 3
        assert model.adaln_mode == "bilinear_dw"
        # First-block-only: exactly one conditioning block, the rest plain.
        assert isinstance(model.pixel_blocks[0], PixelDiTBlock)
        assert sum(isinstance(b, PixelDiTBlock) for b in model.pixel_blocks) == 1

    assert isinstance(model, Module), "PixelDiT should inherit physicsnemo.Module"
    assert isinstance(model.semantic, DiT3D)
    # The semantic output head is dropped (only forward_tokens is used).
    assert not hasattr(model.semantic, "final_layer")
    assert len(model.pixel_blocks) == model.num_layers_pixel


def test_dit3d_invalid_args():
    """Constructor validation for incompatible arguments."""
    with pytest.raises(ValueError):  # embed_dim not divisible by num_heads
        DiT3D(in_channels=4, embed_dim=30, num_heads=4)
    with pytest.raises(ValueError):  # head_dim not divisible by 4 with RoPE
        DiT3D(in_channels=4, embed_dim=8, num_heads=4, rope_mode="axial")
    with pytest.raises(ValueError):  # bad rope_mode
        DiT3D(in_channels=4, rope_mode="bogus")


def test_pixeldit_bilinear_requires_vertical_patch_one():
    """bilinear_dw AdaLN requires the semantic vertical patch size to be 1."""
    with pytest.raises(ValueError):
        PixelDiT(
            semantic_config=dict(
                in_channels=4,
                input_shape=(4, 8, 8),
                patch_size=(2, 2, 2),
                embed_dim=32,
                num_heads=4,
                num_layers=1,
                attn_kernel=-1,
            ),
            embed_dim_pixel=16,
            num_heads_pixel=2,
            adaln_mode="bilinear_dw",
        )


# --------------------------------------------------------------------------- #
# Forward shape tests (CPU + CUDA via the SDPA path)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rope_mode", ["none", "axial", "stereographic"])
def test_dit3d_forward_shape(device, rope_mode):
    """DiT3D forward produces the correct output shape on the SDPA path."""
    torch.manual_seed(0)
    b, c, d, h, w = 2, 4, 4, 8, 8
    model = DiT3D(
        in_channels=c,
        input_shape=(d, h, w),
        patch_size=(1, 2, 2),
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        attn_kernel=-1,
        rope_mode=rope_mode,
    ).to(device)
    x = torch.randn(b, c, d, h, w, device=device)
    pos = _make_pos(b, h, w).to(device) if rope_mode == "stereographic" else None
    out = model(x, pos)
    assert out.shape == (b, c, d, h, w)


@pytest.mark.parametrize("adaln_mode", ["pixel_proj", "bilinear_dw"])
def test_pixeldit_forward_shape(device, adaln_mode):
    """PixelDiT forward produces the correct output shape on the SDPA path."""
    torch.manual_seed(0)
    b, c, d, h, w = 2, 4, 4, 8, 8
    model = PixelDiT(
        semantic_config=dict(
            in_channels=c,
            input_shape=(d, h, w),
            patch_size=(1, 2, 2),
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            attn_kernel=-1,
        ),
        embed_dim_pixel=16,
        num_layers_pixel=2,
        num_heads_pixel=2,
        attn_kernel_pixel=-1,
        adaln_mode=adaln_mode,
    ).to(device)
    x = torch.randn(b, c, d, h, w, device=device)
    assert model(x).shape == (b, c, d, h, w)


# --------------------------------------------------------------------------- #
# Non-regression tests (MOD-008b)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name,builder,golden",
    _FIXTURE_REGISTRY,
    ids=[n for n, _, _ in _FIXTURE_REGISTRY],
)
def test_non_regression(name, builder, golden):
    """Forward outputs match the committed golden (loaded params + inputs)."""
    if not golden.exists():
        pytest.skip(
            f"golden {golden.name} missing; run "
            f"test/experimental/models/strata/data/_generate_dit3d_goldens.py"
        )
    data = torch.load(golden)
    model, _ = builder()
    model.load_state_dict(data["state_dict"])
    model.eval()
    with torch.no_grad():
        y = model(*data["args"])
    assert torch.allclose(y, data["y"], atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------- #
# Checkpoint tests (MOD-008c)
# --------------------------------------------------------------------------- #
def test_dit3d_checkpoint(device):
    """DiT3D save/load/from_checkpoint reproduce the forward output."""
    torch.manual_seed(0)
    kwargs = dict(
        in_channels=4,
        input_shape=(4, 8, 8),
        patch_size=(1, 2, 2),
        embed_dim=32,
        num_heads=4,
        num_layers=3,
        attn_kernel=-1,
        do_alt_depthwise_attn=True,
        rope_mode="axial",
    )
    model_1 = _seed_params(DiT3D(**kwargs), seed=1).to(device)
    model_2 = _seed_params(DiT3D(**kwargs), seed=2).to(device)
    x = torch.randn(2, 4, 4, 8, 8, device=device)
    assert validate_checkpoint(model_1, model_2, (x,))


def test_pixeldit_checkpoint(device):
    """PixelDiT save/load/from_checkpoint reproduce the forward output."""
    torch.manual_seed(0)
    kwargs = dict(
        semantic_config=dict(
            in_channels=4,
            input_shape=(4, 8, 8),
            patch_size=(1, 2, 2),
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            attn_kernel=-1,
        ),
        embed_dim_pixel=16,
        num_layers_pixel=2,
        num_heads_pixel=2,
        attn_kernel_pixel=-1,
        adaln_mode="pixel_proj",
    )
    model_1 = _seed_params(PixelDiT(**kwargs), seed=1).to(device)
    model_2 = _seed_params(PixelDiT(**kwargs), seed=2).to(device)
    x = torch.randn(2, 4, 4, 8, 8, device=device)
    assert validate_checkpoint(model_1, model_2, (x,))


# --------------------------------------------------------------------------- #
# 3D neighborhood-attention (NATTEN) tests (CUDA + natten only)
# --------------------------------------------------------------------------- #
@requires_module(["natten"])
def test_dit3d_natten_forward(device):
    """DiT3D forward on the NA3D path (NATTEN is CUDA-only)."""
    if device == "cpu":
        pytest.skip("natten neighborhood attention is not available on CPU")
    torch.manual_seed(0)
    b, c, d, h, w = 2, 4, 4, 8, 8
    model = DiT3D(
        in_channels=c,
        input_shape=(d, h, w),
        patch_size=(1, 2, 2),
        embed_dim=32,
        num_heads=4,
        num_layers=3,
        attn_kernel=3,
        do_alt_depthwise_attn=True,
        gated_attention=True,
        rope_mode="stereographic",
    ).to(device)
    x = torch.randn(b, c, d, h, w, device=device)
    pos = _make_pos(b, h, w).to(device)
    out = model(x, pos)
    assert out.shape == (b, c, d, h, w)
    assert torch.isfinite(out).all()


@requires_module(["natten"])
def test_pixeldit_natten_forward(device):
    """PixelDiT forward on the NA3D path (NATTEN is CUDA-only)."""
    if device == "cpu":
        pytest.skip("natten neighborhood attention is not available on CPU")
    torch.manual_seed(0)
    b, c, d, h, w = 2, 4, 4, 8, 8
    model = PixelDiT(
        semantic_config=dict(
            in_channels=c,
            input_shape=(d, h, w),
            patch_size=(1, 2, 2),
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            attn_kernel=3,
        ),
        embed_dim_pixel=16,
        num_layers_pixel=2,
        num_heads_pixel=2,
        attn_kernel_pixel=3,
        adaln_mode="bilinear_dw",
    ).to(device)
    x = torch.randn(b, c, d, h, w, device=device)
    out = model(x)
    assert out.shape == (b, c, d, h, w)
    assert torch.isfinite(out).all()


def test_natten3d_attention_kernel_triple():
    """Natten3DSelfAttention accepts both int and per-axis tuple kernels."""
    attn_int = Natten3DSelfAttention(dim=32, num_heads=4, attn_kernel=3)
    assert attn_int.attn_kernel == 3
    attn_tuple = Natten3DSelfAttention(dim=32, num_heads=4, attn_kernel=(3, 5, 5))
    assert attn_tuple.attn_kernel == (3, 5, 5)
