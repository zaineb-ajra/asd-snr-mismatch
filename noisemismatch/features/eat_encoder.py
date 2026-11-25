"""EAT encoder (inference only), partly vendored from the EAT model code.

Source: ``model_core.py`` and ``eat_model.py`` of the Hugging Face repository
worstchan/EAT-base_epoch30_pretrain at revision
6eb95936bf1061cacee1c0b89404a1aca9a3c62a (code of
https://github.com/cwx-worst-one/EAT, MIT License, Copyright (c) 2024 Wenxi
Chen). The vendored parts, the attention, block and positional-encoder modules
(``AltAttention``, ``AltBlock``, ``FixedPositionalEncoder``) and the encoder's
forward pass, come in the upstream code from data2vec 2.0 (fairseq, MIT License,
Copyright (c) Facebook, Inc. and its affiliates). Both licenses are the MIT
License; its terms, which apply to these parts:

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Not vendored: the upstream patch embedding (``PatchEmbed_new``) and sin-cos
positional functions come from AudioMAE / MAE (CC BY-NC 4.0, non-commercial);
the patch projection is written here from its function, and the positional table
is read from the checkpoint instead of being computed. The ``timm`` helpers are
replaced by plain PyTorch (``Mlp``: Linear, exact GELU, Linear).

Changes, none of which alters the computation at inference with the pre-trained
configuration (``layer_norm_first=False``, all dropout and drop-path rates 0):
the ``transformers`` wrapper and the fine-tuning head are removed; dropout,
drop-path, padding-mask, ALiBi and cosine-attention branches are removed. Tensor
names match the checkpoint, which is loaded with ``strict=True``.
"""

from __future__ import annotations

import torch
from torch import nn

EMBED_DIM: int = 768
DEPTH: int = 12
NUM_HEADS: int = 12
PATCH: int = 16
MAX_LENGTH: int = 768  # time patches covered by the positional table
MEL_BINS: int = 128
NORM_EPS: float = 1e-6


class PatchProjection(nn.Module):
    """Linear projection of non-overlapping 16 x 16 patches (strided convolution).

    Written for this repository from the function of the upstream patch
    embedding; the upstream class (``PatchEmbed_new``, from AudioMAE, CC BY-NC
    4.0) is not vendored. Tokens are ordered time-major (index ``t * 8 + f``), the
    order of the checkpoint's positional table.
    """

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(1, EMBED_DIM, kernel_size=PATCH, stride=PATCH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, T, F) -> (B, T/16 * F/16, D)."""
        y = self.proj(x)  # (B, D, T/16, F/16)
        return y.reshape(y.shape[0], EMBED_DIM, -1).permute(0, 2, 1)


class FixedPositionalEncoder(nn.Module):
    """Fixed 2-D sin-cos positional table, loaded from the checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        n = MAX_LENGTH * (MEL_BINS // PATCH)
        self.positions = nn.Parameter(torch.zeros(1, n, EMBED_DIM), requires_grad=False)


class Mlp(nn.Module):
    """``timm.layers.Mlp`` without dropout: Linear, GELU (exact), Linear."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the two layers."""
        return self.fc2(self.act(self.fc1(x)))


class AltAttention(nn.Module):
    """Multi-head self-attention with an explicit float32 softmax."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention over the token sequence ``(B, N, C)``."""
        b, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        dtype = q.dtype
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1, dtype=torch.float32).to(dtype=dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class AltBlock(nn.Module):
    """Transformer block, post-norm variant (``layer_norm_first=False``)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=NORM_EPS)
        self.attn = AltAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=NORM_EPS)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attention, norm, MLP, norm (residual connections as the original)."""
        x = x + self.attn(x)
        r = x = self.norm1(x)
        x = self.mlp(x)
        return self.norm2(r + x)


class EATEncoder(nn.Module):
    """EAT-base encoder: returns the last-layer tokens (CLS first)."""

    def __init__(self) -> None:
        super().__init__()
        self.local_encoder = PatchProjection()
        self.extra_tokens = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
        self.fixed_positional_encoder = FixedPositionalEncoder()
        self.blocks = nn.ModuleList(
            AltBlock(EMBED_DIM, NUM_HEADS) for _ in range(DEPTH)
        )
        self.pre_norm = nn.LayerNorm(EMBED_DIM, eps=NORM_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes normalized filterbanks.

        Args:
            x: Tensor shaped ``(B, 1, T, 128)``, T a multiple of 16.

        Returns:
            Tensor shaped ``(B, 1 + T/16 * 8, 768)``: the CLS token, then the patch
            tokens in time-major order (index ``t * 8 + f``).
        """
        b = x.shape[0]
        x = self.local_encoder(x)
        x = x + self.fixed_positional_encoder.positions[:, : x.size(1), :]
        x = torch.cat((self.extra_tokens.expand(b, -1, -1), x), dim=1)
        x = self.pre_norm(x)
        for block in self.blocks:
            x = block(x)
        return x


def load_encoder(checkpoint: str, device: torch.device) -> EATEncoder:
    """Builds the encoder and loads a ``model.safetensors`` checkpoint (strict).

    Args:
        checkpoint: Path of the safetensors file of the Hugging Face repository.
        device: Torch device.

    Returns:
        The encoder in evaluation mode.
    """
    from safetensors.torch import load_file

    state = load_file(checkpoint)
    prefix = "model."
    if not all(k.startswith(prefix) for k in state):
        raise ValueError(f"{checkpoint}: unexpected tensor names")
    encoder = EATEncoder()
    encoder.load_state_dict(
        {k[len(prefix) :]: v for k, v in state.items()}, strict=True
    )
    return encoder.to(device).eval()
