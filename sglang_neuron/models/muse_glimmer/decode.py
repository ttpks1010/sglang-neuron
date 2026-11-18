# SPDX-License-Identifier: Apache-2.0
"""Muse Glimmer's decode attention over the bf16 paged cache: nkilib's attention_block_tkg runs the QKV projection,
the weightless RMSNorm of each Q and K head, RoPE, attention over the row's pages and its own token, and the K/V write
in one call per layer, and returns the attended heads without the output projection (``W_out=None``), so the graph can
gate them (``sigmoid(x W_gate)``) before ``o_proj``. On a CPU the torch path computes the same attention.

Two things differ from the dense models' wrapper (ops/attention/attention_decode_paged.py):

- A sliding layer attends over the table the runtime trimmed to its window (graph_runner: ``block_table_tensor`` of
  the layer's metadata, with the trimmed table's first position in ``swa_kv_pos_offset``). The kernel masks by
  position in the table's own frame (slot ``j`` of block ``b`` is position ``b * page + j``), so the row's position is
  shifted by the offset, and the window start ``max(0, position - (W - 1))`` is passed as ``swa_start_pos_ids``: the
  kernel then attends to the prior positions in ``[start, position)`` and to the row's own token.
- ``max_context_len`` is not passed. On nkilib's flash-attention path (a prior above its 8K tile) it ends the tile loop
  early, and that early exit gives wrong attention past 8K tokens (sglang-neuron PR #4). Without it the kernel reads
  every tile of the graph's prior, which the decode context ladder bounds.
"""

from dataclasses import dataclass

import nki
import torch
from nkilib.core.utils.common_types import QuantizationType
from nkilib.experimental.transformer.attention_block_tkg import attention_block_tkg
from torch import Tensor

from sglang_neuron.backend.stack import wrap_nki
from sglang_neuron.utils.hardware_config import get_logical_nc_config
from sglang_neuron.utils.neuron_utils import can_run_kernel


@dataclass
class DecodeInputs:
    """A decode step's kernel inputs for the layers of one kind (full, or sliding over one window): ``table`` [B,
    kv_heads, blocks] folded block ids (-1 for a block that starts at or after the row's token, in the table's frame),
    ``update_idx`` the new token's folded slot row per KV head ([B, kv_heads, 1], or [B, 1] with one KV head),
    ``pos_ids`` [B, 1] fp32 the row's position in the table's frame, ``start_ids`` [B, 1] fp32 the window's first
    position in that frame (None for full attention) and ``active_mask`` [1, B, q_heads, 1]."""

    table: Tensor
    update_idx: Tensor
    pos_ids: Tensor
    start_ids: Tensor | None
    active_mask: Tensor


def decode_inputs(block_table: Tensor, slots: Tensor, positions: Tensor, *, kv_heads: int, q_heads: int, page: int,
                  window: int | None = None, offset: Tensor | None = None) -> DecodeInputs:
    """The kernel inputs of a decode step from the runtime's metadata: ``block_table`` [B, blocks] int32 (-1 past a
    row's pages), ``slots`` [B] the new token's slot, ``positions`` [B] its position. For a sliding layer, ``window``
    is W and ``offset`` [B] the trimmed table's first position (``swa_kv_pos_offset``), which the runtime keeps at or
    below the row's position. A padding row (slot 0, position 0, offset 0) writes SGLang's reserved page 0 and attends
    nothing before itself."""
    rows, blocks = block_table.shape
    device = block_table.device
    positions = positions.to(torch.int64)
    frame = positions if offset is None else positions - offset.reshape(rows).to(torch.int64)
    heads = torch.arange(kv_heads, device=device, dtype=torch.int64)
    folded = block_table.to(torch.int64)[:, None, :] * kv_heads + heads[None, :, None]
    starts = torch.arange(blocks, device=device, dtype=torch.int64) * page
    # a block holds prior keys while it starts before the row's token (in the table's frame); the kernel masks the
    # rest by position
    prior = starts[None, None, :] < frame[:, None, None]
    table = torch.where(prior, folded, torch.full_like(folded, -1)).to(torch.int32)
    slots = torch.clamp(slots.to(torch.int64), min=0)
    update = ((slots // page)[:, None] * kv_heads + heads[None, :]) * page + (slots % page)[:, None]
    start_ids = None
    if window is not None:
        start_ids = torch.clamp(frame - (window - 1), min=0).to(torch.float32).view(rows, 1)
    return DecodeInputs(
        table=table,
        update_idx=(update.unsqueeze(-1) if kv_heads > 1 else update).to(torch.int32),
        pos_ids=frame.to(torch.float32).view(rows, 1),
        start_ids=start_ids,
        active_mask=torch.ones((1, rows, q_heads, 1), dtype=torch.uint8, device=device),
    )


@nki.jit
def _attention_block_tkg(X, W_qkv, q_norm, k_norm, norm_eps, cos, sin, active_blocks_table, K_cache, V_cache,
                         attention_mask, kv_cache_update_idx, pos_ids, swa_start_pos_ids, softmax_scale):
    return attention_block_tkg(
        X=X,
        X_hidden_dim_actual=None,
        rmsnorm_X_enabled=False,
        rmsnorm_X_eps=None,
        rmsnorm_X_gamma=None,
        W_qkv=W_qkv,
        bias_qkv=None,
        quantization_type_qkv=QuantizationType.NONE,
        weight_dequant_scale_qkv=None,
        input_dequant_scale_qkv=None,
        rmsnorm_QK_pre_rope_enabled=True,
        rmsnorm_QK_pre_rope_eps=norm_eps,
        rmsnorm_QK_pre_rope_W_Q=q_norm,
        rmsnorm_QK_pre_rope_W_K=k_norm,
        cos=cos,
        sin=sin,
        rope_contiguous_layout=True,
        rmsnorm_QK_post_rope_enabled=False,
        rmsnorm_QK_post_rope_eps=0.0,
        rmsnorm_QK_post_rope_W_Q=None,
        rmsnorm_QK_post_rope_W_K=None,
        K_cache_transposed=False,
        active_blocks_table=active_blocks_table,
        K_cache=K_cache,
        V_cache=V_cache,
        attention_mask=attention_mask,
        sink=None,
        update_cache=True,
        kv_cache_update_idx=kv_cache_update_idx,
        W_out=None,
        bias_out=None,
        quantization_type_out=QuantizationType.NONE,
        weight_dequant_scale_out=None,
        input_dequant_scale_out=None,
        transposed_out=False,
        out_in_sb=False,
        softmax_scale=softmax_scale,
        pos_ids=pos_ids,
        swa_start_pos_ids=swa_start_pos_ids,
    )


def attention_decode(x: Tensor, *, w_qkv: Tensor, q_norm: Tensor, k_norm: Tensor, norm_eps: float, cos: Tensor,
                     sin: Tensor, k_cache: Tensor, v_cache: Tensor, inputs: DecodeInputs, softmax_scale: float,
                     q_heads: int, head_dim: int) -> Tensor:
    """The attended heads of one decode layer, [B, q_heads * d], of x [B, 1, H] (already normed), before the gate and
    the output projection. ``w_qkv`` [H, (q + 2 kv) * d] packs Q, K and V on its output dim, ``q_norm`` / ``k_norm``
    [1, d] the per-head RMSNorm gammas (ones: the model's QK norm has no weight), cos/sin [d / 2, B, 1] the
    contiguous-halves RoPE tables (cos 1 and sin 0 for a layer without RoPE). The caches are written in place."""
    args = dict(W_qkv=w_qkv, q_norm=q_norm, k_norm=k_norm, norm_eps=norm_eps, cos=cos, sin=sin,
                active_blocks_table=inputs.table, K_cache=k_cache, V_cache=v_cache, attention_mask=inputs.active_mask,
                kv_cache_update_idx=inputs.update_idx, pos_ids=inputs.pos_ids, swa_start_pos_ids=inputs.start_ids,
                softmax_scale=softmax_scale)
    rows = x.shape[0]
    if not can_run_kernel(x):
        return _torch_attention_decode(x, q_heads=q_heads, head_dim=head_dim, **args)
    out, _, _ = wrap_nki(_attention_block_tkg)[get_logical_nc_config()](x, **args)
    # the kernel's [B, q_heads, d_head, S_tkg] without an output projection, S_tkg = 1
    return out.reshape(rows, q_heads * head_dim)


def _rms(x: Tensor, gamma: Tensor, eps: float) -> Tensor:
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * gamma.float().reshape(-1)).to(x.dtype)


def _torch_attention_decode(x, *, q_heads, head_dim, W_qkv, q_norm, k_norm, norm_eps, cos, sin, active_blocks_table,
                            K_cache, V_cache, attention_mask, kv_cache_update_idx, pos_ids, swa_start_pos_ids,
                            softmax_scale) -> Tensor:
    """The kernel's result by plain attention over every block the table lists, keys masked by position in the
    table's frame: a prior key at position p is visible when start <= p < pos (start 0 without a window), and the
    row's own token always. Static shapes and no ``.item()``, so it traces."""
    del attention_mask  # the reference masks by position over every listed block
    rows = x.shape[0]
    pages, kv_heads, page, d = K_cache.shape
    group = q_heads // kv_heads
    qkv = x.reshape(rows, -1) @ W_qkv
    q = qkv[:, :q_heads * d].view(rows, q_heads, d)
    k = qkv[:, q_heads * d:(q_heads + kv_heads) * d].view(rows, kv_heads, d)
    v = qkv[:, (q_heads + kv_heads) * d:].view(rows, kv_heads, d)
    q, k = _rms(q, q_norm, norm_eps), _rms(k, k_norm, norm_eps)
    c = cos.permute(1, 2, 0).reshape(rows, 1, d // 2).repeat(1, 1, 2).to(x.dtype)
    s = sin.permute(1, 2, 0).reshape(rows, 1, d // 2).repeat(1, 1, 2).to(x.dtype)

    def rope(t):
        return t * c + torch.cat((-t[..., d // 2:], t[..., :d // 2]), dim=-1) * s

    q, k = rope(q), rope(k)
    table = active_blocks_table.reshape(rows, kv_heads, -1).long()           # [B, kv, blocks]
    blocks = table.shape[-1]
    keys = K_cache.reshape(pages * kv_heads, page, d)[table.clamp(min=0)]      # [B, kv, blocks, page, d]
    values = V_cache.reshape(pages * kv_heads, page, d)[table.clamp(min=0)]
    keys = keys.reshape(rows, kv_heads, blocks * page, d).to(x.dtype)
    values = values.reshape(rows, kv_heads, blocks * page, d).to(x.dtype)
    key_pos = torch.arange(blocks * page, device=x.device)[None, None, :]
    visible = (table.repeat_interleave(page, dim=-1) >= 0) & (key_pos < pos_ids.reshape(rows, 1, 1))
    if swa_start_pos_ids is not None:
        visible &= key_pos >= swa_start_pos_ids.reshape(rows, 1, 1)
    keys = torch.cat([keys, k.unsqueeze(2)], dim=2)                            # the row's own token last
    values = torch.cat([values, v.unsqueeze(2)], dim=2)
    visible = torch.cat([visible, torch.ones(rows, kv_heads, 1, dtype=torch.bool, device=x.device)], dim=-1)
    q = q.view(rows, kv_heads, group, d)
    scores = torch.einsum("bhgd,bhsd->bhgs", q.float(), keys.float()) * softmax_scale
    scores = scores.masked_fill(~visible[:, :, None, :], float("-inf"))
    out = torch.einsum("bhgs,bhsd->bhgd", torch.softmax(scores, dim=-1), values.float()).to(x.dtype)
    index = kv_cache_update_idx.reshape(-1).long()
    K_cache.view(-1, d).index_copy_(0, index, k.reshape(-1, d).to(K_cache.dtype))
    V_cache.view(-1, d).index_copy_(0, index, v.reshape(-1, d).to(V_cache.dtype))
    return out.reshape(rows, q_heads * d)
