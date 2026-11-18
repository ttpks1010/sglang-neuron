# SPDX-License-Identifier: Apache-2.0
"""Muse Glimmer's text model as the runtime's role graphs (prefill: one request's chunk per forward; decode: one token
per row), on attention TP.

The layers are the dense transformer's shape with the model's own pieces (config.py says what they are). A prefill
runs the dense models' QKV kernel (which RMS-normalizes each Q and K head with a gamma of ones, rotates the sliding
layers' heads and writes the chunk's K/V) and the segmented attention kernel, which masks a sliding layer's window over
the request's prior and the chunk. A decode runs the fused attention kernel of decode.py, which returns the attended
heads without the output projection. Both then gate the heads by ``sigmoid(x W_gate)`` and project them out. Each
attention and MLP output is all-reduced once over the attention TP; the embedding and the LM head are sharded by
vocabulary, and the device sampler reads each rank's shard of the logits.

The 13 full-attention layers have no positional encoding: their RoPE tables are the identity (cos 1, sin 0), so the
kernels' RoPE leaves their Q and K as they are. Q's ``qk_scale_factor`` is folded into the softmax scale (RoPE is
linear). The four per-layer norms are centered, ``rms(x) * (1 + w)``, with ``1 + w`` computed in fp32 at run time (a
bf16 ``1 + w`` would round away most of a small ``w``), the two after a sub-layer with ``post_norm_eps``. The logits are
scaled by ``output_multiplier`` and soft-capped by ``final_logit_softcapping`` in fp32."""

import torch
from sglang.srt.distributed.parallel_state import get_attn_tp_group
from torch import nn

import sglang_neuron.ops as NF
from sglang_neuron.layers import ColumnParallelLinear
from sglang_neuron.layers.embedding import VocabDimShardedEmbedding
from sglang_neuron.mem_cache.kv_cache import KVSpec, LayerSpec
from sglang_neuron.models.muse_glimmer import decode
from sglang_neuron.models.muse_glimmer.config import DTYPE, GlimmerShape
from sglang_neuron.models.transformer import weights
from sglang_neuron.models.transformer.model import MLP, RMSNorm
from sglang_neuron.ops.attention.qkv_paged import qkv_paged
from sglang_neuron.parallel import collectives as neuron_collectives
from sglang_neuron.parallel import groups
from sglang_neuron.utils.checkpoints import SafetensorsCheckpoint
from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader, set_weight_loader

# the checkpoint's prefix of the language model's tensors (the multimodal export); the LM head sits at the top level
TEXT_PREFIX = "model.language_model."


def _rms(x32: torch.Tensor, eps: float) -> torch.Tensor:
    return x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)


class CenteredRMSNorm(nn.Module):
    """``rms(x) * (1 + w)``: the checkpoint stores the weight as an offset from one (HF MuseGlimmerTextCenteredRMSNorm).
    The product is computed in fp32, as the reference does, and cast back."""

    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size, dtype=DTYPE))
        self.eps = eps

    def forward(self, x):
        return (_rms(x.float(), self.eps) * (1.0 + self.weight.float())).to(x.dtype)


class WeightlessRMSNorm(nn.Module):
    """``rms(x)`` with no weight: the norm after the embedding."""

    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return _rms(x.float(), self.eps).to(x.dtype)


class Rotary(nn.Module):
    """cos and sin of each position over the rotary half-dimension, [T, d / 2] in DTYPE, for the sliding layers; the
    identity tables (cos 1, sin 0) for the full layers, which have no positional encoding."""

    def __init__(self, shape: GlimmerShape):
        super().__init__()
        d = shape.head_dim
        inv_freq = 1.0 / (shape.rope_theta ** (torch.arange(0, d, 2, dtype=torch.int64).float() / d))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions):
        angles = positions.float()[:, None] * self.inv_freq[None, :]
        return angles.cos().to(DTYPE), angles.sin().to(DTYPE)

    def identity(self, positions):
        half = self.inv_freq.shape[0]
        return (torch.ones(positions.shape[0], half, dtype=DTYPE, device=positions.device),
                torch.zeros(positions.shape[0], half, dtype=DTYPE, device=positions.device))


def column_shard_loader(rows: int) -> SafetensorsWeightLoader:
    """A column-parallel projection ([out, H], the attention gate): this rank's output rows, as [H, out / tp]."""
    return SafetensorsWeightLoader(transform=lambda slices, rank: slices[0][rank * rows:(rank + 1) * rows].t())


class Attention(nn.Module):
    def __init__(self, shape: GlimmerShape, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.name = f"layers.{layer_idx}.self_attn"
        self.tp_group = get_attn_tp_group()
        self.attn_tp_group = groups.get_neuron_attn_tp_group()
        tp = self.tp_group.world_size
        if shape.num_attention_heads % tp:
            raise ValueError(f"{shape.num_attention_heads} attention heads do not shard over an attention TP of {tp}")
        if shape.num_key_value_heads % tp and tp % shape.num_key_value_heads:
            raise ValueError(f"{shape.num_key_value_heads} KV heads neither shard over nor replicate across an "
                             f"attention TP of {tp}")
        self.head_dim = d = shape.head_dim
        self.num_heads = shape.num_attention_heads // tp
        self.num_kv_heads = max(1, shape.num_key_value_heads // tp)
        self.window = shape.window(layer_idx)
        self.has_rope = shape.has_rope(layer_idx)
        self.scaling = shape.softmax_scale
        self.norm_eps = shape.rms_norm_eps
        width = (self.num_heads + 2 * self.num_kv_heads) * d
        self.qkv_weight = nn.Parameter(torch.empty(shape.hidden_size, width, dtype=DTYPE))
        self.gate_weight = nn.Parameter(torch.empty(shape.hidden_size, self.num_heads * d, dtype=DTYPE))
        self.o_weight = nn.Parameter(torch.empty(self.num_heads * d, shape.hidden_size, dtype=DTYPE))
        # the kernels' per-head QK RMSNorm takes a gamma; the model's norm has none, so the gamma is one
        self.register_buffer("qk_gamma", torch.ones(1, d, dtype=DTYPE), persistent=False)
        # bound by bind_kv_cache: [pages, kv_heads, page, d] each
        self.k_cache = None
        self.v_cache = None

        shard = (tp, self.num_heads * d, self.num_kv_heads * d, shape.num_key_value_heads)
        set_weight_loader(self.qkv_weight, weights.fused_qkv_weight_loader(*shard))
        set_weight_loader(self.gate_weight, column_shard_loader(self.num_heads * d))
        set_weight_loader(self.o_weight, weights.row_parallel_loader(self.num_heads * d))

    def _gate_and_project(self, x, attended):
        """``o_proj(attended * sigmoid(x W_gate))``, all-reduced over the attention TP; ``x`` is the layer's normed
        input, ``attended`` [T, heads * d] this rank's attended heads."""
        gate = torch.sigmoid((x @ self.gate_weight).float()).to(attended.dtype)
        return neuron_collectives.all_reduce(self.attn_tp_group, (attended * gate) @ self.o_weight)

    def prefill(self, x, cos, sin, md):
        page = md["block_size"]
        pages = self.k_cache.shape[0]
        tokens = x.shape[0]
        # The runtime pads with slot 0, on SGLang's reserved page. The QKV kernel writes K/V at its slots as DMA offsets
        # with no bounds check, so any slot outside the pool goes to slot 0 too instead of past the cache.
        slots = md["slot_mapping"]
        slots = torch.where((slots < 0) | (slots >= pages * page), torch.zeros_like(slots), slots)
        q, k_cache, v_cache = qkv_paged(
            x.unsqueeze(0),
            weights=self.qkv_weight,
            bias=None,
            # the kernel's RoPE tables cover the whole head, the halves repeated
            cos=torch.cat([cos, cos], dim=-1).unsqueeze(0),
            sin=torch.cat([sin, sin], dim=-1).unsqueeze(0),
            q_norm=self.qk_gamma,
            k_norm=self.qk_gamma,
            norm_eps=self.norm_eps,
            d_head=self.head_dim,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            block_size=page,
            slot_mapping=slots.to(torch.int32).reshape(1, tokens),
        )
        out = NF.segmented_attention(
            q.view(tokens, self.num_heads, self.head_dim).transpose(0, 1),
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=md["block_table_tensor"],
            prior_tokens=md["cached_seq_len"],
            block_size=page,
            kv_segment_size=md["kv_segment_size"],
            scale=self.scaling,
            sliding_window=self.window,
            sink=None,
        )
        return self._gate_and_project(x, out.transpose(0, 1).reshape(tokens, -1))

    def decode(self, x, cos, sin, inputs: decode.DecodeInputs):
        attended = decode.attention_decode(
            x.unsqueeze(1),
            w_qkv=self.qkv_weight,
            q_norm=self.qk_gamma,
            k_norm=self.qk_gamma,
            norm_eps=self.norm_eps,
            # [B, d / 2] -> the kernel's [d / 2, B, 1]
            cos=cos.t().unsqueeze(-1).contiguous(),
            sin=sin.t().unsqueeze(-1).contiguous(),
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            inputs=inputs,
            softmax_scale=self.scaling,
            q_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        return self._gate_and_project(x, attended)


class DecoderLayer(nn.Module):
    def __init__(self, shape: GlimmerShape, layer_idx: int):
        super().__init__()
        self.input_layernorm = CenteredRMSNorm(shape.hidden_size, shape.rms_norm_eps)
        self.self_attn = Attention(shape, layer_idx)
        self.post_attention_layernorm = CenteredRMSNorm(shape.hidden_size, shape.post_norm_eps)
        self.pre_feedforward_layernorm = CenteredRMSNorm(shape.hidden_size, shape.rms_norm_eps)
        # the dense transformer's SiLU MLP: down(silu(gate(x)) * up(x)) with gate and up read as two tensors
        self.mlp = MLP(shape, fused_gate_up=False)
        self.post_feedforward_layernorm = CenteredRMSNorm(shape.hidden_size, shape.post_norm_eps)

    def forward(self, x, attention):
        """``attention(normed)`` is this layer's attention on its phase's inputs."""
        x = x + self.post_attention_layernorm(attention(self.input_layernorm(x)))
        return x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))


class MuseGlimmerGraph(nn.Module):
    """One role graph (``phase`` "prefill" or "decode")."""

    def __init__(self, shape: GlimmerShape, phase: str, checkpoint_prefix: str = TEXT_PREFIX):
        super().__init__()
        if phase not in ("prefill", "decode"):
            raise ValueError(f"a role graph is a prefill or a decode, not {phase!r}")
        self.config = shape
        self.phase = phase
        self.prefix = checkpoint_prefix
        self.tp_group = get_attn_tp_group()
        self.lm_head_tp_group = group = groups.get_neuron_attn_tp_group()
        self.embed_tokens = VocabDimShardedEmbedding(shape.vocab_size, shape.hidden_size, dtype=DTYPE,
                                                     tp_group=group.device_group)
        self.embed_norm = WeightlessRMSNorm(shape.rms_norm_eps)
        self.layers = nn.ModuleList([DecoderLayer(shape, i) for i in range(shape.num_hidden_layers)])
        # the final norm is a plain weighted RMSNorm (not centered)
        self.norm = RMSNorm(shape.hidden_size, shape.rms_norm_eps)
        self.rotary_emb = Rotary(shape)
        self.lm_head = ColumnParallelLinear(shape.hidden_size, shape.vocab_size, dtype=DTYPE, gather_output=True,
                                            tp_group=group.device_group)
        set_weight_loader(self.embed_tokens.weight,
                          weights.vocab_rows_loader(self.embed_tokens.vocab_size_per_rank, shape.hidden_size))
        set_weight_loader(self.lm_head.weight,
                          weights.vocab_rows_loader(self.lm_head.out_features_per_rank, shape.hidden_size))
        # the runtime's per-layer metadata keys and windows (models/contract.declared_layer_kinds)
        self.layer_names = [layer.self_attn.name for layer in self.layers]
        self.sliding_windows = [layer.self_attn.window for layer in self.layers]

    def _first_layer_of_kind(self, sliding: bool):
        for layer in self.layers:
            if (layer.self_attn.window is not None) == sliding:
                return layer.self_attn
        return None

    @torch.no_grad()
    def forward(self, input_ids, positions, attn_metadata, sampling_positions, rank):
        hidden = self.embed_norm(self.embed_tokens(input_ids, scatter_tokens=False, rank=rank))
        tables = {True: self.rotary_emb(positions), False: self.rotary_emb.identity(positions)}
        if self.phase == "decode":
            # one set of kernel inputs per layer kind: the full layers' over the whole table, the sliding layers' over
            # the runtime's window-trimmed table in its own frame
            inputs = {}
            for sliding in (False, True):
                attn = self._first_layer_of_kind(sliding)
                if attn is None:
                    continue
                md = attn_metadata[attn.name]
                inputs[sliding] = decode.decode_inputs(
                    md["block_table_tensor"], md["slot_mapping"], positions, kv_heads=attn.num_kv_heads,
                    q_heads=attn.num_heads, page=md["block_size"], window=attn.window,
                    offset=md.get("swa_kv_pos_offset") if sliding else None)
            for layer in self.layers:
                attn = layer.self_attn
                cos, sin = tables[attn.has_rope]
                step = inputs[attn.window is not None]
                hidden = layer(hidden, lambda x, a=attn, cos=cos, sin=sin, step=step: a.decode(x, cos, sin, step))
        else:
            for layer in self.layers:
                attn = layer.self_attn
                cos, sin = tables[attn.has_rope]
                md = attn_metadata[attn.name]
                hidden = layer(hidden, lambda x, a=attn, cos=cos, sin=sin, md=md: a.prefill(x, cos, sin, md))
        hidden = self.norm(hidden.index_select(0, sampling_positions))
        logits = self.lm_head(hidden).float()
        cap = self.config.final_logit_softcapping
        return cap * torch.tanh(logits * (self.config.output_multiplier / cap))

    def get_kv_spec(self) -> KVSpec:
        return KVSpec(layers=[LayerSpec(name=name) for name in self.layer_names])

    def bind_kv_cache(self, kv_caches: dict) -> None:
        for layer in self.layers:
            attn = layer.self_attn
            attn.k_cache, attn.v_cache = kv_caches[attn.name]

    def hf_weight_mappings(self) -> dict:
        """Parameter name to its checkpoint key, or the keys a fused parameter is built from. The vision tower's
        tensors are never mapped, so they are never read."""
        p = self.prefix
        mappings = {"embed_tokens.weight": f"{p}embed_tokens.weight", "norm.weight": f"{p}norm.weight",
                    "lm_head.weight": "lm_head.weight"}
        for i in range(len(self.layers)):
            src, dst = f"{p}layers.{i}", f"layers.{i}"
            attn = f"{src}.self_attn"
            mappings.update({
                f"{dst}.self_attn.qkv_weight": [f"{attn}.{n}_proj.weight" for n in "qkv"],
                f"{dst}.self_attn.gate_weight": f"{attn}.gate_proj.weight",
                f"{dst}.self_attn.o_weight": f"{attn}.o_proj.weight",
                f"{dst}.mlp.gate_up_weight": [f"{src}.mlp.gate_proj.weight", f"{src}.mlp.up_proj.weight"],
                f"{dst}.mlp.down_weight": f"{src}.mlp.down_proj.weight",
            })
            for norm in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                         "post_feedforward_layernorm"):
                mappings[f"{dst}.{norm}.weight"] = f"{src}.{norm}.weight"
        return mappings

    def load_weights(self, checkpoint_path: str, device: torch.device) -> None:
        """Read this rank's shard of every mapped parameter onto ``device``."""
        checkpoint = SafetensorsCheckpoint(checkpoint_path)
        state = checkpoint.load_sharded_pipelined(self.tp_group.rank_in_group, self.tp_group.world_size, self,
                                                  self.hf_weight_mappings(), device).state_dict
        self.load_state_dict(state, strict=False, assign=True)
