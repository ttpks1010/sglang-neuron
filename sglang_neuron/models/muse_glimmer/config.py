# SPDX-License-Identifier: Apache-2.0
"""The shape of a Muse Glimmer text model, read from the checkpoint's config.json.

Muse Glimmer 30B is a dense decoder: 52 layers of hidden 6656, 32 query heads over 2 KV heads of 128, a SiLU MLP of
19968, a 202048-token vocabulary with its own LM head. Three sliding-window layers (2048 tokens, RoPE theta 5e5 over the
whole head) are followed by one full-attention layer without positional encoding (``layer_rope_theta`` 0), thirteen
times. Each layer's Q and K are RMS-normalized without a weight and Q is scaled by ``qk_scale_factor``; the attention
output is gated by ``sigmoid(x W_gate)`` before the output projection; the four per-layer norms are centered
(``x * (1 + w)``), the two after a sub-layer with ``post_norm_eps``; the embedding is RMS-normalized without a weight;
the logits are scaled by ``output_multiplier`` and soft-capped by ``final_logit_softcapping``.

The shape is read from config.json rather than from the config object SGLang hands over, because that object differs by
SGLang version: SGLang's own ``MuseGlimmerConfig`` (``model_type`` ``muse_glimmer`` in its config registry) flattens
``text_config``, renames ``final_logit_softcapping`` to ``output_soft_cap_temp``, replaces ``layer_rope_theta`` with
``no_rope_layers`` and stores ``qk_scale_factor`` multiplied by ``sqrt(head_dim)``, while transformers' config keeps the
file's values. The file is the one source both agree on."""

import json
import math
import os
from dataclasses import dataclass

import torch

DTYPE = torch.bfloat16
FULL = "full_attention"
SLIDING = "sliding_attention"
ARCHITECTURE = "MuseGlimmerForConditionalGeneration"
NAME = "Muse Glimmer"


def text_config(hf_config):
    """The language model's config: ``text_config`` of transformers' nested config, else the config itself (SGLang's
    flat ``MuseGlimmerConfig``)."""
    return getattr(hf_config, "text_config", None) or hf_config


def raw_text_fields(hf_config) -> dict:
    """config.json's text fields as written: its ``text_config`` merged over the top level."""
    path = getattr(hf_config, "_name_or_path", None) or getattr(text_config(hf_config), "_name_or_path", None)
    file = os.path.join(path or "", "config.json")
    if not path or not os.path.isfile(file):
        raise ValueError(f"{NAME}: the graphs read the model's shape from the checkpoint's config.json, and "
                         f"{file!r} is not a file; pass --model-path a local directory of the checkpoint")
    with open(file) as f:
        raw = json.load(f)
    return {**raw, **(raw.get("text_config") or {})}


@dataclass(frozen=True)
class GlimmerShape:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    # the eps of the two norms after a sub-layer (post-attention, post-feedforward)
    post_norm_eps: float
    sliding_window: int
    rope_theta: float
    # Q is scaled by this after its RMSNorm, on top of 1 / sqrt(head_dim)
    qk_scale_factor: float
    output_multiplier: float
    final_logit_softcapping: float
    # per layer: FULL or SLIDING. The sliding layers rotate their Q and K; the full layers have no positional encoding
    # (glimmer_shape checks the checkpoint's layer_rope_theta says the same), so the kind decides the RoPE too
    layer_types: tuple

    def is_sliding(self, i: int) -> bool:
        return self.layer_types[i] == SLIDING

    def has_rope(self, i: int) -> bool:
        return self.is_sliding(i)

    def window(self, i: int):
        """The layer's window in the runtime's convention (the current token and W - 1 earlier ones), None for a full
        layer."""
        return self.sliding_window if self.is_sliding(i) else None

    @property
    def softmax_scale(self) -> float:
        """``qk_scale_factor / sqrt(head_dim)``: the reference scales Q by qk_scale_factor after its norm, then attends
        with 1 / sqrt(head_dim). RoPE is linear, so the factor moves past it into the scale."""
        return self.qk_scale_factor / math.sqrt(self.head_dim)


def glimmer_shape(hf_config) -> GlimmerShape:
    """The served shape of a Muse Glimmer checkpoint, refusing what the graphs do not compute: a quantized
    checkpoint, a layer kind other than sliding or full attention, a sliding layer without RoPE or a full layer with it,
    a scaled RoPE, attention biases, an activation other than SiLU, tied embeddings, or a config without the model's
    scale factors."""
    raw = raw_text_fields(hf_config)
    cfg = text_config(hf_config)

    def get(name, default=None):
        value = raw.get(name)
        return default if value is None else value

    if raw.get("quantization_config") or getattr(hf_config, "quantization_config", None) \
            or getattr(cfg, "quantization_config", None):
        raise ValueError(f"{NAME}: the graphs serve bf16 checkpoints; this one is quantized")
    for field in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                  "num_key_value_heads", "head_dim", "vocab_size", "sliding_window", "qk_scale_factor",
                  "output_multiplier", "final_logit_softcapping"):
        if get(field) is None:
            raise ValueError(f"{NAME}: config.json's text_config has no {field}")
    n = int(get("num_hidden_layers"))
    layer_types = tuple(get("layer_types") or ())
    if len(layer_types) < n or any(t not in (FULL, SLIDING) for t in layer_types[:n]):
        raise ValueError(f"{NAME}: layer_types must name at least {n} sliding or full attention layers, got "
                         f"{layer_types}")
    layer_types = layer_types[:n]
    thetas = tuple(get("layer_rope_theta") or ())
    if len(thetas) < n:
        raise ValueError(f"{NAME}: layer_rope_theta must give {n} values (0 for a layer without RoPE), got {thetas}")
    for i, (kind, theta_i) in enumerate(zip(layer_types, thetas[:n])):
        if (kind == SLIDING) != bool(theta_i):
            raise ValueError(f"{NAME}: layer {i} is {kind} with layer_rope_theta {thetas[i]}; the graphs rotate the "
                             "sliding layers and not the full ones, as the released checkpoint does")
    rope = get("rope_parameters") or {}
    kind = rope.get("rope_type") or (get("rope_scaling") or {}).get("rope_type") or "default"
    if kind != "default":
        raise ValueError(f"{NAME}: a {kind!r} RoPE scaling is not served; the graphs rotate by plain frequencies")
    theta = float(rope.get("rope_theta", get("rope_theta", 500000.0)))
    rotated = {float(t) for t in thetas[:n] if t}
    if rotated and rotated != {theta}:
        raise ValueError(f"{NAME}: the sliding layers' layer_rope_theta {sorted(rotated)} differ from rope_theta "
                         f"{theta}; the graphs rotate every sliding layer by one theta")
    if get("attention_bias", False):
        raise ValueError(f"{NAME}: attention biases (attention_bias) are not served")
    activation = get("hidden_activation") or get("hidden_act") or "silu"
    if activation != "silu":
        raise ValueError(f"{NAME}: the MLP activation {activation!r} is not served; the graphs compute SiLU")
    if get("tie_word_embeddings", False):
        raise ValueError(f"{NAME}: tied embeddings are not served; the released checkpoint has its own lm_head")
    shape = GlimmerShape(
        vocab_size=int(get("vocab_size")),
        hidden_size=int(get("hidden_size")),
        intermediate_size=int(get("intermediate_size")),
        num_hidden_layers=n,
        num_attention_heads=int(get("num_attention_heads")),
        num_key_value_heads=int(get("num_key_value_heads")),
        head_dim=int(get("head_dim")),
        rms_norm_eps=float(get("rms_norm_eps", 1e-5)),
        post_norm_eps=float(get("post_norm_eps", 1e-8)),
        sliding_window=int(get("sliding_window")),
        rope_theta=theta,
        qk_scale_factor=float(get("qk_scale_factor")),
        output_multiplier=float(get("output_multiplier")),
        final_logit_softcapping=float(get("final_logit_softcapping")),
        layer_types=layer_types,
    )
    if shape.num_attention_heads % shape.num_key_value_heads:
        raise ValueError(f"{NAME}: {shape.num_attention_heads} attention heads do not group over "
                         f"{shape.num_key_value_heads} KV heads")
    if shape.sliding_window <= 0:
        raise ValueError(f"{NAME}: sliding_window must be positive, got {shape.sliding_window}")
    _check_against_config_object(shape, cfg)
    return shape


def _check_against_config_object(shape: GlimmerShape, cfg) -> None:
    """The config object SGLang built (its own flat class or transformers') must describe the same model as
    config.json: the runtime reads its layer_types, sliding_window, heads, hidden and vocab sizes for the KV pool and
    the fleet digest (models/contract.py)."""
    expected = {
        "num_hidden_layers": shape.num_hidden_layers, "hidden_size": shape.hidden_size,
        "vocab_size": shape.vocab_size, "num_attention_heads": shape.num_attention_heads,
        "num_key_value_heads": shape.num_key_value_heads, "head_dim": shape.head_dim,
        "sliding_window": shape.sliding_window,
    }
    for name, value in expected.items():
        found = getattr(cfg, name, None)
        if found is not None and int(found) != value:
            raise ValueError(f"{NAME}: the config SGLang built says {name} = {found} but the checkpoint's config.json "
                             f"says {value}; the launch's --model-path and the config disagree")
    found = tuple(getattr(cfg, "layer_types", None) or ())
    if found and found[:shape.num_hidden_layers] != shape.layer_types:
        raise ValueError(f"{NAME}: the config SGLang built lists layer_types {found} but the checkpoint's config.json "
                         f"lists {shape.layer_types}")
