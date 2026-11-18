# SPDX-License-Identifier: Apache-2.0
"""MuseGlimmerForConditionalGeneration: the host SGLang builds for a Muse Glimmer checkpoint (models/host.py). It
serves the checkpoint's language model, text only, on the graphs in graph.py."""

import copy
import json
import os

from torch import nn

from sglang_neuron import sglang_tree
from sglang_neuron.models.host import ModelHost
from sglang_neuron.models.muse_glimmer.config import NAME, glimmer_shape, text_config
from sglang_neuron.models.muse_glimmer.graph import TEXT_PREFIX, MuseGlimmerGraph
from sglang_neuron.models.transformer.config import check_bf16_launch, check_no_expert_parallel


def _checkpoint_prefix(path: str) -> str:
    """The checkpoint's prefix of the language model's tensors: ``model.language_model.`` in the multimodal export,
    ``model.`` in a text-only one."""
    index = os.path.join(path or "", "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            names = json.load(f).get("weight_map", {})
        return TEXT_PREFIX if any(n.startswith(TEXT_PREFIX) for n in names) else "model."
    try:
        from safetensors import safe_open

        with safe_open(os.path.join(path or "", "model.safetensors"), "pt") as f:
            return TEXT_PREFIX if any(k.startswith(TEXT_PREFIX) for k in f.keys()) else "model."
    except (OSError, ImportError):
        return TEXT_PREFIX


def _host_config(config):
    """The config the runtime reads the model's layers, heads and sizes from (models/contract.py). SGLang's own
    MuseGlimmerConfig is already flat; transformers' nests the language model in ``text_config``, which is copied out
    with the path and dtype the host and the launch checks read from the top level."""
    text = text_config(config)
    if text is config:
        return config
    text = copy.copy(text)
    text._name_or_path = getattr(config, "_name_or_path", "") or getattr(text, "_name_or_path", "")
    for attr in ("dtype", "torch_dtype"):
        if getattr(text, attr, None) is None and getattr(config, attr, None) is not None:
            setattr(text, attr, getattr(config, attr))
    return text


class MuseGlimmerForConditionalGeneration(ModelHost):
    name = NAME
    # the decode attention reads only its graph's table (decode.py), so shorter-context decode graphs serve short rows
    decode_context_ladder = True

    def __init__(self, config, quant_config=None, prefix: str = "", **kwargs):
        super().__init__(_host_config(config), quant_config, prefix, **kwargs)

    def check_launch(self, config) -> None:
        """Refuse, before any weight loads, a checkpoint or layout the graphs do not serve: a dtype other than bf16,
        expert parallelism, DP attention, a TP the heads do not shard over, and the checkpoint shapes config.py
        refuses."""
        check_bf16_launch(config, self.name)
        check_no_expert_parallel(self.name)
        shape = glimmer_shape(config)
        pc = sglang_tree.parallel_config()
        if sglang_tree.attention_dp_size(pc) != 1:
            raise ValueError(f"{self.name} is served on attention TP without DP attention; launch without "
                             "--enable-dp-attention")
        tp = int(pc.tp_size)
        if shape.num_attention_heads % tp:
            raise ValueError(f"--tp-size {tp}: {self.name}'s {shape.num_attention_heads} attention heads do not shard "
                             f"over {tp} ranks; launch with a --tp-size that divides {shape.num_attention_heads}")
        kv = shape.num_key_value_heads
        if kv % tp and tp % kv:
            raise ValueError(f"--tp-size {tp}: {self.name}'s {kv} KV heads neither shard over nor replicate evenly "
                             f"across {tp} ranks")

    def build_graph(self, phase: str, config) -> nn.Module:
        return MuseGlimmerGraph(glimmer_shape(config), phase, _checkpoint_prefix(config._name_or_path))


EntryClass = [MuseGlimmerForConditionalGeneration]
