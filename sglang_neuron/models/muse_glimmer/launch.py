# SPDX-License-Identifier: Apache-2.0
"""Muse Glimmer's launch rules (catalog.ModelSpec.launch_hooks). Module-level imports are stdlib only: the launching
process imports this before it has decided anything else.

The graphs serve the checkpoint's language model, text only. SGLang's own text-only switch for this architecture is
``--language-model-only`` (``ServerArgs.LANGUAGE_MODEL_ONLY_ARCHITECTURES``): with it, SGLang builds no vision tower,
loads no multimodal processor, sends no image warm-up request and refuses media in requests, so the launch is required
to pass it. The KV pool is SGLang's uniform one: the platform turns hybrid SWA memory off itself for a model whose
graphs read no window pool (mem_cache.hybrid_swa.configure)."""

# config.NAME, repeated here: that module imports torch, and this one may import the stdlib only
NAME = "Muse Glimmer"
PAGE_SIZE = 128


def kv_page_size(server_args) -> int:
    return PAGE_SIZE


def launch_problems(server_args, checkpoint_config: dict, environ=None) -> list:
    """Every reason the launch is not served, run in the launching process before any scheduler starts."""
    problems = []
    has_vision = bool(checkpoint_config.get("vision_config"))
    if has_vision and not getattr(server_args, "language_model_only", False):
        problems.append(f"{NAME} on Neuron serves the checkpoint's language model only, and SGLang builds the vision "
                        "tower and the multimodal processor unless the launch says otherwise: launch with "
                        "--language-model-only")
    return problems


def launch_settings(server_args) -> tuple:
    return ()


def rank_runtime_env(role: str) -> dict:
    return {}
