# Muse Glimmer 30B

One aggregated server on one trn2.3xlarge, TP4 on its four NeuronCores, text only. Start the image on the instance and
run the scripts below inside it.

| role | node | script |
|---|---|---|
| server (prefill and decode) | trn2.3xlarge, NeuronCores 0-3, port 30000 | `run_server.sh` |
| client and accuracy check | the same container, second shell | `run_client.sh`, `run_gsm8k.sh` |

## Launch

```bash
bash /opt/sglang-neuron/examples/muse_glimmer_30b/run_server.sh
```

The first launch compiles the decode graphs (four batch sizes over three contexts) and the prefill graphs, about 18
minutes on a trn2.3xlarge; a relaunch with the same flags and compile cache reads them back (13 to 20 minutes).

## Check

```bash
# second shell (docker exec -it sglang bash), once the server is ready
bash /opt/sglang-neuron/examples/muse_glimmer_30b/run_client.sh    # "content" holds "391"; the reasoning is in reasoning_content
bash /opt/sglang-neuron/examples/muse_glimmer_30b/run_gsm8k.sh     # Accuracy: about 0.98 on the first 100 questions
```

## Notes

- The checkpoint: `meta-models/Muse-Glimmer-30B` from Hugging Face, as `<models dir>/Muse-Glimmer-30B`. The server
  loads its language model only (`--language-model-only`); the vision tower's tensors are never read.
- Defaults: `MODEL_PATH=/models/Muse-Glimmer-30B`, `PORT=30000`, `CONTEXT_LENGTH=16384`.
- The server takes 8 requests at a time, a request's prompt and answer together take up to `CONTEXT_LENGTH` tokens,
  and the KV pool holds 32,768 tokens across all requests. `--max-running-requests 16 --max-total-tokens 324608` is
  also tested (pass them after the script's flags: the last value wins).
- The model reasons in its own channel before it answers. `--reasoning-parser muse` puts that text in
  `reasoning_content`; a request's `chat_template_kwargs` `reasoning_strength` (`low`, `medium`, `high`; the template's
  default is `high`) sets how long it reasons.
- Images and video are not served: a request with media is refused.
- Disaggregated prefill and decode is served by the launch checks but not run: the bf16 weights do not fit half a
  trn2.3xlarge.
- [The Muse Glimmer recipe](../../docs/model-recipes/muse-glimmer.md) explains the flags and the model, with the
  measured results.
