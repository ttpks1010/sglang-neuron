# Muse Glimmer on Neuron

`MuseGlimmerForConditionalGeneration` (Muse-Glimmer-30B, its language model only) runs on the
[runtime](../design/runtime.md) in bf16 as an aggregated server, TP4 on the four NeuronCores of a trn2.3xlarge. The
host is `sglang_neuron/models/muse_glimmer/sglang_model.py`, the graphs are
`sglang_neuron/models/muse_glimmer/graph.py` and `decode.py`, and the launch rules are
`sglang_neuron/models/muse_glimmer/launch.py`.

## The model

Muse-Glimmer-30B's language model is a dense decoder: 52 layers of hidden 6656, 32 query heads over 2 KV heads of 128,
a SiLU MLP of 19,968 and a 202,048-token vocabulary with its own LM head. Three sliding-window layers (a window of 2048
tokens, RoPE with theta 5e5 over the whole head) are followed by one full-attention layer with no positional encoding,
thirteen times. Each phase runs its attention in the paged attention kernels. Prefill runs `qkv_paged` (QKV, Q/K norm,
RoPE and the K/V write), then the segmented attention with the sliding windows masked. Decode runs
`attention_block_tkg`, which does the QKV, Q/K norm, RoPE, attention over the row's pages and the K/V write in one call,
without the output projection. In both, the attended heads are gated by `sigmoid(x W_gate)` and projected, then an
all-reduce over the attention TP ranks follows, then the MLP shard and a second all-reduce.

At TP4 each rank holds eight query heads and one KV head (each of the two KV heads is replicated on two ranks). The
embedding and the LM head are sharded by vocabulary, and the device sampler reads each rank's shard of the logits.

The layers:

- **No RoPE on the full layers.** The 13 full-attention layers (3, 7, ..., 51) have no positional encoding: their RoPE
  tables are the identity (cos 1, sin 0), so the kernels' RoPE leaves their Q and K as they are.
- **Q/K norm and scale.** Each Q and K head is RMS-normalized without a weight (the QKV kernel's norm with a gamma of
  ones), and Q's `qk_scale_factor` is folded into the softmax scale.
- **A gated attention output.** The attended heads are multiplied by `sigmoid(x W_gate)` before the output projection.
  The decode kernel is called with `W_out=None`, so it returns the heads and the graph gates and projects them.
- **Centered norms.** The four per-layer norms are `rms(x) * (1 + w)`, with `1 + w` computed in fp32 at run time (a
  bf16 `1 + w`, which folding the 1 into the stored weight gives, would round away most of a small `w`); the two after
  a sub-layer use `post_norm_eps`. The embedding is RMS-normalized without a weight.
- **Logits.** Scaled by `output_multiplier` and soft-capped by `final_logit_softcapping` (`20 * tanh(x / 20)`), in
  fp32.
- **The config.** The shape is read from the checkpoint's `config.json` and checked against the config object SGLang
  builds, which flattens `text_config` and renames or rescales some fields (`config.py`).

### The sliding layers

A sliding layer attends to the last 2048 positions. In prefill the segmented attention kernel reads the request's whole
table and masks the window. In decode it reads the table the runtime trimmed to the window, and the window's first
position is passed as `swa_start_pos_ids`. The decode attention does not pass `max_context_len`, whose early exit on
nkilib's flash-attention path gives wrong attention past 8K tokens; the decode context ladder bounds what it reads.

The KV pool is SGLang's uniform one: every layer's KV in one pool of whole-context pages, since the graphs read no
window pool and the platform turns hybrid SWA memory off for them. A sliding layer therefore keeps KV it no longer
attends to: at a 16384-token context the pool holds about 2.9 times the KV a window pool would need.

## Serving

The commands below run inside the container, with the checkpoint mounted at `/models`; the [container
guide](../guides/container.md) has the `docker run` for one node. The example has the server as scripts
([Muse-Glimmer-30B](../../examples/muse_glimmer_30b/)), and [examples/README.md](../../examples/README.md#run-a-model)
gives the order to run them.

Aggregated, on the four NeuronCores of a trn2.3xlarge, with the validated flags:

```bash
python3 -m sglang.launch_server --model-path /models/Muse-Glimmer-30B \
  --language-model-only --reasoning-parser muse \
  --tp-size 4 --context-length 16384 --chunked-prefill-size 2048 \
  --max-running-requests 8 --max-total-tokens 32768 --mem-fraction-static 0.9 \
  --host 0.0.0.0 --port 30000
```

`--host 0.0.0.0` listens inside the container; publish the port on the host's loopback only
(`-p 127.0.0.1:30000:30000`), as the [container guide](../guides/container.md#3-start-a-container) explains.

- `--language-model-only` is required (the launch is refused without it). It is SGLang's own text-only switch for this
  architecture: SGLang builds no vision tower and no multimodal processor, the vision tower's tensors are never read,
  and a request with an image or a video is refused.
- `--reasoning-parser muse`: the model reasons in a channel of its own before it answers, and the parser puts that text
  in `reasoning_content`. A request's `chat_template_kwargs` `reasoning_strength` (`low`, `medium` or `high`, the
  template's default) sets how long it reasons.
- `--context-length 16384` is tested, and so is 4096 (the same GSM8K score).
- `--max-running-requests 16 --max-total-tokens 324608` is also tested: at `--mem-fraction-static 0.9` a pool of
  326,656 tokens fits, and an oversized pool is refused with that size.

Disaggregated prefill and decode is served by the launch checks but not run ([Limitations](#limitations)).

## Compiled shapes

- **Prefill.** A graph of the chunk (`--chunked-prefill-size`, one of the segmented kernel's KV segment tiles: 512,
  1024, 2048, 4096 or 8192) and one of each smaller tile that splits evenly over the attention TP ranks (three graphs at
  a chunk of 2048).
- **Decode.** A graph per batch size up to `--max-running-requests` (the powers of two below it and the count itself:
  1, 2, 4 and 8 for the example), each over a quarter, a half and the whole context: 12 graphs for the example, 15 at
  16 rows.
- `--max-total-tokens` sizes the KV pool, whose page count is a dimension of every graph; state it, so a relaunch finds
  its graphs in torch-neuronx's compile cache. A token of KV is 52 layers x 1 KV head x 128 x 2 (K and V) x 2 bytes =
  26 KiB on each rank.

## Measured

All numbers are from one trn2.3xlarge, TP4, bf16, the flags above and the greedy device sampler unless a line says
otherwise, on the sglang-neuron image v0.1.0 (Neuron SDK 2.33), with the checkpoint `meta-models/Muse-Glimmer-30B` at
revision `cec94fc`.

Accuracy:

- Greedy tokens of three prompts (6, 70 and 2,532 tokens; the last crosses the 2048-token window and the prefill
  chunk), 32 new tokens each, against Hugging Face transformers in bf16 on CPU: 32 of 32 for the first two; the third
  matches for 9 tokens and then takes the other of two tokens the reference holds at an exact tie. Sent at once, the
  three match the same way. On the `host` sampler all three match, 96 of 96.
- `examples/muse_glimmer_30b/run_gsm8k.sh` (first 100 questions, five-shot, temperature 0): 0.98 at a context of 4096
  and at 16384, none unparsed.
- A code hidden at 10, 50 and 90 percent of filler text is answered exactly at 2,000, 8,000 and 14,000 tokens (9 of 9,
  `reasoning_strength` low). Past 2048 tokens only the 13 full-attention layers see the code.
- Prefix cache: hits on 128, 2,048, 2,176 and 6,000 shared tokens give the cold run's tokens in 8 of 8 cases.
- Samplers: a launch on `greedy` refuses a temperature or a top-k/top-p request with HTTP 400; `topk_topp` serves both
  and keeps its greedy requests deterministic; on `host` (`--disable-overlap-schedule`) a `json_schema` request returns
  a document of the schema.

Speed, one client, random-token prompts:

| Input / output tokens | TTFT p50 | TPOT p50 |
| --- | --- | --- |
| 1024 / 128 | 367 ms | 27.07 ms |
| 4096 / 128 | 743 ms | 27.14 ms |
| 8192 / 256 | 1567 ms | 27.34 ms |

- Steadiness, TPOT p90 and p99 against the median (the bounds of [Performance
  characteristics](../design/disaggregated-inference.md#performance-characteristics) are 1.01 and 1.02): 1.000x and
  1.000x at 1 client (8192 / 1024), 1.115x and 1.116x at 4 clients (4096 / 512), 1.032x and 1.042x at 8 clients
  (1024 / 1024). On an aggregated server a prompt's prefill chunks run between decode steps, so the decode times of
  the other rows spread while prompts arrive.
- Capacity, `--max-running-requests 16 --max-total-tokens 324608`: all 16 rows fill; 16 clients give 253 output
  tokens/s at 1024 / 128 (TPOT p50 43.6 ms) and 275 at 4096 / 512 (44.4 ms).
- Under load: 24 clients on those 16 rows for 30 minutes, prompts of 256 to 3,000 tokens, answers of 64 to 512 with
  `ignore_eos`, 15 percent of the requests cancelled midway: 1,950 requests, no error, every request not cancelled
  returned its full answer, a median TPOT of 56.6 ms in minutes 1 to 6 and 54.7 ms in the last five, and SGLang's idle
  check of the KV pool and request slots stayed silent.
- Startup: 17 to 18 minutes on a cold compile cache (12 decode graphs and 3 prefill graphs), 13 to 20 minutes from a
  warm one; the 16-row launch takes 21 minutes the first time, and the `host` sampler's first launch 41 minutes.

## What is refused

A launch the graphs can't serve fails at startup, naming the setting at fault:

- a launch without `--language-model-only` (the checkpoint's `config.json` has a `vision_config`);
- a model dtype other than bfloat16, and a quantized checkpoint;
- DP attention (`--enable-dp-attention`) and an `--ep-size` other than 1;
- a `--tp-size` the 32 attention heads do not split over, or that the 2 KV heads neither split over nor replicate
  evenly across;
- a `config.json` the graphs do not compute: layer kinds other than sliding and full attention, a sliding layer
  without RoPE or a full layer with it, sliding layers of different thetas, a RoPE scaling, attention biases, an MLP
  activation other than SiLU, tied embeddings, a sliding window that is not positive, or a shape that differs from the
  config SGLang built;
- a KV cache dtype other than `auto`, `bf16` or `bfloat16`;
- the features every model refuses ([What a launch refuses](../design/runtime.md#what-a-launch-refuses)).

A request with an image or a video is refused by SGLang (`--language-model-only`).

## Limitations

- Disaggregated prefill and decode is served by the launch checks but not run: the bf16 weights do not fit half a
  trn2.3xlarge.
- Not run: a decode server alone, the host and device trace checks, a context above 16384, hybrid SWA memory, the
  checkpoint's speculative draft model, quantized weights and Trn3.
