# Method

## Request format

A request is one `state` (the record) and one or more branches. Each branch is rendered as a separate Qwen chat prompt:

```
Read the record and answer the question. Respond with exactly one listed letter and no other text.

Record:
<state>

Question: <instructions>
Answers:
A. <option 1>
B. <option 2>
...

Answer:
```

`noul` branches always use `A. No` / `B. Yes`. The prompt is passed through the Qwen chat template with `enable_thinking=False` and a generation prompt.

## Packing

`pack_request` (`src/system_one/packing.py`) tokenizes every branch prompt, takes the longest common token prefix of all of them as the shared state, and appends each prompt's remaining suffix as a branch. At least one token per branch stays in its suffix. The result is:

- `input_ids`: shared prefix followed by every branch suffix
- `state_len`: length of the shared prefix
- `spans`: the inclusive token range of each branch

## Tree attention mask and positions

Token *i* may attend to token *j* only if *j ≤ i* and *j* is either in the shared prefix or in the same branch as *i*. Position IDs run `0 … state_len-1` over the prefix and restart at `state_len` for each branch, so each branch sees the same positions it would have as a stand-alone prompt.

Both backends build this mask densely (`T × T`, where `T` is the packed length), so memory grows quadratically with packed length.

## Readout

The model runs once over the packed sequence. At the last token of each branch, the logits of the answer-letter tokens `A`, `B`, … (one per listed option) are passed through softmax. Probabilities are conditional on the listed options. The code uses one canonical option order and does not average across option orders.

## Backends

| Backend | File | Weights | Notes |
| --- | --- | --- | --- |
| PyTorch | `inference.py` | Qwen base + PEFT LoRA | bf16 on CUDA, fp16 on MPS, fp32 on CPU; SDPA with a boolean mask |
| MLX | `inference_mlx.py` | Qwen base with LoRA merged in memory | fp16; custom forward over MLX-LM's Qwen3 modules to apply the tree mask and custom positions |

Differences between backends come from precision and kernels; see [RESULTS.md](RESULTS.md).
