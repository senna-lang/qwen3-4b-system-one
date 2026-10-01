# Results

This page separates two kinds of evidence.

| Kind | Who can re-run it | Section |
| --- | --- | --- |
| Behavior checks on the public fixture | Anyone with the adapter | [Reproducible checks](#reproducible-checks) |
| Quality evaluation on undistributed data | Nobody outside the project | [Reported evaluation](#reported-evaluation-not-reproducible) |

## Reproducible checks

[`fixtures/behavior.jsonl`](../fixtures/behavior.jsonl) holds 16 fictional records with 42 branches (16 `choice`, 10 `score`, 16 `noul`), written for this repository. They test behavior, not accuracy: there are no labels.

### Branch isolation

```bash
python scripts/check_branch_isolation.py --backend torch --output outputs/isolation-torch.json
python scripts/check_branch_isolation.py --backend mlx   --output outputs/isolation-mlx.json
```

Each branch is scored packed with the others, packed in reverse order, and alone. The script reports the largest absolute probability difference among the three and whether the argmax matches.

Measured on an Apple Silicon Mac (2026-10-01):

| Backend | Branches | Max abs diff | Argmax agreement |
| --- | ---: | ---: | ---: |
| PyTorch (MPS, fp16) | 42 | 0.0074 | 42/42 |
| MLX (fp16) | 42 | 0.0074 | 42/42 |

Negative control: replacing the tree mask with a plain causal mask (each branch can see earlier branches) and contiguous positions raised the MLX max difference between packed and alone to **0.84**. The check therefore detects leakage between branches; the remaining 0.0074 is fp16 numerical noise from different sequence layouts. The default tolerance is 0.01.

### PyTorch vs MLX

```bash
python scripts/compare_backends.py predict --backend torch --output outputs/torch.jsonl   # torch env
python scripts/compare_backends.py predict --backend mlx   --output outputs/mlx.jsonl     # mlx env
python scripts/compare_backends.py compare outputs/torch.jsonl outputs/mlx.jsonl
```

| Branches | Max abs diff | Mean per-branch max diff | Argmax agreement |
| ---: | ---: | ---: | ---: |
| 42 | 0.0156 | 0.0012 | 42/42 |

The default tolerance is 0.02. Results on other hardware (for example CUDA in bf16) may differ more and have not been measured here.

## Reported evaluation (not reproducible)

These results were measured on data that is not distributed (training texts and teacher outputs). They are reported for context and cannot be re-run from this repository.

On 1,200 held-out branches (397 Choice, 448 Score, 355 Noul), predictions were compared with the soft answer distributions of the DeepSeek Flash teacher, not with human labels. Order-averaged rows pack option-order variants (four for Choice/Score, two for Noul) as extra branches in the same forward pass and average them.

| Model | Teacher top-1 ↑ | Brier ↓ | NLL ↓ | ECE ↓ |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-4B-Instruct-2507, no adapter (order-averaged) | 0.8417 | 0.0663 | 1.374 | 0.102 |
| LoRA (order-averaged) | **0.8792** | **0.0369** | **0.326** | **0.030** |
| LoRA, single canonical order (this code) | 0.8767 | 0.0391 | 0.344 | 0.032 |

The prespecified target of at least +5 points in top-1 was not met (+3.75). Score-type branches remain the weakest; for example, in `examples/requests.jsonl` the adapter rates a deployment that timed out twice as "Not disruptive" (p = 0.725).

A separate small final evaluation (60 private records, 90 branches) and a latency measurement are described in the [model card](https://huggingface.co/sennaLLMLearner/qwen3-4b-system-one-lora); neither is reproducible from this repository.
