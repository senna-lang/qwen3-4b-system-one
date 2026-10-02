# Public accuracy benchmark: protocol

This protocol was fixed and committed **before the first run**. Results are reported as measured, whether the adapter helps or hurts; datasets, prompts, and metrics are not changed after seeing results.

## Systems

- **Base:** Qwen/Qwen3-4B-Instruct-2507 without the adapter.
- **LoRA:** the same model with `sennaLLMLearner/qwen3-4b-system-one-lora`.

Both use the prompt in [METHOD.md](METHOD.md), one canonical option order (the order listed below), the PyTorch backend, and answer-letter readout. The baseline is the same loaded model with the adapter disabled.

## Tasks

500 rows per task, sampled with `dataset.shuffle(seed=0).select(range(500))` from the split below.

| Task | Hugging Face dataset | Split | Kind | State | Question | Options (in order) |
| --- | --- | --- | --- | --- | --- | --- |
| AG News | `fancyzhx/ag_news` | test | choice | article text | What is the topic of this news article? | World, Sports, Business, Sci/Tech |
| MASSIVE scenario | `mteb/amazon_massive_scenario` (`en`) | test | choice | utterance | Which scenario does this request to a voice assistant belong to? | the 18 scenario labels, alphabetical |
| MNLI | `nyu-mll/glue` (`mnli`) | validation_matched | choice | `Premise: …` / `Hypothesis: …` | What is the relationship between the premise and the hypothesis? | entailment, neutral, contradiction |
| BoolQ | `google/boolq` | validation | noul | passage | the dataset question, capitalized, with `?` | No, Yes |
| SST-5 | `SetFit/sst5` | test | score | sentence | How positive is the sentiment of this text? | very negative, negative, neutral, positive, very positive |

Option labels are the datasets' own label names.

## Metrics

- Accuracy of the highest-probability option, with a 95% bootstrap interval (2,000 resamples, seed 0).
- Paired accuracy difference LoRA − base with a 95% bootstrap interval.
- SST-5 additionally reports mean absolute error in levels.

## Data provenance

None of these datasets were used to train or select the adapter (training sources: AMI/ICSI via QMSum, GitHub issues, NASA ASRS, NHTSA, public regulations). AG News, SST-5, and BoolQ were used in earlier experiments of this project with a different, discarded model design; they did not influence this adapter.

## Run

```bash
pip install -e '.[torch,bench]'
python scripts/bench_accuracy.py --output outputs/accuracy
```

Row-level probabilities for both systems are written to `outputs/accuracy/*.predictions.jsonl`, and the summary to `outputs/accuracy/summary.json`.
