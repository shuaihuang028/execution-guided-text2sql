# Execution-Guided Post-Training for Text-to-SQL

QLoRA SFT followed by DPO on `Qwen2.5-Coder-1.5B-Instruct` for the BIRD benchmark, trained on
one 16 GB RTX 5070 Ti. Every predicted query is run in a read-only SQLite sandbox and compared
with the gold result, and the same check is used both for evaluation and for labeling DPO
preference pairs.

```
prepare data -> sandboxed execution -> baseline -> QLoRA SFT
             -> sample candidates -> execution-labeled pairs -> DPO -> error analysis
```

## Results

BIRD dev (corrected 2025-11-06 release, all 1,534 examples). Hyperparameters were tuned on a
validation split taken from train and grouped by database; dev was only used for the final
numbers.

| Checkpoint | Strict local EX | BIRD row-set EX | Executable |
|---|---:|---:|---:|
| Instruct model, prompted | 308 (20.08%) | 347 (22.62%) | 46.57% |
| + QLoRA SFT, 1 epoch | 473 (30.83%) | 529 (34.49%) | 69.30% |
| + DPO | 528 (34.42%) | 589 (38.40%) | 76.88% |

The starting checkpoint is already instruction-tuned, so the first row is a prompted baseline,
not a base model. All three rows use the same prompt settings (6 tables, column descriptions cut
to 96 characters). I checked this by reproducing the recorded input token counts on 60 dev
examples before running DPO.

Paired comparison on the 1,531 examples where every query could be scored (exact two-sided
McNemar test):

| Transition | Fixed | Regressed | Net | p |
|---|---:|---:|---:|---:|
| Baseline -> SFT | 241 | 76 | +165 | 3.8e-21 |
| SFT -> DPO | 111 | 56 | +55 | 2.5e-05 |
| Baseline -> DPO | 295 | 75 | +220 | 5.8e-32 |

Both steps are significant at p < 1e-4.

DPO helps at every difficulty level: simple 41.05 -> 43.72, moderate 20.99 -> 27.31,
challenging 11.69 -> 13.42.

### Error breakdown, SFT vs DPO

| Category | SFT | DPO | Change |
|---|---:|---:|---:|
| Correct | 473 | 528 | +55 |
| missing_column | 397 | 263 | -134 |
| wrong_row_set | 307 | 327 | +20 |
| wrong_values | 230 | 265 | +35 |
| wrong_projection | 51 | 57 | +6 |
| missing_table | 34 | 33 | -1 |
| syntax_error | 30 | 37 | +7 |

Most of the gain is schema linking: missing-column errors drop by about a third. Some of the
queries that used to fail now run but return the wrong rows, which is why `wrong_row_set` and
`wrong_values` go up.

## Two evaluators

- **Strict local EX**: the matcher used for labeling and diagnostics. Order matters only when
  the gold query has `ORDER BY`; rows are compared as multisets, with NULL handling and numeric
  tolerance (`rel_tol=1e-6`, `abs_tol=1e-8`).
- **BIRD row-set EX**: reimplements the official check, `set(pred_rows) == set(gold_rows)`,
  which ignores both row order and duplicates.

| Checkpoint | Agreement | Local-only correct | BIRD-only correct |
|---|---:|---:|---:|
| Baseline | 97.20% | 2 | 41 |
| SFT | 96.22% | 1 | 57 |
| DPO | 95.89% | 1 | 62 |

For SFT, 56 of the 57 BIRD-only matches differ only in duplicate rows and one hits the local
100k-row cap. The single local-only match comes from numeric tolerance.

The BIRD column is computed inside this repo's sandbox (read-only authorizer, timeout, row cap),
not with the official script, and it is not a leaderboard submission. Three gold dev queries take
longer than the 30 s timeout; they are counted as failures.

## Data splits

Train/validation manifests are split by database, not by question, so the same schema never
appears on both sides (BIRD's hidden test set is database-disjoint too).

- Filtered train: 6,601 examples, split into train / validation by database
- Validation is used to pick learning rate, epochs, LoRA rank and DPO beta
- Dev is only used for the final tables above

## Preference pairs

Four candidates per training question are sampled from the SFT adapter at temperature 0.7 and
labeled by execution. A pair is kept only if the model produced both a correct and an incorrect
query for that question; gold SQL is not used as the chosen side (`--allow-gold-fallback` is
there only for ablation). An executable-but-wrong query is preferred as the rejected side, and
each question contributes at most one pair.

| | Count |
|---|---:|
| Training questions sampled | 5,825 |
| Candidates (raw / unique) | 23,300 / 18,246 |
| Questions with at least one correct candidate | 3,897 (66.9%) |
| Questions that produced a pair | 2,448 (42.0%) |
| Hard / easy negatives | 1,444 / 1,004 |

### How reliable are execution labels?

Matching the gold result on one database doesn't mean the query is right. How likely a wrong
query is to match by accident depends on what the gold query returns: a single number is easy to
hit with an unrelated aggregate, a multi-row result is not. I re-ran every gold query and bucketed
the examples by result shape:

| Gold result | Examples | Has a correct candidate | Pairs | Share of pairs |
|---|---:|---:|---:|---:|
| Multi-row or multi-column | 2,143 | 59.1% | 841 | 34.4% |
| Single text value | 1,242 | 74.7% | 547 | 22.3% |
| Single numeric value | 2,436 | 69.9% | 1,060 | 43.3% |
| Gold fails to run | 4 | 0.0% | 0 | 0.0% |

So 43.3% of the DPO pairs are labeled by a single-number match, the weakest kind of evidence.
The model also finds a correct candidate least often on the multi-row examples, where the label
would be most trustworthy. (No pair came from an empty gold result.) BIRD train has no
`difficulty` field, so difficulty can only be compared on dev.

### Does filtering the weak labels help?

Comparing 841 high-confidence pairs against all 2,448 would change both label quality and data
size at once. To separate the two I also trained on 841 pairs drawn at random from the full set
(seed 42; the tier mix stays close to the original: 34.0% multi-row, 42.8% numeric). All arms
start from the same SFT adapter with the same settings (lr 2e-5, beta 0.05, 1 epoch, 3,072-token
cap).

| Arm | Pairs | Used for training | Local EX | BIRD EX | Executable |
|---|---:|---:|---:|---:|---:|
| SFT only | - | - | 30.83% | 34.49% | 69.30% |
| DPO, all pairs | 2,448 | 2,029 | 34.42% | 38.40% | 76.88% |
| DPO, high-confidence | 841 | 691 | 31.88% | 35.79% | 72.24% |
| DPO, random 841 | 841 | 693 | 31.94% | 35.92% | 72.31% |

| Comparison | Delta | b / c | p |
|---|---:|---:|---:|
| high-confidence vs random 841 | +0.07 pp | 26 / 27 | 1.000 |
| all pairs vs high-confidence | -2.55 pp | 86 / 47 | 0.0009 |
| all pairs vs random 841 | -2.48 pp | 72 / 34 | 0.0003 |
| SFT vs high-confidence | +1.05 pp | 26 / 42 | 0.068 |
| SFT vs random 841 | +1.11 pp | 34 / 51 | 0.082 |

At equal size the filtered and random subsets are one example apart (489 vs 490 correct), so
filtering made no measurable difference. Cutting the data to 841 pairs, either way, cost about
2.5 points and left both arms no longer clearly better than SFT. If filtering is worth doing, it probably needs new high-confidence pairs to replace the
ones removed. The reported model uses all pairs.

## DPO setup

With `DPOTrainer(model=peft_model, ref_model=None)` the reference model is not obvious from the
code; it depends on the TRL version and on whether the policy is a PEFT model. On the pinned
versions, TRL saves a frozen copy of the starting adapter as `ref`, so the reference is the SFT
model. I checked this against the saved checkpoints over all 392 LoRA tensors:

| | Relative L2 distance from SFT adapter |
|---|---:|
| `ref` adapter | 0.0 |
| trained policy | 0.0264 |

`text2sql.training.dpo_verification` runs this check and `tests/test_dpo_verification.py`
covers it, including synthetic cases where the reference drifts or the policy doesn't move.

Run details: lr 2e-5, beta 0.05, 1 epoch, 254 optimizer steps, micro-batch 1 with gradient
accumulation 8. 299 of the 2,448 pairs were longer than 3,072 tokens and were dropped instead of
truncated; the remaining 2,149 were split by database into 2,029 train / 120 validation. Final
validation reward margin 0.190, preference accuracy 70.0%, peak 14,150 MiB allocated, 45.6
minutes.

The lr x beta sweep only ran 20 steps per setting, and reward accuracy bounced between 0.45 and
0.62, so the chosen setting is a reasonable pick rather than a tuned optimum.

Versions: `torch 2.11.0+cu128`, `transformers 4.57.6`, `trl 1.12.0`, `peft 0.20.0`,
`bitsandbytes 0.50.2`, Python 3.11.16.

## Training efficiency

### bf16 LoRA vs 4-bit QLoRA

Same 200 optimizer steps on both arms with the same example order (2,766,151 tokens):

| | 4-bit QLoRA (nf4) | bf16 LoRA | Change |
|---|---:|---:|---:|
| Base weights in memory | 1,515 MiB | 2,944 MiB | +94.3% |
| Peak allocated | 13,078 MiB | 13,565 MiB | +3.7% |
| Peak reserved | 28,726 MiB | 29,290 MiB | +2.0% |
| Throughput | 2,182 tok/s | 2,310 tok/s | +5.8% |
| Wall-clock | 21.1 min | 20.0 min | -5.5% |
| Median step | 6.07 s | 5.39 s | -11.2% |
| p95 step | 9.56 s | 11.00 s | +15.1% |
| Trainable parameters | 18.46M (1.18%) | 18.46M (1.18%) | same |

4-bit saves 1,429 MiB of weights but only 488 MiB at peak, because with ~6k-token sequences
the peak is mostly activations and the 151,936-wide LM-head logits. Dequantization also costs
about 6% throughput. For a 1.5B model on this card bf16 LoRA is the better choice; QLoRA would
matter more for a bigger model or shorter sequences. QLoRA does have a better p95 step time.

Peak reserved is above the card's 16,303 MiB on both arms because Windows (WDDM) spills to system
RAM instead of raising OOM. Peak allocated is the number to compare.

### Gradient checkpointing

One step on the same 2,354-token example:

| Gradient checkpointing | Peak allocated | Throughput |
|---|---:|---:|
| Off | 18,288 MiB | 484 tok/s |
| On | 6,584 MiB | 1,797 tok/s |

Without checkpointing the step goes past dedicated VRAM and gets slowed down by paging, so here
checkpointing is faster as well as smaller.

### Prompt length

With 8 tables and 128-character descriptions, training sequences reach p99 7,439 / max 7,536
tokens and the worst-case QLoRA profile peaked at 16,464 MiB, over budget. SFT uses 6 tables and
96 characters instead (p99 5,534, max 5,632) with a 6,144-token limit, so target SQL is never
truncated.

The SFT run: 729 optimizer steps (5,825 micro-steps, 10.2M tokens), 1.80 training hours across
resumed segments. Completion-only loss went from 0.252 (steps 1-20) to 0.131 (steps 710-729);
validation loss 0.1466 on a fixed 128-example subset; peak 13,078 MiB allocated.

## Limitations

- The BIRD number reimplements the official comparison but does not use the official script.
- 43.3% of pairs are labeled by single-number matches, and coverage is skewed toward examples
  with less informative gold results. Filtering didn't matter at 2,448 pairs; that may change
  with more data.
- The data-size effect is measured at two points only (841 and 2,448 pairs).
- The DPO sweep was short (20 steps per setting).
- Wall-clock times from runs at different times on this desktop GPU aren't comparable, so the
  ablation arms have no speed comparison.
- `peak_reserved_vram_mib` is inflated by WDDM spill on Windows.

## Setup

```bash
conda env create -f environment.yml
conda activate text2sql
python -m pytest          # 96 tests
python -m ruff check .
```

## Reproducing the pipeline

Prepare database-grouped manifests and validate that every gold query executes:

```bash
text2sql-prepare-data --data-path /path/to/train.jsonl
```

```bash
text2sql-validate-data --data-path /path/to/train.jsonl --database-root /path/to/train_databases --output results/data_validation.json
```

One-epoch QLoRA SFT (checkpoints every 100 optimizer steps; append
`--resume-from-checkpoint latest` to restore adapter, optimizer, scheduler, RNG and
dataloader position):

```bash
python scripts/train_sft.py --model-path models/Qwen2.5-Coder-1.5B-Instruct --data-path data/raw/train.jsonl --database-root data/raw/bird/train/train_databases --column-meanings data/raw/train_column_meaning.json --train-manifest data/processed/train_manifest.jsonl --validation-manifest data/processed/validation_manifest.jsonl --output-dir artifacts/sft_qlora_final --metrics-output results/sft_train_metrics.json
```

Sample candidates and build execution-labeled preference pairs:

```bash
python scripts/generate_candidates.py --model-path artifacts/sft_qlora_final/final_adapter --data-path data/raw/train.jsonl --database-root data/raw/bird/train/train_databases --column-meanings data/raw/train_column_meaning.json --manifest data/processed/train_manifest.jsonl --output results/train_candidates.jsonl --metrics-output results/train_candidate_metrics.json --batch-size 4 --candidates-per-example 4 --temperature 0.7 --top-p 0.95
```

```bash
python scripts/build_preference_pairs.py --data-path data/raw/train.jsonl --database-root data/raw/bird/train/train_databases --column-meanings data/raw/train_column_meaning.json --manifest data/processed/train_manifest.jsonl --candidates results/train_candidates.jsonl --pairs-output data/processed/preference_pairs.jsonl --labels-output results/train_candidate_labels.jsonl --metrics-output results/preference_pair_metrics.json
```

Stratify those labels by how much each one can be trusted:

```bash
python scripts/analyze_preference_confidence.py --data-path data/raw/train.jsonl --database-root data/raw/bird/train/train_databases --labels results/train_candidate_labels.jsonl --pairs data/processed/preference_pairs.jsonl --signatures-output results/preference_confidence_signatures.jsonl --metrics-output results/preference_confidence_metrics.json --high-confidence-pairs-output data/processed/preference_pairs_high_confidence.jsonl
```

DPO from the SFT adapter:

```bash
python scripts/train_dpo.py --adapter-path artifacts/sft_qlora_final/final_adapter --preferences data/processed/preference_pairs.jsonl --output-dir artifacts/dpo_qlora_final --metrics-output results/dpo_train_metrics.json --learning-rate 2e-5 --beta 0.05
```

Evaluate a checkpoint on dev, then reproduce BIRD row-set semantics and diff the errors:

```bash
python scripts/generate_baseline.py --model-path artifacts/dpo_qlora_final/final_adapter --data-path data/raw/dev/dev_20251106.jsonl --database-root data/raw/dev/package/dev_20240627/dev_databases --output results/dpo_predictions.jsonl --metrics-output results/dpo_generation_metrics.json --batch-size 16 --max-new-tokens 1100 --max-tables 6 --max-description-chars 96
```

```bash
python scripts/evaluate_predictions.py --data-path data/raw/dev/dev_20251106.jsonl --database-root data/raw/dev/package/dev_20240627/dev_databases --predictions results/dpo_predictions.jsonl --details-output results/dpo_evaluation.jsonl --summary-output results/dpo_summary.json
```

```bash
python scripts/compare_official_evaluator.py --data-path data/raw/dev/dev_20251106.jsonl --database-root data/raw/dev/package/dev_20240627/dev_databases --predictions results/dpo_predictions.jsonl --local-details results/dpo_evaluation.jsonl --details-output results/dpo_official_parity_details.jsonl --summary-output results/dpo_official_parity_summary.json
```

```bash
python scripts/analyze_errors.py --data-path data/raw/dev/dev_20251106.jsonl --baseline results/post_sft_evaluation.jsonl --candidate results/dpo_evaluation.jsonl --summary-output results/dpo_vs_sft_error_analysis.json --details-output results/dpo_vs_sft_error_details.jsonl
```

Draw the size-matched random control and compare arms with paired McNemar tests:

```bash
python scripts/sample_preference_subset.py --pairs data/processed/preference_pairs.jsonl --output data/processed/preference_pairs_random841.jsonl --size 841 --seed 42 --signatures results/preference_confidence_signatures.jsonl --metrics-output results/random_subset_metrics.json
```

```bash
python scripts/compare_checkpoints.py --evaluation "sft=results/post_sft_evaluation.jsonl" --evaluation "dpo_full=results/dpo_evaluation.jsonl" --evaluation "dpo_high_conf=results/dpo_hc_evaluation.jsonl" --evaluation "dpo_random=results/dpo_rand_evaluation.jsonl" --compare "dpo_high_conf:dpo_random" --compare "dpo_full:dpo_high_conf" --compare "sft:dpo_full" --output results/significance_data_selection.json
```

Run the LoRA/QLoRA ablation (drop `--no-load-in-4bit` for the QLoRA arm):

```bash
python scripts/profile_sft.py --model-path models/Qwen2.5-Coder-1.5B-Instruct --data-path data/raw/train.jsonl --database-root data/raw/bird/train/train_databases --column-meanings data/raw/train_column_meaning.json --manifest data/processed/train_manifest.jsonl --max-steps 200 --no-load-in-4bit --metrics-output results/ablation_lora_bf16.json --step-log-output results/ablation_lora_bf16_steps.jsonl
```

This repository does not download or redistribute BIRD databases; point the scripts at a
local BIRD download.
