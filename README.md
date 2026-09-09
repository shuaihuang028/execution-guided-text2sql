# Execution-Guided Post-Training for Text-to-SQL

This repository builds a reproducible post-training pipeline for a 1.5B code LLM on a
single 16 GB GPU. The first milestone implements the data contract, schema serialization,
prompt construction, and guarded SQLite execution needed for trustworthy evaluation and
preference-data generation.

## Current status

- [x] BIRD JSON/JSONL loader with stable example IDs
- [x] SQLite schema extraction (columns, types, PKs, and FKs)
- [x] Deterministic prompt construction and lexical schema pruning
- [x] Read-only execution sandbox with authorizer, timeout, and row limit
- [x] Ordered and multiset result comparison with NULL and numeric handling
- [x] Unit tests for the evaluation contract
- [x] Deterministic, resumable pre-SFT generation command
- [x] Pre-SFT benchmark with the guarded local evaluator
- [x] Official BIRD row-set evaluator parity analysis
- [x] QLoRA memory/throughput profiling on the target 16 GB GPU
- [x] Full one-epoch QLoRA SFT with resumable checkpoints
- [x] Full post-SFT execution evaluation
- [x] Paired pre/post-SFT SQL error taxonomy
- [x] Resumable train-only candidate sampling and execution-labeling pipeline
- [ ] Full execution-labeled preference-pair generation and analysis (running)
- [ ] Execution-labeled preference generation and DPO

## Preliminary baseline

These numbers use the repository's guarded local evaluator on all 1,534 examples from the
corrected BIRD dev release (2025-11-06). They are not labeled as official BIRD leaderboard
scores until evaluator parity is verified.

| Checkpoint | Strict local EX | BIRD row-set EX reproduction | Executable rate |
|---|---:|---:|---:|
| Pre-SFT, 8 tables / 128-char descriptions | 317 / 1,534 (20.66%) | — | 47.09% |
| Matched Pre-SFT, 6 tables / 96-char descriptions | 308 / 1,534 (20.08%) | 347 / 1,534 (22.62%) | 46.57% |
| One-epoch QLoRA SFT, matched prompt | 473 / 1,534 (30.83%) | 529 / 1,534 (34.49%) | 69.30% |

Against the matched prompt baseline, one-epoch SFT improves local EX by 10.76 percentage
points and executable rate by 22.73 points. It fixes 241 previously incorrect examples
while regressing 76 previously correct examples, for a net gain of 165. The largest
database-level gains are codebase_community (+42), student_club (+26), superhero (+26),
thrombosis_prediction (+17), and european_football_2 (+16).

The official BIRD evaluator compares `set(predicted_rows) == set(gold_rows)`, ignoring row
order and duplicate multiplicity. Reproducing that comparison inside the read-only sandbox
raises matched Base from 20.08% to 22.62% and SFT from 30.83% to 34.49%; SFT's gain is
11.86 percentage points (+182 correct). On SFT predictions, the strict and BIRD-style
labels agree on 96.22% of examples. Of 57 BIRD-style-only matches, 56 are duplicate-row
differences and one exceeds the local 100k-row cap; one strict-only match comes from local
numeric tolerance. These are evaluator-semantics reproductions, not official leaderboard
submissions, because execution still uses this project's guarded timeout and row limits.

Post-SFT error analysis identifies missing columns (397), wrong row sets (307), wrong
values (230), wrong projections (51), missing tables (34), and syntax errors (30) as the
largest remaining categories. Local EX by difficulty is 41.05% for simple, 20.99% for
moderate, and 11.69% for challenging examples. The paired analysis artifact preserves all
985 persistent failures and 76 regressions for targeted preference-data construction.

Three unusually expensive gold queries still exceed a 30-second local safety budget. They
remain in the denominator, so excluding them cannot inflate the reported local EX.

With question-aware pruning to at most eight tables and column descriptions capped at 128
characters, the 6,601-example training set has sequence lengths of p50 1,753, p95 5,226,
p99 7,439, and max 7,536 tokens with the 8-table baseline prompt. A worst-case QLoRA
profile reached 16,464 MiB allocated, so SFT uses a safer 6-table/96-character prompt:
p99 5,534, max 5,632 tokens, and a 6,144-token ceiling without
silently truncating target SQL.

The finalized QLoRA profile ran 200 optimizer steps with micro-batch 1 and gradient
accumulation 8. It processed 2,766,151 tokens in 20.7 minutes (2,228 tokens/s), peaked at
13,078 MiB allocated VRAM, and reduced mean completion-only loss from 0.247 over the first
20 steps to 0.187 over the final 20. The saved adapter contains 18,464,768 trainable
parameters (1.18% of the loaded model).

### Training-efficiency ablation

A paired one-step QLoRA profile on the same 2,354-token example isolates gradient
checkpointing. On this Windows/WDDM setup, the non-checkpointed run spills beyond the
GPU's 16,303 MiB dedicated-memory budget, while checkpointing stays safely inside it and
also avoids the severe paging slowdown.

| Gradient checkpointing | Sequence tokens | Peak allocated | Throughput |
|---|---:|---:|---:|
| Off | 2,354 | 18,288 MiB | 484 tok/s |
| On | 2,354 | 6,584 MiB | 1,797 tok/s |

The production 200-step configuration includes optimizer-state growth and a much broader
length distribution, so its measured peak is higher (13,078 MiB) than the one-step row.

Run the full one-epoch SFT job (checkpoints are written every 100 optimizer steps):

```bash
python scripts/train_sft.py \
  --model-path models/Qwen2.5-Coder-1.5B-Instruct \
  --data-path data/raw/train.jsonl \
  --database-root data/raw/bird/train/train_databases \
  --column-meanings data/raw/train_column_meaning.json \
  --train-manifest data/processed/train_manifest.jsonl \
  --validation-manifest data/processed/validation_manifest.jsonl \
  --output-dir artifacts/sft_qlora_final \
  --metrics-output results/sft_train_metrics.json
```

If interrupted, append `--resume-from-checkpoint latest`; the trainer restores adapter,
optimizer, scheduler, RNG, and dataloader progress from the newest checkpoint.

The full SFT run completed all 729 optimizer steps (5,825 micro-steps and 10,212,428
tokens) in 1.80 logged training hours across resumable segments. Mean completion-only
loss moved from 0.252 over steps 1-20 to 0.131 over steps 710-729. The fixed 128-example
held-out subset reached 0.1466 validation loss, with 13,078 MiB peak allocated VRAM. The
final adapter is saved under `artifacts/sft_qlora_final/final_adapter`.

Sample four SQL candidates per training example from that adapter and construct the main
generated-only preference dataset:

```bash
python scripts/generate_candidates.py \
  --model-path artifacts/sft_qlora_final/final_adapter \
  --data-path data/raw/train.jsonl \
  --database-root data/raw/bird/train/train_databases \
  --column-meanings data/raw/train_column_meaning.json \
  --manifest data/processed/train_manifest.jsonl \
  --output results/train_candidates.jsonl \
  --metrics-output results/train_candidate_metrics.json \
  --batch-size 4 --max-batch-input-tokens 10000 \
  --vectorized-candidate-max-prompt-tokens 3000 \
  --candidates-per-example 4 --temperature 0.7 --top-p 0.95

python scripts/build_preference_pairs.py \
  --data-path data/raw/train.jsonl \
  --database-root data/raw/bird/train/train_databases \
  --column-meanings data/raw/train_column_meaning.json \
  --manifest data/processed/train_manifest.jsonl \
  --candidates results/train_candidates.jsonl \
  --pairs-output data/processed/preference_pairs.jsonl \
  --labels-output results/train_candidate_labels.jsonl \
  --metrics-output results/preference_pair_metrics.json
```

Only examples containing both a generated execution-correct SQL and a generated negative
become main-experiment pairs. Executable-but-wrong SQL is preferred as the rejected hard
negative. Ground-truth fallback is disabled by default and is available only as an explicit
ablation via `--allow-gold-fallback`. All candidates come from the database-grouped Train
manifest; Dev remains held out for final evaluation.

Run DPO from the SFT adapter after preference construction:

```bash
python scripts/train_dpo.py \
  --adapter-path artifacts/sft_qlora_final/final_adapter \
  --preferences data/processed/preference_pairs.jsonl \
  --output-dir artifacts/dpo_qlora_final \
  --metrics-output results/dpo_train_metrics.json \
  --learning-rate 1e-5 --beta 0.1
```

The DPO train/validation split is grouped by database. The trainer removes overlength pairs
instead of silently truncating their SQL completion and saves loss, reward margin, validation,
throughput, runtime, and peak-VRAM metrics.

## Setup

```bash
conda env create -f environment.yml
conda activate text2sql
python -m pytest
python -m ruff check .
```

Inspect BIRD prompt lengths with the exact model tokenizer:

```bash
python scripts/inspect_data.py \
  --data-path /path/to/train.jsonl \
  --database-root /path/to/train_databases \
  --column-meanings /path/to/train_column_meaning.json \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct
```

The script does not download or redistribute BIRD databases. Point it at a local BIRD
download. Final benchmark numbers will use the official BIRD evaluator; the local matcher
is intended for guarded candidate labeling and diagnostics.

Create train/validation manifests grouped by database (not by question) to prevent schema
leakage:

```bash
text2sql-prepare-data --data-path /path/to/train.jsonl
```

Validate that every referenced database exists and smoke-test gold SQL through the guarded
executor:

```bash
text2sql-validate-data \
  --data-path /path/to/train.jsonl \
  --database-root /path/to/train_databases \
  --output results/data_validation.json
```
