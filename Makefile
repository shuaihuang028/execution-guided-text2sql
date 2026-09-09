.PHONY: test prepare-data validate-data inspect-data generate-baseline profile train-sft generate-preferences analyze-confidence train-dpo evaluate ablation

test:
	python -m pytest -q

prepare-data:
	python scripts/prepare_data.py --data-path $(DATA_PATH) --output-directory data/processed

validate-data:
	python scripts/validate_data.py --data-path $(DATA_PATH) --database-root $(DATABASE_ROOT) --output results/data_validation.json

inspect-data:
	python scripts/inspect_data.py --data-path $(DATA_PATH) --database-root $(DATABASE_ROOT) --model $(MODEL)

generate-baseline:
	python scripts/generate_baseline.py --model-path $(MODEL_PATH) --data-path $(DATA_PATH) --database-root $(DATABASE_ROOT) --column-meanings $(COLUMN_MEANINGS) --manifest $(MANIFEST) --output results/baseline_predictions.jsonl --metrics-output results/baseline_generation_metrics.json

profile:
	python scripts/profile_sft.py --model-path $(MODEL_PATH) --data-path $(DATA_PATH) --database-root $(DATABASE_ROOT) --column-meanings $(COLUMN_MEANINGS) --manifest data/processed/train_manifest.jsonl --metrics-output results/sft_profile.json --step-log-output results/sft_profile_steps.jsonl

train-sft:
	python scripts/train_sft.py --model-path $(MODEL_PATH) --data-path $(DATA_PATH) --database-root $(DATABASE_ROOT) --column-meanings $(COLUMN_MEANINGS) --train-manifest data/processed/train_manifest.jsonl --validation-manifest data/processed/validation_manifest.jsonl --output-dir artifacts/sft_qlora_final --metrics-output results/sft_train_metrics.json

generate-preferences:
	python scripts/generate_candidates.py --model-path artifacts/sft_qlora_final/final_adapter --data-path $(DATA_PATH) --database-root $(DATABASE_ROOT) --column-meanings $(COLUMN_MEANINGS) --manifest data/processed/train_manifest.jsonl --output results/train_candidates.jsonl --metrics-output results/train_candidate_metrics.json
	python scripts/build_preference_pairs.py --data-path $(DATA_PATH) --database-root $(DATABASE_ROOT) --column-meanings $(COLUMN_MEANINGS) --manifest data/processed/train_manifest.jsonl --candidates results/train_candidates.jsonl --pairs-output data/processed/preference_pairs.jsonl --labels-output results/train_candidate_labels.jsonl --metrics-output results/preference_pair_metrics.json

analyze-confidence:
	python scripts/analyze_preference_confidence.py --data-path $(DATA_PATH) --database-root $(DATABASE_ROOT) --labels results/train_candidate_labels.jsonl --pairs data/processed/preference_pairs.jsonl --signatures-output results/preference_confidence_signatures.jsonl --metrics-output results/preference_confidence_metrics.json --high-confidence-pairs-output data/processed/preference_pairs_high_confidence.jsonl

train-dpo:
	python scripts/train_dpo.py --adapter-path artifacts/sft_qlora_final/final_adapter --preferences data/processed/preference_pairs.jsonl --output-dir artifacts/dpo_qlora_final --metrics-output results/dpo_train_metrics.json

ablation:
	python scripts/profile_sft.py --model-path $(MODEL_PATH) --data-path $(DATA_PATH) --database-root $(DATABASE_ROOT) --column-meanings $(COLUMN_MEANINGS) --manifest data/processed/train_manifest.jsonl --max-steps 200 --load-in-4bit --metrics-output results/ablation_qlora_4bit.json --step-log-output results/ablation_qlora_4bit_steps.jsonl
	python scripts/profile_sft.py --model-path $(MODEL_PATH) --data-path $(DATA_PATH) --database-root $(DATABASE_ROOT) --column-meanings $(COLUMN_MEANINGS) --manifest data/processed/train_manifest.jsonl --max-steps 200 --no-load-in-4bit --metrics-output results/ablation_lora_bf16.json --step-log-output results/ablation_lora_bf16_steps.jsonl

evaluate:
	python scripts/evaluate_predictions.py --data-path $(DATA_PATH) --database-root $(DATABASE_ROOT) --predictions $(PREDICTIONS) --details-output results/evaluation_details.jsonl --summary-output results/evaluation_summary.json
