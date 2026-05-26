# TriLens

[![Homepage](https://img.shields.io/badge/Homepage-TriLens-blue)](https://tosakaucw.github.io/TriLens/)

TriLens is a white-box hallucination detection toolkit based on per-layer logit-lens entropy. It extracts module-wise entropy trajectories from transformer decoder layers and trains lightweight probes for hallucination detection.

## Core idea

For each generated response, TriLens records three per-layer readouts:

- `H_a`: entropy from self-attention outputs
- `H_m`: entropy from MLP / FFN outputs
- `H_x`: entropy from residual stream states

These features are written as JSONL rows and consumed by evaluation scripts for linear or MLP probe training.

## Repository layout

```text
src/
  kca.py            # TriLens hook capture and logit-lens entropy features
  utils.py          # TriLens probe model definitions
  config.py         # training configuration dataclass
scripts/
  compute_kca_*.py  # dataset-specific TriLens feature extraction
  data_utils.py     # shared dataset parsing and candidate construction helpers
  run_kca_eval.py   # within-dataset probe evaluation
  run_cross_dataset_eval.py
  run_multi_dataset_sweep.py
  analyze_kca_diagnostic.py
```

## Requirements

This repository intentionally does not pin an environment. Install the ML stack appropriate for your CUDA / PyTorch setup. The code expects at least:

- `torch`
- `transformers`
- `numpy`
- `scikit-learn`
- `rich`
- `jsonlines`

TriLens feature extraction is designed for Llama/Qwen/Gemma-style causal language models whose decoder layers are available under `model.model.layers`.

## Quick start

Compute TriLens features for HaluEval:

```bash
python scripts/compute_kca_halueval.py \
  --model_name_or_path "$MODEL" \
  --data_path "data/halueval/qa_data.json" \
  --output_path "outputs/MODEL/kca_halueval_10k.jsonl"
```

Compute features for other supported datasets:

```bash
python scripts/compute_kca_squad2.py \
  --model_name_or_path "$MODEL" \
  --data_path "data/squad/dev-v2.0.json" \
  --output_path "outputs/MODEL/kca_squad2_10k.jsonl"

python scripts/compute_kca_hotpotqa.py \
  --model_name_or_path "$MODEL" \
  --data_path "data/hotpotqa/hotpot_dev_distractor_v1.json" \
  --output_path "outputs/MODEL/kca_hotpotqa_distractor_10k.jsonl"

python scripts/compute_kca_triviaqa.py \
  --model_name_or_path "$MODEL" \
  --data_path "data/triviaqa/verified-web-dev.json" \
  --output_path "outputs/MODEL/kca_triviaqa_10k.jsonl"
```

Evaluate a probe on extracted features:

```bash
python scripts/run_kca_eval.py \
  --input "outputs/MODEL/kca_halueval_10k.jsonl" \
  --features H_a,H_m,H_x \
  --aggregation first \
  --probe mlp \
  --seeds 5 \
  --output_json "outputs/eval/kca_eval_results_10k.jsonl" \
  --tag "MODEL_halueval_10k"
```

## Notes

- Keep `--attn_implementation eager` when attention tensors are required by the selected model/backend.
- Use `--lens_dtype float32` if logit-lens entropy produces overflow or NaN values in fp16.
- Generated datasets, model outputs, logs, and checkpoints should live under ignored directories such as `data/`, `outputs/`, `saves/`, and `logs/`.

## License

MIT License. See [LICENSE](LICENSE).
