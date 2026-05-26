# TriLens

[![Homepage](https://img.shields.io/badge/Homepage-TriLens-blue)](https://anonymous.4open.science/w/TriLens)

TriLens is a lightweight white-box hallucination detection toolkit based on per-layer logit-lens entropy. Instead of training probes on high-dimensional hidden states, TriLens summarizes how internal certainty evolves across transformer depth by reading three module-wise states through the model's vocabulary lens.

## Overview

For each generated response, TriLens records three entropy trajectories at every decoder layer:

- `H_a`: entropy from self-attention outputs
- `H_m`: entropy from MLP / FFN outputs
- `H_x`: entropy from residual-stream states

These compact `3L`-dimensional features are exported as JSONL rows and used by lightweight linear or MLP probes for hallucination detection. The design is intended to make the detector efficient, easy to inspect, and reproducible across datasets and models.

## Repository layout

```text
src/
  kca.py            # TriLens hook capture and logit-lens entropy extraction
  utils.py          # probe model definitions
  config.py         # training configuration dataclass
scripts/
  compute_kca_*.py  # dataset-specific TriLens feature extraction
  data_utils.py     # shared dataset parsing and candidate construction helpers
  run_kca_eval.py   # within-dataset probe evaluation
  run_cross_dataset_eval.py
  run_multi_dataset_sweep.py
  analyze_kca_diagnostic.py
requirements.txt    # Python dependencies used in our experiments
```

## Installation

We recommend creating a clean Python environment before installing the dependencies:

```bash
conda create -n trilens python=3.10 -y
conda activate trilens
pip install -r requirements.txt
```

The pinned `requirements.txt` is provided to make review and reproduction easier. Depending on your CUDA driver and hardware, you may need to install the matching PyTorch wheel from the official PyTorch index before or after installing the remaining dependencies.

TriLens feature extraction is designed for Llama/Qwen/Gemma-style causal language models whose decoder layers are available under `model.model.layers`.

## Quick start

Set the model path or Hugging Face model identifier:

```bash
export MODEL="path-or-hf-id-to-your-causal-lm"
```

Compute TriLens features for HaluEval:

```bash
python scripts/compute_kca_halueval.py \
  --model_name_or_path "$MODEL" \
  --data_path "data/halueval/qa_data.json" \
  --output_path "outputs/MODEL/kca_halueval_10k.jsonl"
```

Evaluate a probe on the extracted features:

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

## Supported dataset scripts

TriLens includes feature extraction scripts for the QA benchmarks used in our experiments:

```bash
python scripts/compute_kca_halueval.py \
  --model_name_or_path "$MODEL" \
  --data_path "data/halueval/qa_data.json" \
  --output_path "outputs/MODEL/kca_halueval_10k.jsonl"

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

## Evaluation utilities

Run within-dataset evaluation:

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

Run cross-dataset evaluation or multi-dataset sweeps:

```bash
python scripts/run_cross_dataset_eval.py --help
python scripts/run_multi_dataset_sweep.py --help
```

Inspect diagnostic patterns:

```bash
python scripts/analyze_kca_diagnostic.py --help
```

## Practical notes

- Keep `--attn_implementation eager` when attention tensors are required by the selected model/backend.
- Use `--lens_dtype float32` if logit-lens entropy produces overflow or NaN values in fp16.
- Store datasets, extracted features, logs, and checkpoints under ignored directories such as `data/`, `outputs/`, `saves/`, and `logs/`.
- The repository does not include benchmark datasets or model weights; please obtain them from their original sources and follow their licenses.

## License

MIT License. See [LICENSE](LICENSE).
