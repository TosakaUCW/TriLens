<div align="center">
  <h1>TriLens: Per-Layer Logit-Lens Entropy for White-Box Hallucination Detection</h1>

  <p>
    <strong>Bohan Yang<sup>1,2,3</sup>, Yijun Gong<sup>5</sup>, Zhi Zhang<sup>6</sup>, Ge Zhang<sup>1</sup>, Wenpeng Xing<sup>1,2,*</sup>, Meng Han<sup>1,2,4</sup></strong>
  </p>

  <p>
    <sup>1</sup>Zhejiang University &nbsp;·&nbsp;
    <sup>2</sup>Binjiang Institute of Zhejiang University &nbsp;·&nbsp;
    <sup>3</sup>Beijing Normal-Hong Kong Baptist University<br>
    <sup>4</sup>GenTel.io &nbsp;·&nbsp;
    <sup>5</sup>Great Bay University &nbsp;·&nbsp;
    <sup>6</sup>University of California San Diego<br>
    <sup>*</sup>Corresponding author
  </p>

  <p><strong>Findings of EMNLP 2026</strong></p>

  <p>
    <a href="docs/assets/paper.pdf"><img src="https://img.shields.io/badge/Paper-PDF-b31b1b.svg" alt="Paper PDF"></a>
    <a href="https://arxiv.org/abs/2606.01033"><img src="https://img.shields.io/badge/arXiv-2606.01033-b31b1b.svg" alt="arXiv"></a>
    <a href="https://tosakaucw.github.io/TriLens/"><img src="https://img.shields.io/badge/Project-Page-2f80ed.svg" alt="Project page"></a>
    <a href="https://2026.emnlp.org/"><img src="https://img.shields.io/badge/EMNLP_2026-Findings-8a2be2.svg" alt="Findings of EMNLP 2026"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
  </p>
</div>

## Overview

TriLens is a lightweight white-box hallucination detection toolkit based on per-layer logit-lens entropy. Instead of training probes on high-dimensional hidden states, TriLens summarizes how internal certainty evolves across transformer depth by reading three module-wise states through the model's vocabulary lens.

For each generated response, TriLens records three entropy trajectories at every decoder layer:

- `H_a`: entropy from self-attention outputs
- `H_m`: entropy from MLP / FFN outputs
- `H_x`: entropy from residual-stream states

These compact `3L`-dimensional features are used by lightweight linear or MLP probes. The resulting detector is efficient, inspectable, and reproducible across datasets and models.

## Quick Start

Create a clean environment and install the dependencies:

```bash
conda create -n trilens python=3.10 -y
conda activate trilens
pip install -r requirements.txt
```

Set a local model path or Hugging Face model identifier, then extract TriLens features from HaluEval:

```bash
export MODEL="path-or-hf-id-to-your-causal-lm"
python scripts/compute_kca_halueval.py \
  --model_name_or_path "$MODEL" \
  --data_path "data/halueval/qa_data.json" \
  --output_path "outputs/MODEL/kca_halueval_10k.jsonl"
```

Train and evaluate a lightweight probe on the extracted features:

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

The pinned `requirements.txt` supports review and reproduction. Depending on your CUDA driver and hardware, you may need to install the matching PyTorch wheel from the official PyTorch index. Feature extraction currently targets Llama/Qwen/Gemma-style causal language models whose decoder layers are exposed under `model.model.layers`.

## Additional Usage

<details>
<summary><strong>Output Format / Custom Dataset</strong></summary>

Feature extraction writes one JSON object per candidate response to a JSONL file. The core fields are:

| Field | Meaning |
| --- | --- |
| `index` | Source-example group ID. Give paired supported and hallucinated responses the same value so the evaluator keeps them in the same train/test partition. |
| `candidate_index` | Candidate number within the source example. |
| `task`, `pairing`, `response_type` | Dataset/task metadata and candidate construction mode. |
| `label` | Binary target: `0` for a supported/correct response and `1` for a hallucinated response. |
| `num_layers`, `num_response_tokens` | Dimensions `L` and `T` of the extracted trajectories. |
| `core_positions` | Token boundaries used to locate the prompt and response; `response_start` is the first response token. |
| `H_a`, `H_m`, `H_x` | Main entropy trajectories, each stored as an `L × T` nested list. |
| `H_pre`, `H_p`, `JSD_am` | Auxiliary per-layer, per-token entropy or divergence features with the same `L × T` shape. |
| `JSD_to_final` | Optional DoLa-style layer-contrast feature, emitted with `--emit_dola`. |
| `meta`, `id` | Dataset-specific metadata and an optional original example ID. |

A schematic row looks like this:

```json
{
  "index": 0,
  "candidate_index": 0,
  "task": "custom",
  "response_type": "right",
  "label": 0,
  "num_layers": 2,
  "num_response_tokens": 2,
  "core_positions": {
    "user_prompt_start": 0,
    "user_prompt_end": 128,
    "response_start": 128
  },
  "H_a": [[2.91, 2.74], [2.61, 2.43]],
  "H_m": [[3.05, 2.88], [2.77, 2.52]],
  "H_x": [[2.83, 2.65], [2.48, 2.21]]
}
```

To use a custom dataset:

1. Adapt the closest `scripts/compute_kca_*.py` loader so each source example yields candidates containing `prompt`, `response`, `label`, and `response_type`. Use `0` for supported/correct responses and `1` for hallucinated responses.
2. Tokenize the prompt and response separately, concatenate them, and pass the first response-token position as `response_start` to `compute_kca_features` in `src/kca.py`.
3. Write the returned feature arrays using the schema above. Keep the same `index` for candidates derived from the same source example.
4. Pass the resulting JSONL file to `scripts/run_kca_eval.py`. The evaluator expects `index`, `label`, `H_x`, every feature named by `--features`, and at least 50 rows.

`scripts/data_utils.py` provides JSON/JSONL loading and candidate-construction helpers. There is not yet a universal custom-data extraction CLI, so adapting a dataset script is the supported integration path.

</details>

<details>
<summary><strong>Repository layout</strong></summary>

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

</details>

<details>
<summary><strong>Dataset-specific extraction scripts</strong></summary>

The repository includes extraction scripts for the four QA benchmarks used in our experiments:

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

</details>

<details>
<summary><strong>Evaluation utilities and feature options</strong></summary>

The within-dataset evaluator accepts the raw feature blocks `H_a`, `H_m`, `H_x`, `H_pre`, `H_p`, `JSD_am`, and `JSD_to_final`, plus the fixed-layer reductions `TriMeanL`, `TriMaxL`, and `TriMinL`. Use `--aggregation first`, `mean`, or `mean_top3` to select the first response token, all response tokens, or the first three response tokens. Both `linear` and `mlp` probes are available.

Run cross-dataset evaluation or multi-dataset sweeps:

```bash
python scripts/run_cross_dataset_eval.py --help
python scripts/run_multi_dataset_sweep.py --help
```

Inspect diagnostic patterns:

```bash
python scripts/analyze_kca_diagnostic.py --help
```

Show all within-dataset evaluation options:

```bash
python scripts/run_kca_eval.py --help
```

</details>

## Practical Notes

- Keep `--attn_implementation eager` when attention tensors are required by the selected model/backend.
- Use `--lens_dtype float32` if logit-lens entropy produces overflow or NaN values in fp16.
- Store datasets, extracted features, logs, and checkpoints under ignored directories such as `data/`, `outputs/`, `saves/`, and `logs/`.
- The repository does not include benchmark datasets or model weights; obtain them from their original sources and follow their licenses.

## Citation

The EMNLP 2026 proceedings metadata is not yet available. Please cite the current arXiv version:

```bibtex
@misc{yang2026trilens,
  title         = {{TriLens}: Per-Layer Logit-Lens Entropy for White-Box Hallucination Detection},
  author        = {Yang, Bohan and Gong, Yijun and Zhang, Zhi and Zhang, Ge and Xing, Wenpeng and Han, Meng},
  year          = {2026},
  eprint        = {2606.01033},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2606.01033},
  url           = {https://arxiv.org/abs/2606.01033}
}
```

## License

MIT License. See [LICENSE](LICENSE).
