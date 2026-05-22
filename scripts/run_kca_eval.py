#!/usr/bin/env python3
"""Evaluation runner for TriLens feature probes.

The runner builds feature vectors from module-wise entropy readouts and
trains either a linear classifier or a lightweight MLP probe.

    * Split: grouped train/test split by sample index when available.
    * Classifiers:
        - 'linear' -- sklearn LogisticRegression (L2, C=1, max_iter=2000).
        - 'mlp'    -- src.utils.TriLensProbe.
    * Features: any subset of {H_a, H_m, H_x, JSD_am}, plus fixed
      L-dimensional reductions {TriMeanL, TriMaxL, TriMinL}.
    * Aggregation:
        - 'first'     -- response position 0 only.
        - 'mean'      -- mean over all response tokens.
        - 'mean_top3' -- mean over first min(T, 3) tokens.
    * Optional post-assembly reduction:
        - 'none'      -- keep the assembled feature vector as is.
        - 'pca_to_l'  -- fit PCA on the training split and project to L dimensions.
    * Repeats over N seeds, reports test AUROC mean +/- std.

Usage examples::

    # Single cell: Qwen2.5 SQuAD2, H_x alone, linear probe, 5 seeds.
    python scripts/run_kca_eval.py \
        --input outputs/Qwen2.5-7B-Instruct/kca_squad2_500.jsonl \
        --features H_x --aggregation mean --probe linear --seeds 5

    # Match TriLens setup exactly (MLP probe, mean aggregation).
    python scripts/run_kca_eval.py \
        --input outputs/Qwen2.5-7B-Instruct/kca_squad2_500.jsonl \
        --features H_a,H_m,H_x,JSD_am --aggregation mean --probe mlp --seeds 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils import TriLensProbe


FEATURE_KEYS = ("H_a", "H_m", "H_x", "H_pre", "H_p", "JSD_am", "JSD_to_final")
SPECIAL_FEATURE_KEYS = ("TriMeanL", "TriMaxL", "TriMinL")


# --------------------------------------------------------------------------- #
# Feature assembly
# --------------------------------------------------------------------------- #

def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def aggregate(arr: np.ndarray, mode: str) -> np.ndarray:
    """arr: [L, T]; returns [L] vector."""
    if arr.size == 0 or arr.shape[1] == 0:
        return np.full(arr.shape[0], np.nan, dtype=np.float32)
    if mode == "first":
        return arr[:, 0].astype(np.float32)
    if mode == "mean":
        return np.nanmean(arr, axis=1).astype(np.float32)
    if mode == "mean_top3":
        k = min(arr.shape[1], 3)
        return np.nanmean(arr[:, :k], axis=1).astype(np.float32)
    raise ValueError(f"unknown aggregation: {mode}")


def parse_feature_spec(spec: str) -> Tuple[str, int]:
    """Allow 'H_x' or 'H_x*3' to repeat a feature block for matched-dim controls."""
    if "*" not in spec:
        return spec, 1
    name, mult = spec.split("*", 1)
    name = name.strip()
    mult = mult.strip()
    if not mult.isdigit() or int(mult) <= 0:
        raise ValueError(f"invalid feature repetition spec: {spec}")
    return name, int(mult)


def _aggregate_raw_feature(rows: Sequence[dict], feat: str, aggregation: str) -> np.ndarray:
    per_sample = []
    for r in rows:
        arr = np.array(r[feat], dtype=np.float32)  # [L, T]
        per_sample.append(aggregate(arr, aggregation))
    return np.stack(per_sample)  # [N, L]


def _assemble_special_feature(rows: Sequence[dict], feat: str, aggregation: str) -> np.ndarray:
    trio = np.stack(
        [
            _aggregate_raw_feature(rows, "H_a", aggregation),
            _aggregate_raw_feature(rows, "H_m", aggregation),
            _aggregate_raw_feature(rows, "H_x", aggregation),
        ],
        axis=1,  # [N, 3, L]
    )
    if feat == "TriMeanL":
        return np.mean(trio, axis=1, dtype=np.float32)
    if feat == "TriMaxL":
        return np.max(trio, axis=1)
    if feat == "TriMinL":
        return np.min(trio, axis=1)
    raise ValueError(f"unknown special feature: {feat}")


def infer_layer_dim(rows: Sequence[dict]) -> int:
    if not rows or "H_x" not in rows[0]:
        raise KeyError("cannot infer layer dimension without H_x in the input rows")
    arr = np.array(rows[0]["H_x"], dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected H_x to have shape [L, T], got {arr.shape}")
    return int(arr.shape[0])


def assemble(rows: Sequence[dict], features: Sequence[str], aggregation: str) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (X [N, sum_L], y [N])."""
    if not rows:
        raise ValueError("no rows loaded from input jsonl")
    parsed_features = [parse_feature_spec(f) for f in features]
    for f, _repeat in parsed_features:
        if f in FEATURE_KEYS:
            if f not in rows[0]:
                available = sorted(rows[0].keys())
                raise KeyError(
                    f"feature '{f}' is missing from input rows. "
                    f"Available keys include: {available}"
                )
            continue
        if f in SPECIAL_FEATURE_KEYS:
            missing = [base for base in ("H_a", "H_m", "H_x") if base not in rows[0]]
            if missing:
                available = sorted(rows[0].keys())
                raise KeyError(
                    f"special feature '{f}' requires {missing}, but they are missing. "
                    f"Available keys include: {available}"
                )
            continue
        raise ValueError(f"unknown feature {f}; must be in {FEATURE_KEYS + SPECIAL_FEATURE_KEYS}")
    X_chunks: List[np.ndarray] = []
    for feat, repeat in parsed_features:
        if feat in FEATURE_KEYS:
            chunk = _aggregate_raw_feature(rows, feat, aggregation)
        else:
            chunk = _assemble_special_feature(rows, feat, aggregation)
        for _ in range(repeat):
            X_chunks.append(chunk.copy())
    X = np.concatenate(X_chunks, axis=1)
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


def extract_groups(rows: Sequence[dict], prefix: str | None = None) -> np.ndarray:
    """Returns one group id per row, using the original sample index."""
    groups = []
    for i, row in enumerate(rows):
        if "index" not in row:
            raise KeyError(f"row {i} is missing 'index'; cannot do group-aware split")
        group = row["index"]
        groups.append(f"{prefix}:{group}" if prefix is not None else str(group))
    return np.asarray(groups, dtype=object)


def grouped_split_indices(
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split rows while keeping each original example pair in one partition."""
    if len(y) != len(groups):
        raise ValueError(f"y/groups length mismatch: {len(y)} vs {len(groups)}")
    if len(y) < 2:
        raise ValueError("need at least 2 rows to split")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y, groups=groups))

    overlap = np.intersect1d(groups[train_idx], groups[test_idx])
    if overlap.size != 0:
        raise RuntimeError(f"group leakage detected for {overlap.size} groups")
    return train_idx, test_idx


def grouped_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_idx, test_idx = grouped_split_indices(y, groups, test_size=test_size, seed=seed)
    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
        groups[train_idx],
        groups[test_idx],
    )


# --------------------------------------------------------------------------- #
# Classifiers
# --------------------------------------------------------------------------- #

def train_linear(X_tr, y_tr, X_te, y_te, seed: int) -> float:
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    clf.fit(X_tr, y_tr)
    scores = clf.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(y_te, scores))


def train_mlp(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    seed: int,
    groups_tr: np.ndarray | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 5e-4,
    device: str = "cuda",
) -> float:
    """Train the exact TriLens Probe architecture on the given features."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tr_t = torch.from_numpy(X_tr).float()
    y_tr_t = torch.from_numpy(y_tr).float()
    X_te_t = torch.from_numpy(X_te).float()

    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")

    # Hold out 10% of train for LR scheduler, keeping paired rows together.
    if groups_tr is not None and len(np.unique(groups_tr)) >= 2:
        tr_idx_np, val_idx_np = grouped_split_indices(y_tr, groups_tr, test_size=0.1, seed=seed)
        tr_idx = torch.from_numpy(tr_idx_np).long()
        val_idx = torch.from_numpy(val_idx_np).long()
    else:
        n_tr = len(X_tr_t)
        n_val = max(1, int(0.1 * n_tr))
        perm = torch.randperm(n_tr, generator=torch.Generator().manual_seed(seed))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

    train_ds = TensorDataset(X_tr_t[tr_idx], y_tr_t[tr_idx])
    val_ds = TensorDataset(X_tr_t[val_idx], y_tr_t[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=max(8, batch_size))

    model = TriLensProbe(input_dim=X_tr.shape[1]).to(dev)
    criterion = nn.BCELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=5)

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optim.zero_grad()
            out = model(xb).squeeze(1)
            loss = criterion(out, yb)
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            val_losses = []
            for xb, yb in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                out = model(xb).squeeze(1)
                val_losses.append(criterion(out, yb).item())
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        sched.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = model(X_te_t.to(dev)).squeeze(1).cpu().numpy()

    return float(roc_auc_score(y_te, scores))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TriLens eval runner (TriLens-aligned protocol).")
    p.add_argument("--input", type=str, required=True, help="TriLens jsonl file.")
    p.add_argument(
        "--features",
        type=str,
        default="H_x",
        help="Comma-separated feature specs. Raw blocks: {H_a,H_m,H_x,H_pre,H_p,JSD_am,JSD_to_final}; "
             "fixed L-dim reductions: {TriMeanL,TriMaxL,TriMinL}; repetition syntax H_x*3 is also supported.",
    )
    p.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        choices=["first", "mean", "mean_top3"],
    )
    p.add_argument(
        "--reduce",
        type=str,
        default="none",
        choices=["none", "pca_to_l"],
        help="Optional post-assembly dimensionality control. pca_to_l fits PCA on the training split only.",
    )
    p.add_argument("--probe", type=str, default="linear", choices=["linear", "mlp"])
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seeds", type=int, default=5, help="Number of independent splits / MLP initializations.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_json", type=str, default=None, help="Where to append the result (optional).")
    p.add_argument("--tag", type=str, default=None, help="Optional tag for the result row (e.g. model_dataset).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    features = [f.strip() for f in args.features.split(",") if f.strip()]

    rows = load_jsonl(Path(args.input))
    X, y = assemble(rows, features, args.aggregation)
    layer_dim = infer_layer_dim(rows)
    groups = extract_groups(rows)
    if X.shape[0] < 50:
        raise ValueError(f"too few rows ({X.shape[0]}) to split; need >=50.")

    test_aurocs: List[float] = []
    for s in range(args.seeds):
        X_tr, X_te, y_tr, y_te, g_tr, _ = grouped_train_test_split(
            X, y, groups, test_size=args.test_size, seed=s
        )
        if args.reduce == "pca_to_l":
            if X_tr.shape[1] <= layer_dim:
                raise ValueError(
                    f"pca_to_l requires input dim > L, but got d={X_tr.shape[1]} and L={layer_dim}"
                )
            projector = PCA(n_components=layer_dim, svd_solver="full", random_state=s)
            X_tr = projector.fit_transform(X_tr).astype(np.float32, copy=False)
            X_te = projector.transform(X_te).astype(np.float32, copy=False)
        if args.probe == "linear":
            auc = train_linear(X_tr, y_tr, X_te, y_te, seed=s)
        else:
            auc = train_mlp(
                X_tr, y_tr, X_te, y_te,
                seed=s,
                groups_tr=g_tr,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
            )
        test_aurocs.append(auc)

    mean = float(np.mean(test_aurocs))
    std = float(np.std(test_aurocs))

    result = {
        "tag": args.tag,
        "input": args.input,
        "features": features,
        "aggregation": args.aggregation,
        "reduce": args.reduce,
        "probe": args.probe,
        "n_rows": int(X.shape[0]),
        "n_groups": int(np.unique(groups).size),
        "n_features": int(layer_dim if args.reduce == "pca_to_l" else X.shape[1]),
        "layer_dim": int(layer_dim),
        "seeds": args.seeds,
        "split_type": "group_index",
        "test_auroc_mean": mean,
        "test_auroc_std": std,
        "test_auroc_per_seed": test_aurocs,
    }
    print(json.dumps(result))

    if args.output_json is not None:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
