#!/usr/bin/env python3
"""Cross-dataset generalization runner.

Train probe on the 80% split of <train_input>; evaluate on the 20% split
of <test_input>. When the two paths point to the same file the result
matches the in-domain row of the main results table; when they differ,
the result measures transfer AUROC.

Reuses the feature-assembly and probe-training code from run_kca_eval.py
to keep protocol identical to in-domain numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.run_kca_eval import (
    FEATURE_KEYS,
    assemble,
    extract_groups,
    grouped_train_test_split,
    load_jsonl,
    train_linear,
    train_mlp,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-dataset eval runner.")
    p.add_argument("--train_input", type=str, required=True)
    p.add_argument("--test_input", type=str, required=True)
    p.add_argument(
        "--features",
        type=str,
        default="H_a,H_m,H_x",
        help="Comma-separated subset of {H_a,H_m,H_x,JSD_am,JSD_to_final}.",
    )
    p.add_argument(
        "--aggregation",
        type=str,
        default="first",
        choices=["first", "mean", "mean_top3"],
    )
    p.add_argument("--probe", type=str, default="mlp", choices=["linear", "mlp"])
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--tag", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    features = [f.strip() for f in args.features.split(",") if f.strip()]
    for f in features:
        if f not in FEATURE_KEYS:
            raise ValueError(f"unknown feature {f}")

    train_rows = load_jsonl(Path(args.train_input))
    test_rows = load_jsonl(Path(args.test_input))

    X_tr_full, y_tr_full = assemble(train_rows, features, args.aggregation)
    X_te_full, y_te_full = assemble(test_rows, features, args.aggregation)
    g_tr_full = extract_groups(train_rows, prefix=Path(args.train_input).stem)
    g_te_full = extract_groups(test_rows, prefix=Path(args.test_input).stem)

    if X_tr_full.shape[0] < 50 or X_te_full.shape[0] < 50:
        raise ValueError("Need at least 50 samples in each input.")

    same_dataset = Path(args.train_input).resolve() == Path(args.test_input).resolve()

    aurocs: List[float] = []
    for s in range(args.seeds):
        # In-domain (same dataset): standard 80/20 to reproduce main table.
        # Cross-domain (different): independently 80/20 each side; train on
        # train-side's 80%, test on test-side's 20%, both seeded so the splits
        # are stable.
        X_tr, _, y_tr, _, g_tr, _ = grouped_train_test_split(
            X_tr_full, y_tr_full, g_tr_full, test_size=args.test_size, seed=s
        )
        if same_dataset:
            # When inputs match, use the canonical 80/20 (single split for
            # both training and evaluation). This makes the diagonal of the
            # cross-dataset matrix exactly comparable to in-domain numbers.
            _, X_te, _, y_te, _, _ = grouped_train_test_split(
                X_tr_full, y_tr_full, g_tr_full, test_size=args.test_size, seed=s
            )
        else:
            _, X_te, _, y_te, _, _ = grouped_train_test_split(
                X_te_full, y_te_full, g_te_full, test_size=args.test_size, seed=s
            )

        if args.probe == "linear":
            auc = train_linear(X_tr, y_tr, X_te, y_te, seed=s)
        else:
            auc = train_mlp(
                X_tr, y_tr, X_te, y_te, seed=s, groups_tr=g_tr,
                epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, device=args.device,
            )
        aurocs.append(auc)

    mean = float(np.mean(aurocs))
    std = float(np.std(aurocs))

    result = {
        "tag": args.tag,
        "train_input": args.train_input,
        "test_input": args.test_input,
        "features": features,
        "aggregation": args.aggregation,
        "probe": args.probe,
        "n_train_rows": int(X_tr_full.shape[0]),
        "n_test_rows": int(X_te_full.shape[0]),
        "n_train_groups": int(np.unique(g_tr_full).size),
        "n_test_groups": int(np.unique(g_te_full).size),
        "seeds": args.seeds,
        "split_type": "group_index",
        "test_auroc_mean": mean,
        "test_auroc_std": std,
        "test_auroc_per_seed": aurocs,
        "same_dataset": bool(same_dataset),
    }
    print(json.dumps(result))

    if args.output_json is not None:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
