#!/usr/bin/env python3
"""Multi-dataset training sweep.

For each model, concatenate the 80% train splits of all four benchmarks
into a single ~60K-row training set, train ONE MLP probe on this union,
and then evaluate it on each benchmark's held-out 20% test split.

The result is an extra column to the cross-dataset table: a probe that
sees representative training data from every benchmark and is asked to
hold up across all of them simultaneously. If the per-layer entropy
feature is dataset-invariant when given a representative training mixture,
this single probe should match (rather than fall below) per-dataset
probes.

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/run_multi_dataset_sweep.py \\
      --output_json outputs/eval/multi_dataset_results.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.run_kca_eval import (
    assemble,
    extract_groups,
    grouped_split_indices,
    load_jsonl,
    train_linear,
)


MODELS = [
    "Qwen2.5-7B-Instruct",
    "Meta-Llama-3-8B-Instruct",
    "gemma-2-9b-it",
]
DATASETS = [
    "halueval_10k",
    "squad2_10k",
    "hotpotqa_distractor_10k",
    "triviaqa_10k",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-dataset training sweep.")
    p.add_argument("--features", type=str, default="H_a,H_m,H_x")
    p.add_argument("--aggregation", type=str, default="first",
                   choices=["first", "mean", "mean_top3"])
    p.add_argument("--probe", type=str, default="mlp", choices=["linear", "mlp"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_json", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    features = [f.strip() for f in args.features.split(",") if f.strip()]
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("")

    # ---- Phase 1: pre-assemble all (model, dataset) features ----
    print(f"[load] {len(MODELS) * len(DATASETS)} feature files...", flush=True)
    cache: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for model in MODELS:
        for ds in DATASETS:
            fpath = Path("outputs") / model / f"kca_{ds}.jsonl"
            if not fpath.exists():
                print(f"  [miss] {fpath}", flush=True)
                continue
            t0 = time.time()
            rows = load_jsonl(fpath)
            X, y = assemble(rows, features, args.aggregation)
            groups = extract_groups(rows, prefix=ds)
            cache[(model, ds)] = (X, y, groups)
            print(f"  [ok] {model:30s} | {ds:30s} | {X.shape} | {time.time()-t0:.1f}s",
                  flush=True)

    # ---- Phase 2: per-model multi-dataset training ----
    total = len(MODELS) * args.seeds
    print(f"\n[sweep] {total} multi-dataset trainings (MLP, 3-entropy, first-token)\n", flush=True)
    i = 0
    t_start = time.time()

    for model in MODELS:
        # Aggregate per-test-ds AUROCs across seeds.
        per_test = {ds: [] for ds in DATASETS}

        for s in range(args.seeds):
            i += 1
            t_seed = time.time()

            # Build the 80/20 split per dataset (with the same seed)
            # then concatenate the 80% halves into a single training set.
            X_tr_parts, y_tr_parts, g_tr_parts = [], [], []
            test_splits = {}  # ds -> (X_te, y_te)
            for ds in DATASETS:
                X_full, y_full, g_full = cache.get((model, ds), (None, None, None))
                if X_full is None:
                    continue
                train_idx, test_idx = grouped_split_indices(
                    y_full, g_full, test_size=args.test_size, seed=s
                )
                X_tr, X_te = X_full[train_idx], X_full[test_idx]
                y_tr, y_te = y_full[train_idx], y_full[test_idx]
                g_tr = g_full[train_idx]
                X_tr_parts.append(X_tr)
                y_tr_parts.append(y_tr)
                g_tr_parts.append(g_tr)
                test_splits[ds] = (X_te, y_te)

            X_train = np.concatenate(X_tr_parts, axis=0)
            y_train = np.concatenate(y_tr_parts, axis=0)
            g_train = np.concatenate(g_tr_parts, axis=0)
            print(
                f"[seed {s}] {model:30s} | union train shape: {X_train.shape}",
                flush=True,
            )

            # Train ONE probe on the union.
            if args.probe == "linear":
                # Hack: train_linear takes (X_tr, y_tr, X_te, y_te). To recover
                # a fitted classifier we re-import its body. Easiest: pick
                # halueval as a "dummy" test, but evaluate per-test below
                # using sklearn API instead.
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(C=1.0, max_iter=2000, random_state=s).fit(X_train, y_train)
                from sklearn.metrics import roc_auc_score
                for ds, (Xt, yt) in test_splits.items():
                    auc = float(roc_auc_score(yt, clf.predict_proba(Xt)[:, 1]))
                    per_test[ds].append(auc)
            else:
                # Train MLP once, then evaluate on each test split.
                # We re-implement the train loop here so we can evaluate on
                # multiple test sets after a single fit.
                import torch
                import torch.nn as nn
                from torch.utils.data import DataLoader, TensorDataset
                from torch.optim.lr_scheduler import ReduceLROnPlateau
                from sklearn.metrics import roc_auc_score
                from src.utils import TriLensProbe

                torch.manual_seed(s)
                np.random.seed(s)

                X_tr_t = torch.from_numpy(X_train).float()
                y_tr_t = torch.from_numpy(y_train).float()

                dev = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

                # 10% validation split off the union for LR scheduler.
                tr_idx_np, val_idx_np = grouped_split_indices(
                    y_train, g_train, test_size=0.1, seed=s
                )
                tr_idx = torch.from_numpy(tr_idx_np).long()
                val_idx = torch.from_numpy(val_idx_np).long()
                train_ds = TensorDataset(X_tr_t[tr_idx], y_tr_t[tr_idx])
                val_ds = TensorDataset(X_tr_t[val_idx], y_tr_t[val_idx])
                train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=max(8, args.batch_size))

                model_probe = TriLensProbe(input_dim=X_train.shape[1]).to(dev)
                criterion = nn.BCELoss()
                optim = torch.optim.Adam(model_probe.parameters(), lr=args.lr)
                sched = ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=5)

                best_val = float("inf")
                best_state = {k: v.detach().clone() for k, v in model_probe.state_dict().items()}

                for _ in range(args.epochs):
                    model_probe.train()
                    for xb, yb in train_loader:
                        xb, yb = xb.to(dev), yb.to(dev)
                        optim.zero_grad()
                        out = model_probe(xb).squeeze(1)
                        loss = criterion(out, yb)
                        loss.backward()
                        optim.step()
                    model_probe.eval()
                    with torch.no_grad():
                        vlosses = []
                        for xb, yb in val_loader:
                            xb, yb = xb.to(dev), yb.to(dev)
                            out = model_probe(xb).squeeze(1)
                            vlosses.append(criterion(out, yb).item())
                    vloss = float(np.mean(vlosses)) if vlosses else 0.0
                    sched.step(vloss)
                    if vloss < best_val:
                        best_val = vloss
                        best_state = {k: v.detach().clone() for k, v in model_probe.state_dict().items()}

                model_probe.load_state_dict(best_state)
                model_probe.eval()
                with torch.no_grad():
                    for ds, (Xt, yt) in test_splits.items():
                        Xt_t = torch.from_numpy(Xt).float().to(dev)
                        scores = model_probe(Xt_t).squeeze(1).cpu().numpy()
                        auc = float(roc_auc_score(yt, scores))
                        per_test[ds].append(auc)

            print(f"  -> seed {s} done in {time.time()-t_seed:.1f}s", flush=True)

        # Write one record per (model, test_ds) summarizing across seeds.
        for ds, aurocs in per_test.items():
            if not aurocs:
                continue
            mean = float(np.mean(aurocs))
            std = float(np.std(aurocs))
            rec = {
                "model": model,
                "training": "multi_dataset_union",
                "test_ds": ds,
                "features": features,
                "aggregation": args.aggregation,
                "probe": args.probe,
                "seeds": args.seeds,
                "split_type": "group_index",
                "test_auroc_mean": mean,
                "test_auroc_std": std,
                "test_auroc_per_seed": aurocs,
            }
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        elapsed = time.time() - t_start
        eta = elapsed / max(i, 1) * (total - i)
        print(f"[model done] {model} | elapsed {elapsed/60:.1f}m | eta {eta/60:.1f}m\n", flush=True)

    print(f"[done] wrote {len(MODELS) * len(DATASETS)} cells to {out_path}", flush=True)


if __name__ == "__main__":
    main()
