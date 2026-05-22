#!/usr/bin/env python3
"""analysis with length-confound controls.

HaluEval QA has a large response-length asymmetry (right ≈ 3 tokens,
hallu ≈ 14 tokens). Length alone gives AUROC ~0.97. Any per-layer
feature that averages over tokens can inherit this signal through the
differing token-vocab distributions. So we report:

 1. Raw token-averaged features (unfair — for the record).
 2. FIRST-TOKEN only (completely length-neutral, same position for all).
 3. MIN-LENGTH truncation (each sample uses only the first K=min(T_i, 3)
    tokens — matches right-answer length and removes length confound).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def load(path: Path):
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def stack_feature(rows, key, agg: str, k: int = 3):
    """Return [N, L] matrix after pooling over response tokens.

    agg:
      'mean'      -- mean over all response tokens
      'first'     -- first response token only
      'mean_topk' -- mean over first min(T, k) tokens
    """
    out = []
    for r in rows:
        arr = np.array(r[key], dtype=np.float32)  # [L, T]
        T = arr.shape[1]
        if T == 0:
            out.append(np.full(arr.shape[0], np.nan))
            continue
        if agg == "mean":
            v = np.nanmean(arr, axis=1)
        elif agg == "first":
            v = arr[:, 0]
        elif agg == "mean_topk":
            v = np.nanmean(arr[:, : min(T, k)], axis=1)
        else:
            raise ValueError(agg)
        out.append(v)
    return np.stack(out)


def cv_auroc(X, y, n_splits: int = 5):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    aurocs = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tr], y[tr])
        aurocs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aurocs)), float(np.std(aurocs))


def per_layer_auroc(X, y, L):
    """Single-feature AUROC for every layer."""
    out = []
    for l in range(L):
        x = X[:, l]
        ok = np.isfinite(x)
        if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
            out.append(np.nan)
            continue
        auc = roc_auc_score(y[ok], x[ok])
        out.append(max(auc, 1 - auc))
    return np.array(out)


def main(path: str):
    rows = load(Path(path))
    L = rows[0]["num_layers"]
    logV = rows[0]["meta"]["log_vocab_nats"]
    y = np.array([r["label"] for r in rows])
    lens = np.array([r["num_response_tokens"] for r in rows])
    print(f"rows={len(rows)} right={(y==0).sum()} hallu={(y==1).sum()} L={L} log|V|={logV:.3f}")
    print(f"response length: right mean={lens[y==0].mean():.2f}, hallu mean={lens[y==1].mean():.2f}")

    # Single baseline: response length alone
    auc_len = roc_auc_score(y, lens); auc_len = max(auc_len, 1 - auc_len)
    print(f"\n[baseline] response_length AUROC = {auc_len:.4f}\n")

    print("== H(p_a) uniformity check (sanity: does MHSA logit lens actually vary?) ==")
    Ha_first = stack_feature(rows, "H_a", "first")
    print(f"  H_a / log|V| across layers (mean over samples, first-token only):")
    mean_per_layer = np.nanmean(Ha_first, axis=0) / logV
    print("  " + " ".join(f"L{l}:{mean_per_layer[l]:.2f}" for l in range(0, L, 4)))
    print(f"  min ratio: {mean_per_layer.min():.3f}  max ratio: {mean_per_layer.max():.3f}")
    if mean_per_layer.min() < 0.85:
        print("  => MHSA logit lens is NOT uniform. The 'MHSA has vocab preferences' narrative survives.\n")
    else:
        print("  => WARNING: MHSA logit lens ~uniform, KCA narrative is at risk.\n")

    for agg, label in [
        ("mean", "(confounded) MEAN over all response tokens"),
        ("first", "FIRST response token only"),
        ("mean_topk", "MEAN over first min(T, 3) tokens"),
    ]:
        print(f"\n================ {label} ================")
        feats = {k: stack_feature(rows, k, agg) for k in ["H_a", "H_m", "H_x", "JSD_am"]}

        # Per-layer best AUROC (peak layer)
        print(f"{'feat':>8} {'best_layer':>10} {'best_AUROC':>10} {'all-layer_AUROC':>18}")
        for k, X in feats.items():
            per_layer = per_layer_auroc(X, y, L)
            best_l = int(np.nanargmax(per_layer))
            m, s = cv_auroc(X, y)
            print(f"{k:>8} {best_l:>10} {per_layer[best_l]:>10.4f} {f'{m:.4f}±{s:.4f}':>18}")

        X_all = np.concatenate([feats[k] for k in ["H_a", "H_m", "H_x", "JSD_am"]], axis=1)
        m, s = cv_auroc(X_all, y)
        print(f"{'[all]':>8} {'-':>10} {'-':>10} {f'{m:.4f}±{s:.4f}':>18}")

    # Sanity: length-controlled — use first-token features only, also include length as a baseline.
    print("\n================ LENGTH-CONTROLLED: first-token features vs length-only ================")
    feats_first = np.concatenate([stack_feature(rows, k, "first") for k in ["H_a", "H_m", "H_x", "JSD_am"]], axis=1)
    m, s = cv_auroc(feats_first, y)
    print(f"  first-token features alone:           AUROC = {m:.4f} ± {s:.4f}")

    # Paired-subset: only rows within overlapping length range (e.g. length 2-4)
    mask_short = lens <= 4
    print(f"\n  subset (response_length <= 4): n={mask_short.sum()}, right={(y[mask_short]==0).sum()}, hallu={(y[mask_short]==1).sum()}")
    if (y[mask_short] == 0).sum() >= 20 and (y[mask_short] == 1).sum() >= 20:
        y_sub = y[mask_short]
        feats_sub = np.concatenate([stack_feature([rows[i] for i in np.where(mask_short)[0]], k, "first") for k in ["H_a", "H_m", "H_x", "JSD_am"]], axis=1)
        m, s = cv_auroc(feats_sub, y_sub)
        print(f"    first-token features (length-matched subset): AUROC = {m:.4f} ± {s:.4f}")
        auc_len_sub = roc_auc_score(y_sub, lens[mask_short]); auc_len_sub = max(auc_len_sub, 1 - auc_len_sub)
        print(f"    length alone on same subset:                   AUROC = {auc_len_sub:.4f}")
    else:
        print("    insufficient samples in length-matched subset, skipping.")


if __name__ == "__main__":
    main(sys.argv[1])
