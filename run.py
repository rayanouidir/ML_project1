"""Project 1 - pipeline: preprocessing, bagged regularized logistic regression,
F1-optimal threshold, CSV submission.

Usage:
    python run.py        # CV (to get the threshold) + final training + submission.csv
    python run.py cv     # ablation study (one change at a time w.r.t. BASE)
"""

import sys
import time
import warnings
import numpy as np
import os
from helpers import create_csv_submission
from implementations import reg_logistic_regression, sigmoid

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH = "data/dataset_to_release/dataset/"
SEED = 0
K_FOLDS = 5
FIT_SAMPLE = 60000     
CHUNK = 20000           
CV_BAGS = 3             
FINAL_BAGS = 10        

BASE = dict(max_nan=0.5, onehot_max=6, special=True, poly=False,
            zero88=True, inter_k=8,
            lambda_=1e-3, gamma=0.1, max_iters=1000, ratio=3, n_bags=1)


# ----------------------------------------------------------------------------
# Data loading (chunked, float32, to limit memory)
# ----------------------------------------------------------------------------
def _read_chunked(path, chunk=20000):
    blocks = []
    with open(path) as f:
        f.readline()
        while True:
            b = np.genfromtxt(f, delimiter=",", max_rows=chunk, dtype=np.float32)
            if b.size == 0:
                break
            blocks.append(np.atleast_2d(b))
    return np.vstack(blocks)


def load_data_light(data_path):
    y = np.genfromtxt(os.path.join(data_path, "y_train.csv"), delimiter=",",
                      skip_header=1, dtype=int, usecols=1)
    xtr = _read_chunked(os.path.join(data_path, "x_train.csv"))
    xte = _read_chunked(os.path.join(data_path, "x_test.csv"))
    return xtr[:, 1:], xte[:, 1:], y, xtr[:, 0].astype(int), xte[:, 0].astype(int)



def find_special_codes(x):
    """Columns whose max is 9/99/999 use 7/77/777 ("don't know") and 9/99/999
    ("refused") as codes -> NaN. In "number of days" columns (max 99, with 77 and
    88 present) 88 means "none", i.e. 0."""
    codes, zero88 = {}, []
    mx = np.nanmax(x, axis=0)
    for j, m in enumerate(mx):
        if m in (9, 99, 999):
            codes[j] = np.array([m, 7 * m // 9], dtype=x.dtype)
            if m == 99 and np.any(x[:, j] == 88) and np.any(x[:, j] == 77):
                zero88.append(j)
    return codes, zero88


def apply_special_codes(x, codes, zero88):
    x = x.copy()
    for j in zero88:
        x[x[:, j] == 88, j] = 0
    for j, c in codes.items():
        x[np.isin(x[:, j], c), j] = np.nan
    return x


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------
def fit_preprocessor(x, y, max_nan=0.5, onehot_max=6, special=True, poly=False,
                     zero88=True, inter_k=8, miss_ind_min=0.01, clip=5.0):
    """Learn all preprocessing statistics on (x, y)."""
    p = {"poly": poly}
    codes, z88 = find_special_codes(x) if special else ({}, [])
    p["codes"], p["zero88"] = codes, (z88 if zero88 else [])
    x = apply_special_codes(x, p["codes"], p["zero88"])

    nan_frac = np.isnan(x).mean(axis=0)
    std = np.nanstd(x, axis=0)
    p["keep"] = np.where((nan_frac <= max_nan) & (std > 0))[0]
    x, nan_frac = x[:, p["keep"]], nan_frac[p["keep"]]

    p["miss_cols"] = np.where(nan_frac > miss_ind_min)[0]    # missing-value indicators
    p["median"] = np.nanmedian(x, axis=0)

    cats = {}                                                 # few distinct values -> one-hot
    for j in range(x.shape[1]):
        vals = np.unique(x[~np.isnan(x[:, j]), j])
        if len(vals) <= onehot_max:
            cats[j] = vals
    p["cats"] = cats
    p["num_cols"] = np.array([j for j in range(x.shape[1]) if j not in cats], dtype=int)

    xi = np.where(np.isnan(x), p["median"], x)
    p["mean"] = xi[:, p["num_cols"]].mean(axis=0)
    p["std"] = xi[:, p["num_cols"]].std(axis=0) + 1e-8

    # pairwise products of the inter_k numeric features most correlated with y
    p["ia"] = p["ib"] = np.array([], dtype=int)
    if inter_k > 1:
        z = np.clip((xi[:, p["num_cols"]] - p["mean"]) / p["std"], -clip, clip)
        score = np.abs(z.T @ (y - y.mean()))
        top = np.argsort(-score)[:inter_k]
        ia, ib = np.triu_indices(len(top), 1)
        p["ia"], p["ib"] = top[ia], top[ib]
    return p


def transform(x, p, clip=5.0):
    x = apply_special_codes(x, p["codes"], p["zero88"])[:, p["keep"]]
    miss = np.isnan(x[:, p["miss_cols"]]).astype(np.float32)
    xi = np.where(np.isnan(x), p["median"], x)
    num = np.clip((xi[:, p["num_cols"]] - p["mean"]) / p["std"], -clip, clip)
    blocks = [np.ones((x.shape[0], 1), dtype=np.float32), num, miss]
    if p["poly"]:
        blocks.append(num ** 2)
    if len(p["ia"]):
        blocks.append(num[:, p["ia"]] * num[:, p["ib"]])
    blocks += [(xi[:, [j]] == v[None, :]).astype(np.float32) for j, v in p["cats"].items()]
    return np.hstack(blocks).astype(np.float64)


# ----------------------------------------------------------------------------
# Metric / threshold
# ----------------------------------------------------------------------------
def f1_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return 2 * tp / (2 * tp + fp + fn + 1e-12)


def best_threshold(y_true, probs, grid=np.arange(0.05, 0.95, 0.005)):
    """Decision threshold maximizing F1 (crucial with ~9% positives)."""
    scores = [f1_score(y_true, (probs >= t).astype(int)) for t in grid]
    i = int(np.argmax(scores))
    return grid[i], scores[i]


# ----------------------------------------------------------------------------
# Training / prediction (bagging of undersampled models)
# ----------------------------------------------------------------------------
def prep_args(params):
    return {k: params[k] for k in ("max_nan", "onehot_max", "special", "poly",
                                   "zero88", "inter_k")}


def fit_and_train(x_raw, y, idx, params, rng):
    """Fit the preprocessor on a subsample of rows idx, then train n_bags
    regularized logistic regressions, each on all positives + a different random
    subset of negatives. Returns (preprocessor, W) with W of shape (D, n_bags)."""
    sub = idx if len(idx) <= FIT_SAMPLE else rng.choice(idx, FIT_SAMPLE, replace=False)
    p = fit_preprocessor(x_raw[sub], y[sub], **prep_args(params))

    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    n_neg = min(len(neg), int(params["ratio"] * len(pos)))
    ws = []
    for _ in range(params["n_bags"]):
        neg_s = rng.choice(neg, size=n_neg, replace=False)
        sel = rng.permutation(np.concatenate([pos, neg_s]))
        xs = transform(x_raw[sel], p)
        w, _ = reg_logistic_regression(y[sel], xs, params["lambda_"],
                                       np.zeros(xs.shape[1]),
                                       params["max_iters"], params["gamma"])
        ws.append(w)
    return p, np.column_stack(ws)


def predict_proba(x_raw, idx, p, W):
    """Mean probability over the bagged models, computed chunk by chunk."""
    out = [sigmoid(transform(x_raw[idx[s:s + CHUNK]], p) @ W).mean(axis=1)
           for s in range(0, len(idx), CHUNK)]
    return np.concatenate(out)


# ----------------------------------------------------------------------------
# Cross-validation / submission
# ----------------------------------------------------------------------------
def cross_validate(x_raw, y, params, k=K_FOLDS):
    """K-fold CV (preprocessing refitted in each fold, no leakage).
    Returns (mean F1 at the best threshold, median best threshold)."""
    rng = np.random.RandomState(SEED)
    folds = np.array_split(rng.permutation(len(y)), k)
    f1s, thrs = [], []
    for i in range(k):
        va = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        p, W = fit_and_train(x_raw, y, tr, params, rng)
        t, f = best_threshold(y[va], predict_proba(x_raw, va, p, W))
        f1s.append(f)
        thrs.append(t)
        print(f"   fold {i + 1}/{k}: F1={f:.4f} thr={t:.3f}", flush=True)
    return np.mean(f1s), np.median(thrs)


def make_submission(x_raw, y, x_test, test_ids, params, thr, f1):
    """Train on the full training set, predict test with threshold thr, write CSVs."""
    rng = np.random.RandomState(SEED)
    p, W = fit_and_train(x_raw, y, np.arange(len(y)), params, rng)
    probs = predict_proba(x_test, np.arange(len(x_test)), p, W)
    y_pred = np.where(probs >= thr, 1, -1)              # the platform expects {-1, 1}
    name = f"submission_F1_{f1:.4f}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    create_csv_submission(test_ids, y_pred, name)
    create_csv_submission(test_ids, y_pred, "submission.csv")
    print(f"CSV written: {name} (and submission.csv), "
          f"predicted positive rate: {np.mean(y_pred == 1):.3f}", flush=True)


def run_submit(x_raw, y, x_test, test_ids):
    """CV (CV_BAGS bags) to get the threshold, then final training (FINAL_BAGS bags)."""
    f1, thr = cross_validate(x_raw, y, dict(BASE, n_bags=CV_BAGS))
    print(f"CV F1 = {f1:.4f}, threshold = {thr:.3f}", flush=True)
    make_submission(x_raw, y, x_test, test_ids, dict(BASE, n_bags=FINAL_BAGS), thr, f1)


def run_cv(x_raw, y, x_test, test_ids):
    """Ablation: one hyperparameter changed at a time w.r.t. BASE (single model,
    n_bags=1), then a CSV with the best configuration."""
    f1, thr = cross_validate(x_raw, y, BASE)
    print(f"BASE {BASE}\n  -> F1={f1:.4f} (threshold {thr:.3f})", flush=True)
    results = [(f1, thr, BASE)]
    grid = {"n_bags": [3, 10], "zero88": [False], "inter_k": [0, 15],
            "max_iters": [300, 3000], "lambda_": [1e-5, 1e-4, 1e-2],
            "ratio": [1, 5], "onehot_max": [0, 10], "max_nan": [0.2, 0.8],
            "special": [False], "poly": [True]}
    for name, values in grid.items():
        for v in values:
            params = dict(BASE, **{name: v})
            f1, thr = cross_validate(x_raw, y, params)
            print(f"{name}={v}: F1={f1:.4f} (threshold {thr:.3f})", flush=True)
            results.append((f1, thr, params))
    f1, thr, best = max(results, key=lambda r: r[0])
    print(f"Best configuration: F1={f1:.4f}, {best}", flush=True)
    make_submission(x_raw, y, x_test, test_ids, dict(best, n_bags=max(best["n_bags"], FINAL_BAGS)),
                    thr, f1)


if __name__ == "__main__":
    print("loading data...", flush=True)
    x_train, x_test, y_train, train_ids, test_ids = load_data_light(DATA_PATH)
    y01 = (y_train == 1).astype(int)                    # labels {-1,1} -> {0,1}
    print("train:", x_train.shape, "positive rate:", y01.mean(), flush=True)
    if len(sys.argv) > 1 and sys.argv[1] == "cv":
        run_cv(x_train, y01, x_test, test_ids)
    else:
        run_submit(x_train, y01, x_test, test_ids)