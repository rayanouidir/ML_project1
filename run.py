"""
Project 1 - Step 3 : preprocessing -> cross-validation -> final training -> submission.
Only NumPy. Uses reg_logistic_regression and sigmoid from implementations.py.

Memory-friendly: raw data kept in float32, never copied as a whole. The
preprocessor is fitted on a subsample, only the (undersampled) training rows
are transformed, and predictions are made chunk by chunk.

Usage :
    python run.py        # CV on BASE, final training, writes a CSV
    python run.py cv     # ablation by CV, then writes a CSV with the best configuration
"""
import sys
import warnings
import numpy as np
import os
import time
from helpers import create_csv_submission
from implementations import reg_logistic_regression, sigmoid

warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN columns etc.
warnings.filterwarnings("ignore", category=UserWarning)      # genfromtxt end-of-file notice

DATA_PATH = "data/dataset_to_release/dataset/"   # folder with x_train.csv, y_train.csv, x_test.csv
SEED = 0
K_FOLDS = 5
FIT_SAMPLE = 60000            # rows used to fit the preprocessor
CHUNK = 20000                 # rows per chunk at prediction time

# Hyperparameters used for the final submission (and as baseline in the ablation)
BASE = dict(max_nan=0.5, onehot_max=6, special=True, poly=False,
            lambda_=1e-3, gamma=0.05, max_iters=500, ratio=3)



# ----------------------------------------------------------------------------
# Memory-friendly data loading (np.genfromtxt on the whole file needs ~10 GB)
# ----------------------------------------------------------------------------
def _read_chunked(path, chunk=20000):
    """Read a numeric CSV (with header) block by block, as float32."""
    blocks = []
    with open(path) as f:
        f.readline()                                    # skip header
        while True:
            b = np.genfromtxt(f, delimiter=",", max_rows=chunk, dtype=np.float32)
            if b.size == 0:
                break
            blocks.append(np.atleast_2d(b))
    return np.vstack(blocks)


def load_data_light(data_path):
    """Same output as helpers.load_csv_data (x_train, x_test, y_train, train_ids,
    test_ids) but chunked and in float32."""
    y = np.genfromtxt(os.path.join(data_path, "y_train.csv"), delimiter=",",
                      skip_header=1, dtype=int, usecols=1)
    xtr = _read_chunked(os.path.join(data_path, "x_train.csv"))
    xte = _read_chunked(os.path.join(data_path, "x_test.csv"))
    return xtr[:, 1:], xte[:, 1:], y, xtr[:, 0].astype(int), xte[:, 0].astype(int)

# ----------------------------------------------------------------------------
# Preprocessing (all statistics are learned on training data only)
# ----------------------------------------------------------------------------
def find_special_codes(x):
    """BRFSS heuristic: a column whose max is 9, 99 or 999 usually encodes
    'refused' (9/99/999) and 'don't know' (7/77/777). Returns {col: codes}."""
    codes = {}
    mx = np.nanmax(x, axis=0)
    for j, m in enumerate(mx):
        if m in (9, 99, 999):
            codes[j] = np.array([m, 7 * m // 9], dtype=x.dtype)
    return codes


def apply_special_codes(x, codes):
    """Replace special codes by NaN (they are then imputed like missing values)."""
    x = x.copy()
    for j, c in codes.items():
        x[np.isin(x[:, j], c), j] = np.nan
    return x


def fit_preprocessor(x, max_nan=0.5, onehot_max=6, special=True, poly=False,
                     miss_ind_min=0.01):
    """Learn the cleaning parameters on x of shape (n, d) (may contain NaN)."""
    p = {"poly": poly, "codes": find_special_codes(x) if special else {}}
    x = apply_special_codes(x, p["codes"])

    nan_frac = np.isnan(x).mean(axis=0)
    std = np.nanstd(x, axis=0)
    p["keep"] = np.where((nan_frac <= max_nan) & (std > 0))[0]    # drop empty/constant columns
    x, nan_frac = x[:, p["keep"]], nan_frac[p["keep"]]

    p["miss_cols"] = np.where(nan_frac > miss_ind_min)[0]          # 'is missing' indicators
    p["median"] = np.nanmedian(x, axis=0)

    # categorical columns = few distinct values -> one-hot
    cats = {}
    for j in range(x.shape[1]):
        vals = np.unique(x[~np.isnan(x[:, j]), j])
        if len(vals) <= onehot_max:
            cats[j] = vals
    p["cats"] = cats
    p["num_cols"] = np.array([j for j in range(x.shape[1]) if j not in cats], dtype=int)

    xi = np.where(np.isnan(x), p["median"], x)
    p["mean"] = xi[:, p["num_cols"]].mean(axis=0)
    p["std"] = xi[:, p["num_cols"]].std(axis=0) + 1e-8
    return p


def transform(x, p, clip=5.0):
    """Imputation, missing indicators, one-hot, standardization (+ squares), bias.
    Returns a float64 matrix (call it on subsets / chunks only)."""
    x = apply_special_codes(x, p["codes"])[:, p["keep"]]
    miss = np.isnan(x[:, p["miss_cols"]]).astype(np.float32)
    xi = np.where(np.isnan(x), p["median"], x)
    num = np.clip((xi[:, p["num_cols"]] - p["mean"]) / p["std"], -clip, clip)
    blocks = [np.ones((x.shape[0], 1), dtype=np.float32), num, miss]
    if p["poly"]:
        blocks.append(num ** 2)                                    # degree-2 features
    blocks += [(xi[:, [j]] == v[None, :]).astype(np.float32) for j, v in p["cats"].items()]
    return np.hstack(blocks).astype(np.float64)


# ----------------------------------------------------------------------------
# Metrics, training, prediction
# ----------------------------------------------------------------------------
def f1_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return 2 * tp / (2 * tp + fp + fn + 1e-12)


def best_threshold(y_true, probs, grid=np.arange(0.05, 0.9, 0.01)):
    """Decision threshold maximizing F1 (crucial with ~9% positives)."""
    scores = [f1_score(y_true, (probs >= t).astype(int)) for t in grid]
    i = int(np.argmax(scores))
    return grid[i], scores[i]


def prep_args(params):
    return {k: params[k] for k in ("max_nan", "onehot_max", "special", "poly")}


def fit_and_train(x_raw, y, idx, params, rng):
    """Fit the preprocessor on a subsample of rows idx, undersample the negatives,
    train the model. Returns (preprocessor, w)."""
    sub = idx if len(idx) <= FIT_SAMPLE else rng.choice(idx, FIT_SAMPLE, replace=False)
    p = fit_preprocessor(x_raw[sub], **prep_args(params))

    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    neg = rng.choice(neg, size=min(len(neg), int(params["ratio"] * len(pos))), replace=False)
    sel = rng.permutation(np.concatenate([pos, neg]))
    xs = transform(x_raw[sel], p)

    w, _ = reg_logistic_regression(y[sel], xs, params["lambda_"], np.zeros(xs.shape[1]),
                                   params["max_iters"], params["gamma"])
    return p, w


def predict_proba(x_raw, idx, p, w):
    """Probabilities for rows idx of x_raw, computed chunk by chunk."""
    out = [sigmoid(transform(x_raw[idx[s:s + CHUNK]], p) @ w)
           for s in range(0, len(idx), CHUNK)]
    return np.concatenate(out)



def cross_validate(x_raw, y, params, k=K_FOLDS):
    """K-fold CV (preprocessing refitted in each fold, no leakage).
    Returns (mean F1 at the best threshold, mean best threshold)."""
    rng = np.random.RandomState(SEED)
    folds = np.array_split(rng.permutation(len(y)), k)
    f1s, thrs = [], []
    for i in range(k):
        va = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        p, w = fit_and_train(x_raw, y, tr, params, rng)
        t, f = best_threshold(y[va], predict_proba(x_raw, va, p, w))
        f1s.append(f)
        thrs.append(t)
    return np.mean(f1s), np.mean(thrs)


def make_submission(x_raw, y, x_test, test_ids, params, thr, f1):
    """Train on the full training set with `params`, predict the test set with
    threshold `thr`, and write a CSV (timestamped copy + latest 'submission.csv')."""
    rng = np.random.RandomState(SEED)
    p, w = fit_and_train(x_raw, y, np.arange(len(y)), params, rng)
    probs = predict_proba(x_test, np.arange(len(x_test)), p, w)
    y_pred = np.where(probs >= thr, 1, -1)              # the platform expects {-1, 1}
    name = f"submission_F1_{f1:.4f}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    create_csv_submission(test_ids, y_pred, name)
    create_csv_submission(test_ids, y_pred, "submission.csv")
    print(f"CSV written: {name} (and submission.csv), "
          f"predicted positive rate: {np.mean(y_pred == 1):.3f}", flush=True)


def run_submit(x_raw, y, x_test, test_ids):
    """CV on BASE (to get the threshold), then final training + CSV."""
    f1, thr = cross_validate(x_raw, y, BASE)
    print(f"CV F1 = {f1:.4f}, threshold = {thr:.2f}", flush=True)
    make_submission(x_raw, y, x_test, test_ids, BASE, thr, f1)


def run_cv(x_raw, y, x_test, test_ids):
    """Ablation (one hyperparameter changed at a time w.r.t. BASE), then a CSV
    is written with the best configuration found."""
    f1, thr = cross_validate(x_raw, y, BASE)
    print(f"BASE {BASE}\n  -> F1={f1:.4f} (threshold {thr:.2f})", flush=True)
    results = [(f1, thr, BASE)]
    grid = {"lambda_": [1e-5, 1e-4, 1e-2, 1e-1], "ratio": [1, 5],
            "onehot_max": [0, 10], "max_nan": [0.2, 0.8],
            "special": [False], "poly": [True]}
    for name, values in grid.items():
        for v in values:
            params = dict(BASE, **{name: v})
            f1, thr = cross_validate(x_raw, y, params)
            print(f"{name}={v}: F1={f1:.4f} (threshold {thr:.2f})", flush=True)
            results.append((f1, thr, params))
    f1, thr, best = max(results, key=lambda r: r[0])
    print(f"Best configuration: F1={f1:.4f}, {best}", flush=True)
    make_submission(x_raw, y, x_test, test_ids, best, thr, f1)


if __name__ == "__main__":

    print("loading data...", flush=True)
    x_train, x_test, y_train, train_ids, test_ids = load_data_light(DATA_PATH)
    y01 = (y_train == 1).astype(int)                    # labels {-1,1} -> {0,1}
    print("train:", x_train.shape, "positive rate:", y01.mean(), flush=True)
    if len(sys.argv) > 1 and sys.argv[1] == "cv":
        run_cv(x_train, y01, x_test, test_ids)
    else:
        run_submit(x_train, y01, x_test, test_ids)