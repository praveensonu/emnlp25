import re
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import hmean

_FILE_RE = re.compile(
    r"""
    ^(?P<method>.+?)_                          # method  (greedy-as-little-as-possible)
    (?:(?P<setting>[^_]+)_)?                   # optional setting
    (?P<dataset>forget|retain|test)            # dataset token
    (?:_results)?                              # optional literal '_results'
    \.csv$                                     # file extension
    """,
    re.IGNORECASE | re.VERBOSE,
)



def split_filename(fname: str):
    """
    Accepts any of the following shapes (case-insensitive):

        <method>_<dataset>.csv
        <method>_<dataset>_results.csv
        <method>_<setting>_<dataset>.csv
        <method>_<setting>_<dataset>_results.csv

    Returns (method, setting or None, dataset)
    """
    m = _FILE_RE.match(fname)
    if not m:
        raise ValueError(f"Unrecognised naming pattern: {fname}")

    return m.group("method"), m.group("setting"), m.group("dataset").lower()


def harmonic_mean(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[arr > 0]                   # avoid division-by-zero
    return hmean(arr) if arr.size else np.nan


def compute_metric_block(df: pd.DataFrame,
                         dataset: str,          # 'forget' | 'retain' | 'test'
                         metric_kind: str):     # 'FE' | 'MU'
    """Return a Series of scores.

    * forget  → single value (index 'overall')
    * retain/test → one value per `type` (domain/entity/general)
    """
    # -------- locate mandatory score columns -------------------------------
    col_map = {c.lower(): c for c in df.columns}      # case-insensitive lookup
    rouge = col_map.get("rouge_l") or col_map.get("rouge-l")
    prob  = col_map.get("probability") or col_map.get("probs") or col_map.get("prob")
    cos   = col_map.get("cos_sim")   or col_map.get("cosine_sim") or col_map.get("cosine")

    if not all([rouge, prob, cos]):
        raise KeyError(
            "Missing one of required columns rouge_l / probability / cos_sim "
            f"in {df.columns.tolist()}"
        )

    triplet = df[[rouge, prob, cos]].astype(float)

    # -------- per-row metric ------------------------------------------------
    if metric_kind == "FE":
        score = 1.0 - triplet.mean(axis=1)
    else:                                  # MU-R or MU-T
        score = triplet.apply(harmonic_mean, axis=1)

    # -------- aggregate -----------------------------------------------------
    if dataset == "forget":                # no `type` column expected
        return pd.Series({"overall": score.mean()})
    else:                                  # retain / test must have `type`
        if "type" not in df.columns:
            raise KeyError(
                f"`type` column missing in a {dataset} file that should have it"
            )
        return score.groupby(df["type"]).mean()


def summarise_folder(folder=".", pattern="*.csv"):
    rows = []

    for csv_path in glob.glob(os.path.join(folder, pattern)):
        try:
            method, setting, dataset = split_filename(os.path.basename(csv_path))
        except ValueError:
            continue                                  # skip non-matching names

        df = pd.read_csv(csv_path)

        metric_kind = "FE" if dataset == "forget" else "MU"
        metric_name = {"forget": "FE", "retain": "MU-R", "test": "MU-T"}[dataset]

        # ---- NEW: pass `dataset` into compute_metric_block ---------------
        for cat, val in compute_metric_block(df, dataset, metric_kind).items():
            rows.append(
                {
                    "method": method,
                    "setting": setting or "(none)",
                    "dataset": dataset,
                    "category": cat,      # domain / entity / general / overall
                    "metric": metric_name,
                    "score": val,
                }
            )

    return pd.DataFrame(rows)


if __name__ == '__main__':
    summary = summarise_folder('/home/praveen/theoden/emnlp25/results/datasets')
    summary.to_csv("/home/praveen/theoden/emnlp25/results/datasets/granular_dir_vs_ind.csv", index=False)
    # neat table: method × setting × metric  with one column per category
    print(
        summary.pivot_table(
            index=['method', 'setting', 'metric'],
            columns='category',
            values='score',
            aggfunc='first',
        ).round(4)
    )