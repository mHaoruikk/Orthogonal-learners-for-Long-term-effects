"""Convert data/GAIN/quarterly.mat to data/GAIN/quarterly.csv.

Each variable in the .mat file is an (n, 1) numeric column. We stack them
in the order given by data/GAIN/GAIN_vars_description.log and write a
single CSV with one row per subject.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parent.parent
MAT_PATH = REPO_ROOT / "data" / "GAIN" / "quarterly.mat"
CSV_PATH = REPO_ROOT / "data" / "GAIN" / "quarterly.csv"

VAR_ORDER = [
    "county", "age", "remsamp", "xsexf", "xhsdip", "white", "hisp", "black",
    "x1chld", "xchld05", "alameda", "river", "sandiego", "la", "e",
    "adcpc4", "adcpc3", "adcpc2", "adcpc1",
    "agesq", "afdcfg", "single",
    "grde911", "grade12", "grd1315", "grade16", "grd1720", "dumkids",
    *[f"tcedd{i}" for i in range(1, 37)],
    *[f"tcprn{i}" for i in range(1, 11)],
    "grew1", "gepop1",
    *[f"paid{i}" for i in range(4, 0, -1)],
    *[f"aid{i}" for i in range(1, 37)],
    *[f"tcpp{i}" for i in range(1, 11)],
    *[f"padcpc{i}" for i in range(1, 5)],
    *[f"ptcedd{i}" for i in range(1, 37)],
    *[f"ymw{i}" for i in range(1, 37)],
    "tcddyp13", "tcddyp46", "tcddyp79",
    "tcddw13", "tcddw46", "tcddw79",
    "tcddy13", "tcddy46", "tcddy79",
    "o299013", "o299046", "o299079",
    "aidy13", "aidy46", "aidy79",
    "aidw13", "aidw46", "aidw79",
    "unemp10", "aidcon4",
]


def main() -> None:
    mat = loadmat(MAT_PATH)
    missing = [v for v in VAR_ORDER if v not in mat]
    extra = [k for k in mat if not k.startswith("__") and k not in VAR_ORDER]
    if missing:
        raise KeyError(f"Missing variables in .mat: {missing}")
    if extra:
        raise KeyError(f"Unexpected variables in .mat: {extra}")

    columns = {}
    for name in VAR_ORDER:
        arr = np.asarray(mat[name]).reshape(-1)
        if arr.dtype.byteorder == ">":
            arr = arr.astype(arr.dtype.newbyteorder("="))
        columns[name] = arr

    df = pd.DataFrame(columns)
    print(f"Shape: {df.shape}")
    df.to_csv(CSV_PATH, index=False)
    print(f"Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
