"""Build the re-sampled real-world IST-3 dataset per docs/IST-realword.md.

Outputs src/data/ist_real.csv with the full 3034-patient cohort, columns:
  - X (raw, not standardised; categoricals kept raw for one-hot at model time):
      age, gender, nihss, pred_nihss, gcs_score_rand, stroketype,
      sbprand, dbprand, glucose, glucose_missing, weight,
      atrialfib_rand, stroke_pre, antiplat_rand, livealone_rand, indepinadl_rand,
      vis_infarct, R_infarct_size, konprob, randdelay, country
  - S: gcs_score_7, indepinadl_7, ablewalk_7, sich7, dead7
  - A (1 = rt-PA, 0 = Control), Y (in [0,1]; NaN where R=0), R, retained, pi_star

Two overlap mechanisms:
  - R  : plan18 / form-return / death-within-548-days split (§3)
  - retained : rejection sampling against clinically-motivated pi_star (§4)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat
from sklearn.linear_model import Ridge

SEED = 42
SAS_PATH = Path(r"C:\Users\ma\Research\Long-term-effects\IST-3-Dataset\datashare_aug2015.sas7bdat")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / "src" / "data" / "ist_real.csv"


def recode_yn(s: pd.Series) -> pd.Series:
    """IST-3 YNDQ: 1=Yes, 2=No, 20/30/40=missing-codes. Returns 0/1 with NaN elsewhere."""
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s.isin([1, 2])).map({1: 1, 2: 0})


def build_cohort(df: pd.DataFrame) -> pd.DataFrame:
    no_form = (df[["gcs_eye_7", "gcs_motor_7", "gcs_verbal_7"]] == 30).any(axis=1)
    return df.loc[~no_form].reset_index(drop=True)


def build_X(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["age"] = df["age"].astype(float)
    X["gender"] = (df["gender"] == 1).astype(int)
    X["nihss"] = df["nihss"].astype(float)
    X["pred_nihss"] = (df["pred_nihss"] == 1).astype(int)
    X["gcs_score_rand"] = df["gcs_score_rand"].astype(float)
    X["stroketype"] = pd.to_numeric(df["stroketype"], errors="coerce").astype("Int64")
    X["sbprand"] = df["sbprand"].astype(float)

    db = pd.to_numeric(df["dbprand"], errors="coerce")
    db = db.where(~db.isin([35, 36]))
    X["dbprand"] = db.fillna(db.median())

    glu = pd.to_numeric(df["glucose"], errors="coerce")
    X["glucose_missing"] = glu.isna().astype(int)
    glu = glu.groupby(df["country"]).transform(lambda s: s.fillna(s.median()))
    X["glucose"] = glu.fillna(glu.median())

    w = pd.to_numeric(df["weight"], errors="coerce")
    X["weight"] = w.fillna(w.median())

    for col in ("atrialfib_rand", "stroke_pre", "antiplat_rand",
                "livealone_rand", "indepinadl_rand"):
        v = recode_yn(df[col])
        X[col] = v.fillna(v.mode().iloc[0]).astype(int)

    X["vis_infarct"] = (df["vis_infarct"] == 1).astype(int)

    r_is = pd.to_numeric(df["R_infarct_size"], errors="coerce")
    X["R_infarct_size"] = r_is.fillna(r_is.median()).astype(float)

    kp = pd.to_numeric(df["konprob"], errors="coerce")
    X["konprob"] = kp.fillna(kp.median())

    rd = pd.to_numeric(df["randdelay"], errors="coerce")
    X["randdelay"] = rd.fillna(rd.median())

    X["country"] = df["country"].astype(str)
    return X


def _gcs_comp(s: pd.Series) -> pd.Series:
    """GCS 7-day component: valid in {0..6, 10=died}; 20/30/40 are non-answers."""
    s = pd.to_numeric(s, errors="coerce")
    s = s.where(s.isin([0, 1, 2, 3, 4, 5, 6, 10]))
    return s.replace({10: 0})


def _bin_7(s: pd.Series) -> pd.Series:
    """7-day YNDQ with 10=died: map Yes->1, No->0, died->0; other codes NaN."""
    s = pd.to_numeric(s, errors="coerce")
    s = s.where(s.isin([1, 2, 10]))
    return s.map({1: 1, 2: 0, 10: 0})


def build_S(df: pd.DataFrame) -> pd.DataFrame:
    S = pd.DataFrame(index=df.index)
    gcs7 = _gcs_comp(df["gcs_eye_7"]) + _gcs_comp(df["gcs_motor_7"]) + _gcs_comp(df["gcs_verbal_7"])
    S["gcs_score_7"] = gcs7.fillna(gcs7.median())
    for col in ("indepinadl_7", "ablewalk_7"):
        v = _bin_7(df[col])
        S[col] = v.fillna(v.mode().iloc[0]).astype(int)
    for col in ("sich7", "dead7"):
        v = pd.to_numeric(df[col], errors="coerce")
        S[col] = v.fillna(v.mode().iloc[0]).astype(int)
    return S


def build_R_and_Y(df: pd.DataFrame, X: pd.DataFrame):
    eq = pd.to_numeric(df["euroqol18"], errors="coerce")
    ohs18 = pd.to_numeric(df["ohs18"], errors="coerce")
    receigh = pd.to_numeric(df["receighteen"], errors="coerce")
    plan18 = pd.to_numeric(df["plan18"], errors="coerce")
    censor18 = pd.to_numeric(df["censor18"], errors="coerce")
    surv18 = pd.to_numeric(df["surv18"], errors="coerce")

    is_dead18 = (ohs18 == 6)
    died_548 = (censor18 == 0) & (surv18 <= 548)
    form_returned = (receigh == 1)
    policy_excluded = (plan18 == 2)

    R = (~policy_excluded & (form_returned | died_548)).astype(int)

    Y = eq / 100.0
    Y = Y.where(~is_dead18, 0.0)
    Y = Y.where(~died_548, 0.0)

    alive_missing = (
        form_returned & ~is_dead18 & ~died_548 & (plan18 == 1) & Y.isna()
    )
    alive_observed = (
        form_returned & ~is_dead18 & ~died_548 & (plan18 == 1) & Y.notna()
    )

    if alive_missing.any():
        feat = pd.get_dummies(X, columns=["country", "stroketype"], drop_first=False)
        feat["ohs18"] = ohs18.fillna(-1.0)
        feat = feat.astype(float).fillna(0.0)
        model = Ridge(alpha=1.0, random_state=SEED)
        model.fit(feat.loc[alive_observed].values, Y.loc[alive_observed].values)
        pred = np.clip(model.predict(feat.loc[alive_missing].values), 0.0, 1.0)
        Y.loc[alive_missing] = pred

    return R.astype(int), Y, int(alive_missing.sum())


def pi_star(age, nihss, gcs, af):
    z = 0.3 - 0.04 * (age - 70) - 0.015 * (nihss - 12) ** 2 \
        + 0.03 * (gcs - 12) - 0.4 * af
    return 1.0 / (1.0 + np.exp(-z))


def main():
    print(f"Loading {SAS_PATH}")
    df, _ = pyreadstat.read_sas7bdat(str(SAS_PATH))
    print(f"Raw shape: {df.shape}")

    df = build_cohort(df)
    df["A"] = (df["itt_treat"] == 0).astype(int)
    print(f"Cohort (3035 - 1 missing 7-day form): {len(df)}")

    X = build_X(df)
    S = build_S(df)
    R, Y, n_imputed = build_R_and_Y(df, X)
    print(f"Regression-imputed EQ-5D for {n_imputed} alive-form-returned missing cases")

    pi = pi_star(X["age"].to_numpy(), X["nihss"].to_numpy(),
                 X["gcs_score_rand"].to_numpy(), X["atrialfib_rand"].to_numpy())
    pi_max = float(max(pi.max(), (1.0 - pi).max()))
    A = df["A"].to_numpy()
    p_keep = np.where(A == 1, pi / pi_max, (1.0 - pi) / pi_max)
    rng = np.random.default_rng(SEED)
    retained = (rng.random(len(A)) < p_keep).astype(int)

    out = pd.concat([X, S], axis=1)
    out["A"] = A
    out["Y"] = Y.to_numpy()
    out["R"] = R.to_numpy()
    out["retained"] = retained
    out["pi_star"] = pi

    print("--- diagnostics ---")
    print(f"R=1: {int(out['R'].sum())}, R=0: {int((1 - out['R']).sum())}")
    print(f"retained: {int(out['retained'].sum())} / {len(out)} "
          f"(expected ~1650)")
    print(f"pi_star: min={pi.min():.3f}, mean={pi.mean():.3f}, "
          f"max={pi.max():.3f}, pi_max={pi_max:.3f}")
    rsub = out.loc[out["retained"] == 1]
    print(f"retained A=1 share: {rsub['A'].mean():.3f}")
    print(f"retained R=1 count: {int(rsub['R'].sum())}")
    print(f"Y NaN total: {int(out['Y'].isna().sum())} "
          f"(all should have R=0)")
    assert (out.loc[out["Y"].isna(), "R"] == 0).all(), "NaN Y must only occur at R=0"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(out)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()