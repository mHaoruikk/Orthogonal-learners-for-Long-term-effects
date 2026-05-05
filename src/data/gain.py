"""GAIN real-world dataset for HLTE experiments.

Implements the dataset construction pipeline in docs/GAIN_experiment_plan.md
§3 and the pseudo-oracle CATE in §4. The pseudo-oracle is averaged across
`oracle_seeds` independent cross-fitted DR runs for stability, then cached to
disk.

Usage:
    ds = RealWorldGAIN(cfg)
    data, gt = ds.sample()          # D1_train (R=0) + D2 (R=1)
    X_test = ds.X_test              # 20% held-out Riverside
    tau_star = ds.tau_star_test     # pseudo-oracle tau* on the held-out set
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

from src.data.base_dataset import BaseDataset, GroundTruth, TwoSampleDataSplit

logger = logging.getLogger(__name__)


# Covariate block per docs/GAIN_experiment_plan.md §2.6.
X_COLS = [
    "age", "agesq",
    "xsexf", "xhsdip", "x1chld", "xchld05",
    "single", "dumkids",
    "white", "hisp", "black",
    "grde911", "grade12", "grd1315", "grade16", "grd1720",
    *[f"tcprn{i}" for i in range(1, 11)],
    *[f"paid{i}" for i in range(1, 5)],
    *[f"adcpc{i}" for i in range(1, 5)],
    *[f"padcpc{i}" for i in range(1, 5)],
    "grew1", "gepop1",
]

A_COL = "e"
RIVER_COL = "river"

VALID_Y_KINDS = ("mean", "q36", "last_year", "mean_30_36",
                 "emp_mean", "emp_q36", "emp_30_36")


def _build_surrogate(df: pd.DataFrame, q_start: int, q_end: int) -> np.ndarray:
    """S = (tcedd_{q_start..q_end}, aid_{...}, 1{tcedd_k > 0}_{...})."""
    qs = list(range(q_start, q_end + 1))
    tcedd = df[[f"tcedd{k}" for k in qs]].to_numpy(dtype=float)
    aid = df[[f"aid{k}" for k in qs]].to_numpy(dtype=float)
    emp = (tcedd > 0).astype(float)
    return np.concatenate([tcedd, aid, emp], axis=1)


def _build_y(df: pd.DataFrame, kind: str, scale: float = 1.0) -> np.ndarray:
    """Long-term outcome variants, multiplied by `scale`.

    Earnings kinds (raw $/quarter, scaled by `scale`):
    - "mean"       : Y = mean(tcedd13..tcedd36)   (Athey-style long-run, default)
    - "q36"        : Y = tcedd36                  (terminal quarter)
    - "last_year"  : Y = mean(tcedd33..tcedd36)   (final-year mean)
    - "mean_30_36" : Y = mean(tcedd30..tcedd36)   (terminal 7-quarter mean)

    Employment kinds (each quarter's employment indicator = 1{tcedd_k > 0};
    the dataset has no native employment field, this matches Athey et al.):
    - "emp_mean"   : Y = mean_{k=13..36} 1{tcedd_k > 0}   (mean employment rate)
    - "emp_q36"    : Y = 1{tcedd_36 > 0}                  (terminal-quarter employment)
    - "emp_30_36"  : Y = mean_{k=30..36} 1{tcedd_k > 0}   (terminal 7-quarter mean employment)
    """
    if kind == "mean":
        cols = [f"tcedd{k}" for k in range(13, 37)]
        y = df[cols].to_numpy(dtype=float).mean(axis=1)
    elif kind == "q36":
        y = df["tcedd36"].to_numpy(dtype=float)
    elif kind == "last_year":
        cols = [f"tcedd{k}" for k in range(33, 37)]
        y = df[cols].to_numpy(dtype=float).mean(axis=1)
    elif kind == "mean_30_36":
        cols = [f"tcedd{k}" for k in range(30, 37)]
        y = df[cols].to_numpy(dtype=float).mean(axis=1)
    elif kind == "emp_mean":
        cols = [f"tcedd{k}" for k in range(13, 37)]
        y = (df[cols].to_numpy(dtype=float) > 0).astype(float).mean(axis=1)
    elif kind == "emp_q36":
        y = (df["tcedd36"].to_numpy(dtype=float) > 0).astype(float)
    elif kind == "emp_30_36":
        cols = [f"tcedd{k}" for k in range(30, 37)]
        y = (df[cols].to_numpy(dtype=float) > 0).astype(float).mean(axis=1)
    else:
        raise ValueError(f"y_kind must be one of {VALID_Y_KINDS}, got {kind!r}")
    return y * float(scale)


def _zscore(x: np.ndarray) -> np.ndarray:
    mu = float(x.mean())
    sd = float(x.std())
    return (x - mu) / sd if sd > 1e-8 else x - mu


def _pseudo_oracle_tau(
    X: np.ndarray, A: np.ndarray, Y: np.ndarray,
    z_pi: np.ndarray, seeds: list[int],
) -> np.ndarray:
    """Cross-fitted DR pseudo-oracle averaged over `seeds` runs (§4).

    For each seed: 5-fold CV; within each fold fit
      - mu_a(x) via XGBoost (500 rounds, depth 4)
      - pi(x) via logistic regression on z_pi = (x_earn, x_age)
    Build OOF pseudo-outcomes
      psi_i = (A_i - pi_i) / (pi_i(1-pi_i)) * (Y_i - mu_{A_i}(x_i)) + mu_1(x_i) - mu_0(x_i)
    Fit a final-stage XGBoost regressor on (X, psi) and predict on all X.
    Return the mean of per-seed predictions.
    """
    n = X.shape[0]
    mu_params = dict(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, verbosity=0, tree_method="hist",
    )
    final_params = dict(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, verbosity=0, tree_method="hist",
    )

    preds = np.zeros((len(seeds), n), dtype=float)
    for s_idx, seed in enumerate(seeds):
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        psi = np.empty(n, dtype=float)

        for fold, (tr, te) in enumerate(kf.split(X)):
            pi_clf = LogisticRegression(
                penalty="l2", C=1.0, max_iter=2000, tol=1e-4,
                random_state=seed * 100 + fold,
            )
            pi_clf.fit(z_pi[tr], A[tr])
            pi_te = np.clip(pi_clf.predict_proba(z_pi[te])[:, 1], 0.05, 0.95)

            tr0 = tr[A[tr] == 0]
            tr1 = tr[A[tr] == 1]
            mu0 = XGBRegressor(**mu_params, random_state=seed * 100 + 10 * fold + 1)
            mu1 = XGBRegressor(**mu_params, random_state=seed * 100 + 10 * fold + 2)
            mu0.fit(X[tr0], Y[tr0])
            mu1.fit(X[tr1], Y[tr1])
            mu0_te = mu0.predict(X[te])
            mu1_te = mu1.predict(X[te])

            A_te = A[te]
            mu_a_te = np.where(A_te == 1, mu1_te, mu0_te)
            ipw = (A_te - pi_te) / (pi_te * (1.0 - pi_te))
            psi[te] = ipw * (Y[te] - mu_a_te) + (mu1_te - mu0_te)

        final = XGBRegressor(**final_params, random_state=seed * 100 + 9999)
        final.fit(X, psi)
        preds[s_idx] = final.predict(X)
        logger.info(
            "  oracle run %d/%d (seed=%d): tau* mean=%.2f, std=%.2f",
            s_idx + 1, len(seeds), seed,
            float(preds[s_idx].mean()), float(preds[s_idx].std()),
        )

    return preds.mean(axis=0)


def _pseudo_oracle_ra_tau(
    X: np.ndarray, A: np.ndarray, Y: np.ndarray, seeds: list[int],
) -> np.ndarray:
    """Cross-fitted RA (T-learner) pseudo-oracle averaged over `seeds` runs.

    Same skeleton as `_pseudo_oracle_tau` but with the regression-only pseudo-
    outcome psi_i = mu_1_hat(x_i) - mu_0_hat(x_i) (no IPW correction). Lower
    variance than DR when N is small or pi is near {0, 1}; higher bias if mu_a
    is misspecified. Useful as a stability sanity check on the DR oracle.
    """
    n = X.shape[0]
    mu_params = dict(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, verbosity=0, tree_method="hist",
    )
    final_params = dict(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, verbosity=0, tree_method="hist",
    )

    preds = np.zeros((len(seeds), n), dtype=float)
    for s_idx, seed in enumerate(seeds):
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        psi = np.empty(n, dtype=float)

        for fold, (tr, te) in enumerate(kf.split(X)):
            tr0 = tr[A[tr] == 0]
            tr1 = tr[A[tr] == 1]
            mu0 = XGBRegressor(**mu_params, random_state=seed * 100 + 10 * fold + 1)
            mu1 = XGBRegressor(**mu_params, random_state=seed * 100 + 10 * fold + 2)
            mu0.fit(X[tr0], Y[tr0])
            mu1.fit(X[tr1], Y[tr1])
            psi[te] = mu1.predict(X[te]) - mu0.predict(X[te])

        final = XGBRegressor(**final_params, random_state=seed * 100 + 9999)
        final.fit(X, psi)
        preds[s_idx] = final.predict(X)
        logger.info(
            "  RA oracle run %d/%d (seed=%d): tau* mean=%.4f, std=%.4f",
            s_idx + 1, len(seeds), seed,
            float(preds[s_idx].mean()), float(preds[s_idx].std()),
        )

    return preds.mean(axis=0)


class RealWorldGAIN(BaseDataset):
    """HLTE dataset built from the GAIN CSV (Athey-style Riverside vs. others).

    Pipeline (docs/GAIN_experiment_plan.md §3):
      §3.2  Split: D1 = Riverside (retain X, A, S), D2 = the other three sites
            (retain X, S, Y).
      §3.3  Induce low treatment overlap inside Riverside via covariate-
            dependent rejection sampling controlled by `gamma_pi`.
      §3.4  80/20 train/test split on Riverside, stratified on A.

    Pseudo-oracle (§4): fit once on the full unfiltered Riverside sample; the
    cached values survive `resample()` calls so the MC loop never refits.
    """

    def __init__(self, config):
        self.config = config
        self.seed = int(config.seed)

        self.s_quarter_start = int(getattr(config, "s_quarter_start", 1))
        self.s_quarter_end = int(getattr(config, "s_quarter_end", 6))
        if not (1 <= self.s_quarter_start <= self.s_quarter_end <= 36):
            raise ValueError(
                f"s_quarter_start/end must satisfy 1 <= start <= end <= 36; "
                f"got [{self.s_quarter_start}, {self.s_quarter_end}]"
            )

        self.y_kind = str(getattr(config, "y_kind", "mean"))
        if self.y_kind not in VALID_Y_KINDS:
            raise ValueError(f"y_kind must be one of {VALID_Y_KINDS}, got {self.y_kind!r}")
        self.y_scale = float(getattr(config, "y_scale", 1.0))
        if self.y_kind.startswith("emp_") and self.y_scale != 1.0:
            logger.info(
                "RealWorldGAIN: y_kind=%r is an employment rate in [0,1]; overriding "
                "y_scale %g -> 1.0", self.y_kind, self.y_scale,
            )
            self.y_scale = 1.0

        self.gamma_pi = float(getattr(config, "gamma_pi", 2.0))
        self.beta_earn = float(getattr(config, "beta_earn", 0.7))
        self.beta_age = float(getattr(config, "beta_age", 0.3))
        self.trim_eps = float(getattr(config, "trim_eps", 0.01))
        self.train_frac = float(getattr(config, "train_frac", 0.8))
        self.oracle_seeds = list(getattr(config, "oracle_seeds", [0, 1, 2, 3, 4]))

        self.data_path = Path(config.data_path)
        df_full = pd.read_csv(self.data_path)
        missing = [c for c in X_COLS + [A_COL, RIVER_COL] if c not in df_full.columns]
        if missing:
            raise ValueError(f"CSV {self.data_path} missing columns: {missing}")
        logger.info("RealWorldGAIN: loaded %s (%d rows)", self.data_path, len(df_full))

        # §3.2 D1 / D2 split
        river_mask = df_full[RIVER_COL].to_numpy() == 1
        df_river = df_full.loc[river_mask].reset_index(drop=True)
        df_other = df_full.loc[~river_mask].reset_index(drop=True)
        logger.info("RealWorldGAIN: Riverside n=%d, Others n=%d",
                    len(df_river), len(df_other))

        # Covariates, treatment, surrogates, long-term outcome
        self.X_river = df_river[X_COLS].to_numpy(dtype=float)
        self.A_river = df_river[A_COL].to_numpy(dtype=int)
        self.S_river = _build_surrogate(df_river, self.s_quarter_start, self.s_quarter_end)
        self.Y_river_full = _build_y(df_river, self.y_kind, self.y_scale)       # oracle input only

        self.X_other = df_other[X_COLS].to_numpy(dtype=float)
        self.S_other = _build_surrogate(df_other, self.s_quarter_start, self.s_quarter_end)
        self.Y_other = _build_y(df_other, self.y_kind, self.y_scale)

        self.dim_x = self.X_river.shape[1]

        # Standardized earn/age for m(x) — fit on full Riverside
        earn = df_river[[f"tcprn{i}" for i in range(1, 11)]].to_numpy().mean(axis=1)
        age = df_river["age"].to_numpy(dtype=float)
        self.x_earn_river = _zscore(earn)
        self.x_age_river = _zscore(age)

        # §4 pseudo-oracle (cached). Two flavours: DR (default) and RA. Both
        # are functions of (X, A, Y) only — surrogate-agnostic — so they share
        # cache lifetimes with `y_kind`/`y_scale`/`dim_x`.
        self.tau_star_river = self._compute_or_load_oracle()
        self.tau_star_ra_river = self._compute_or_load_oracle_ra()

        # §3.3 + §3.4
        self.resample(self.seed)

    # ------------------------------------------------------------
    def _oracle_cache_path(self) -> Path:
        key = "-".join(str(s) for s in self.oracle_seeds)
        return (
            self.data_path.parent
            / f"tau_star_gain_oseeds{key}_y-{self.y_kind}"
              f"_sc{self.y_scale:g}_d{self.dim_x}.npy"
        )

    def _compute_or_load_oracle(self) -> np.ndarray:
        path = self._oracle_cache_path()
        if path.exists():
            tau = np.load(path)
            if tau.shape == (self.X_river.shape[0],):
                logger.info("RealWorldGAIN: loaded cached pseudo-oracle %s", path)
                return tau
            logger.info("RealWorldGAIN: oracle cache shape mismatch, recomputing")

        logger.info(
            "RealWorldGAIN: fitting pseudo-oracle (DR 5-fold CV, %d averaged runs)",
            len(self.oracle_seeds),
        )
        z_pi = np.column_stack([self.x_earn_river, self.x_age_river])
        tau = _pseudo_oracle_tau(
            self.X_river, self.A_river, self.Y_river_full, z_pi, self.oracle_seeds,
        )
        np.save(path, tau)
        logger.info(
            "RealWorldGAIN: cached pseudo-oracle to %s  (mean=%.2f std=%.2f)",
            path, float(tau.mean()), float(tau.std()),
        )
        return tau

    # ------------------------------------------------------------
    def _oracle_ra_cache_path(self) -> Path:
        key = "-".join(str(s) for s in self.oracle_seeds)
        return (
            self.data_path.parent
            / f"tau_star_gain_ra_oseeds{key}_y-{self.y_kind}"
              f"_sc{self.y_scale:g}_d{self.dim_x}.npy"
        )

    def _compute_or_load_oracle_ra(self) -> np.ndarray:
        path = self._oracle_ra_cache_path()
        if path.exists():
            tau = np.load(path)
            if tau.shape == (self.X_river.shape[0],):
                logger.info("RealWorldGAIN: loaded cached RA pseudo-oracle %s", path)
                return tau
            logger.info("RealWorldGAIN: RA oracle cache shape mismatch, recomputing")

        logger.info(
            "RealWorldGAIN: fitting RA pseudo-oracle (T-learner 5-fold CV, %d averaged runs)",
            len(self.oracle_seeds),
        )
        tau = _pseudo_oracle_ra_tau(
            self.X_river, self.A_river, self.Y_river_full, self.oracle_seeds,
        )
        np.save(path, tau)
        logger.info(
            "RealWorldGAIN: cached RA pseudo-oracle to %s  (mean=%.4f std=%.4f)",
            path, float(tau.mean()), float(tau.std()),
        )
        return tau

    # ------------------------------------------------------------
    def resample(self, seed: int) -> "RealWorldGAIN":
        """Redraw §3.3 rejection sampling and §3.4 train/test split."""
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)

        logits = self.gamma_pi * (
            self.beta_earn * self.x_earn_river + self.beta_age * self.x_age_river
        )
        m = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        m = np.clip(m, self.trim_eps, 1.0 - self.trim_eps)

        keep_prob = np.where(self.A_river == 1, m, 1.0 - m)
        keep = rng.random(self.X_river.shape[0]) < keep_prob

        X_k = self.X_river[keep]
        A_k = self.A_river[keep]
        S_k = self.S_river[keep]
        tau_k = self.tau_star_river[keep]
        tau_ra_k = self.tau_star_ra_river[keep]
        pi_post = (m[keep] * 0.81) / (m[keep] * 0.81 + (1 - m[keep]) * (1 - 0.81))

        logger.info(
            "RealWorldGAIN: rejection-sampled %d/%d Riverside (A=1: %d, A=0: %d); "
            "post-filter pi_mean=%.3f",
            int(keep.sum()), len(keep),
            int((A_k == 1).sum()), int((A_k == 0).sum()), float(pi_post.mean()),
        )

        idx = np.arange(len(A_k))
        tr_idx, te_idx = train_test_split(
            idx, train_size=self.train_frac, stratify=A_k,
            random_state=self.seed, shuffle=True,
        )

        self.X_train, self.A_train, self.S_train = X_k[tr_idx], A_k[tr_idx], S_k[tr_idx]
        self.X_test, self.A_test, self.S_test = X_k[te_idx], A_k[te_idx], S_k[te_idx]
        self.tau_star_train = tau_k[tr_idx]
        self.tau_star_test = tau_k[te_idx]
        self.tau_star_ra_train = tau_ra_k[tr_idx]
        self.tau_star_ra_test = tau_ra_k[te_idx]

        logger.info(
            "RealWorldGAIN: Riverside train n=%d, test n=%d",
            len(tr_idx), len(te_idx),
        )
        return self

    # ------------------------------------------------------------
    def sample(self) -> Tuple[TwoSampleDataSplit, GroundTruth]:
        """Training-time view: D1 (Riverside train, R=0) + D2 (others, R=1).

        The pseudo-oracle is only defined on Riverside rows, so
        `GroundTruth.tau` returns NaN for arbitrary X. Evaluate pseudo-PEHE
        against `self.tau_star_test` on `self.X_test` directly.
        """
        data = TwoSampleDataSplit(
            X_e=self.X_train.copy(), A_e=self.A_train.copy(), S_e=self.S_train.copy(),
            X_o=self.X_other.copy(), S_o=self.S_other.copy(), Y_o=self.Y_other.copy(),
        )
        gt = GroundTruth(
            tau=lambda X: np.full(X.shape[0], np.nan),
            pi_E=None, e_O=None,
        )
        logger.info(
            "RealWorldGAIN.sample: n_e=%d (R=0), n_o=%d (R=1), dim_x=%d, dim_s=%d",
            data.X_e.shape[0], data.X_o.shape[0], self.dim_x, data.S_e.shape[1],
        )
        return data, gt
