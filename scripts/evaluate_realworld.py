"""Evaluate any meta-learner on real IST-3 data.

Two metrics per run (docs/IST-realword.md §6.i and §6.ii):

(i)  DR pseudo-outcome MSE (plug-in). Fixed approximated ground truth
     tau_tilde(x) = mu_1(x) - mu_0(x), where mu_a is fit on the retained
     R==1 rows (where Y is observed). One PEHE_s per replicate.

(ii) Subgroup stability across overlap strata. Stratify R==0 points by
     deciles of s(x) = pi_hat(x)*(1-pi_hat(x))*rho_hat(x); within each
     stratum report mean over x of Var_s(tau_hat_s(x)) across replicates.

Structure (simplified): no bootstrap, no external CV. Single fixed
`retained` cohort (resample at --seed). A single overlap-bin vector
and a single ground-truth tau_tilde vector, both frozen across
replicates. S replicates (default 20), each a fresh random seed
threaded through the model cfg so every sub-regressor / nuisance KFold
gets randomized. For every replicate the learner is fit on the full
retained TwoSampleDataSplit (it cross-fits its own nuisances
internally); we then read off tau_hat_s on the R==0 rows.

Usage (lte conda env):
    ~/AppData/Local/miniconda3/envs/lte/python.exe -m scripts.evaluate_realworld \\
        --model T-learner --S_seeds 20 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
logging.getLogger("sklearn").setLevel(logging.ERROR)

from src.data.base_dataset import TwoSampleDataSplit
from src.data.real_world import RealWorldIST
from src.data.utils import load_config
from src.model.ipw_learner import IPW_Learner
from src.model.lto_learner import LTO_Learner
from src.model.ra_learner import RA_Learner
from src.model.t_learner import TLearner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate_realworld")

LEARNER_REGISTRY = {
    "T-learner":   TLearner,
    "DR-learner":  LTO_Learner,   # cfg sets weight_type=identity
    "TO-learner":  LTO_Learner,   # cfg sets weight_type=to
    "LO-learner":  LTO_Learner,   # cfg sets weight_type=lo
    "DO-learner":  LTO_Learner,   # cfg sets weight_type=dual
    "IPW-learner": IPW_Learner,
    "RA-learner":  RA_Learner,
}

DEFAULT_MODELS = [
    "T-learner", "DR-learner", "TO-learner", "LO-learner",
    "DO-learner", "IPW-learner", "RA-learner",
]

N_STRATA = 10
GT_SEED = 20260422       # fixed seed for ground-truth mu_a and overlap pi/rho
OUTPUT_PATH = Path("outputs/real-world.json")


def seed_model_cfg(cfg: DictConfig, seed: int) -> DictConfig:
    """Deep-copy cfg and inject `seed` into every random_state / seed field.

    - Top-level: `cfg.random_state = seed` (read by NuisanceFactory for its KFold).
    - Every sub-node with a `type` field: `parameters.random_state = seed`,
      plus `parameters.seed = seed` for xgboost.
    """
    new_cfg = deepcopy(cfg)
    OmegaConf.set_struct(new_cfg, False)
    new_cfg.random_state = int(seed)

    # sklearn base types that accept a `random_state` kwarg.
    _RS_TYPES = {"random_forest", "logistic", "lasso", "ridge", "elasticnet"}

    def _walk(node):
        if not isinstance(node, DictConfig):
            return
        if "type" in node:
            t = str(node.type).lower()
            if t in _RS_TYPES or t == "xgboost":
                if node.get("parameters") is None:
                    node.parameters = OmegaConf.create({})
                node.parameters.random_state = int(seed)
                if t == "xgboost":
                    node.parameters.seed = int(seed)
            # linear, solve_linear, torch_mlp: no random_state kwarg; skip.
        for k in list(node.keys()):
            _walk(node[k])

    _walk(new_cfg)
    return new_cfg


def fit_plug_in_mu(
    X_o: np.ndarray, A_o: np.ndarray, Y_o: np.ndarray,
) -> tuple[XGBRegressor, XGBRegressor]:
    """Plug-in T-learner ground truth: mu_a(x) = E[Y | X=x, A=a] on R==1 rows."""
    xgb_params = dict(n_estimators=200, max_depth=5, learning_rate=0.1,
                      subsample=0.8, colsample_bytree=0.8,
                      n_jobs=-1, verbosity=0)
    mu0 = XGBRegressor(**xgb_params, random_state=GT_SEED, seed=GT_SEED)
    mu1 = XGBRegressor(**xgb_params, random_state=GT_SEED + 1, seed=GT_SEED + 1)
    mu0.fit(X_o[A_o == 0], Y_o[A_o == 0])
    mu1.fit(X_o[A_o == 1], Y_o[A_o == 1])
    return mu0, mu1


def compute_overlap_bins(
    X_all: np.ndarray, A_all: np.ndarray, R_all: np.ndarray,
    X_eval: np.ndarray, n_bins: int = N_STRATA,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit pi_hat, rho_hat on full retained data; return (s(x), bin_ids) at X_eval."""
    lr_params = dict(penalty="l2", C=1.0, max_iter=2000, tol=1e-4, fit_intercept=True)
    pi_clf = LogisticRegression(**lr_params, random_state=GT_SEED)
    pi_clf.fit(X_all, A_all)
    pi = pi_clf.predict_proba(X_eval)[:, 1]

    rho_clf = LogisticRegression(**lr_params, random_state=GT_SEED + 1)
    rho_clf.fit(X_all, R_all)
    rho = rho_clf.predict_proba(X_eval)[:, 1]

    score = pi * (1.0 - pi) * rho
    n = len(score)
    order = np.argsort(score, kind="stable")
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)
    bin_ids = (ranks * n_bins) // n
    return score, bin_ids


def fit_and_predict(
    learner_cls, seeded_cfg: DictConfig,
    X: np.ndarray, A: np.ndarray, S: np.ndarray, Y: np.ndarray,
    r0_idx: np.ndarray, r1_idx: np.ndarray,
) -> np.ndarray:
    """Fit learner on the full retained TwoSampleDataSplit, predict at X[R==0].

    The learner cross-fits its own nuisances internally; we treat it as a
    black box and read off tau_hat on the experiment rows.
    """
    train_split = TwoSampleDataSplit(
        X_e=X[r0_idx], A_e=A[r0_idx], S_e=S[r0_idx],
        X_o=X[r1_idx], S_o=S[r1_idx], Y_o=Y[r1_idx],
    )
    learner = learner_cls(seeded_cfg)
    learner.fit(train_split)
    return learner.predict_cate(X[r0_idx])


def stratum_variances(var_per_x: np.ndarray, bin_ids: np.ndarray,
                      n_bins: int = N_STRATA) -> np.ndarray:
    out = np.full(n_bins, np.nan, dtype=float)
    for s in range(n_bins):
        mask = bin_ids == s
        if mask.any():
            out[s] = float(var_per_x[mask].mean())
    return out


def evaluate_one_model(
    model_name: str, dataset: str, trainer: str, S_seeds: int,
    X: np.ndarray, A: np.ndarray, S: np.ndarray, Y: np.ndarray,
    r0_idx: np.ndarray, r1_idx: np.ndarray,
    tau_tilde: np.ndarray, bin_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run S_seeds replicates for one learner; return (pehe_arr, strat_var)."""
    cfg = load_config(dataset=dataset, model=model_name, trainer=trainer)
    learner_cls = LEARNER_REGISTRY[model_name]

    tau_matrix = np.empty((len(r0_idx), S_seeds), dtype=float)
    pehe_list: list[float] = []
    for s in range(S_seeds):
        seeded_cfg = seed_model_cfg(cfg.model, seed=s)
        tau_hat_s = fit_and_predict(
            learner_cls, seeded_cfg, X, A, S, Y, r0_idx, r1_idx,
        )
        tau_matrix[:, s] = tau_hat_s
        pehe_s = float(np.mean((tau_hat_s - tau_tilde) ** 2))
        pehe_list.append(pehe_s)
        logger.info("[%s] seed=%d  PEHE=%.5f", model_name, s, pehe_s)

    pehe_arr = np.asarray(pehe_list)
    var_per_x = tau_matrix.var(axis=1, ddof=1)
    strat_var = stratum_variances(var_per_x, bin_ids)

    print()
    print("=" * 60)
    print(f"Model:           {model_name}")
    print(f"Replicates:      {S_seeds}")
    print(f"PEHE values:     {np.round(pehe_arr, 5).tolist()}")
    print(f"PEHE mean:       {pehe_arr.mean():.5f}")
    print(f"PEHE std:        {pehe_arr.std(ddof=1):.5f}")
    print("-" * 60)
    print(f"Subgroup stability ({N_STRATA} overlap deciles)")
    print(f"  bin 0 = lowest overlap (hardest), bin 9 = highest overlap")
    print(f"  mean over x in bin of Var_s(tau_hat_s(x)) across {S_seeds} replicates:")
    print(f"  {np.round(strat_var, 6).tolist()}")
    print("=" * 60)

    return pehe_arr, strat_var


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   choices=sorted(LEARNER_REGISTRY),
                   help="one or more learners to evaluate")
    p.add_argument("--dataset", default="real-world-ist")
    p.add_argument("--trainer", default="default")
    p.add_argument("--S_seeds", type=int, default=20,
                   help="number of random-seed replicates")
    p.add_argument("--seed", type=int, default=42,
                   help="fixes the retained-cohort resample; replicate seeds are 0..S_seeds-1")
    args = p.parse_args()

    # Fixed resampling + frozen quantities (model-independent)
    ds_cfg = load_config(dataset=args.dataset, model=args.models[0],
                         trainer=args.trainer).dataset
    ds = RealWorldIST(ds_cfg)
    ds.resample_retained(int(args.seed))
    X, A, S, R, Y = ds.X, ds.A, ds.S, ds.R, ds.Y
    r0_idx = np.where(R == 0)[0]
    r1_idx = np.where(R == 1)[0]
    X_r0 = X[r0_idx]
    logger.info("Retained cohort: n=%d  (R=0: %d, R=1: %d)",
                len(X), len(r0_idx), len(r1_idx))

    mu0, mu1 = fit_plug_in_mu(X[r1_idx], A[r1_idx], Y[r1_idx])
    tau_tilde = mu1.predict(X_r0) - mu0.predict(X_r0)
    logger.info("Plug-in ground truth: tau_tilde mean=%.4f  std=%.4f",
                tau_tilde.mean(), tau_tilde.std())

    score, bin_ids = compute_overlap_bins(X, A, R, X_r0)
    logger.info("Overlap score range: [%.4f, %.4f]; bin sizes=%s",
                score.min(), score.max(),
                np.bincount(bin_ids, minlength=N_STRATA).tolist())

    results: dict[str, dict[str, list[float]]] = {}
    for model_name in args.models:
        pehe_arr, strat_var = evaluate_one_model(
            model_name, args.dataset, args.trainer, args.S_seeds,
            X, A, S, Y, r0_idx, r1_idx, tau_tilde, bin_ids,
        )
        results[model_name] = {
            "pehe": pehe_arr.tolist(),
            "strat_var": strat_var.tolist(),
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s (models: %s)", OUTPUT_PATH, list(results))


if __name__ == "__main__":
    main()
