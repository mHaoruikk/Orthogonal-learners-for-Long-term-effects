"""GAIN Experiment A (docs/GAIN_experiment_plan.md §5.1) — pseudo-PEHE and
estimator variance per test unit.

For each learner:
  * Fix Riverside rejection + train/test split via --master_seed.
  * Compute a model-agnostic overlap score w_hat(x_i) on X_test using
    random-forest pi_hat and rho_hat (§5.1 step 1). Saved so stratification
    can be done offline.
  * Across B replicates (default 10), each with a fresh seed injected into
    the learner's random_state fields, fit on D1_train + D2 and predict
    tau_hat on X_test (R=0 only).
  * Per test unit i:
        pehe_per_x[i] = mean_b ((tau_hat_b(x_i) - tau_star(x_i))^2)
        var_per_x[i]  = Var_b  (tau_hat_b(x_i))

Output: outputs/gain_pehe.json.

Usage:
    ~/AppData/Local/miniconda3/envs/lte/python.exe -m scripts.eval_gain_pehe \
        --B 10 --master_seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
logging.getLogger("sklearn").setLevel(logging.ERROR)

from src.data.gain import RealWorldGAIN
from src.data.utils import load_config
from src.eval_utils import DEFAULT_MODELS, LEARNER_REGISTRY, seed_model_cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_gain_pehe")

OUTPUT_PATH = Path("outputs/gain_pehe.json")
OVERLAP_RF_SEED = 20260423  # fixed across runs so omega_hat is reproducible


def compute_overlap_score(ds: RealWorldGAIN) -> np.ndarray:
    """omega_hat(x) = pi_hat(x)(1-pi_hat(x)) * rho_hat(x) on X_test.

    pi_hat fit on D1_train (post-filter Riverside 80%); rho_hat fit on
    D1_train U D2. Both via RandomForest with a fixed seed — no cross-fitting
    needed because X_test is held out. This is the "single, fixed" estimator
    per §5.1 step 1.
    """
    X_train, A_train = ds.X_train, ds.A_train
    X_other = ds.X_other
    X_test = ds.X_test

    rf_params = dict(
        n_estimators=500, max_depth=None, min_samples_leaf=5,
        n_jobs=-1, random_state=OVERLAP_RF_SEED,
    )

    pi_rf = RandomForestClassifier(**rf_params)
    pi_rf.fit(X_train, A_train)
    pi_te = np.clip(pi_rf.predict_proba(X_test)[:, 1], 1e-3, 1.0 - 1e-3)

    X_all = np.concatenate([X_train, X_other], axis=0)
    R_all = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_other))])
    rho_rf = RandomForestClassifier(**rf_params)
    rho_rf.fit(X_all, R_all)
    rho_te = rho_rf.predict_proba(X_test)[:, 1]

    return pi_te * (1.0 - pi_te) * rho_te


def run_one_model(
    model_name: str, dataset: str, trainer: str, B: int, ds: RealWorldGAIN,
) -> dict:
    cfg = load_config(dataset=dataset, model=model_name, trainer=trainer)
    learner_cls = LEARNER_REGISTRY[model_name]

    data, _ = ds.sample()
    n_test = ds.X_test.shape[0]
    tau_matrix = np.empty((B, n_test), dtype=float)

    for b in range(B):
        seeded_cfg = seed_model_cfg(cfg.model, seed=b)
        learner = learner_cls(seeded_cfg)
        learner.fit(data)
        tau_hat = learner.predict_cate(ds.X_test)
        tau_matrix[b] = tau_hat
        logger.info(
            "[%s] replicate %d/%d  tau_hat mean=%.3f std=%.3f",
            model_name, b + 1, B,
            float(tau_hat.mean()), float(tau_hat.std()),
        )

    tau_star = ds.tau_star_test
    sq_err = (tau_matrix - tau_star[None, :]) ** 2
    pehe_per_x = sq_err.mean(axis=0)
    pehe_per_seed = sq_err.mean(axis=1)  # length B: overall PEHE for each replicate
    var_per_x = tau_matrix.var(axis=0, ddof=1) if B > 1 else np.zeros(n_test)

    return {
        "pehe_per_x": pehe_per_x.tolist(),
        "pehe_per_seed": pehe_per_seed.tolist(),
        "var_per_x": var_per_x.tolist(),
        "pehe_mean": float(pehe_per_x.mean()),
        "var_mean": float(var_per_x.mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   choices=sorted(LEARNER_REGISTRY),
                   help="learners to evaluate")
    p.add_argument("--dataset", default="gain",
                   help="dataset config name (config/dataset/<name>.yaml)")
    p.add_argument("--trainer", default="default")
    p.add_argument("--B", type=int, default=10,
                   help="number of replicates per learner")
    p.add_argument("--master_seed", type=int, default=42,
                   help="fixes Riverside rejection sampling + 80/20 split")
    p.add_argument("--y_kind", choices=["mean", "q36", "last_year"], default=None,
                   help="override dataset.y_kind (otherwise read from config)")
    p.add_argument("--output", default=None,
                   help="default: outputs/gain_pehe_<y_kind>.json")
    args = p.parse_args()

    ds_cfg = load_config(dataset=args.dataset, model=args.models[0],
                         trainer=args.trainer).dataset
    if args.y_kind is not None:
        ds_cfg.y_kind = args.y_kind
    ds = RealWorldGAIN(ds_cfg)
    ds.resample(args.master_seed)

    logger.info(
        "GAIN Experiment A: n_test=%d (R=0)  gamma_pi=%s  s=[%d,%d]  y_kind=%s  y_scale=%g",
        len(ds.X_test), ds.gamma_pi,
        ds.s_quarter_start, ds.s_quarter_end, ds.y_kind, ds.y_scale,
    )

    overlap_score = compute_overlap_score(ds)
    logger.info(
        "omega_hat on X_test: mean=%.4f  min=%.4f  max=%.4f",
        float(overlap_score.mean()),
        float(overlap_score.min()), float(overlap_score.max()),
    )

    results = {
        "config": {
            "dataset": args.dataset,
            "trainer": args.trainer,
            "B": args.B,
            "master_seed": args.master_seed,
            "models": list(args.models),
            "gamma_pi": float(ds.gamma_pi),
            "s_quarter_range": [ds.s_quarter_start, ds.s_quarter_end],
            "y_kind": ds.y_kind,
            "y_scale": float(ds.y_scale),
            "overlap_rf_seed": OVERLAP_RF_SEED,
        },
        "test_size": int(len(ds.X_test)),
        "overlap_score": overlap_score.tolist(),
        "tau_star_test": ds.tau_star_test.tolist(),
        "models": {},
    }

    for model_name in args.models:
        logger.info("=== %s ===", model_name)
        results["models"][model_name] = run_one_model(
            model_name, args.dataset, args.trainer, args.B, ds,
        )
        logger.info(
            "[%s] pehe_mean=%.4f  var_mean=%.4f",
            model_name,
            results["models"][model_name]["pehe_mean"],
            results["models"][model_name]["var_mean"],
        )

    out_path = Path(args.output) if args.output else Path(f"outputs/gain_pehe_{ds.y_kind}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
