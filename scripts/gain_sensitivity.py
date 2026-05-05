"""GAIN Experiment C — surrogacy sensitivity via T-sweep.

For each T in T_VALUES:
  * Build a fresh RealWorldGAIN with s_quarter_end=T (s_quarter_start=1) and
    `--master_seed`. The Riverside rejection mask + 80/20 train/test split do
    NOT depend on T (they only use covariates), so X_train / X_test are
    identical across T values. The two pseudo-oracles tau*_DR and tau*_RA are
    functions of Y only and are therefore also identical across T — the same
    cached vectors are reused.
  * For each learner, run B replicates (seeds 0..B-1). Record per-replicate
        var_per_x[i]  = Var_b  (tau_hat_b(x_i))
        ate_per_seed  = mean_x tau_hat_b(x)
    and per-oracle metrics (DR and RA, both surrogate-agnostic):
        pehe_per_x[i] = mean_b ((tau_hat_b(x_i) - tau_star(x_i))^2)
        ate_bias      = mean(ate_per_seed) - mean(tau_star_test)
    Reporting both lets us check whether the IPW term in the DR oracle is
    introducing finite-sample noise that drives the observed PEHE rankings.

Output: outputs/gain-surrogacy-sensitivity-<y_kind>.json.

Usage:
    ~/AppData/Local/miniconda3/envs/lte/python.exe -m scripts.gain_sensitivity \\
        --B 5 --master_seed 42 --y_kind emp_mean
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

warnings.filterwarnings("ignore")
logging.getLogger("sklearn").setLevel(logging.ERROR)

from src.data.gain import RealWorldGAIN
from src.data.utils import load_config
from src.eval_utils import LEARNER_REGISTRY, seed_model_cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gain_sensitivity")

# §5.3: T in {2, 4, 6, 12}. T sets s_quarter_end with s_quarter_start fixed at 1.
T_VALUES = [2, 4, 6, 12]

DEFAULT_MODELS = [
    "T-learner", "DR-learner", "TO-learner", "LO-learner",
    "DO-learner", "IPW-learner", "RA-learner",
]

DEFAULT_OUTPUT_PATH = Path("outputs/gain-surrogacy-sensitivity.json")


def _pehe_block(tau_matrix: np.ndarray, tau_star: np.ndarray) -> dict:
    """PEHE / ATE-bias block against a single pseudo-oracle vector."""
    sq_err = (tau_matrix - tau_star[None, :]) ** 2
    pehe_per_x = sq_err.mean(axis=0)
    pehe_per_seed = sq_err.mean(axis=1)
    ate_mean = float(tau_matrix.mean(axis=1).mean())
    return {
        "pehe_per_x": pehe_per_x.tolist(),
        "pehe_per_seed": pehe_per_seed.tolist(),
        "pehe_mean": float(pehe_per_x.mean()),
        "ate_bias": ate_mean - float(tau_star.mean()),
    }


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
            "[%s] replicate %d/%d  tau_hat mean=%.4f std=%.4f",
            model_name, b + 1, B,
            float(tau_hat.mean()), float(tau_hat.std()),
        )

    var_per_x = tau_matrix.var(axis=0, ddof=1) if B > 1 else np.zeros(n_test)
    ate_per_seed = tau_matrix.mean(axis=1)

    return {
        "var_per_x": var_per_x.tolist(),
        "ate_per_seed": ate_per_seed.tolist(),
        "var_mean": float(var_per_x.mean()),
        "ate_mean": float(ate_per_seed.mean()),
        "vs_dr": _pehe_block(tau_matrix, ds.tau_star_test),
        "vs_ra": _pehe_block(tau_matrix, ds.tau_star_ra_test),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   choices=sorted(LEARNER_REGISTRY),
                   help="learners to evaluate")
    p.add_argument("--dataset", default="gain",
                   help="dataset config name (config/dataset/<name>.yaml)")
    p.add_argument("--trainer", default="default")
    p.add_argument("--B", type=int, default=5,
                   help="number of replicates per learner")
    p.add_argument("--master_seed", type=int, default=42,
                   help="fixes Riverside rejection sampling + 80/20 split")
    p.add_argument("--T_values", nargs="+", type=int, default=T_VALUES,
                   help="surrogate horizons to sweep (s_quarter_end values)")
    p.add_argument("--y_kind", default="mean",
                   help="long-term outcome variant (see src/data/gain.py)")
    p.add_argument("--gamma_pi", type=float, default=2.0,
                   help="rejection-sampling intensity (Experiment C fixes this)")
    p.add_argument("--output", default=None,
                   help="output JSON path (default: outputs/gain-surrogacy-sensitivity-<y_kind>.json)")
    args = p.parse_args()
    if args.output is None:
        args.output = f"outputs/gain-surrogacy-sensitivity-{args.y_kind}.json"

    base_ds_cfg = load_config(
        dataset=args.dataset, model=args.models[0], trainer=args.trainer,
    ).dataset

    test_size: int | None = None
    tau_star_test_ref: np.ndarray | None = None
    tau_star_ra_test_ref: np.ndarray | None = None
    results_by_T: dict = {}

    for T in args.T_values:
        logger.info("######## T = %d ########", T)
        ds_cfg = deepcopy(base_ds_cfg)
        OmegaConf.set_struct(ds_cfg, False)
        ds_cfg.s_quarter_start = 1
        ds_cfg.s_quarter_end = int(T)
        ds_cfg.y_kind = args.y_kind
        ds_cfg.gamma_pi = float(args.gamma_pi)

        ds = RealWorldGAIN(ds_cfg)
        ds.resample(args.master_seed)

        if test_size is None:
            test_size = int(len(ds.X_test))
            tau_star_test_ref = ds.tau_star_test.copy()
            tau_star_ra_test_ref = ds.tau_star_ra_test.copy()
        else:
            # Cross-T sanity: split is covariate-only, oracles are Y-only — all
            # must be invariant to T. If not, something upstream broke.
            if not np.allclose(ds.tau_star_test, tau_star_test_ref):
                raise RuntimeError("DR tau_star_test changed between T values")
            if not np.allclose(ds.tau_star_ra_test, tau_star_ra_test_ref):
                raise RuntimeError("RA tau_star_test changed between T values")

        logger.info(
            "GAIN Experiment C: T=%d  y_kind=%s  y_scale=%g  n_test=%d  gamma_pi=%s",
            T, ds.y_kind, ds.y_scale, len(ds.X_test), ds.gamma_pi,
        )
        logger.info(
            "  oracle ATE: DR=%+.4f  RA=%+.4f",
            float(tau_star_test_ref.mean()), float(tau_star_ra_test_ref.mean()),
        )

        models_results: dict = {}
        for model_name in args.models:
            logger.info("=== %s [T=%d] ===", model_name, T)
            res = run_one_model(model_name, args.dataset, args.trainer, args.B, ds)
            models_results[model_name] = res
            logger.info(
                "[%s][T=%d] var=%.4f  pehe_DR=%.4f bias_DR=%+.4f  "
                "pehe_RA=%.4f bias_RA=%+.4f",
                model_name, T, res["var_mean"],
                res["vs_dr"]["pehe_mean"], res["vs_dr"]["ate_bias"],
                res["vs_ra"]["pehe_mean"], res["vs_ra"]["ate_bias"],
            )

        results_by_T[str(T)] = {
            "s_quarter_range": [1, int(T)],
            "models": models_results,
        }

    out = {
        "config": {
            "dataset": args.dataset,
            "trainer": args.trainer,
            "B": args.B,
            "master_seed": args.master_seed,
            "models": list(args.models),
            "T_values": [int(t) for t in args.T_values],
            "y_kind": args.y_kind,
            "gamma_pi": float(args.gamma_pi),
        },
        "test_size": test_size,
        "tau_star_test_dr": tau_star_test_ref.tolist() if tau_star_test_ref is not None else [],
        "tau_star_test_ra": tau_star_ra_test_ref.tolist() if tau_star_ra_test_ref is not None else [],
        "ate_oracle_dr": float(tau_star_test_ref.mean()) if tau_star_test_ref is not None else None,
        "ate_oracle_ra": float(tau_star_ra_test_ref.mean()) if tau_star_ra_test_ref is not None else None,
        "results_by_T": results_by_T,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
