"""Evaluate meta-learners on a fully synthetic (NieWager) dataset.

For each learner, across B replicate runs:
  * Re-sample the DGP with seed = master_seed + b (fresh X, A, S, Y).
  * Split the sampled TwoSampleDataSplit into (train, test) via
    split_two_sample_data with val_fraction=0.
  * Seed the learner's random_state fields with b and fit on train.
  * Predict tau_hat on test.X_e and compute
        PEHE_b = mean_x ((tau_hat(x) - tau_true(x))^2)

Report per learner: PEHE mean and std across B runs.

The two crucial overlap-controlling parameters
  gamma     = γ_π  (treatment overlap)
  gamma_rho = γ_ρ  (long-term outcome overlap)
are recorded in the output config block and printed to the log.

Usage (lte conda env):
    ~/AppData/Local/miniconda3/envs/lte/python.exe -m scripts.eval_syn \\
        --dataset syn-dual-overlap --B 10 --master_seed 42
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

warnings.filterwarnings("ignore")
logging.getLogger("sklearn").setLevel(logging.ERROR)

from src.data.base_dataset import TwoSampleDataSplit
from src.data.synthetic import NieWagerSyntheticDataset
from src.data.utils import load_config, split_two_sample_data
from src.eval_utils import (
    DEFAULT_MODELS,
    DEFAULT_MODELS_NN,
    LEARNER_REGISTRY,
    seed_model_cfg,
    swap_cate_regressor_to_nn,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_syn")

OUTPUT_DIR = Path("outputs")


def sample_and_split(
    ds_cfg: DictConfig, seed: int, test_fraction: float,
) -> tuple[TwoSampleDataSplit, TwoSampleDataSplit, callable]:
    """Sample a fresh synthetic dataset at `seed` and split train/test."""
    cfg = deepcopy(ds_cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.seed = int(seed)
    ds = NieWagerSyntheticDataset(cfg)
    data, gt = ds.sample()
    train, _, test = split_two_sample_data(
        data,
        val_fraction=0.0,
        test_fraction=test_fraction,
        random_state=int(seed),
    )
    return train, test, gt.tau


def run_one_replicate(
    model_name: str, cfg_model: DictConfig,
    train: TwoSampleDataSplit, test: TwoSampleDataSplit,
    tau_true_fn, seed: int,
) -> float:
    learner_cls = LEARNER_REGISTRY[model_name]
    seeded_cfg = seed_model_cfg(cfg_model, seed=seed)
    learner = learner_cls(seeded_cfg)
    learner.fit(train)
    tau_hat = learner.predict_cate(test.X_e)
    tau_true = tau_true_fn(test.X_e)
    return float(np.mean((tau_hat - tau_true) ** 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=None,
                   choices=sorted(LEARNER_REGISTRY),
                   help="learners to evaluate; defaults to DEFAULT_MODELS(_NN)")
    p.add_argument("--nn_backbone",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="use NN (torch_mlp) backbone for the CATE regressor "
                        "(default); pass --no-nn_backbone to use each model's "
                        "stock cate_regressor")
    p.add_argument("--dataset", default="syn-dual-overlap",
                   help="synthetic dataset cfg (config/dataset/<name>.yaml)")
    p.add_argument("--trainer", default="default")
    p.add_argument("--B", type=int, default=10,
                   help="number of replicate runs")
    p.add_argument("--master_seed", type=int, default=42,
                   help="per-run seed is master_seed + b")
    p.add_argument("--test_fraction", type=float, default=0.2)
    p.add_argument("--output", default=None,
                   help="output JSON path; defaults to outputs/<dataset>_pehe_<backbone>.json")
    args = p.parse_args()

    if args.models is None:
        args.models = list(DEFAULT_MODELS_NN if args.nn_backbone else DEFAULT_MODELS)

    backbone_tag = "nn" if args.nn_backbone else "stock"
    output_path = (
        Path(args.output) if args.output
        else OUTPUT_DIR / f"{args.dataset}_pehe_{backbone_tag}.json"
    )

    base_cfg = load_config(dataset=args.dataset, model=args.models[0],
                           trainer=args.trainer)
    ds_cfg = base_cfg.dataset
    if ds_cfg.type != "synthetic_surrogate":
        raise ValueError(
            f"--dataset={args.dataset} has type={ds_cfg.type}; "
            "eval_syn.py only supports synthetic_surrogate datasets."
        )

    gamma = float(ds_cfg.get("gamma", 0.0))
    gamma_rho = float(ds_cfg.get("gamma_rho", 0.0))
    logger.info(
        "Synthetic dataset '%s': gamma_pi=%.3f  gamma_rho=%.3f  n=%d  dim_x=%d",
        args.dataset, gamma, gamma_rho,
        int(ds_cfg.n), int(ds_cfg.dim_x),
    )

    # Preload per-model cfg once; seed_model_cfg returns a fresh deepcopy each call.
    # When --nn_backbone is set, swap the cate_regressor for a torch_mlp.
    model_cfgs = {}
    for m in args.models:
        cfg_m = load_config(dataset=args.dataset, model=m, trainer=args.trainer).model
        if args.nn_backbone:
            cfg_m = swap_cate_regressor_to_nn(cfg_m)
        model_cfgs[m] = cfg_m
    logger.info(
        "CATE regressor backbone: %s",
        "NN (torch_mlp)" if args.nn_backbone else "per-config (stock)",
    )

    pehe_by_model: dict[str, list[float]] = {m: [] for m in args.models}
    for b in range(args.B):
        seed = int(args.master_seed) + b
        train, test, tau_true_fn = sample_and_split(
            ds_cfg, seed=seed, test_fraction=args.test_fraction,
        )
        logger.info(
            "[run %d/%d  seed=%d]  n_train(e,o)=(%d,%d)  n_test(e,o)=(%d,%d)",
            b + 1, args.B, seed,
            train.X_e.shape[0], train.X_o.shape[0],
            test.X_e.shape[0],  test.X_o.shape[0],
        )
        for m in args.models:
            pehe = run_one_replicate(
                m, model_cfgs[m], train, test, tau_true_fn, seed=b,
            )
            pehe_by_model[m].append(pehe)
            logger.info("  %-14s PEHE=%.5f", m, pehe)

    results = {
        "config": {
            "dataset": args.dataset,
            "trainer": args.trainer,
            "B": args.B,
            "master_seed": args.master_seed,
            "models": list(args.models),
            "nn_backbone": bool(args.nn_backbone),
            "gamma": gamma,
            "gamma_rho": gamma_rho,
            "eta_pi": float(ds_cfg.get("eta_pi", 0.0)),
            "eta_rho": float(ds_cfg.get("eta_rho", 0.0)),
            "n": int(ds_cfg.n),
            "dim_x": int(ds_cfg.dim_x),
            "test_fraction": float(args.test_fraction),
        },
        "models": {},
    }
    for m in args.models:
        arr = np.asarray(pehe_by_model[m])
        results["models"][m] = {
            "pehe": arr.tolist(),
            "pehe_mean": float(arr.mean()),
            "pehe_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        }

    print()
    print("=" * 72)
    print(f"Synthetic dataset:  {args.dataset}")
    print(f"Overlap parameters: gamma_pi={gamma}  gamma_rho={gamma_rho}")
    print(f"CATE backbone:      {'NN (torch_mlp)' if args.nn_backbone else 'stock per-config'}")
    print(f"Replicates:         {args.B}  (seeds {args.master_seed}..{args.master_seed + args.B - 1})")
    print("-" * 72)
    print(f"{'Model':<16}  {'PEHE mean':>10}  {'PEHE std':>10}")
    print("-" * 72)
    for m in args.models:
        r = results["models"][m]
        print(f"{m:<16}  {r['pehe_mean']:>10.5f}  {r['pehe_std']:>10.5f}")
    print("=" * 72)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
