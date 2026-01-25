import logging
from typing import List, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.data.utils import split_two_sample_data
from src.model.dr_learner import DRLearner
from src.model.t_learner import TLearner
from src.model.t_r_learner import tRlearner
from src.utils import simulate_dataset, evaluate_mse
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _clone_cfg(cfg: DictConfig, seed: int, gamma: float) -> DictConfig:
    cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_copy.dataset.seed = seed
    cfg_copy.dataset.gamma = gamma
    return cfg_copy


def _train_single(cfg: DictConfig, eval_X: np.ndarray, seed: int) -> Tuple[np.ndarray, float, float]:
    """Train DR-learner once, return predictions on eval_X, MSE, and average overlap on test set."""
    data, ground_truth = simulate_dataset(cfg)

    val_fraction = cfg.trainer.get("val_fraction", 0.2)
    test_fraction = cfg.trainer.get("test_fraction", 0.2)
    train_data, _, test_data = split_two_sample_data(
        data, val_fraction=val_fraction, test_fraction=test_fraction, random_state=seed
    )
    if cfg.model.name == "t_learner":
        learner = TLearner(cfg.model)
    elif cfg.model.name == "dr_learner":
        learner = DRLearner(cfg.model)
    elif cfg.model.name == "r_learner":
        learner = tRlearner(cfg.model)
    else:
        learner = None

    learner.fit(train_data)

    # Estimate average overlap pi(X)(1-pi(X)) on test experimental covariates
    if test_data.X_e.shape[0] > 0:
        cf_nuis = learner.nuisance_factory.crossfit_nuisance(train_data)
        pi_pred_folds = [nm.pi_x.predict(test_data.X_e) for nm in cf_nuis.folds]
        pi_mean = np.mean(np.stack(pi_pred_folds, axis=0), axis=0)
        overlap = pi_mean * (1.0 - pi_mean)
        avg_overlap = float(np.mean(overlap))
    else:
        avg_overlap = float("nan")

    preds = learner.predict_cate(eval_X)
    mse = evaluate_mse(learner, test_data, ground_truth)
    return preds, mse, avg_overlap


@hydra.main(version_base="1.1", config_name="config.yaml", config_path="../config/")
def main(args: DictConfig):
    logger.info("Loaded configuration:\n%s", OmegaConf.to_yaml(args))

    gammas = [0.0, 1.0, 2., 3., 4.0]
    num_trials = args.get("num_trials", 5)
    base_seed = int(args.dataset.seed)

    results = []

    for gamma in gammas:
        logger.info("=== Gamma %.2f ===", gamma)

        # Fixed eval set per gamma
        eval_cfg = _clone_cfg(args, seed=base_seed, gamma=gamma)
        eval_data, _ = simulate_dataset(eval_cfg)
        X_eval = eval_data.X_e

        preds_across_trials: List[np.ndarray] = []
        mses: List[float] = []
        overlaps: List[float] = []

        for trial in range(num_trials):
            trial_seed = base_seed + trial
            cfg_run = _clone_cfg(args, seed=trial_seed, gamma=gamma)
            preds, mse, overlap = _train_single(cfg=cfg_run, eval_X=X_eval, seed=trial_seed)
            preds_across_trials.append(preds)
            mses.append(mse)
            overlaps.append(overlap)

        pred_matrix = np.stack(preds_across_trials, axis=0)
        if pred_matrix.shape[0] > 1:
            pointwise_var = np.var(pred_matrix, axis=0, ddof=1)
        else:
            pointwise_var = np.zeros_like(pred_matrix[0])

        avg_pointwise_var = float(np.mean(pointwise_var))
        mse_mean = float(np.mean(mses))
        mse_std = float(np.std(mses, ddof=1)) if len(mses) > 1 else 0.0
        overlap_mean = float(np.nanmean(overlaps)) if len(overlaps) > 0 else float("nan")

        results.append({
            "gamma": gamma,
            "avg_pointwise_var": avg_pointwise_var,
            "mse_mean": mse_mean,
            "mse_std": mse_std,
            "overlap_mean": overlap_mean,
        })

        logger.info(
            "Gamma %.2f trials -> var=%.6f, mse=%.6f+/-%.6f, overlap_mean=%.6f",
            gamma, avg_pointwise_var, mse_mean, mse_std, overlap_mean
        )

    logger.info("Completed. Results array (len=%d):", len(results))
    logger.info("%s", results)


if __name__ == "__main__":
    main()
