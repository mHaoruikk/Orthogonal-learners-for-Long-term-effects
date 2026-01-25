import logging
from typing import List, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.data.base_dataset import TwoSampleDataSplit, GroundTruth
from src.data.utils import split_two_sample_data
from src.model.dr_learner import DRLearner
from src.utils import simulate_dataset, evaluate_mse
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _clone_with_seed(cfg: DictConfig, seed: int) -> DictConfig:
    """Make a resolved copy of cfg with an updated dataset seed."""
    cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_copy.dataset.seed = seed
    return cfg_copy


def _train_single(cfg: DictConfig, eval_X: np.ndarray, seed: int) -> Tuple[np.ndarray, float, float]:
    """Train a single DR-learner, return predictions, MSE, and average overlap on the test set."""
    cfg_seeded = _clone_with_seed(cfg, seed)
    data, ground_truth = simulate_dataset(cfg_seeded)

    val_fraction = cfg_seeded.trainer.get("val_fraction", 0.2)
    test_fraction = cfg_seeded.trainer.get("test_fraction", 0.2)
    train_data, _, test_data = split_two_sample_data(
        data, val_fraction=val_fraction, test_fraction=test_fraction, random_state=seed
    )

    learner = DRLearner(cfg_seeded.model)
    learner.fit(train_data)

    # Estimate average overlap pi(X)(1-pi(X)) on the test experimental covariates
    if test_data.X_e.shape[0] > 0:
        cf_nuis = learner.nuisance_factory.crossfit_nuisance(train_data)
        pi_pred_folds = [nm.pi_x.predict(test_data.X_e) for nm in cf_nuis.folds]
        rho_pred_folds = [nm.rho_x.predict(test_data.X_e) for nm in cf_nuis.folds]
        pi_mean = np.mean(np.stack(pi_pred_folds, axis=0), axis=0)
        rho_mean = np.mean(np.stack(rho_pred_folds, axis=0), axis=0)
        overlap = pi_mean * (1.0 - pi_mean) * rho_mean
        avg_overlap = float(np.mean(overlap))
    else:
        avg_overlap = float("nan")

    preds = learner.predict_cate(eval_X)
    mse = evaluate_mse(learner, test_data, ground_truth)
    return preds, mse, avg_overlap


@hydra.main(version_base="1.1", config_name="config.yaml", config_path="../config/")
def main(args: DictConfig):
    logger.info("Loaded configuration:\n%s", OmegaConf.to_yaml(args))

    num_trials = args.get("num_trials", 10)
    base_seed = int(args.dataset.seed)

    logger.info("Preparing fixed evaluation set with seed=%d", base_seed)
    eval_cfg = _clone_with_seed(args, base_seed)
    eval_data, eval_ground_truth = simulate_dataset(eval_cfg)
    X_eval = eval_data.X_e
    true_cate = eval_ground_truth.tau(X_eval)

    preds_across_trials: List[np.ndarray] = []
    mses: List[float] = []
    overlaps: List[float] = []

    for trial in range(num_trials):
        trial_seed = base_seed + trial
        #logger.info("Starting trial %d/%d (seed=%d)", trial + 1, num_trials, trial_seed)
        preds, mse, overlap = _train_single(args, X_eval, seed=trial_seed)
        preds_across_trials.append(preds)
        mses.append(mse)
        overlaps.append(overlap)
        #logger.info("Trial %d MSE: %.6f", trial + 1, mse)

    pred_matrix = np.stack(preds_across_trials, axis=0)
    if pred_matrix.shape[0] > 1:
        pointwise_var = np.var(pred_matrix, axis=0, ddof=1)
    else:
        pointwise_var = np.zeros_like(pred_matrix[0])

    avg_pointwise_var = float(np.mean(pointwise_var))
    mse_mean = float(np.mean(mses))
    mse_std = float(np.std(mses, ddof=1)) if len(mses) > 1 else 0.0
    overlap_mean = float(np.nanmean(overlaps)) if len(overlaps) > 0 else float("nan")
    overlap_std = float(np.nanstd(overlaps, ddof=1)) if len(overlaps) > 1 else 0.0

    logger.info("Point-wise CATE variance: mean=%.6f, min=%.6f, max=%.6f",
                avg_pointwise_var, float(np.min(pointwise_var)), float(np.max(pointwise_var)))
    logger.info("MSE across trials: mean=%.6f, std=%.6f", mse_mean, mse_std)
    logger.info("Overlap pi(x)(1-pi(x)) on test set: mean=%.6f, std=%.6f", overlap_mean, overlap_std)
    logger.info("Average ground-truth CATE on eval set: %.6f", float(np.mean(true_cate)))


if __name__ == "__main__":
    main()

    # Example usage:
    # python -m scripts.train_t_learner_variance dataset=setupA model=T-learner trainer=default num_trials=10
