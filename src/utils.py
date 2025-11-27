import logging
from typing import Tuple

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from src.data.base_dataset import TwoSampleDataSplit, GroundTruth

from src.data.synthetic import NieWagerSyntheticDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simulate_dataset(cfg: DictConfig) -> Tuple[TwoSampleDataSplit, GroundTruth]:
    """Simulate a dataset using the configured synthetic setup."""
    if cfg.dataset.type != "synthetic_surrogate":
        raise NotImplementedError(
            f"Unsupported dataset type: {cfg.dataset.type}. Only synthetic_surrogate is available."
        )

    logger.info("Sampling dataset with configuration '%s'", cfg.dataset.name)
    dataset = NieWagerSyntheticDataset(cfg.dataset)
    return dataset.sample()


def evaluate_mse(
    learner, data: TwoSampleDataSplit, ground_truth: GroundTruth
) -> float:
    """Compute mean squared error between predicted and true CATE."""
    logger.info("Predicting CATE on experimental covariates")
    cate_pred = learner.predict_cate(data.X_e)
    cate_true = ground_truth.tau(data.X_e)
    mse = float(np.mean((cate_pred - cate_true) ** 2))
    logger.info("Computed CATE MSE: %.6f", mse)
    return mse