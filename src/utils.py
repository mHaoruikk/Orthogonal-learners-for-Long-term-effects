import logging
from typing import Tuple

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from src.data.base_dataset import TwoSampleDataSplit, GroundTruth

from src.data.synthetic import NieWagerSyntheticDataset
from src.data.semi_synthetic import IST3SemiSyntheticDataset
from src.data.real_world import RealWorldIST

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


_DATASET_REGISTRY = {
    "synthetic_surrogate": NieWagerSyntheticDataset,
    "semi_synthetic_ist": IST3SemiSyntheticDataset,
    "real_world": RealWorldIST,
}


def simulate_dataset(cfg: DictConfig) -> Tuple[TwoSampleDataSplit, GroundTruth]:
    """Build a dataset from cfg.dataset.type and return (data, ground_truth)."""
    cls = _DATASET_REGISTRY.get(cfg.dataset.type)
    if cls is None:
        raise NotImplementedError(
            f"Unsupported dataset type: {cfg.dataset.type}. "
            f"Known: {sorted(_DATASET_REGISTRY)}"
        )
    logger.info("Sampling dataset with configuration '%s'", cfg.dataset.name)
    dataset = cls(cfg.dataset)
    return dataset.sample()


def evaluate_mse(
    learner, data: TwoSampleDataSplit, ground_truth: GroundTruth
) -> float:
    """Compute mean squared error between predicted and true CATE."""
    #logger.info("Predicting CATE on experimental covariates")
    cate_pred = learner.predict_cate(data.X_e)
    cate_true = ground_truth.tau(data.X_e)
    mse = float(np.mean((cate_pred - cate_true) ** 2))
    logger.info("Computed CATE MSE: %.6f", mse)
    return mse