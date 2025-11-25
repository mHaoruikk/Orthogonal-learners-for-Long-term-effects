import logging
from typing import Tuple

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.data.base_dataset import TwoSampleDataSplit, GroundTruth
from src.data.synthetic import NieWagerSyntheticDataset
from src.model.t_learner import TLearner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(version_base="1.1", config_name="config.yaml", config_path="../config/")
def main(args: DictConfig):
    logger.info("Loaded configuration:\n%s", OmegaConf.to_yaml(args))


def simulate_dataset(cfg: DictConfig) -> Tuple[TwoSampleDataSplit, GroundTruth]:
    """Simulate a dataset using the configured synthetic setup."""
    if cfg.dataset.type != "synthetic_surrogate":
        raise NotImplementedError(
            f"Unsupported dataset type: {cfg.dataset.type}. Only synthetic_surrogate is available."
        )

    logger.info("Sampling dataset with configuration '%s'", cfg.dataset.name)
    dataset = NieWagerSyntheticDataset(cfg.dataset)
    return dataset.sample()


def train_t_learner(cfg: DictConfig, data: TwoSampleDataSplit) -> TLearner:
    """Train the T-learner along with its nuisance models."""
    logger.info("Initializing T-learner model and building nuisance models")
    learner = TLearner(cfg.model)
    learner.fit(data)
    return learner


def evaluate_mse(
    learner: TLearner, data: TwoSampleDataSplit, ground_truth: GroundTruth
) -> float:
    """Compute mean squared error between predicted and true CATE."""
    logger.info("Predicting CATE on experimental covariates")
    cate_pred = learner.predict_cate(data.X_e)
    cate_true = ground_truth.tau(data.X_e)
    mse = float(np.mean((cate_pred - cate_true) ** 2))
    logger.info("Computed CATE MSE: %.6f", mse)
    return mse


@hydra.main(version_base="1.1", config_name="config.yaml", config_path="../config/")
def main(args: DictConfig):
    logger.info("Loaded configuration:\n%s", OmegaConf.to_yaml(args))

    logger.info("Starting dataset simulation")
    data, ground_truth = simulate_dataset(args)

    logger.info("Training T-learner and associated nuisances")
    learner = train_t_learner(args, data)

    logger.info("Evaluating model performance using mean squared error")
    mse = evaluate_mse(learner, data, ground_truth)

    metrics = {"cate_mse": mse}
    logger.info("Final metrics: %s", metrics)


if __name__ == "__main__":
    main()