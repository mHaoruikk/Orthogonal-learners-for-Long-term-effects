import logging

import hydra
import numpy as np
from omegaconf import DictConfig

from src.data.base_dataset import TwoSampleDataSplit, GroundTruth
from src.data.utils import split_two_sample_data
from src.model.lto_learner import LTO_Learner
from src.utils import simulate_dataset, evaluate_mse
from src.visualization import plot_cate_predictions

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_lto_learner(cfg: DictConfig, data: TwoSampleDataSplit) -> LTO_Learner:
    """Train the LTO_Learner along with its nuisance models."""
    logger.info("Initializing LTO_Learner (weight_type=%s)", cfg.model.get("weight_type", "to"))
    learner = LTO_Learner(cfg.model)
    learner.fit(data)
    return learner


@hydra.main(version_base="1.1", config_name="config.yaml", config_path="../config/")
def main(args: DictConfig):
    logger.info("Starting dataset simulation")
    data, ground_truth = simulate_dataset(args)
    val_fraction = args.trainer.get("val_fraction", 0.2)
    test_fraction = args.trainer.get("test_fraction", 0.2)
    train_data, val_data, test_data = split_two_sample_data(data, val_fraction, test_fraction)

    logger.info("Training LTO-learner and associated nuisances")
    learner = train_lto_learner(args, train_data)

    X_test = test_data.X_e
    true_cate = ground_truth.tau(X_test)
    plot_cate_predictions(X_test, true_cate=true_cate, pred_cate=learner.predict_cate(X_test))
    mse = evaluate_mse(learner, test_data, ground_truth)

    metrics = {"cate_mse": mse}
    logger.info("Final metrics: %s", metrics)


if __name__ == "__main__":
    main()

    # commands to run
    # python -m scripts.train_lto dataset=setupA model=LTO-learner trainer=default
