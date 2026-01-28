import logging
from typing import Tuple

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from scripts.train_r_learner import train_t_r_learner
from src.data.base_dataset import TwoSampleDataSplit, GroundTruth
from src.data.synthetic import NieWagerSyntheticDataset
from src.data.utils import split_two_sample_data
from src.model.dr_learner import DRLearner
from src.utils import simulate_dataset, evaluate_mse
from src.visualization import plot_cate_predictions

#suppress warnings
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_dr_learner(cfg: DictConfig, data: TwoSampleDataSplit) -> DRLearner:
    """Train the DR-learner along with its nuisance models."""
    logger.info("Initializing DR-learner model and building nuisance models")
    learner = DRLearner(cfg.model)
    learner.fit(data)
    return learner


@hydra.main(version_base="1.1", config_name="config.yaml", config_path="../config/")
def main(args: DictConfig):
    #logger.info("Loaded configuration:\n%s", OmegaConf.to_yaml(args))

    logger.info("Starting dataset simulation")
    data, ground_truth = simulate_dataset(args)
    val_fraction = args.trainer.get('val_fraction', 0.2)
    test_fraction = args.trainer.get('test_fraction', 0.2)
    train_data, val_data, test_data = split_two_sample_data(data, val_fraction, test_fraction)

    logger.info("Training DR-learner and associated nuisances")
    learner = train_dr_learner(args, train_data)

    X_test = test_data.X_e
    true_cate = ground_truth.tau(X_test)
    plot_cate_predictions(X_test, true_cate = true_cate, pred_cate=learner.predict_cate(X_test))

    logger.info("Evaluating model performance using mean squared error")
    mse = evaluate_mse(learner, test_data, ground_truth)

    metrics = {"cate_mse": mse}
    logger.info("Final metrics: %s", metrics) #0.00224 DR-learner #0.001 R-learner


if __name__ == "__main__":
    main()


    # python -m scripts.train_r_learner dataset=setupA model=R-learner trainer=default