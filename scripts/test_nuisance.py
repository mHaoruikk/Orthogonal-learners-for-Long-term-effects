import logging

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.data.synthetic import NieWagerSyntheticDataset
from src.model.t_r_learner import tRlearner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


@hydra.main(version_base="1.1", config_name="config.yaml", config_path="../config/")
def main(cfg: DictConfig):
    logger.info("Loaded configuration:\n%s", OmegaConf.to_yaml(cfg))

    # 1) Instantiate dataset and R-learner
    dataset = NieWagerSyntheticDataset(cfg.dataset)
    data, _ = dataset.sample()
    learner = tRlearner(cfg.model)

    # 2) Fit nuisance models (from the R-learner) and evaluate MSE against ground truth
    cf_nuis = learner.nuisance_factory.crossfit_nuisance(data)
    if len(cf_nuis.folds) == 0:
        raise RuntimeError("No nuisance models were produced by cross-fitting.")

    nm = cf_nuis.folds[0]  # use nuisances from the first fold
    pi_true = dataset.pi_E(data.X_e)
    pi_pred = nm.pi_x.predict(data.X_e)
    pi_mse = _mse(pi_true, pi_pred)

    SX_o = np.column_stack([data.S_o, data.X_o])
    h_true = dataset.true_h(data.S_o, data.X_o)
    h_pred = nm.h.predict(SX_o)
    h_mse = _mse(h_true, h_pred)

    logger.info("First-fold nuisance MSEs -> pi_x (E): %.6f | h (O): %.6f", pi_mse, h_mse)
    print(f"pi_x MSE on experimental set: {pi_mse:.6f}")
    print(f"h MSE on observational set: {h_mse:.6f}")
    print(f"total Num samples: {data.X_e.shape[0] + data.X_o.shape[0]}")


if __name__ == "__main__":
    main()

    #python -m scripts.test_nuisance dataset=setupA model=R-learner trainer=default
