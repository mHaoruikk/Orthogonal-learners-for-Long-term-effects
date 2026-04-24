"""Shared utilities for evaluation scripts (eval_syn, eval_gain_pehe, evaluate_ist_real).

Provides:
  * LEARNER_REGISTRY          — maps a learner name to its class.
  * DEFAULT_MODELS            — canonical default list of learners to evaluate.
  * DEFAULT_MODELS_NN         — same learner names, intended to be used with
                                the NN cate_regressor swap below.
  * NN_CATE_REGRESSOR_CFG     — torch_mlp spec used when swapping in a NN
                                backbone for the CATE regressor.
  * swap_cate_regressor_to_nn — deep-copy a model cfg and replace its
                                cate_regressor with NN_CATE_REGRESSOR_CFG.
  * seed_model_cfg            — deep-copy a model cfg and inject a seed into
                                every random_state / seed field.
"""
from __future__ import annotations

from copy import deepcopy

from omegaconf import DictConfig, OmegaConf

from src.model.ipw_learner import IPW_Learner
from src.model.lto_learner import LTO_Learner
from src.model.ra_learner import RA_Learner
from src.model.t_learner import TLearner


LEARNER_REGISTRY = {
    "T-learner":    TLearner,
    "DR-learner":   LTO_Learner,   # cfg sets weight_type=identity
    "TO-learner":   LTO_Learner,   # cfg sets weight_type=to
    "TO-learner-2": LTO_Learner,   # cfg sets weight_type=to-sq
    "LO-learner":   LTO_Learner,   # cfg sets weight_type=lo
    "DO-learner":   LTO_Learner,   # cfg sets weight_type=dual
    "DO-learner-2": LTO_Learner,   # cfg sets weight_type=dual-sq
    "IPW-learner":  IPW_Learner,
    "RA-learner":   RA_Learner,
}

DEFAULT_MODELS = [
    "T-learner", "DR-learner", "TO-learner", "TO-learner-2", "LO-learner",
    "DO-learner", "DO-learner-2", "IPW-learner", "RA-learner",
]

# Same roster of learner names; used with swap_cate_regressor_to_nn so every
# learner's CATE regressor backbone becomes a torch MLP. Useful for stressing
# learners that rely on a flexible stage-2 regressor.
DEFAULT_MODELS_NN = list(DEFAULT_MODELS)

# torch_mlp spec mirroring the cate_regressor block in config/model/*-NN.yaml.
NN_CATE_REGRESSOR_CFG = {
    "type": "torch_mlp",
    "parameters": {
        "hidden_layers": [20, 20, 10, 10],
        "output_activation": "identity",
        "epochs": 20,
        "batch_size": 64,
        "lr": 1e-3,
        "device": "cpu",
    },
}


def swap_cate_regressor_to_nn(cfg: DictConfig) -> DictConfig:
    """Deep-copy cfg and replace cate_regressor with NN_CATE_REGRESSOR_CFG."""
    new_cfg = deepcopy(cfg)
    OmegaConf.set_struct(new_cfg, False)
    new_cfg.cate_regressor = OmegaConf.create(NN_CATE_REGRESSOR_CFG)
    return new_cfg


def seed_model_cfg(cfg: DictConfig, seed: int) -> DictConfig:
    """Deep-copy cfg and inject `seed` into every random_state / seed field.

    - Top-level: `cfg.random_state = seed` (read by NuisanceFactory for its KFold).
    - Every sub-node with a `type` field: `parameters.random_state = seed`,
      plus `parameters.seed = seed` for xgboost.
    """
    new_cfg = deepcopy(cfg)
    OmegaConf.set_struct(new_cfg, False)
    new_cfg.random_state = int(seed)

    # sklearn base types that accept a `random_state` kwarg.
    _RS_TYPES = {"random_forest", "logistic", "lasso", "ridge", "elasticnet"}

    def _walk(node):
        if not isinstance(node, DictConfig):
            return
        if "type" in node:
            t = str(node.type).lower()
            if t in _RS_TYPES or t == "xgboost":
                if node.get("parameters") is None:
                    node.parameters = OmegaConf.create({})
                node.parameters.random_state = int(seed)
                if t == "xgboost":
                    node.parameters.seed = int(seed)
            # linear, solve_linear, torch_mlp: no random_state kwarg; skip.
        for k in list(node.keys()):
            _walk(node[k])

    _walk(new_cfg)
    return new_cfg
