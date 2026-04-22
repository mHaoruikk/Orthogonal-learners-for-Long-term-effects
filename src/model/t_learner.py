import numpy as np
from dataclasses import dataclass

from src.data.base_dataset import TwoSampleDataSplit
from src.model.nuisances import NuisanceFactory, CrossFittedNuisances
from src.model.utils import build_regressor


class TLearner:

    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.nuisance_factory = NuisanceFactory(self.model_cfg)
        self.y0_estimator = None
        self.y1_estimator = None

    def fit(self, data: TwoSampleDataSplit):
        cf_nuis = self.nuisance_factory.crossfit_nuisance(data)
        X_e, A_e, S_e = data.X_e, data.A_e, data.S_e
        K = self.model_cfg.num_crossfit

        #For simplicity, we can use a single global cate_regressor trained on all E-sample,
        #but with pseudo-outcomes built using *out-of-fold* nuisances.

        #surrogate T-learner based on h(S,X)
        g_cfg = self.model_cfg.cate_regressor
        reg0 = build_regressor(g_cfg)
        reg1 = build_regressor(g_cfg)

        # Build pseudo-outcomes using cross-fitted h
        Y_tilde = np.empty(X_e.shape[0], dtype=float)
        for k in range(K):
            nm_k = cf_nuis.folds[k]
            idx_k = np.where(cf_nuis.fold_id_e == k)[0]
            SX_e_k = np.column_stack([S_e[idx_k], X_e[idx_k]])
            Y_tilde[idx_k] = nm_k.h.predict(SX_e_k)

        # Now T-learner on (X_e, A_e, Y_tilde)
        reg0.fit(X_e[A_e == 0], Y_tilde[A_e == 0])
        reg1.fit(X_e[A_e == 1], Y_tilde[A_e == 1])

        self.y0_estimator = reg0
        self.y1_estimator = reg1
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        m1 = self.y1_estimator.predict(X)
        m0 = self.y0_estimator.predict(X)
        return m1 - m0