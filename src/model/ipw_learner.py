import numpy as np

from src.data.base_dataset import TwoSampleDataSplit
from src.model.nuisances import NuisanceFactory
from src.model.utils import build_regressor


class IPW_Learner:
    """
    Long-term IPW-learner (Inverse Propensity Weighting).

    Stage 1: cross-fit nuisances (h, pi_x).
    Stage 2: regress the IPW pseudo-outcome on X using experimental data only.

    Pseudo-outcome:
        T_IPW = (A / pi(X) - (1-A) / (1-pi(X))) * h(S,X)

    Loss:
        L_IPW = E[ 1(R=0) * (T_IPW - g(X))^2 ]
    """

    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.cate_regressor_cfg = model_cfg.cate_regressor
        self.nuisance_factory = NuisanceFactory(self.model_cfg)

    def fit(self, data: TwoSampleDataSplit):
        cf_nuis = self.nuisance_factory.crossfit_nuisance(data)
        X_e, A_e, S_e = data.X_e, data.A_e, data.S_e
        K = self.model_cfg.num_crossfit

        PO_e = np.empty(X_e.shape[0], dtype=float)

        for k in range(K):
            nm = cf_nuis.folds[k]
            idx = np.where(cf_nuis.fold_id_e == k)[0]
            Xk, Ak, Sk = X_e[idx], A_e[idx], S_e[idx]

            pi_x = np.clip(nm.pi_x.predict(Xk), 1e-3, 1 - 1e-3)
            h_hat = nm.h.predict(np.column_stack([Sk, Xk]))

            # T_IPW = (A/pi - (1-A)/(1-pi)) * h(S,X)
            PO_e[idx] = (Ak / pi_x - (1 - Ak) / (1 - pi_x)) * h_hat

        solver = build_regressor(self.cate_regressor_cfg)
        shuffle = np.random.permutation(X_e.shape[0])
        solver.fit(X_e[shuffle], PO_e[shuffle])
        self.cate_estimator = solver
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        return self.cate_estimator.predict(X)
