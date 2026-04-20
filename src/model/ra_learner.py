import numpy as np

from src.data.base_dataset import TwoSampleDataSplit
from src.model.nuisances import NuisanceFactory
from src.model.utils import build_regressor


class RA_Learner:
    """
    Long-term RA-learner (Regression Adjustment).

    Stage 1: cross-fit nuisances (h, mu_0, mu_1).
    Stage 2: regress the RA pseudo-outcome on X using experimental data only.

    Pseudo-outcome:
        T_RA = A * (h(S,X) - mu_0(X)) + (1-A) * (mu_1(X) - h(S,X))

    Loss:
        L_RA = E[ 1(R=0) * (T_RA - g(X))^2 ]
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

            mu0 = nm.mu_0.predict(Xk)
            mu1 = nm.mu_1.predict(Xk)
            h_hat = nm.h.predict(np.column_stack([Sk, Xk]))

            # T_RA = A*(h - mu_0) + (1-A)*(mu_1 - h)
            PO_e[idx] = Ak * (h_hat - mu0) + (1 - Ak) * (mu1 - h_hat)

        solver = build_regressor(self.cate_regressor_cfg)
        shuffle = np.random.permutation(X_e.shape[0])
        solver.fit(X_e[shuffle], PO_e[shuffle])
        self.cate_estimator = solver
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        return self.cate_estimator.predict(X)
