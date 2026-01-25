import numpy as np

from src.data.base_dataset import TwoSampleDataSplit
from src.model.nuisances import NuisanceFactory
from src.model.utils import build_regressor


class DRLearner:
    """
    Doubly-robust learner for long-term CATE.

    Stage 1: cross-fitted nuisances (pi_x, pi_s_x, rho_s_x, h, mu_0, mu_1).
    Stage 2: regress DR pseudo-outcomes on X to estimate tau(X).
    """

    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.nuisance_factory = NuisanceFactory(self.model_cfg)
        self.cate_regressor_cfg = self.model_cfg.cate_regressor
        self.cate_regressor = build_regressor(self.cate_regressor_cfg)

    def fit(self, data: TwoSampleDataSplit):
        cf_nuis = self.nuisance_factory.crossfit_nuisance(data)
        X_e, A_e, S_e = data.X_e, data.A_e, data.S_e
        X_o, S_o, Y_o = data.X_o, data.S_o, data.Y_o
        K = self.model_cfg.num_crossfit

        phi_e = np.empty_like(S_e, dtype=float)
        phi_o = np.empty_like(Y_o, dtype=float)

        for k in range(K):
            nm_k = cf_nuis.folds[k]

            # Experimental fold: tau_AIPW(Z;eta)
            idx_e = np.where(cf_nuis.fold_id_e == k)[0]
            if idx_e.size > 0:
                X_e_k, A_e_k, S_e_k = X_e[idx_e], A_e[idx_e], S_e[idx_e]
                pi_x = np.clip(nm_k.pi_x.predict(X_e_k), 1e-3, 1 - 1e-3)
                mu0 = nm_k.mu_0.predict(X_e_k)
                mu1 = nm_k.mu_1.predict(X_e_k)
                h_hat = nm_k.h.predict(np.column_stack([S_e_k, X_e_k]))
                mu_a = np.where(A_e_k == 1, mu1, mu0)
                tau_aipw = mu1 - mu0 + (A_e_k - pi_x) / (pi_x * (1 - pi_x)) * (h_hat - mu_a)
                phi_e[idx_e] = tau_aipw

            # Observational fold: psi_obs(Z;eta)
            idx_o = np.where(cf_nuis.fold_id_o == k)[0]
            if idx_o.size > 0:
                X_o_k, S_o_k, Y_o_k = X_o[idx_o], S_o[idx_o], Y_o[idx_o]
                pi_x_o = np.clip(nm_k.pi_x.predict(X_o_k), 1e-3, 1 - 1e-3)
                pi_sx_o = np.clip(
                    nm_k.pi_s_x.predict(np.column_stack([S_o_k, X_o_k])),
                    1e-3,
                    1 - 1e-3,
                )
                rho_sx_o = np.clip(
                    nm_k.rho_s_x.predict(np.column_stack([S_o_k, X_o_k])),
                    1e-3,
                    1 - 1e-3,
                )
                h_o = nm_k.h.predict(np.column_stack([S_o_k, X_o_k]))
                kappa = (1 - rho_sx_o) / rho_sx_o * (pi_sx_o - pi_x_o) / (pi_x_o * (1 - pi_x_o))
                psi_obs = kappa * (Y_o_k - h_o)
                phi_o[idx_o] = psi_obs

        # Stage 2: regress pseudo-outcomes on X to learn tau(X)
        if self.cate_regressor_cfg.type.lower() == "solve_linear":
            # Use closed-form solution treating DR pseudo-outcomes as targets.
            rho_e = np.ones_like(phi_e)
            self.cate_regressor.fit(
                X_e=X_e,
                X_o=X_o,
                rho_e=rho_e,
                phi=phi_e,
                psi=phi_o,
            )
        else:
            X_all = np.vstack([X_e, X_o])
            phi_all = np.concatenate([phi_e, phi_o])
            self.cate_regressor.fit(X_all, phi_all)
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        return self.cate_regressor.predict(X)
