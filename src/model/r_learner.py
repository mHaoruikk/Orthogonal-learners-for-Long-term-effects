import numpy as np
from dataclasses import dataclass

from src.data.base_dataset import TwoSampleDataSplit
from src.model.nuisances import NuisanceFactory, CrossFittedNuisances
from src.model.utils import build_regressor

class Rlearner:
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        lambda_type = model_cfg.get('lambda_type', 'identity')
        self.cate_regressor_cfg = model_cfg.cate_regressor
        if lambda_type == "overlap":
            self.lambda_pi = lambda pi: pi * (1 - pi)
            self.rho_A_pi = lambda A, pi: (A - pi) ** 2
        elif lambda_type == "treat":
            self.lambda_pi = lambda pi: pi
            self.rho_A_pi = lambda A, pi: A
        elif lambda_type == "control":
            self.lambda_pi = lambda pi: 1 - pi
            self.rho_A_pi = lambda A, pi: 1 - A
        elif lambda_type == "identity":
            self.lambda_pi = lambda pi: np.ones_like(pi)
            self.rho_A_pi = lambda A, pi: np.ones_like(A)
        else:
            raise NotImplementedError(f"Unknown lambda_type: {lambda_type}")

        self.nuisance_factory = NuisanceFactory(self.model_cfg)
        

    def fit(self, data: TwoSampleDataSplit):
        cf_nuis = self.nuisance_factory.crossfit_nuisance(data)
        X_e, A_e, S_e = data.X_e, data.A_e, data.S_e
        X_o, S_o, Y_o = data.X_o, data.S_o, data.Y_o
        K = self.model_cfg.num_crossfit

        #For simplicity, we can use a single global cate_regressor trained on all E-sample,
        #but with pseudo-outcomes built using *out-of-fold* nuisances.

        pi_x_e = np.empty_like(A_e, dtype = float)
        pseudo_outcome_e = np.empty_like(S_e, dtype = float)
        pseudo_outcome_o = np.empty_like(Y_o, dtype = float)

        for k in range(K):
            nm_k = cf_nuis.folds[k]
            
            #Build pseudo-outcome on experimental dataset \phi_\lambda^0(Z_i)
            #\phi^0_\lambda(Z)=\mu(1,X)-\mu(0,X)+\frac{\lambda(\pi(X))}{\rho(A,\pi)}\cdot \Delta_h(S,X).
            idx_k = np.where(cf_nuis.fold_id_e == k)[0]
            X_e_k, A_e_k, S_e_k = X_e[idx_k], A_e[idx_k], S_e[idx_k]

            mu_0_pred = nm_k.mu_0.predict(X_e_k)
            mu_1_pred = nm_k.mu_1.predict(X_e_k)
            h_pred = nm_k.h.predict(np.column_stack([S_e_k, X_e_k]))
            pi_x_pred = nm_k.pi_x.predict(X_e_k)
            pi_x_e[idx_k] = pi_x_pred
            #truncate pi to avoid division by zero
            pi_x_pred = np.clip(pi_x_pred, 1e-3, 1 - 1e-3)
            Delta_k = A_e_k / pi_x_pred * (h_pred - mu_1_pred) - \
                        (1 - A_e_k) / (1 - pi_x_pred) * (h_pred - mu_0_pred)
            phi_k = mu_1_pred - mu_0_pred + self.lambda_pi(pi_x_pred) / self.rho_A_pi(A_e_k, pi_x_pred) * Delta_k
            pseudo_outcome_e[idx_k] = phi_k

            #Build pseudo-outcome \psi_lambda^1(Z_i)
            #\psi_\lambda^1(Z;\eta)=\lambda(\pi(X)) \kappa(S,X) (Y-h(S,X))
            # with \kappa(S,X) = \frac{1-\rho(S,X)}{\rho(S,X)} \frac{\pi(S,X)-\pi(X)}{\pi(X)(1-\pi(X))}
            idx_k_o = np.where(cf_nuis.fold_id_o == k)[0]
            X_o_k, S_o_k, Y_o_k = data.X_o[idx_k_o], data.S_o[idx_k_o], data.Y_o[idx_k_o]
            h_o_pred = nm_k.h.predict(np.column_stack([S_o_k, X_o_k]))
            pi_x_o_pred = nm_k.pi_x(X_o_k)
            pi_s_x_o_pred = nm_k.pi_s_x(np.column_stack([S_o_k, X_o_k]))
            pi_x_o_pred = np.clip(pi_x_o_pred, 1e-3, 1 - 1e-3)
            pi_s_x_o_pred = np.clip(pi_s_x_o_pred, 1e-3, 1 - 1e-3)
            rho_s_x_o_pred = nm_k.rho_s_x(np.column_stack([S_o_k, X_o_k]))
            rho_s_x_o_pred = np.clip(rho_s_x_o_pred, 1e-3, 1 - 1e-3)
            kappa_o = (1 - rho_s_x_o_pred) / rho_s_x_o_pred * (pi_s_x_o_pred - pi_x_o_pred) / (pi_x_o_pred * (1 - pi_x_o_pred))
            psi_o_k = self.lambda_pi(pi_x_o_pred) * kappa_o * (Y_o_k - h_o_pred)
            pseudo_outcome_o[idx_k_o] = psi_o_k

        # ---- cate_regressor: CATE regressor fitted on experimental data + pseudo-outcomes ----
        
        if self.cate_regressor_cfg.type == "solve_linear":
            solver = build_regressor(self.cate_regressor_cfg)
            solver.fit(X_e, X_o, 
                       rho_e = self.rho_A_pi(A_e, pi_x_e),
                       phi = pseudo_outcome_e,
                       psi = pseudo_outcome_o)
            self.cate_estimator = solver
            return self
    
    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        return self.cate_estimator.predict(X)