import numpy as np
from dataclasses import dataclass

from src.data.base_dataset import TwoSampleDataSplit
from src.model.nuisances import NuisanceFactory, CrossFittedNuisances
from src.model.utils import build_regressor

class DualOverlapWeightedOrthogonalLearner: #t for treatment
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        lambda_type = model_cfg.get('lambda_type', 'identity')
        self.cate_regressor_cfg = model_cfg.cate_regressor

        self.omega = None #weight function \omega(pi,rho) = lambda(pi) * r(rho)
        self.omega_pi = None #\frac{\partial \omega}{\partial \pi}(pi, rho)
        self.omega_rho = None # \frac{\partial \omega}{\partial \rho}(pi, rho)

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
        
        r_type = model_cfg.get('r_type', 'identity')
        if r_type == "proportional":
            self.r_rho = lambda rho: rho
        elif r_type == "identity":
            self.r_rho = lambda rho: np.ones_like(rho)
        else:
            raise NotImplementedError(f"Unknown r_type: {r_type}")
        
        self.omega = lambda pi, rho: self.lambda_pi(pi) * self.r_rho(rho)
        if lambda_type == "overlap":
            self.omega_pi = lambda pi, rho: (1 - 2 * pi) * self.r_rho(rho)
        elif lambda_type == "identity":
            self.omega_pi = lambda pi, rho: np.zeros_like(pi)
        else:
            raise NotImplementedError(f"omega_pi not implemented for lambda_type: {lambda_type}")
        
        if r_type == "proportional":
            self.omega_rho = lambda pi, rho: self.lambda_pi(pi)
        elif r_type == "identity":
            self.omega_rho = lambda pi, rho: np.zeros_like(rho)
        else:
            raise NotImplementedError(f"omega_rho not implemented for r_type: {r_type}")
        
        # \Omega(Z;\eta) &= \1(R=0) \frac{\partial \omega}{\partial \pi} (A - \pi(X)) + (1-\rho(X)) \frac{\partial \omega}{\partial \rho} (R - \rho(X))
        self.Omega = lambda A, pi, R, rho: \
            (1 - R) * self.omega_pi(pi, rho) * (A - pi) + \
            (1 - rho) * self.omega_rho(pi, rho) * (R - rho)

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
        weights_e = np.empty_like(S_e, dtype = float)
        weights_o = np.empty_like(Y_o, dtype = float)

        for k in range(K):
            nm_k = cf_nuis.folds[k]
            
            #Build pseudo-outcome on experimental dataset 
            # \tilde{Y} = \frac{1}{\omega^*(Z;\eta)} \Bigg[ \1(R=0) \omega(X) \hat{\tau}_{DR}(Z;\eta) \\
            # + (\mu(1,X)-\mu(0,X)) \Omega(Z;\eta) 
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
            
            tau_dr_k = mu_1_pred - mu_0_pred + Delta_k
            omega_k = self.omega(pi_x_pred, nm_k.rho_x.predict(X_e_k))
            Omega_k = self.Omega(A_e_k, pi_x_pred, R = 0, rho = nm_k.rho_x.predict(X_e_k))
            omega_star = np.clip(omega_k + Omega_k, 1e-3, np.inf)

            PO_k = omega_k * tau_dr_k + (mu_1_pred - mu_0_pred) * Omega_k
            
            pseudo_outcome_e[idx_k] = PO_k / omega_star
            weights_e[idx_k] = 1 / omega_star

            #Build pseudo-outcome \psi_lambda^1(Z_i)
            #\tilde{Y} = \frac{1}{\omega^*(Z;\eta)} \Bigg[ \1(R=0) \omega(X) \hat{\tau}_{DR}(Z;\eta) \\
            # + (\mu(1,X)-\mu(0,X)) \Omega(Z;\eta) 
            idx_k_o = np.where(cf_nuis.fold_id_o == k)[0]
            X_o_k, S_o_k, Y_o_k = data.X_o[idx_k_o], data.S_o[idx_k_o], data.Y_o[idx_k_o]
            mu_1_pred_o = nm_k.mu_1.predict(X_o_k)
            mu_0_pred_o = nm_k.mu_0.predict(X_o_k)
            h_o_pred = nm_k.h.predict(np.column_stack([S_o_k, X_o_k]))
            pi_x_o_pred = nm_k.pi_x(X_o_k)
            pi_s_x_o_pred = nm_k.pi_s_x(np.column_stack([S_o_k, X_o_k]))
            pi_x_o_pred = np.clip(pi_x_o_pred, 1e-3, 1 - 1e-3)
            pi_s_x_o_pred = np.clip(pi_s_x_o_pred, 1e-3, 1 - 1e-3)
            rho_s_x_o_pred = nm_k.rho_s_x(np.column_stack([S_o_k, X_o_k]))
            rho_s_x_o_pred = np.clip(rho_s_x_o_pred, 1e-3, 1 - 1e-3)
            kappa_o = (1 - rho_s_x_o_pred) / rho_s_x_o_pred * (pi_s_x_o_pred - pi_x_o_pred) / (pi_x_o_pred * (1 - pi_x_o_pred))
            psi_o_k = kappa_o * (Y_o_k - h_o_pred)

            omega_o_k = self.omega(pi_x_o_pred, rho_s_x_o_pred)
            Omega_o_k = self.Omega(A = 0, pi = pi_x_o_pred, R = 1, rho = rho_s_x_o_pred)
            omega_star_o = np.clip(omega_o_k + Omega_o_k, 1e-3, np.inf)

            PO_o_k = omega_o_k * psi_o_k + (mu_1_pred_o - mu_0_pred_o) * Omega_o_k
            pseudo_outcome_o[idx_k_o] = PO_o_k / omega_star_o
            weights_o[idx_k_o] = 1 / omega_star_o

        # ---- cate_regressor: CATE regressor fitted on experimental data + pseudo-outcomes ----
        
        if self.cate_regressor_cfg.type == "xgboost":
            solver = build_regressor(self.cate_regressor_cfg)
            #concat experimental and observational data and shuffle
            X = np.vstack([X_e, X_o])
            PO = np.concatenate([pseudo_outcome_e, pseudo_outcome_o])
            weights = np.concatenate([weights_e, weights_o])
            shuffle_idx = np.random.permutation(X.shape[0])
            solver.fit(X[shuffle_idx], PO[shuffle_idx], sample_weight=weights[shuffle_idx])
            self.cate_estimator = solver
            return self
    
    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        return self.cate_estimator.predict(X)