import numpy as np

from src.data.base_dataset import TwoSampleDataSplit
from src.model.nuisances import NuisanceFactory, CrossFittedNuisances
from src.model.utils import build_regressor


class LTO_Learner:
    """
    LT-O-learner: Long-Term Orthogonal learner for CATE estimation.

    Minimises the orthogonal loss
        L_omega(g, eta) = E[ omega*(Z;eta) g(X)^2 - 2 T_LT(Z;eta) g(X) ]
    where omega* is the LT weighting function and T_LT is the LT pseudo-outcome.

    Six weight_type choices for omega(X):
        "identity" : omega = 1
        "to"       : omega = pi(1-pi)           (treatment overlap)
        "to-sq"    : omega = pi^2 (1-pi)^2
        "lo"       : omega = rho                 (long-term overlap)
        "dual"     : omega = pi(1-pi) rho
        "dual-sq"  : omega = pi^2 (1-pi)^2 rho
    """

    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.cate_regressor_cfg = model_cfg.cate_regressor

        weight_type = model_cfg.get("weight_type", "to")
        self._setup_weight_functions(weight_type)

        self.nuisance_factory = NuisanceFactory(self.model_cfg)

    # ------------------------------------------------------------------
    # weight functions omega, partial derivatives, and Omega
    # ------------------------------------------------------------------
    def _setup_weight_functions(self, weight_type: str):
        if weight_type == "identity":
            self.omega     = lambda pi, rho: np.ones_like(pi)
            self.omega_pi  = lambda pi, rho: np.zeros_like(pi)
            self.omega_rho = lambda pi, rho: np.zeros_like(pi)
        elif weight_type == "to":
            self.omega     = lambda pi, rho: pi * (1 - pi)
            self.omega_pi  = lambda pi, rho: 1 - 2 * pi
            self.omega_rho = lambda pi, rho: np.zeros_like(pi)
        elif weight_type == "to-sq":
            self.omega     = lambda pi, rho: (pi * (1 - pi)) ** 2
            self.omega_pi  = lambda pi, rho: 2 * pi * (1 - pi) * (1 - 2 * pi)
            self.omega_rho = lambda pi, rho: np.zeros_like(pi)
        elif weight_type == "lo":
            self.omega     = lambda pi, rho: rho
            self.omega_pi  = lambda pi, rho: np.zeros_like(pi)
            self.omega_rho = lambda pi, rho: np.ones_like(pi)
        elif weight_type == "dual":
            self.omega     = lambda pi, rho: pi * (1 - pi) * rho
            self.omega_pi  = lambda pi, rho: (1 - 2 * pi) * rho
            self.omega_rho = lambda pi, rho: pi * (1 - pi)
        elif weight_type == "dual-sq":
            self.omega     = lambda pi, rho: (pi * (1 - pi)) ** 2 * rho
            self.omega_pi  = lambda pi, rho: 2 * pi * (1 - pi) * (1 - 2 * pi) * rho
            self.omega_rho = lambda pi, rho: (pi * (1 - pi)) ** 2
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")

    def _compute_Omega(self, A, pi, R, rho):
        """
        Omega(Z;eta) = 1(R=0) * d_omega/d_pi * (A - pi)
                     + (1 - rho) * d_omega/d_rho * (R - rho)
        """
        return (1 - R) * self.omega_pi(pi, rho) * (A - pi) \
             + (1 - rho) * self.omega_rho(pi, rho) * (R - rho)

    # ------------------------------------------------------------------
    # pseudo-outcome components
    # ------------------------------------------------------------------
    @staticmethod
    def _tau_aipw(A, pi, h, mu_0, mu_1):
        """
        tau_AIPW = mu(1,X) - mu(0,X)
                 + (A - pi) / (pi(1-pi)) * (h(S,X) - mu(A,X))
        """
        mu_a = np.where(A == 1, mu_1, mu_0)
        return mu_1 - mu_0 + (A - pi) / (pi * (1 - pi)) * (h - mu_a)

    @staticmethod
    def _psi_obs(pi_x, pi_sx, rho_sx, h, Y):
        """
        psi_obs = (1 - rho_s(S,X)) / rho_s(S,X)
                * (pi_s(S,X) - pi(X)) / (pi(X)(1 - pi(X)))
                * (Y - h(S,X))
        """
        kappa = (1 - rho_sx) / rho_sx * (pi_sx - pi_x) / (pi_x * (1 - pi_x))
        return kappa * (Y - h)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, data: TwoSampleDataSplit):
        cf_nuis = self.nuisance_factory.crossfit_nuisance(data)
        X_e, A_e, S_e = data.X_e, data.A_e, data.S_e
        X_o, S_o, Y_o = data.X_o, data.S_o, data.Y_o
        K = self.model_cfg.num_crossfit

        T_LT_e = np.empty(X_e.shape[0], dtype=float)
        T_LT_o = np.empty(X_o.shape[0], dtype=float)
        omega_star_e = np.empty(X_e.shape[0], dtype=float)
        omega_star_o = np.empty(X_o.shape[0], dtype=float)

        for k in range(K):
            nm = cf_nuis.folds[k]

            # ---- experimental fold (R = 0) ----
            idx_e = np.where(cf_nuis.fold_id_e == k)[0]
            Xk, Ak, Sk = X_e[idx_e], A_e[idx_e], S_e[idx_e]

            pi_x  = np.clip(nm.pi_x.predict(Xk), 1e-3, 1 - 1e-3)
            rho_x = np.clip(nm.rho_x.predict(Xk), 1e-3, 1 - 1e-3)
            mu0   = nm.mu_0.predict(Xk)
            mu1   = nm.mu_1.predict(Xk)
            h_hat = nm.h.predict(np.column_stack([Sk, Xk]))

            tau_hat = mu1 - mu0
            tau_aipw = self._tau_aipw(Ak, pi_x, h_hat, mu0, mu1)

            omega_k = self.omega(pi_x, rho_x)
            Omega_k = self._compute_Omega(A=Ak, pi=pi_x, R=0, rho=rho_x)
            omega_star_k = omega_k + Omega_k

            T_LT_e[idx_e] = omega_k * tau_aipw + tau_hat * Omega_k
            omega_star_e[idx_e] = omega_star_k

            # ---- observational fold (R = 1) ----
            idx_o = np.where(cf_nuis.fold_id_o == k)[0]
            Xok, Sok, Yok = X_o[idx_o], S_o[idx_o], Y_o[idx_o]

            pi_x_o  = np.clip(nm.pi_x.predict(Xok), 1e-3, 1 - 1e-3)
            rho_x_o = np.clip(nm.rho_x.predict(Xok), 1e-3, 1 - 1e-3)
            pi_sx_o = np.clip(
                nm.pi_s_x.predict(np.column_stack([Sok, Xok])), 1e-3, 1 - 1e-3
            )
            rho_sx_o = np.clip(
                nm.rho_s_x.predict(np.column_stack([Sok, Xok])), 1e-3, 1 - 1e-3
            )
            mu0_o = nm.mu_0.predict(Xok)
            mu1_o = nm.mu_1.predict(Xok)
            h_o   = nm.h.predict(np.column_stack([Sok, Xok]))

            tau_hat_o = mu1_o - mu0_o
            psi_obs = self._psi_obs(pi_x_o, pi_sx_o, rho_sx_o, h_o, Yok)
            omega_o = self.omega(pi_x_o, rho_x_o)
            Omega_o = self._compute_Omega(A=0, pi=pi_x_o, R=1, rho=rho_x_o)
            omega_star_ok = Omega_o  # 1(R=0)*omega is 0 for R=1

            T_LT_o[idx_o] = omega_o * psi_obs + tau_hat_o * Omega_o
            omega_star_o[idx_o] = omega_star_ok

        # ---- stage 2: minimise L_omega via weighted regression ----
        # PO = T_LT / omega*,  weight = omega*
        omega_star_e = np.clip(omega_star_e, 1e-3, np.inf)
        omega_star_o = np.clip(omega_star_o, 1e-3, np.inf)

        PO_e = T_LT_e / omega_star_e
        PO_o = T_LT_o / omega_star_o

        X_all  = np.vstack([X_e, X_o])
        PO_all = np.concatenate([PO_e, PO_o])
        w_all  = np.concatenate([omega_star_e, omega_star_o])

        shuffle = np.random.permutation(X_all.shape[0])
        solver = build_regressor(self.cate_regressor_cfg)
        solver.fit(X_all[shuffle], PO_all[shuffle], sample_weight=w_all[shuffle])
        self.cate_estimator = solver
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        return self.cate_estimator.predict(X)
