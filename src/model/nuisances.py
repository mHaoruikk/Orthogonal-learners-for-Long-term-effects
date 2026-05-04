from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.model_selection import KFold

from src.data.base_dataset import TwoSampleDataSplit
from src.model.base_model import BaseEstimator, SklearnClassifier, XGBoostClassifier
from src.model.utils import build_classifier, build_regressor

@dataclass
class NuisanceModels:
    pi_x: Optional[BaseEstimator] = None
    pi_s_x: Optional[BaseEstimator] = None
    rho_x: Optional[BaseEstimator] = None
    rho_s_x: Optional[BaseEstimator] = None
    h: Optional[BaseEstimator] = None
    mu_0: Optional[BaseEstimator] = None   
    mu_1: Optional[BaseEstimator] = None

@dataclass
class CrossFittedNuisances:
    """
    Result of cross-fitting K sets of nuisances.

    folds[k] contains NuisanceModels trained on fold k's *training* data.
    fold_id_e[i] gives which fold observation i in the experimental sample belongs to.
    fold_id_o[j] gives which fold observation j in the observational sample belongs to.
    """
    folds: list[NuisanceModels]
    fold_id_e: np.ndarray  # shape (n_e,), values in {0,...,K-1}
    fold_id_o: np.ndarray  # shape (n_o,), values in {0,...,K-1}


class NuisanceFactory:
    """
    Factory that, given Model Config and data, fits nuisances via K-fold cross-fitting.

    - For each fold k, we fit nuisances on the training portion of that fold
      (separately for experimental and observational parts where needed).
    - We return K sets of NuisanceModels plus fold assignments.
    """

    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.K = model_cfg.num_crossfit

    def _make_folds(self, n_e: int, n_o: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Create K-fold assignments for E and O separately.
        Seed read from `model_cfg.random_state` (default 42) so the caller can
        thread a fresh seed per replicate.
        Returns:
            fold_id_e: (n_e,) in {0,...,K-1}
            fold_id_o: (n_o,) in {0,...,K-1}
        """
        rs = int(self.model_cfg.get("random_state", 42))
        kf_e = KFold(n_splits=self.K, shuffle=True, random_state=rs)
        kf_o = KFold(n_splits=self.K, shuffle=True, random_state=rs + 1)

        fold_id_e = np.empty(n_e, dtype=int)
        fold_id_o = np.empty(n_o, dtype=int)

        for k, (_, val_idx) in enumerate(kf_e.split(np.arange(n_e))):
            fold_id_e[val_idx] = k
        for k, (_, val_idx) in enumerate(kf_o.split(np.arange(n_o))):
            fold_id_o[val_idx] = k

        return fold_id_e, fold_id_o

    def crossfit_nuisance(self, data: TwoSampleDataSplit) -> CrossFittedNuisances:
        X_e, A_e, S_e = data.X_e, data.A_e, data.S_e
        X_o, S_o, Y_o = data.X_o, data.S_o, data.Y_o


        # ---- h: E[Y | S,X,R=1] using full observational train data ----
        if "h" in self.model_cfg.nuisances:
            if not self.model_cfg.nuisances["h"].get("crossfit", True):
                #fit h on full O-data
                SX_o = np.column_stack([S_o, X_o])
                h_est = build_regressor(self.model_cfg.nuisances["h"])
                h_est.fit(SX_o, Y_o)

        n_e, n_o = X_e.shape[0], X_o.shape[0]
        fold_id_e, fold_id_o = self._make_folds(n_e, n_o)

        folds: list[NuisanceModels] = []

        for k in range(self.K):
            # Training indices for this fold
            idx_e = np.where(fold_id_e != k)[0]
            idx_o = np.where(fold_id_o != k)[0]

            # Build nuisances according to config
            nuis_cfg = self.model_cfg.nuisances

            nm = NuisanceModels()
            if "h" in nuis_cfg:
                if not self.model_cfg.nuisances["h"].get("crossfit", True):
                    nm.h = h_est
                else: #crossfit h on obs data
                    SX_o = np.column_stack([S_o, X_o])
                    h_est = build_regressor(nuis_cfg["h"])
                    h_est.fit(SX_o[idx_o], Y_o[idx_o])
                    nm.h = h_est

            # ---- pi_x: P(A=1 | X, R=0) using experimental train data ----
            if "pi_x" in nuis_cfg:
                pi_x_est = build_classifier(nuis_cfg["pi_x"])
                pi_x_est.fit(X_e[idx_e], A_e[idx_e])
                nm.pi_x = pi_x_est

            # ---- pi_s_x: P(A=1 | S,X,R=0) ----
            if "pi_s_x" in nuis_cfg:
                Z_e = np.column_stack([S_e, X_e])  # shape (n_e, 1+d)
                pi_s_x_est = build_classifier(nuis_cfg["pi_s_x"])
                pi_s_x_est.fit(Z_e[idx_e], A_e[idx_e])
                nm.pi_s_x = pi_s_x_est

            # ---- rho_x: P(R=1 | X) using combined E+O ----
            if "rho_x" in nuis_cfg:
                X_comb = np.vstack([X_e, X_o])
                R_comb = np.concatenate([np.zeros(n_e, dtype=int), np.ones(n_o, dtype=int)])

                # training indices for combined set based on folds:
                # E with fold != k and O with fold != k
                train_comb = np.concatenate([
                    np.where(fold_id_e != k)[0],
                    n_e + np.where(fold_id_o != k)[0],
                ])

                rho_x_est = build_classifier(nuis_cfg["rho_x"])
                rho_x_est.fit(X_comb[train_comb], R_comb[train_comb])
                nm.rho_x = rho_x_est
            
            if "rho_s_x" in nuis_cfg:
                X_comb = np.vstack([X_e, X_o])
                R_comb = np.concatenate([np.zeros(n_e, dtype=int), np.ones(n_o, dtype=int)])
                train_comb = np.concatenate([
                    np.where(fold_id_e != k)[0],
                    n_e + np.where(fold_id_o != k)[0],
                ])

                rho_s_x_est = build_classifier(nuis_cfg["rho_s_x"])
                rho_s_x_est.fit(X_comb[train_comb], R_comb[train_comb])
                nm.rho_s_x = rho_s_x_est

            folds.append(nm)

        # ---- mu_mean: \mu(a,x) fitted on experimental train data + \hat{h} as pseudo-outcome ----
        if "mu_0" in nuis_cfg and nm.h is not None:
            #Get \hat{h}(S,X) on E-train using h_est fitted on R=1 train data
            SX_e = np.column_stack([S_e, X_e])
            #\hat{h} is the mean from all the k splits' predictions (this is fine because h is fit only on R=1 data)
            h_hat_e_train = np.zeros(X_e.shape[0], dtype=float)
            for k in range(self.K):
                h_hat = folds[k].h.predict(SX_e)
                h_hat_e_train += h_hat
            h_hat_e_train /= self.K

            for k in range(self.K):
                nm = folds[k]
                idx_e = np.where(fold_id_e != k)[0]
                #Fit separate models for A=0 and A=1
                mu_cfg_0 = nuis_cfg["mu_0"]

                # A=0
                idx0 = idx_e[A_e[idx_e] == 0]
                if idx0.size > 0:
                    mu0_est = build_regressor(mu_cfg_0)
                    mu0_est.fit(X_e[idx0], h_hat_e_train[idx0])
                    nm.mu_0 = mu0_est
                else:
                    raise ValueError("No training samples with A=0 in {k}-th fold")

                # A=1
                mu_cfg_1 = nuis_cfg["mu_1"]
                idx1 = idx_e[A_e[idx_e] == 1]
                if idx1.size > 0:
                    mu1_est = build_regressor(mu_cfg_1)
                    mu1_est.fit(X_e[idx1], h_hat_e_train[idx1])
                    nm.mu_1 = mu1_est
                else:
                    raise ValueError("No training samples with A=1 in {k}-th fold")

        return CrossFittedNuisances(
            folds=folds,
            fold_id_e=fold_id_e,
            fold_id_o=fold_id_o
        )
