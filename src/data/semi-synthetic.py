from src.data.base_dataset import BaseDataset, TwoSampleDataSplit, GroundTruth, MixedDataSample
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import logging
import pandas as pd
logger = logging.getLogger(__name__)


class BaseSemiSyntheticDataset(BaseDataset):
    """
    Base class for semi-synthetic dataset.
    Load covariates from real data and simulate.
    Implements the general DGP:

    1. use a sample function to decide belonging to E or O, p(R=1 | X) = \rho(X)
    2. A^E | X ~ Ber(pi(X))
    3. A^O | X ~ Ber(e(X))   (implicit treatment in O)
    4. S = a(X) + (A - 0.5) * tau_S(X) + delta
    5. Y^O = b(X, S) + (A^O - 0.5) * tau_Y(X) + eps
       under surrogacy tau_Y = 0 => Y^O = b(X, S) + eps
    """
    def __init__(self, config):
        super().__init__(config)

        self.sigma_s = config.sigma_s
        self.sigma_y = config.sigma_y

        self.rho_x = self.create_rho_x()

        self.dim_x = config.dim_x

        self.df = self.load_real_data(config.real_data_path)
        self.X_cols = config.X_cols
        
        self.X_e, self.X_o = None, None
        self.A_e, self.A_o = None, None
        self.S_e, self.S_o = None, None
        self.Y_e, self.Y_o = None, None

        self.X = self.df[self.X_cols].to_numpy()
        self.A = None
        self.S = None
        self.R = None
        self.Y = None

    @abstractmethod
    def load_real_data(self, path: str) -> np.ndarray:
        """Load real data from path and return as a numpy array."""
        ...

    @abstractmethod
    def create_rho_x(self):
        """Create the function rho(X) = P(R=1 | X)."""
        ...

    def Sample_belonging(self, X: np.ndarray) -> np.ndarray:
        """Sample belonging to E or O.
        returns R ~ Ber(rho(X)).
        """
        return np.random.binomial(n=1, p=self.rho_x(X), size=X.shape[0])

    @abstractmethod
    def pi_E(self, X: np.ndarray) -> np.ndarray:
        """
        True treatment propensity in experimental sample: pi^*(X) = P(A=1 | X, R=0).
        """
        ...

    @abstractmethod
    def e_O(self, X: np.ndarray) -> np.ndarray:
        """
        Implicit treatment propensity in observational sample: e^*(X) = P(A=1 | X, R=1).
        """
        ...

    @abstractmethod
    def a(self, X: np.ndarray) -> np.ndarray:
        """
        Main effect of S: a^*(X).
        """
        ...

    @abstractmethod
    def tau_S(self, X: np.ndarray) -> np.ndarray:
        """
        Short-term CATE: tau_S^*(X).
        """
        ...

    @abstractmethod
    def b(self, X: np.ndarray, S: np.ndarray) -> np.ndarray:
        """
        Long-term regression: b^*(X, S).
        """
        ...
    @abstractmethod
    def tau_Y(self, X: np.ndarray) -> np.ndarray:
        """
        Direct long-term effect of A on Y, tau_Y^*(X).
        Under surrogacy we set this to zero by default.
        """
        ...

    @abstractmethod
    def true_cate(self, X: np.ndarray) -> np.ndarray:
        """
        τ(X) = E[Y^1 - Y^0 | X].
        """
        ...

    def sample(self) -> Tuple[TwoSampleDataSplit, GroundTruth]:
        """
        Sample experimental and observational data according to the
        generic DGP using subclass-specific building blocks.
        """
        rng = np.random.default_rng(self.seed)

        # 1. Pre-treatment covariates
        X_e, X_o = self.sample_covariates(rng)   # (n_e, dim_x), (n_o, dim_x)

        # 2. Treatment in E: A^E | X ~ Ber(pi(X))
        pi_e = self.pi_E(X_e)
        A_e = rng.binomial(n=1, p=pi_e, size=self.n_e)

        # 3. Implicit treatment in O: A^O | X ~ Ber(e(X))
        e_o = self.e_O(X_o)
        A_o = rng.binomial(n=1, p=e_o, size=self.n_o)
        # 4. Short-term outcome S
        a_e = self.a(X_e)
        a_o = self.a(X_o)
        tau_S_e = self.tau_S(X_e)
        tau_S_o = self.tau_S(X_o)

        delta_e = rng.normal(loc=0.0, scale=self.sigma_s, size=self.n_e)
        delta_o = rng.normal(loc=0.0, scale=self.sigma_s, size=self.n_o)

        S_e = a_e + (A_e - 0.5) * tau_S_e + delta_e
        S_o = a_o + (A_o - 0.5) * tau_S_o + delta_o

        # 5. Long-term outcome in O
        tau_Y_o = self.tau_Y(X_o)
        eps_o = rng.normal(loc=0.0, scale=self.sigma_y, size=self.n_o)
        b_o = self.b(X_o, S_o)
        Y_o = b_o + (A_o - 0.5) * tau_Y_o + eps_o

        # Long-term outcome in E (not observed)
        tau_Y_e = self.tau_Y(X_e)
        eps_e = rng.normal(loc=0.0, scale=self.sigma_y, size=self.n_e)
        b_e = self.b(X_e, S_e)
        Y_e = b_e + (A_e - 0.5) * tau_Y_e + eps_e

        data = TwoSampleDataSplit(
            X_e=X_e.copy(),
            A_e=A_e.copy(),
            S_e=S_e.copy(),
            X_o=X_o.copy(),
            S_o=S_o.copy(),
            Y_o=Y_o.copy(),
        )

        gt = GroundTruth(
            tau=self.true_cate,
            pi_E=self.pi_E,
            e_O=self.e_O
        )

        self.X_e, self.A_e, self.S_e, self.Y_e = X_e, A_e, S_e, Y_e
        self.X_o, self.A_o, self.S_o, self.Y_o = X_o, A_o, S_o, Y_o

        return data, gt
    
    def _trim(self, u, eta = 0.1):
        return np.clip(u, eta, 1 - eta)
    


class IST3SemiSyntheticDataset(BaseSemiSyntheticDataset):
    """
    Parametric semi-synthetic dataset based on IST-3 baseline covariates.

    Implements:
      S = 10 * sigmoid( S_lat )
      Y =  6 * sigmoid( Y_lat )
      S_lat = a^*(X) + (A-0.5) tau_S^*(X) + delta
      Y_lat = b^*(X,S) + (A-0.5) tau_Y^*(X) + eps
    """

    def __init__(self, config):
        super().__init__(config)

        self.x_idx = {c: i for i, c in enumerate(self.X_cols)}
        self.numerical_cols = config.numerical_cols

        X_np = self.df[self.X_cols].to_numpy(dtype=float)
        self.x_mean = np.nanmean(X_np, axis=0)
        self.x_std = np.nanstd(X_np, axis=0)
        self.x_std = np.where(self.x_std < 1e-8, 1.0, self.x_std)

        self._cate_mc = getattr(config, "cate_mc", 200)

    def load_real_data(self, path: str) -> pd.DataFrame:
        import pandas as pd
        df = pd.read_csv(path)

        # ensure X columns are numeric (coerce non-numeric to NaN)
        for c in self.X_cols:
            if c in self.numerical_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            else:
                df[c] = df[c].astype("category").cat.codes.astype(float)
        #report data info
        logger.info(f"Loaded real data from {path} with shape {df.shape}")
        logger.info(f"Numerical Columns: {self.numerical_cols}")
        logger.info(f"Categorical Columns: {[c for c in self.X_cols if c not in self.numerical_cols]}")

        return df

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-z))

    def _z(self, X: np.ndarray, col: str) -> np.ndarray:
        j = self.x_idx[col]
        return (X[:, j] - self.x_mean[j]) / self.x_std[j]

    def create_rho_x(self):
        def rho_x(X: np.ndarray) -> np.ndarray:

            x_age = self._z(X, "age")
            x_nihss = self._z(X, "nihss")

            logits = 1.2 - 0.25 * x_nihss + 0.10 * x_age
            p = self._sigmoid(logits)

            return self._trim(p, eta=0.01)
        return rho_x

    def pi_E(self, X: np.ndarray) -> np.ndarray:
        x_age = self._z(X, "age")
        x_nihss = self._z(X, "nihss")
        x_randdelay = self._z(X, "randdelay")
        return self._trim(np.sin(np.pi * (x_age * x_randdelay + 1) / 2))

    def e_O(self, X: np.ndarray) -> np.ndarray:
        x_nihss = self._z(X, "nihss")
        logits = -0.3 + 0.03 * x_nihss
        return self._sigmoid(logits)

    # ---------- range enforcement ----------
    def squash_S(self, S_lat: np.ndarray) -> np.ndarray:
        # S in [0,10]
        return 10.0 * self._sigmoid(S_lat)

    def squash_Y(self, Y_lat: np.ndarray) -> np.ndarray:
        # Y in [0,6]
        return 6.0 * self._sigmoid(Y_lat)

    # ---------- latent a^*(X): exp + quadratic ----------
    def a(self, X: np.ndarray) -> np.ndarray:
        x_age   = self._z(X, "age")
        x_nihss = self._z(X, "nihss")
        x_delay = self._z(X, "randdelay")
        x_gcs   = self._z(X, "gcs_score_rand")
        x_glu   = self._z(X, "glucose")

        x_hypo  = self._z(X, "R_hypodensity")
        x_swell = self._z(X, "R_swelling")
        x_infar = self._z(X, "R_infarct_size")

        x_af    = self._z(X, "atrialfib_rand")
        x_dm    = self._z(X, "diabetes_pre")
        x_preI  = self._z(X, "indepinadl_rand")

        alpha0 = 0.0
        alpha1, alpha2, alpha3, alpha4 = 0.9, 0.5, 0.4, 0.25
        alpha5, alpha6, alpha7, alpha8 = 0.18, 0.08, 0.10, 0.12
        alpha9, alpha10, alpha11 = 0.25, 0.20, 0.10
        alpha12, alpha13, alpha14 = 0.20, 0.12, 0.15

        a_lat = (
            alpha0
            + alpha1 * np.exp(-0.7 * x_nihss)
            + alpha2 * np.exp(-0.4 * x_age)
            + alpha3 * np.exp(-0.6 * x_delay)
            + alpha4 * x_gcs
            - alpha5 * (x_nihss ** 2)
            - alpha6 * (x_age ** 2)
            - alpha7 * (x_delay ** 2)
            - alpha8 * (x_nihss * x_age)
            - alpha9  * x_hypo
            - alpha10 * x_swell
            - alpha11 * x_infar
            - alpha12 * x_af
            - alpha13 * x_dm
            + alpha14 * x_preI
            - 0.05 * x_glu 
        )
        return a_lat

    # ---------- latent tau_S^*(X): sparse linear (<= 5 vars) ----------
    def tau_S(self, X: np.ndarray) -> np.ndarray:
        x_nihss = self._z(X, "nihss")
        x_delay = self._z(X, "randdelay")
        x_age   = self._z(X, "age")
        x_gcs   = self._z(X, "gcs_score_rand")
        x_hypo  = self._z(X, "R_hypodensity")

        beta0 = 0.9
        beta1, beta2, beta3, beta4, beta5 = 0.30, 0.25, 0.15, 0.15, 0.10

        tau_lat = (
            beta0
            - beta1 * x_nihss
            - beta2 * x_delay
            - beta3 * x_age
            + beta4 * x_gcs
            - beta5 * x_hypo
        )
        return tau_lat

    # ---------- latent b^*(X,S): same style as a^*, plus S terms ----------
    def b(self, X: np.ndarray, S: np.ndarray) -> np.ndarray:
        # scale S from [0,10] to O(1)
        s = (S - 5.0) / 2.0

        x_age   = self._z(X, "age")
        x_nihss = self._z(X, "nihss")
        x_delay = self._z(X, "randdelay")
        x_gcs   = self._z(X, "gcs_score_rand")

        x_hypo  = self._z(X, "R_hypodensity")
        x_swell = self._z(X, "R_swelling")

        x_af    = self._z(X, "atrialfib_rand")
        x_dm    = self._z(X, "diabetes_pre")
        x_preI  = self._z(X, "indepinadl_rand")

        gamma0 = -0.2
        gamma1, gamma2, gamma3 = 0.5, 0.9, 0.15
        gamma4, gamma5, gamma6, gamma7 = 0.6, 0.25, 0.25, 0.15
        gamma8, gamma9 = 0.10, 0.08
        gamma10, gamma11 = 0.15, 0.12
        gamma12, gamma13, gamma14 = 0.12, 0.08, 0.08

        b_lat = (
            gamma0
            + gamma1 * np.exp(0.6 * s)
            + gamma2 * s
            - gamma3 * (s ** 2)
            + gamma4 * np.exp(-0.6 * x_nihss)
            + gamma5 * np.exp(-0.3 * x_age)
            + gamma6 * np.exp(-0.4 * x_delay)
            + gamma7 * x_gcs
            - gamma8 * (x_nihss ** 2)
            - gamma9 * (x_nihss * x_age)
            - gamma10 * x_hypo
            - gamma11 * x_swell
            - gamma12 * x_af
            - gamma13 * x_dm
            + gamma14 * x_preI
        )
        return b_lat

    # ---------- latent tau_Y^*(X): sparse linear, smaller magnitude ----------
    def tau_Y(self, X: np.ndarray) -> np.ndarray:
        x_nihss = self._z(X, "nihss")
        x_delay = self._z(X, "randdelay")
        x_age   = self._z(X, "age")
        x_gcs   = self._z(X, "gcs_score_rand")

        theta0 = 0.25
        theta1, theta2, theta3, theta4 = 0.10, 0.10, 0.06, 0.06

        tauY_lat = (
            theta0
            - theta1 * x_nihss
            - theta2 * x_delay
            - theta3 * x_age
            + theta4 * x_gcs
        )
        return tauY_lat

    # ---------- ground-truth CATE via Monte Carlo (recommended) ----------
    def true_cate(self, X: np.ndarray) -> np.ndarray:
        """
        τ(X) = E[Y^1 - Y^0 | X] under the full nonlinear (sigmoid) DGP.
        Approximated by MC. Vectorized.
        """
        rng = np.random.default_rng(12345)
        n = X.shape[0]
        M = int(self._cate_mc)

        X_rep = np.repeat(X, M, axis=0)

        # independent noises for the two potential outcomes
        delta1 = rng.normal(0.0, self.sigma_s, size=n * M)
        delta0 = rng.normal(0.0, self.sigma_s, size=n * M)
        eps1   = rng.normal(0.0, self.sigma_y, size=n * M)
        eps0   = rng.normal(0.0, self.sigma_y, size=n * M)

        A1 = np.ones(n * M)
        A0 = np.zeros(n * M)

        # S^1, S^0
        S1_lat = self.a(X_rep) + (A1 - 0.5) * self.tau_S(X_rep) + delta1
        S0_lat = self.a(X_rep) + (A0 - 0.5) * self.tau_S(X_rep) + delta0
        S1 = self.squash_S(S1_lat)
        S0 = self.squash_S(S0_lat)

        # Y^1, Y^0
        Y1_lat = self.b(X_rep, S1) + (A1 - 0.5) * self.tau_Y(X_rep) + eps1
        Y0_lat = self.b(X_rep, S0) + (A0 - 0.5) * self.tau_Y(X_rep) + eps0
        Y1 = self.squash_Y(Y1_lat)
        Y0 = self.squash_Y(Y0_lat)

        tau = (Y1 - Y0).reshape(n, M).mean(axis=1)
        return tau