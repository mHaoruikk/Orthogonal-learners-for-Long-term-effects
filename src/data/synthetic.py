from src.data.base_dataset import BaseDataset, TwoSampleDataSplit, GroundTruth
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseSyntheticDataset(BaseDataset):
    """
    Base class for fully synthetic dataset.
    Two-sample dataset with experimental and observational data.
    Implements the general DGP:

    1. X^E ~ P_d^E, X^O ~ P_d^O
    2. A^E | X ~ Ber(pi(X))
    3. A^O | X ~ Ber(e(X))   (implicit treatment in O)
    4. S = a(X) + (A - 0.5) * tau_S(X) + delta
    5. Y^O = b(X, S) + (A^O - 0.5) * tau_Y(X) + eps
       under surrogacy tau_Y = 0 => Y^O = b(X, S) + eps
    """
    def __init__(self, config):
        super().__init__(config)
        self.n_e = getattr(config, 'n_e', None)
        self.n_o = getattr(config, 'n_o', None)
        if self.n_e is None or self.n_o is None:
            self.n = getattr(config, 'n', 1000)
        else:
            self.n = self.n_e + self.n_o
        
        self.sigma_s = config.sigma_s
        self.sigma_y = config.sigma_y

        self.dim_x = config.dim_x
        
        self.X_e, self.X_o = None, None
        self.A_e, self.A_o = None, None
        self.S_e, self.S_o = None, None
        self.Y_e, self.Y_o = None, None

    @abstractmethod
    def sample_covariates(self, ):
        """Sample covariates X."""

    @abstractmethod
    def sample_covariates(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample (X_e, X_o) for experimental and observational datasets.
        """
        ...

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

        X_e, X_o = self.sample_covariates(rng)

        pi_e = self.pi_E(X_e)
        A_e = rng.binomial(n=1, p=pi_e, size=self.n_e)

        e_o = self.e_O(X_o)
        A_o = rng.binomial(n=1, p=e_o, size=self.n_o)

        a_e = self.a(X_e)
        a_o = self.a(X_o)
        tau_S_e = self.tau_S(X_e)
        tau_S_o = self.tau_S(X_o)

        delta_e = rng.normal(loc=0.0, scale=self.sigma_s, size=self.n_e)
        delta_o = rng.normal(loc=0.0, scale=self.sigma_s, size=self.n_o)

        S_e = a_e + (A_e - 0.5) * tau_S_e + delta_e
        S_o = a_o + (A_o - 0.5) * tau_S_o + delta_o

        tau_Y_o = self.tau_Y(X_o)
        eps_o = rng.normal(loc=0.0, scale=self.sigma_y, size=self.n_o)
        b_o = self.b(X_o, S_o)
        Y_o = b_o + (A_o - 0.5) * tau_Y_o + eps_o

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


class NieWagerSyntheticDataset(BaseSyntheticDataset):
    """
    Nie & Wager (2021) type synthetic dataset.
    """
    def __init__(self, config):
        super().__init__(config)
        self.gamma = config.get("gamma", 0.0)          # treatment overlap difficulty (γ_π)
        self.gamma_rho = config.get("gamma_rho", 0.0)  # long-term outcome overlap difficulty (γ_ρ)
        self.eta_pi = config.get("eta_pi", 0.05)       # lower/upper bound for π: π ∈ (η_π, 1−η_π)
        self.eta_rho = config.get("eta_rho", 0.01)     # lower bound for ρ: ρ ∈ (η_ρ, 1)

    def sample_covariates(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        # Always draw from a single uniform pool and split by R ~ Ber(rho(X)).
        # X ~ Unif([-1,1]^dim_x) per spec.
        X = rng.uniform(-1, 1, size=(self.n, self.dim_x))
        R = rng.binomial(n=1, p=self.rho_x(X), size=X.shape[0])
        X_e = X[R == 0]
        X_o = X[R == 1]
        self.n_e = X_e.shape[0]
        self.n_o = X_o.shape[0]
        return X_e, X_o
    
    # Analytical moments of h(X) = X0·X1 + X2 + X3 + X7²  with X ~ Unif(-1,1):
    #   E[h]   = 0 + 0 + 0 + 1/3  = 1/3
    #   Var[h] = 1/9 + 1/3 + 1/3 + 4/45  = 13/15
    _H_MEAN = 1.0 / 3.0
    _H_STD  = (13.0 / 15.0) ** 0.5   # ≈ 0.9309

    def pi_E(self, X: np.ndarray) -> np.ndarray:
        # Option A — standardised Nie-Wager feature map (zero-mean, unit-variance logit).
        # π(X) = η_π + (1 − 2η_π) · σ(γ_π · Z(X))
        # Z(X) = (X0·X1 + X2 + X3 + X7² − 1/3) / √(13/15)  ≈ N(0,1) by CLT
        # Analytical boundary mass: P(π within 0.1 of bound) ≈ 2·Φ(−2.20/γ_π)
        #   → γ_π ≈ 2 gives ~28%, γ_π ≈ 3 gives ~47%
        h = X[:, 0] * X[:, 1] + X[:, 2] + X[:, 3] + X[:, 7] ** 2
        logits = self.gamma * (h - self._H_MEAN) / self._H_STD
        return self.eta_pi + (1 - 2 * self.eta_pi) / (1 + np.exp(-logits))
        
    def rho_x(self, X: np.ndarray) -> np.ndarray:
        # Smooth lower-bounded propensity: ρ(X) = η_ρ + (1 − η_ρ)·σ(X0 + X1 + γ_ρ)
        # Maps smoothly to (η_ρ, 1); larger γ_ρ → more units in observational sample (R=1),
        # i.e., sparser experimental coverage → worse long-term overlap.
        logits = X[:, 0] + X[:, 1] - self.gamma_rho
        return self.eta_rho + (1 - self.eta_rho) / (1 + np.exp(-logits))

    def e_O(self, X:np.ndarray) -> np.ndarray:
        # Spec: trim_{0.1}(sigma(X1 + X2 + X3))  (1-indexed → 0-indexed: X[:,1]+X[:,2]+X[:,3])
        logits = X[:, 1] + X[:, 2] + X[:, 3]
        e = 1 / (1 + np.exp(-logits))
        return self._trim(e)
        
    def a(self, X: np.ndarray) -> np.ndarray:
        return (
            np.sin(np.pi * X[:, 0] * X[:, 1])
            + 2 * (X[:, 2] - 0.5) ** 2
            + X[:, 3]
            + 0.5 * X[:, 4]
            + X[:, 5]
        )

    def tau_S(self, X: np.ndarray) -> np.ndarray:
        return 0.25 * (X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3]) + 1

    def b(self, X: np.ndarray, S: np.ndarray) -> np.ndarray:
        # Spec: sin(X0*X1) + X6^2 + X7 + S  (1-indexed X7,X8 → 0-indexed X[:,6],X[:,7])
        return np.sin(X[:, 0] * X[:, 1]) + X[:, 6] ** 2 + X[:, 7] + S

    def tau_Y(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])  

    def true_h(self, S: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Ground-truth h(S, X) = E[Y | S, X, R=1] under the data-generating process.
        """
        return self.b(X, S) + (self.e_O(X) - 0.5) * self.tau_Y(X)

    def true_cate(self, X: np.ndarray) -> np.ndarray:
        return self.tau_S(X)  
    


    
    
    


    
        
