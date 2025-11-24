from base_dataset import BaseDataset, TwoSampleDataSplit, GroundTruth
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
        self.n_e = config.n_e
        self.n_o = config.n_o
        self.sigma_s = config.sigma_s
        self.sigma_y = config.sigma_y

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

        data = TwoSampleDataSplit(
            X_e=X_e,
            A_e=A_e,
            S_e=S_e,
            X_o=X_o,
            S_o=S_o,
            Y_o=Y_o,
        )

        gt = GroundTruth(
            tau=self.true_cate,
            pi_E=self.pi_E,
            e_O=self.e_O
        )
        return data, gt
    
    def _trim(self, u, eta = 0.1):
        return np.clip(u, eta, 1 - eta)


class NieWagerSyntheticDataset(BaseSyntheticDataset):
    """
    Nie & Wager (2021) type synthetic dataset.
    """
    def __init__(self, config):
        super().__init__(config)
        self.propensity_E = config.propensity_E
        self.propensity_O = config.propensity_O

    def sample_covariates(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        X_e = rng.uniform(-1, 1, size=(self.n_e, self.dim_x))
        X_o = rng.uniform(-1, 1, size=(self.n_o, self.dim_x))
        return X_e, X_o
    
    def pi_E(self, X:np.ndarray) -> np.ndarray:
        if self.propensity_E == "sin":
            return self._trim(np.sin(np.pi * X[:, 0] * X[:, 1]))
        elif self.propensity_E == "exp":
            logits = 0.5 * X[:, 0] + 0.5 * X[:, 1]
            pi = 1 / (1 + np.exp(-logits))
            return self._trim(pi)
        else:
            raise NotImplementedError(f"Unknown propensity_E: {self.propensity_E}")
        
    def e_O(self, X:np.ndarray) -> np.ndarray:
        if self.propensity_O == "exp":
            logits = 0.5 * X[:, 0] - 0.5 * X[:, 1]
            e = 1 / (1 + np.exp(-logits))
            return self._trim(e)
        else:
            raise NotImplementedError(f"Unknown propensity_O: {self.propensity_O}")
        
    def a(self, X: np.ndarray) -> np.ndarray:
        return (
            np.sin(np.pi * X[:, 0] * X[:, 1])
            + 2 * (X[:, 2] - 0.5) ** 2
            + X[:, 3]
            + 0.5 * X[:, 4]
            + X[:, 5]
        )

    def tau_S(self, X: np.ndarray) -> np.ndarray:
        return 0.5 * (X[:, 0] + X[:, 1])

    def b(self, X: np.ndarray, S: np.ndarray) -> np.ndarray:
        return X[:, 6] ** 2 + X[:, 7] + S

    def tau_Y(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])  

    def true_cate(self, X: np.ndarray) -> np.ndarray:
        return self.tau_S(X)  
    


    
    
    


    
        