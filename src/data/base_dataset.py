import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, random_split
from sklearn.model_selection import KFold, train_test_split
from dataclasses import dataclass
from typing import Tuple, Callable, Optional
import logging
from abc import ABC, abstractmethod
logger = logging.getLogger(__name__)

@dataclass
class TwoSampleDataSplit:
    """Base class for a two-sample dataset."""
    # Experimental: (X, A, S)
    X_e: np.ndarray
    A_e: np.ndarray
    S_e: np.ndarray

    # Observational: (X, S, Y)
    X_o: np.ndarray
    S_o: np.ndarray
    Y_o: np.ndarray

@dataclass
class MixedDataSample:
    """data class for mixed experimental and observational data."""
    X: np.ndarray
    A: np.ndarray
    S: np.ndarray
    R: np.ndarray  # R=0 for E, R=1 for O
    Y: np.ndarray  # Y observed only for R=1

@dataclass
class GroundTruth:
    """
    Holds ground truth functions for simulations.
    """
    tau: Optional[Callable[[np.ndarray], np.ndarray]]  # tau(X) = E[Y^1 - Y^0 | X]
    pi_E: Optional[Callable[[np.ndarray], float]]  # pi_E(X) = P(A=1 | X, R=0)
    e_O: Optional[Callable[[np.ndarray], float]]  # e_O(X) = P(A=1 | X, R=1)
    

class BaseDataset(ABC):
    """
    Base dataset class for all kinds of datasets.
    """
    def __init__(self, config):
        """Initialize the dataset with hydra config data."""
        self.config = config
        self.seed = config.seed
        self.dim_x = config.dim_x


    @abstractmethod
    def sample(self):
        """Sample dataset for synthetic/semi-synthetic datasets."""


