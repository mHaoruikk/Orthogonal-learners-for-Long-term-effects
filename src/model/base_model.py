from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
from sklearn import RandomForestRegressor

class BaseEstimator(ABC):
    """
    Baseclass for all estimators, to provide interface of fit() and predict()
    """

    @abstractmethod
    def fit(self, X:np.ndarray, y:np.ndarray, **kargs):
        ...

    @abstractmethod
    def predict(self, X:np.ndarray, **kargs):
        ...

    
class SklearnRegressor(BaseEstimator):
    """
    Wrapper for sklearn regressors
    """
    def __init__(self, base = None, **kargs):
        if base == None:
            self.model = RandomForestRegressor(**kargs)
        else:
            self.model = base(**kargs)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
    

class TorchRegressor(BaseEstimator):
    def __init__(self, net: nn.Module, optimizer_ctor, loss_fn, epochs: int = 100, batch_size: int = 128, device="cpu"):
        self.net = net
        self.optimizer_ctor = optimizer_ctor
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

    def fit(self, X, y, sample_weight=None):
        # move tensors, run SGD loop
        ...
        return self

    def predict(self, X):
        # eval mode forward pass
        self.net.eval()
        ...


    


