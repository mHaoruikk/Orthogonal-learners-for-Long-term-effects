from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn

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

    def __call__(self, X:np.ndarray, **kargs):
        return self.predict(X, **kargs)

class SklearnRegressor(BaseEstimator):
    """
    Wrapper for sklearn regressors
    """
    def __init__(self, model):
        self.model = model
        
    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model.predict(X)
    
class SklearnClassifier(BaseEstimator):
    """
    Wrapper for sklearn classifiers
    """
    def __init__(self, model):
        self.model = model
        
    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]
    
class XGBoostRegressor(BaseEstimator):
    def __init__(self, model):
        self.model = model
        
    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model.predict(X)

class XGBoostClassifier(BaseEstimator):
    def __init__(self, model):
        self.model = model
        
    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

class ClosedFormLinear:
    """
    CATE estimator is linear: g_\theta(x) = \theta^\top f(x)
    Closed-form linear regression solution to \theta (only suitable when dim(f) is small):
    \bigl(\sum_{R_i=0}\rho_i f_if_i^\top\bigr)\theta^* = \sum_{R_i=0}\rho_i\phi_i f_i + \sum_{R_i=1}\psi_i f_i.
    linear: f(x) = (x^', 1)'
    """
    def __init__(self, feature_type: str = "linear", 
                 regularization: float = 0.0, 
                 fit_intercept: bool = False):
        
        self.feature_type = feature_type
        self.regularization = regularization
        self.fit_intercept = fit_intercept
        if self.feature_type == "linear":
            if self.fit_intercept:
                def linear_fn(x):
                    if len(x.shape) == 1:
                        return np.concatenate((x, np.array([1.0])))
                    else:
                        return np.column_stack((x, np.ones(x.shape[0])))
            else:
                def linear_fn(x):
                    return x
            self.f = linear_fn
        else:
            raise NotImplementedError(f"Unknown feature_type: {self.feature_type}")
        self.theta_ = None

    def fit(
        self,
        X_e: np.ndarray, X_o: np.ndarray,
        rho_e: np.ndarray,
        phi: np.ndarray,
        psi: np.ndarray,
    ):
        f_e = self.f(X_e)  # (n_e, d)
        f_o = self.f(X_o)  # (n_o, d)

        H = f_e.T @ (rho_e[:, None] * f_e) + self.regularization * np.eye(f_e.shape[1])  # (d, d)

        b = f_e.T @ (rho_e * phi) + f_o.T @ psi  # (d,)

        self.theta_ = np.linalg.solve(H, b)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        f_X = self.f(X) 
        return f_X @ self.theta_

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)


#Implement a single layer torch neural network (which is basically linear regression)
class TorchLinearModel(nn.Module):
    def __init__(self, input_dim: int):
        super(TorchLinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze()  # Squeeze to get shape (N,)


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

    


    


