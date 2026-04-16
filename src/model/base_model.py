from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
        return self.linear(x).squeeze(-1)  # (N,)


class TorchMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers, output_activation: str | None = None):
        super().__init__()
        if hidden_layers is None:
            raise ValueError("hidden_layers must be provided when building an MLP.")

        layers = []
        prev_dim = input_dim
        for width in hidden_layers:
            layers.append(nn.Linear(prev_dim, width))
            layers.append(nn.ReLU())
            prev_dim = width
        layers.append(nn.Linear(prev_dim, 1))

        if output_activation is None or output_activation == "identity":
            pass
        elif output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        else:
            raise ValueError(f"Unknown output_activation: {output_activation}")

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TorchRegressor(BaseEstimator):
    def __init__(
        self,
        input_dim: int | None = None,
        hidden_layers = None,
        output_activation: str | None = None,
        net: nn.Module | None = None,
        optimizer_ctor = torch.optim.Adam,
        loss_fn: nn.Module | None = None,
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str | torch.device | None = None,
    ):
        if net is None and hidden_layers is None:
            raise ValueError("hidden_layers must be provided when net is None.")

        self._hidden_layers = hidden_layers
        self._output_activation = output_activation

        if net is None:
            # If input_dim is not known at construction time, we lazily build the MLP in fit().
            self.net = (
                TorchMLP(input_dim=input_dim, hidden_layers=hidden_layers, output_activation=output_activation)
                if input_dim is not None
                else None
            )
        else:
            self.net = net

        self.optimizer_ctor = optimizer_ctor
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if loss_fn is None:
            if output_activation == "sigmoid":
                self.loss_fn = nn.BCELoss(reduction="none")
            else:
                self.loss_fn = nn.MSELoss(reduction="none")
        else:
            self.loss_fn = loss_fn

    def fit(self, X, y, sample_weight=None):
        # move tensors, run SGD loop
        if self.net is None:
            if X is None:
                raise ValueError("X must be provided to infer input_dim when net is None.")
            if not hasattr(X, "shape") or len(X.shape) != 2:
                raise ValueError(f"Expected X to be a 2D array, got shape={getattr(X, 'shape', None)}")
            self.net = TorchMLP(
                input_dim=int(X.shape[1]),
                hidden_layers=self._hidden_layers,
                output_activation=self._output_activation,
            )

        self.net.to(self.device)
        self.net.train()

        # Keep tensors on CPU for DataLoader; move to device per-batch.
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.float32).view(-1)

        if sample_weight is not None:
            w_tensor = torch.as_tensor(sample_weight, dtype=torch.float32).view(-1)
            dataset = TensorDataset(X_tensor, y_tensor, w_tensor)
        else:
            dataset = TensorDataset(X_tensor, y_tensor)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = self.optimizer_ctor(self.net.parameters(), lr=self.lr)

        for _ in range(self.epochs):
            for batch in loader:
                optimizer.zero_grad()
                if sample_weight is None:
                    xb, yb = batch
                    wb = None
                else:
                    xb, yb, wb = batch
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                wb = wb.to(self.device) if wb is not None else None
                preds = self.net(xb).view(-1)
                loss = self._compute_loss(preds, yb, wb)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        # eval mode forward pass
        if self.net is None:
            raise ValueError("Model is not initialized. Call fit() before predict().")
        self.net.eval()
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            preds = self.net(X_tensor).view(-1)
        return preds.cpu().numpy()

    def _compute_loss(self, preds, targets, weights):
        losses = self.loss_fn(preds, targets)
        if weights is None:
            return losses.mean() if losses.dim() > 0 else losses
        if losses.dim() == 0:
            raise ValueError("sample_weight requires loss_fn with reduction='none'.")
        return (losses.view(-1) * weights).mean()

    


    


