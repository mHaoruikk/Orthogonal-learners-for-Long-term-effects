from src.model.base_model import SklearnClassifier, SklearnRegressor, XGBoostClassifier, XGBoostRegressor, \
    ClosedFormLinear, TorchRegressor

from sklearn.linear_model import (
    LogisticRegression,
    Lasso,
    Ridge,
    ElasticNet,
    LinearRegression,
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

def build_classifier(cfg) -> SklearnClassifier | XGBoostClassifier:
    t = cfg.type.lower()
    params = cfg.parameters or {}

    if t == "logistic":
        model = LogisticRegression(**params)
        return SklearnClassifier(model)
    elif t == "random_forest":
        model = RandomForestClassifier(**params)
        return SklearnClassifier(model)
    elif t == "xgboost":
        model = XGBClassifier(**params)
        return XGBoostClassifier(model)
    elif t == "torch_mlp":
        # TorchRegressor with sigmoid output behaves as a binary classifier returning P(Y=1|X).
        return TorchRegressor(
            input_dim=params.get("input_dim", None),
            hidden_layers=params.get("hidden_layers", None),
            output_activation=params.get("output_activation", "sigmoid"),
            epochs=int(params.get("epochs", 20)),
            batch_size=int(params.get("batch_size", 64)),
            lr=float(params.get("lr", 1e-3)),
            device=params.get("device", None),
        )
    else:
        raise ValueError(f"Unknown classifier type: {t}")
    
def build_regressor(cfg) -> SklearnRegressor | XGBoostRegressor | ClosedFormLinear | TorchRegressor:
    t = cfg.type.lower()
    params = cfg.parameters or {}

    if t == "lasso":
        model = Lasso(**params)
        return SklearnRegressor(model)
    elif t == "ridge":
        model = Ridge(**params)
        return SklearnRegressor(model)
    elif t == "elasticnet":
        model = ElasticNet(**params)
        return SklearnRegressor(model)
    elif t == "linear":
        model = LinearRegression(**params)
        return SklearnRegressor(model)
    elif t == "random_forest":
        model = RandomForestRegressor(**params)
        return SklearnRegressor(model)
    elif t == "xgboost":
        model = XGBRegressor(**params)
        return XGBoostRegressor(model)
    elif t == "solve_linear":
        model = ClosedFormLinear(**params)
        return model
    elif t == "torch_mlp":
        return TorchRegressor(
            input_dim=params.get("input_dim", None),
            hidden_layers=params.get("hidden_layers", None),
            output_activation=params.get("output_activation", "identity"),
            epochs=int(params.get("epochs", 20)),
            batch_size=int(params.get("batch_size", 64)),
            lr=float(params.get("lr", 1e-3)),
            device=params.get("device", None),
        )
    else:
        raise ValueError(f"Unknown regressor type: {t}")
