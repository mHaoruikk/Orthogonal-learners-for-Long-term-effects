from src.model.base_model import SklearnClassifier, SklearnRegressor, XGBoostClassifier, XGBoostRegressor

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
    else:
        raise ValueError(f"Unknown classifier type: {t}")
    
def build_regressor(cfg) -> SklearnRegressor | XGBoostRegressor:
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
    else:
        raise ValueError(f"Unknown regressor type: {t}")