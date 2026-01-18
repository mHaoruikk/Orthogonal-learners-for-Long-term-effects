from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor


@dataclass
class FittedSModel:
    s_col: str
    a_val: int
    target_type: str  # "regression" or "classification"
    model: Pipeline
    metric_name: str
    metric_value: float
    label_encoder: Optional[LabelEncoder] = None

    def predict_expectation(self, X: pd.DataFrame) -> np.ndarray:
        if self.target_type == "regression":
            return self.model.predict(X)

        if self.label_encoder is None:
            raise ValueError("label_encoder missing for classification target")

        proba = self.model.predict_proba(X)
        classes = self.label_encoder.classes_

        if classes.dtype.kind in ("i", "u", "f"):
            return proba @ classes.astype(float)

        preds = np.argmax(proba, axis=1)
        return self.label_encoder.inverse_transform(preds)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.target_type == "regression":
            return self.model.predict(X)

        if self.label_encoder is None:
            raise ValueError("label_encoder missing for classification target")

        preds = self.model.predict(X)
        return self.label_encoder.inverse_transform(preds.astype(int))


class ConditionalSModeler:
    """
    Fit E[S | A=a, X] separately for treated/untreated groups and for each short-term outcome S.
    Uses XGBoost with normalization on numeric covariates and one-hot encoding for categoricals.
    """

    def __init__(
        self,
        a_col: str,
        x_numeric_cols: Sequence[str],
        x_categorical_cols: Sequence[str],
        s_numeric_cols: Sequence[str],
        s_categorical_cols: Sequence[str],
        test_size: float = 0.2,
        random_state: int = 42,
        reg_params: Optional[dict] = None,
        clf_params: Optional[dict] = None,
    ) -> None:
        self.a_col = a_col
        self.x_numeric_cols = list(x_numeric_cols)
        self.x_categorical_cols = list(x_categorical_cols)
        self.s_numeric_cols = list(s_numeric_cols)
        self.s_categorical_cols = list(s_categorical_cols)
        self.test_size = test_size
        self.random_state = random_state

        self.default_reg_params = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "n_jobs": -1,
            "random_state": random_state,
        }
        self.default_clf_params = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "n_jobs": -1,
            "random_state": random_state,
            "tree_method": "hist",
        }

        self.reg_params = reg_params or {}
        self.clf_params = clf_params or {}
        self.models: Dict[Tuple[str, int], FittedSModel] = {}

    def _build_preprocessor(self) -> ColumnTransformer:
        transformers = []

        if self.x_numeric_cols:
            transformers.append(
                ("num", Pipeline([("scaler", StandardScaler())]), self.x_numeric_cols)
            )
        if self.x_categorical_cols:
            transformers.append(
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.x_categorical_cols)
            )

        return ColumnTransformer(transformers, remainder="drop")

    def _fit_single(self, df: pd.DataFrame, s_col: str, a_val: int) -> FittedSModel:
        mask = df[self.a_col] == a_val
        if mask.sum() < 10:
            raise ValueError(f"Not enough samples for A={a_val} to fit {s_col}")

        X = df.loc[mask, self.x_numeric_cols + self.x_categorical_cols]
        y = df.loc[mask, s_col]
        preprocessor = self._build_preprocessor()
        is_cat_target = (
            s_col in self.s_categorical_cols
            or str(y.dtype).startswith("category")
            or y.dtype == object
        )

        if is_cat_target:
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            if np.unique(y_enc).size < 2:
                raise ValueError(f"Target {s_col} has one class for A={a_val}")

            stratify = y_enc if np.unique(y_enc).size > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y_enc,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=stratify,
            )

            clf_params = {**self.default_clf_params, **self.clf_params}
            num_classes = np.unique(y_train).size
            if num_classes == 2:
                clf_params.setdefault("objective", "binary:logistic")
            else:
                clf_params.setdefault("objective", "multi:softprob")
                clf_params["num_class"] = num_classes

            pipe = Pipeline(
                [("prep", preprocessor), ("model", XGBClassifier(**clf_params))]
            )
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            acc = accuracy_score(y_test, preds)
            return FittedSModel(
                s_col=s_col,
                a_val=a_val,
                target_type="classification",
                model=pipe,
                metric_name="accuracy",
                metric_value=float(acc),
                label_encoder=le,
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        reg_params = {**self.default_reg_params, **self.reg_params}
        pipe = Pipeline([("prep", preprocessor), ("model", XGBRegressor(**reg_params))])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        return FittedSModel(
            s_col=s_col,
            a_val=a_val,
            target_type="regression",
            model=pipe,
            metric_name="mse",
            metric_value=float(mse),
        )

    def fit(self, df: pd.DataFrame, s_cols: Optional[Sequence[str]] = None) -> Dict[Tuple[str, int], FittedSModel]:
        targets = list(s_cols) if s_cols is not None else self.s_numeric_cols + self.s_categorical_cols
        for s_col in targets:
            for a_val in (0, 1):
                self.models[(s_col, a_val)] = self._fit_single(df, s_col, a_val)
        return self.models

    def predict_expectation(self, X: pd.DataFrame, s_col: str, a_val: int) -> np.ndarray:
        key = (s_col, a_val)
        if key not in self.models:
            raise KeyError(f"No model fitted for {key}")
        return self.models[key].predict_expectation(X)

    def metrics(self) -> Dict[Tuple[str, int], Tuple[str, float]]:
        return {
            key: (model.metric_name, model.metric_value)
            for key, model in self.models.items()
        }
