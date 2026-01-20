"""Model training utilities for churn classification and value regression."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def split_features_target(
    features: pd.DataFrame, target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split features and target column."""
    X = features.drop(columns=[target_col])
    y = features[target_col]
    return X, y


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Build a preprocessing pipeline for numeric and categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def build_churn_models(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Dict[str, Pipeline]:
    """Create churn model pipelines."""
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models: Dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=config.RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        random_state=config.RANDOM_STATE,
                        max_depth=6,
                    ),
                ),
            ]
        ),
    }
    return models


def build_value_models(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Dict[str, Pipeline]:
    """Create regression model pipelines for value proxy."""
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models: Dict[str, Pipeline] = {
        "ridge": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", Ridge(random_state=config.RANDOM_STATE)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=config.RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    return models


def select_feature_columns(features: pd.DataFrame) -> Tuple[list[str], list[str]]:
    """Split numeric and categorical features based on dtype."""
    categorical_features = ["country_mode"]
    numeric_features = [
        col
        for col in features.columns
        if col not in {"CustomerID", "country_mode"}
        and pd.api.types.is_numeric_dtype(features[col])
    ]
    return numeric_features, categorical_features


def predict_churn_probabilities(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Predict churn probabilities for the positive class."""
    return model.predict_proba(X)[:, 1]
