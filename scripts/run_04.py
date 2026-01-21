"""Run final value-proxy and action list steps from notebooks/04.

This script loads cleaned transactions and customer features produced
by earlier notebooks, trains churn and value models on a training set,
evaluates on a holdout set, predicts scores on the full set, and writes
`reports/customer_action_list.csv`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error

from src import config
from src.features import (
    compute_future_revenue,
    build_time_based_labels,
)
from src.modeling import (
    select_feature_columns,
    build_churn_models,
    build_value_models,
    predict_churn_probabilities,
)


def main():
    processed = Path(config.PROCESSED_DIR)
    reports = Path(config.REPORTS_DIR)
    reports.mkdir(parents=True, exist_ok=True)

    clean_df = pd.read_parquet(processed / "transactions_clean.parquet")
    features = pd.read_parquet(processed / "customer_features.parquet")

    snapshot_date = pd.to_datetime(features["snapshot_date"]).max()
    cutoff_date = snapshot_date - pd.Timedelta(days=config.TIME_SPLIT.cutoff_days_before_snapshot)
    
    # Create train/validation split: earlier customers for training, later for validation
    train_cutoff = cutoff_date - pd.Timedelta(days=60)
    features["train_flag"] = features["last_purchase_date"] <= train_cutoff

    numeric_features, categorical_features = select_feature_columns(features)
    X = features[numeric_features + categorical_features]
    X_train = X[features["train_flag"]]
    X_val = X[~features["train_flag"]]

    # Build churn labels using time-based split
    labels = build_time_based_labels(clean_df, cutoff_date, snapshot_date)
    labels_map = labels.to_dict()
    y = features["CustomerID"].map(labels_map).astype(int)
    y_train = y[features["train_flag"]]
    y_val = y[~features["train_flag"]]

    # Train churn model on training set and evaluate on validation set
    churn_model = build_churn_models(numeric_features, categorical_features)["hist_gradient_boosting"]
    churn_model.fit(X_train, y_train)
    y_val_pred_proba = predict_churn_probabilities(churn_model, X_val)
    if y_val.sum() > 0:
        val_auc = roc_auc_score(y_val, y_val_pred_proba)
        print(f"Churn model validation AUC: {val_auc:.4f}")
    
    # Predict on full dataset
    features["churn_probability"] = predict_churn_probabilities(churn_model, X)

    # Future revenue and value model with train/validation split
    future_revenue = compute_future_revenue(clean_df, cutoff_date, window_days=config.DEFAULT_PREDICTION_WINDOW_DAYS)
    features = features.merge(future_revenue, on="CustomerID", how="left")
    rev_col = f"future_revenue_{config.DEFAULT_PREDICTION_WINDOW_DAYS}d"
    features[rev_col] = features[rev_col].fillna(0)
    
    y_val_revenue = features.loc[~features["train_flag"], rev_col]

    value_model = build_value_models(numeric_features, categorical_features)["ridge"]
    value_model.fit(X_train, features.loc[features["train_flag"], rev_col])
    y_val_pred_value = value_model.predict(X_val)
    if len(y_val_revenue) > 0:
        mse = mean_squared_error(y_val_revenue, y_val_pred_value)
        val_rmse = mse ** 0.5
        print(f"Value model validation RMSE: {val_rmse:.4f}")
    
    features["value_score"] = value_model.predict(X)

    out_cols = ["CustomerID", "churn_probability", "value_score", rev_col]
    out_path = reports / "customer_action_list.csv"
    features[out_cols].to_csv(out_path, index=False)
    print(f"Wrote action list to {out_path}")


if __name__ == "__main__":
    main()
