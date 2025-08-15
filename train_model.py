import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

def train_and_evaluate(X, y, save_dir="models"):
    """
    Train blended LGBM + Ridge model and save both models + feature list.
    """
    os.makedirs(save_dir, exist_ok=True)
    feature_cols = X.columns.tolist()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train LGBM model
    lgbm = MultiOutputRegressor(
        LGBMRegressor(
            n_estimators=100,
            random_state=42,
            reg_alpha=1.0,
            reg_lambda=1.0,
            max_depth=6,
            num_leaves=20
        )
    )
    lgbm.fit(X_train, y_train)

    # Train Ridge model
    ridge = MultiOutputRegressor(Ridge(alpha=1.0))
    ridge.fit(X_train, y_train)

    # Predictions
    y_pred_lgbm = lgbm.predict(X_test)
    y_train_pred_lgbm = lgbm.predict(X_train)
    y_pred_ridge = ridge.predict(X_test)
    y_train_pred_ridge = ridge.predict(X_train)

    # Blend predictions (70% LGBM + 30% Ridge)
    y_pred = 0.7 * y_pred_lgbm + 0.3 * y_pred_ridge
    y_train_pred = 0.7 * y_train_pred_lgbm + 0.3 * y_train_pred_ridge

    # Metrics
    r2_test = r2_score(y_test, y_pred, multioutput='raw_values')
    r2_train = r2_score(y_train, y_train_pred, multioutput='raw_values')
    mae_scores = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    rmse_scores = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))

    for i, ticker in enumerate(y.columns):
        print(
            f"{ticker} → Train R²: {r2_train[i]:.3f} | "
            f"Test R²: {r2_test[i]:.3f} | "
            f"MAE: {mae_scores[i]:.5f} | "
            f"RMSE: {rmse_scores[i]:.5f}"
        )

    # Save models and feature list
    joblib.dump(lgbm, os.path.join(save_dir, "lgbm.pkl"))
    joblib.dump(ridge, os.path.join(save_dir, "ridge.pkl"))
    joblib.dump(feature_cols, os.path.join(save_dir, "feature_cols.pkl"))

    print(f"✅ Models and features saved to '{save_dir}'")

    return lgbm, ridge, feature_cols