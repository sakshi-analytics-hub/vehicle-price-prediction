"""
Train & Compare Multiple Regression Models
Used Vehicle Resale Price Prediction
"""

import pandas as pd
import numpy as np
import joblib
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

DATA_PATH = "data/car_data_processed.csv"
MODEL_DIR = "models"
REPORT_DIR = "reports"

TARGET = "selling_price"
NUMERIC_FEATURES = ["km_driven", "age"]
ORDINAL_FEATURES = ["owner"]
NOMINAL_FEATURES = ["fuel", "seller_type", "transmission", "brand"]

OWNER_ORDER = [["Test Drive Car", "First Owner", "Second Owner",
                "Third Owner", "Fourth & Above Owner"]]


def build_preprocessor():
    numeric_transform = StandardScaler()
    ordinal_transform = OrdinalEncoder(categories=OWNER_ORDER,
                                        handle_unknown="use_encoded_value",
                                        unknown_value=-1)
    nominal_transform = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transform, NUMERIC_FEATURES),
            ("ord", ordinal_transform, ORDINAL_FEATURES),
            ("nom", nominal_transform, NOMINAL_FEATURES),
        ]
    )
    return preprocessor


def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.001, max_iter=10000),
        "Polynomial Regression (deg2)": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("lin", LinearRegression())
        ]),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=12,
                                                 random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300,
                                                          max_depth=3,
                                                          learning_rate=0.05,
                                                          random_state=42),
        "XGBoost": XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=42, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                    random_state=42, verbose=-1),
        "SVR (RBF)": SVR(kernel="rbf", C=10, epsilon=0.05),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=7),
        "Neural Net (MLP)": MLPRegressor(hidden_layer_sizes=(64, 32),
                                           max_iter=2000, random_state=42,
                                           early_stopping=True),
    }


def evaluate_model(pipeline, X_train, X_test, y_train_log, y_test_log, y_test_actual):
    start = time.time()
    pipeline.fit(X_train, y_train_log)
    train_time = time.time() - start

    pred_log = pipeline.predict(X_test)
    pred_actual = np.expm1(pred_log)  # invert log1p transform

    r2 = r2_score(y_test_actual, pred_actual)
    n, p = X_test.shape[0], X_test.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    mae = mean_absolute_error(y_test_actual, pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, pred_actual))

    # 5-fold CV on log target (R2)
    cv_scores = cross_val_score(pipeline, X_train, y_train_log, cv=5, scoring="r2", n_jobs=-1)

    return {
        "R2": round(r2, 4),
        "Adjusted_R2": round(adj_r2, 4),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "CV_R2_mean": round(cv_scores.mean(), 4),
        "CV_R2_std": round(cv_scores.std(), 4),
        "Train_time_sec": round(train_time, 2),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Log-transform target (right-skewed) — inverted back at evaluation time
    y_log = np.log1p(y)

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    y_test_actual = np.expm1(y_test_log)

    preprocessor = build_preprocessor()
    models = get_models()

    results = {}
    fitted_pipelines = {}

    for name, model in models.items():
        print(f"Training: {name} ...")
        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])
        metrics = evaluate_model(pipe, X_train, X_test, y_train_log, y_test_log, y_test_actual)
        results[name] = metrics
        fitted_pipelines[name] = pipe
        print(f"  -> R2={metrics['R2']}  RMSE={metrics['RMSE']}  CV_R2={metrics['CV_R2_mean']}")

    # Save results table
    results_df = pd.DataFrame(results).T.sort_values("R2", ascending=False)
    results_df.to_csv(f"{REPORT_DIR}/model_comparison.csv")
    print("\n=== FINAL COMPARISON (sorted by R2) ===")
    print(results_df)

    with open(f"{REPORT_DIR}/model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot comparison chart
    plt.figure(figsize=(10, 6))
    results_df_sorted = results_df.sort_values("R2")
    bars = plt.barh(results_df_sorted.index, results_df_sorted["R2"], color="#4C72B0")
    plt.xlabel("R² Score (Test Set)")
    plt.title("Model Comparison — R² Score")
    plt.xlim(0, 1)
    for bar, val in zip(bars, results_df_sorted["R2"]):
        plt.text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center")
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/model_comparison_chart.png")
    plt.close()

    # RMSE chart too
    plt.figure(figsize=(10, 6))
    results_df_sorted_rmse = results_df.sort_values("RMSE", ascending=False)
    bars = plt.barh(results_df_sorted_rmse.index, results_df_sorted_rmse["RMSE"], color="#C44E52")
    plt.xlabel("RMSE (lower is better)")
    plt.title("Model Comparison — RMSE")
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/model_comparison_rmse.png")
    plt.close()

    # Best model
    best_name = results_df.index[0]
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name}")

    joblib.dump(best_pipeline, f"{MODEL_DIR}/best_model.pkl")
    joblib.dump(fitted_pipelines, f"{MODEL_DIR}/all_models.pkl")

    with open(f"{MODEL_DIR}/best_model_name.json", "w") as f:
        json.dump({"best_model": best_name}, f)

    print("Models saved to models/")


if __name__ == "__main__":
    main()
