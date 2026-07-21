"""
EDA + Preprocessing for Used Vehicle Resale Price Prediction
Dataset: CAR DETAILS FROM CAR DEKHO
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

DATA_PATH = "data/car_data.csv"
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Raw shape: {df.shape}")
    return df


def clean_data(df):
    # Drop exact duplicate rows (this dataset has many)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dropped {before - len(df)} duplicate rows -> {len(df)} rows remain")

    # Basic sanity filters
    df = df[df["selling_price"] > 0]
    df = df[df["km_driven"] > 0]
    df = df[df["year"] >= 1990]

    # Remove extreme outliers in price and km (top/bottom 0.5%)
    low_p, high_p = df["selling_price"].quantile([0.005, 0.995])
    df = df[(df["selling_price"] >= low_p) & (df["selling_price"] <= high_p)]

    low_k, high_k = df["km_driven"].quantile([0.005, 0.995])
    df = df[(df["km_driven"] >= low_k) & (df["km_driven"] <= high_k)]

    return df.reset_index(drop=True)


def feature_engineering(df):
    CURRENT_YEAR = 2026  # dataset scraped historically; using project year for age calc
    df["age"] = CURRENT_YEAR - df["year"]

    # Extract brand from name (first word)
    df["brand"] = df["name"].apply(lambda x: str(x).split()[0].strip())

    # Group rare brands into "Other" to avoid huge one-hot dimensionality
    brand_counts = df["brand"].value_counts()
    rare_brands = brand_counts[brand_counts < 15].index
    df["brand"] = df["brand"].apply(lambda b: "Other" if b in rare_brands else b)

    # Price per km driven (helps EDA, not used directly as leakage-free model feature list
    # since it's derived from target — we WON'T feed this into the model)
    df["price_per_km"] = df["selling_price"] / (df["km_driven"] + 1)

    # Drop raw name & year now that we have brand & age
    df_model = df.drop(columns=["name", "year", "price_per_km"])

    return df, df_model


def run_eda(df):
    # 1. Target distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(df["selling_price"], bins=50, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Selling Price Distribution")
    sns.histplot(np.log1p(df["selling_price"]), bins=50, ax=axes[1], color="#55A868")
    axes[1].set_title("Log(Selling Price) Distribution")
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/price_distribution.png")
    plt.close()

    # 2. Price vs Age
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="age", y="selling_price", alpha=0.4, color="#C44E52")
    plt.title("Selling Price vs Vehicle Age")
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/price_vs_age.png")
    plt.close()

    # 3. Price vs Km Driven
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="km_driven", y="selling_price", alpha=0.4, color="#8172B2")
    plt.title("Selling Price vs Km Driven")
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/price_vs_km.png")
    plt.close()

    # 4. Categorical boxplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.boxplot(data=df, x="fuel", y="selling_price", ax=axes[0])
    axes[0].set_title("Price by Fuel Type")
    axes[0].tick_params(axis="x", rotation=30)
    sns.boxplot(data=df, x="transmission", y="selling_price", ax=axes[1])
    axes[1].set_title("Price by Transmission")
    sns.boxplot(data=df, x="owner", y="selling_price", ax=axes[2])
    axes[2].set_title("Price by Owner Type")
    axes[2].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/categorical_boxplots.png")
    plt.close()

    # 5. Top 15 brands by count
    plt.figure(figsize=(9, 5))
    df["brand"].value_counts().head(15).plot(kind="bar", color="#4C72B0")
    plt.title("Top 15 Brands by Listing Count")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/top_brands.png")
    plt.close()

    print("EDA plots saved to reports/")


if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df_full, df_model = feature_engineering(df)
    run_eda(df_full)

    df_model.to_csv("data/car_data_processed.csv", index=False)
    print(f"Processed data saved: {df_model.shape}")
    print(df_model.head())
