"""
Depreciation Curve Analysis
Shows how predicted resale price changes as Age / Km Driven increase,
using the best trained model. This is the project's unique differentiator.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_DIR = "models"
REPORT_DIR = "reports"

model = joblib.load(f"{MODEL_DIR}/best_model.pkl")


def predict_price(brand, fuel, seller_type, transmission, owner, km_driven, age):
    row = pd.DataFrame([{
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "age": age,
        "brand": brand,
    }])
    pred_log = model.predict(row)[0]
    return np.expm1(pred_log)


def depreciation_vs_age(brand="Maruti", fuel="Petrol", seller_type="Individual",
                         transmission="Manual", owner="First Owner", km_driven=40000):
    ages = np.arange(0, 16)
    prices = [predict_price(brand, fuel, seller_type, transmission, owner, km_driven, a)
              for a in ages]

    plt.figure(figsize=(8, 5))
    plt.plot(ages, prices, marker="o", color="#C44E52", linewidth=2)
    plt.title(f"Predicted Depreciation Curve — {brand} ({fuel}, {transmission})\n"
              f"Fixed km_driven={km_driven}, owner={owner}")
    plt.xlabel("Vehicle Age (years)")
    plt.ylabel("Predicted Selling Price (₹)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/depreciation_vs_age.png")
    plt.close()
    return ages, prices


def depreciation_vs_km(brand="Maruti", fuel="Petrol", seller_type="Individual",
                        transmission="Manual", owner="First Owner", age=5):
    kms = np.arange(0, 220000, 10000)
    prices = [predict_price(brand, fuel, seller_type, transmission, owner, k, age)
              for k in kms]

    plt.figure(figsize=(8, 5))
    plt.plot(kms, prices, marker="o", color="#4C72B0", linewidth=2)
    plt.title(f"Predicted Price vs Km Driven — {brand} ({fuel}, {transmission})\n"
              f"Fixed age={age} yrs, owner={owner}")
    plt.xlabel("Km Driven")
    plt.ylabel("Predicted Selling Price (₹)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/depreciation_vs_km.png")
    plt.close()
    return kms, prices


def compare_brands_depreciation(brands=("Maruti", "Hyundai", "Honda", "Toyota"),
                                 fuel="Petrol", seller_type="Individual",
                                 transmission="Manual", owner="First Owner",
                                 km_driven=40000):
    ages = np.arange(0, 16)
    plt.figure(figsize=(9, 6))
    for brand in brands:
        prices = [predict_price(brand, fuel, seller_type, transmission, owner, km_driven, a)
                  for a in ages]
        # Normalize to % of value retained (age=0 baseline) for fair comparison
        base = prices[0]
        retained_pct = [p / base * 100 for p in prices]
        plt.plot(ages, retained_pct, marker="o", label=brand, linewidth=2)

    plt.title("Brand Value Retention Comparison (% of original predicted price)")
    plt.xlabel("Vehicle Age (years)")
    plt.ylabel("% Value Retained")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/brand_depreciation_comparison.png")
    plt.close()


if __name__ == "__main__":
    depreciation_vs_age()
    depreciation_vs_km()
    compare_brands_depreciation()
    print("Depreciation analysis plots saved to reports/")
