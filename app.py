"""
Streamlit App — Used Vehicle Resale Price Prediction with Depreciation Analysis
Run locally:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

st.set_page_config(page_title="Vehicle Resale Price Predictor", page_icon="🚗", layout="wide")

MODEL_DIR = "models"
REPORT_DIR = "reports"

BRANDS = ['Audi', 'BMW', 'Chevrolet', 'Datsun', 'Fiat', 'Ford', 'Honda', 'Hyundai',
          'Mahindra', 'Maruti', 'Mercedes-Benz', 'Nissan', 'Other', 'Renault',
          'Skoda', 'Tata', 'Toyota', 'Volkswagen']
FUELS = ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric']
SELLER_TYPES = ['Individual', 'Dealer', 'Trustmark Dealer']
TRANSMISSIONS = ['Manual', 'Automatic']
OWNERS = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car']


@st.cache_resource
def load_model():
    return joblib.load(f"{MODEL_DIR}/best_model.pkl")


@st.cache_data
def load_comparison():
    return pd.read_csv(f"{REPORT_DIR}/model_comparison.csv", index_col=0)


@st.cache_data
def load_best_model_name():
    with open(f"{MODEL_DIR}/best_model_name.json") as f:
        return json.load(f)["best_model"]


def predict_price(model, brand, fuel, seller_type, transmission, owner, km_driven, age):
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
    return float(np.expm1(pred_log))


# ---------------- Sidebar Inputs ----------------
st.sidebar.header("🚘 Vehicle Details")
brand = st.sidebar.selectbox("Brand", BRANDS, index=BRANDS.index("Maruti"))
fuel = st.sidebar.selectbox("Fuel Type", FUELS)
transmission = st.sidebar.selectbox("Transmission", TRANSMISSIONS)
seller_type = st.sidebar.selectbox("Seller Type", SELLER_TYPES)
owner = st.sidebar.selectbox("Owner", OWNERS)
year = st.sidebar.slider("Year of Purchase", 1995, 2026, 2018)
km_driven = st.sidebar.number_input("Km Driven", min_value=0, max_value=400000, value=40000, step=1000)

age = 2026 - year

st.title("🚗 Used Vehicle Resale Price Predictor")
st.caption("Regression-based ML project · trained on CarDekho used vehicle listings · "
           "compares 12 regression models · includes depreciation curve analysis")

tab1, tab2, tab3 = st.tabs(["💰 Price Prediction", "📉 Depreciation Curves", "📊 Model Comparison"])

model = load_model()
best_model_name = load_best_model_name()

# ---------------- TAB 1: Prediction ----------------
with tab1:
    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.subheader("Predict Selling Price")
        if st.button("Predict Price", type="primary", use_container_width=True):
            price = predict_price(model, brand, fuel, seller_type, transmission,
                                   owner, km_driven, age)
            st.metric("Estimated Resale Price", f"₹ {price:,.0f}")
            st.caption(f"Predicted using best model: **{best_model_name}**")
        else:
            st.info("Set vehicle details in the sidebar and click **Predict Price**.")

    with col2:
        st.subheader("Current Configuration")
        st.write(pd.DataFrame([{
            "Brand": brand, "Fuel": fuel, "Transmission": transmission,
            "Seller Type": seller_type, "Owner": owner,
            "Year": year, "Age (yrs)": age, "Km Driven": km_driven
        }]).T.rename(columns={0: "Value"}))

# ---------------- TAB 2: Depreciation Curves ----------------
with tab2:
    st.subheader("How does predicted price change with Age and Mileage?")
    st.write("Fixing all other features to your current sidebar selection, "
             "the model predicts price across a range of ages and km-driven values — "
             "showing the *depreciation curve* instead of just one point estimate.")

    c1, c2 = st.columns(2)

    with c1:
        ages = np.arange(0, 16)
        prices_age = [predict_price(model, brand, fuel, seller_type, transmission,
                                     owner, km_driven, a) for a in ages]
        chart_df = pd.DataFrame({"Age (years)": ages, "Predicted Price (₹)": prices_age})
        st.line_chart(chart_df.set_index("Age (years)"))
        st.caption(f"Price vs Age, fixed km_driven = {km_driven:,}")

    with c2:
        kms = np.arange(0, 220000, 10000)
        prices_km = [predict_price(model, brand, fuel, seller_type, transmission,
                                    owner, k, age) for k in kms]
        chart_df2 = pd.DataFrame({"Km Driven": kms, "Predicted Price (₹)": prices_km})
        st.line_chart(chart_df2.set_index("Km Driven"))
        st.caption(f"Price vs Km Driven, fixed age = {age} yrs")

    st.divider()
    st.subheader("Brand Value Retention Comparison")
    compare_brands = st.multiselect("Compare brands", BRANDS,
                                     default=["Maruti", "Hyundai", "Honda", "Toyota"])
    if compare_brands:
        retention_df = pd.DataFrame({"Age (years)": ages})
        for b in compare_brands:
            p = [predict_price(model, b, fuel, seller_type, transmission,
                                owner, km_driven, a) for a in ages]
            base = p[0] if p[0] != 0 else 1
            retention_df[b] = [x / base * 100 for x in p]
        st.line_chart(retention_df.set_index("Age (years)"))
        st.caption("% of original predicted value retained as vehicle ages (normalized at age=0)")

# ---------------- TAB 3: Model Comparison ----------------
with tab3:
    st.subheader("Regression Model Performance Comparison")
    comparison_df = load_comparison()
    st.dataframe(
        comparison_df.style.highlight_max(subset=["R2", "Adjusted_R2", "CV_R2_mean"], color="#c6f6d5")
                             .highlight_min(subset=["MAE", "RMSE"], color="#c6f6d5"),
        use_container_width=True
    )
    st.bar_chart(comparison_df["R2"].sort_values())
    st.caption(f"🏆 Best performing model on this dataset: **{best_model_name}**")

st.divider()
st.caption("Built as an end-to-end regression ML project: EDA → preprocessing → "
           "12-model comparison → depreciation analysis → Streamlit deployment.")
