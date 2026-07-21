# 🚗 Used Vehicle Resale Price Prediction with Depreciation Analysis

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-006400)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Model-9ACD32)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](#license)

An end-to-end **regression machine learning project** that predicts the resale price of
used cars and visualizes **how predicted price depreciates with age and mileage** —
instead of returning just a single point estimate like most similar projects.

📊 Trains and compares **12 regression models** · 📉 Includes a **depreciation curve
module** · 🚀 Deployed as an interactive **Streamlit web app**

**Dataset:** [CAR DETAILS FROM CAR DEKHO](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) (Kaggle)

---

## 📑 Table of Contents
- [Demo](#-demo)
- [What Makes This Project Different](#-what-makes-this-project-different)
- [Project Structure](#-project-structure)
- [Models Trained & Compared](#-models-trained--compared)
- [Results](#-results-on-this-dataset-test-set-sorted-by-r²)
- [How to Run](#️-how-to-run)
- [Deployment](#️-deployment)
- [Tech Stack](#️-tech-stack)
- [Possible Extensions](#-possible-extensions)
- [License](#license)

---

## 🎬 Demo

The app has 3 interactive tabs:

| Tab | What it does |
|---|---|
| 💰 **Price Prediction** | Enter vehicle details in the sidebar → get an instant predicted resale price |
| 📉 **Depreciation Curves** | See how predicted price changes as age (0–15 yrs) and km driven (0–220,000) increase, plus a brand-vs-brand value retention comparison |
| 📊 **Model Comparison** | Full metrics table + R² bar chart comparing all 12 trained regression models |

---

## 🔑 What Makes This Project Different

Most resale-price projects stop at "predict one number." This project adds a
**depreciation curve module**: fixing a vehicle's brand/fuel/transmission, it sweeps
age and km driven through the trained model to plot how predicted value decays over
time — turning a single prediction into an interpretable trend a buyer or seller can
actually reason about.

---

## 📁 Project Structure

```
vehicle-price-prediction/
├── data/
│   ├── car_data.csv                 # raw dataset
│   └── car_data_processed.csv       # cleaned + feature-engineered
├── src/
│   ├── eda_preprocessing.py         # EDA + cleaning + feature engineering
│   ├── train_models.py              # trains & compares 12 regression models
│   └── depreciation_analysis.py     # depreciation curve generation
├── models/
│   ├── best_model.pkl               # best pipeline (preprocessing + model)
│   ├── all_models.pkl               # all 12 trained pipelines
│   └── best_model_name.json
├── reports/
│   ├── price_distribution.png
│   ├── price_vs_age.png
│   ├── price_vs_km.png
│   ├── categorical_boxplots.png
│   ├── top_brands.png
│   ├── model_comparison.csv / .json
│   ├── model_comparison_chart.png
│   ├── model_comparison_rmse.png
│   ├── depreciation_vs_age.png
│   ├── depreciation_vs_km.png
│   └── brand_depreciation_comparison.png
├── app.py                           # Streamlit deployment app
├── requirements.txt
└── README.md
```

---

## 🧠 Models Trained & Compared

| # | Model |
|---|---|
| 1 | Linear Regression |
| 2 | Ridge Regression |
| 3 | Lasso Regression |
| 4 | Polynomial Regression (degree 2) |
| 5 | Decision Tree Regressor |
| 6 | Random Forest Regressor |
| 7 | Gradient Boosting Regressor |
| 8 | XGBoost Regressor |
| 9 | LightGBM Regressor |
| 10 | Support Vector Regressor (RBF) |
| 11 | K-Nearest Neighbors Regressor |
| 12 | Neural Network (MLPRegressor) |

**Target transform:** `selling_price` is log1p-transformed before training (right-skewed
target) and inverse-transformed (`expm1`) before evaluation/prediction, so all reported
metrics are in real ₹ terms.

**Metrics computed for every model:** R², Adjusted R², MAE, RMSE, and 5-fold
cross-validated R² (mean ± std) — not just a single train/test split.

---

## 📈 Results (Test Set, sorted by R²)

| Model | R² | Adj. R² | RMSE | CV R² (mean) |
|---|---|---|---|---|
| **Polynomial Regression (deg2)** 🏆 | 0.787 | 0.784 | 174,449 | 0.772 |
| SVR (RBF) | 0.778 | 0.776 | 177,830 | 0.767 |
| Neural Net (MLP) | 0.778 | 0.776 | 177,970 | 0.762 |
| XGBoost | 0.751 | 0.748 | 188,582 | 0.784 |
| Ridge Regression | 0.747 | 0.745 | 189,799 | 0.767 |
| Linear Regression | 0.747 | 0.744 | 190,034 | 0.767 |
| Lasso Regression | 0.740 | 0.737 | 192,642 | 0.765 |
| LightGBM | 0.738 | 0.735 | 193,306 | 0.764 |
| Gradient Boosting | 0.737 | 0.734 | 193,672 | 0.779 |
| KNN Regressor | 0.718 | 0.715 | 200,635 | 0.740 |
| Random Forest | 0.705 | 0.702 | 205,243 | 0.761 |
| Decision Tree | 0.547 | 0.543 | 254,048 | 0.693 |

Full table with MAE + CV std: [`reports/model_comparison.csv`](reports/model_comparison.csv)

> **Note:** on this dataset (small — ~3,500 rows after cleaning, few numeric features),
> simple/kernel models (Polynomial, SVR, MLP) edge out tree ensembles. Dataset size and
> feature richness matter as much as model complexity. XGBoost has the best
> cross-validated R², suggesting it generalizes most consistently even though its
> single test-split R² is slightly lower — a good talking point on train/test-split vs.
> cross-validated performance.

---

## ▶️ How to Run

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/vehicle-price-prediction.git
cd vehicle-price-prediction
```

### 2. Create a virtual environment & install dependencies
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. (Optional) Retrain the models from scratch
```bash
python src/eda_preprocessing.py     # cleans data, saves EDA plots
python src/train_models.py          # trains & compares all 12 models
python src/depreciation_analysis.py # generates depreciation curve plots
```
This regenerates everything in `reports/` and `models/`. Skip this step if you just
want to run the app using the pre-trained models already included in the repo.

### 4. Launch the app
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`.

---

## ☁️ Deployment

### Option A — Streamlit Community Cloud (recommended, free)
1. Push this repo to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Point it to your repo, branch, and `app.py`
4. Deploy — live in ~2 minutes

### Option B — Hugging Face Spaces
1. Create a new Space → SDK: **Streamlit**
2. Upload this repo's contents (including `models/` and `reports/`)
3. Space auto-builds and deploys

> ⚠️ Make sure `models/*.pkl` and `reports/model_comparison.csv` are committed to the
> repo (they're the artifacts the app loads at runtime) — don't `.gitignore` them.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data & ML | pandas, numpy, scikit-learn, XGBoost, LightGBM |
| Visualization | matplotlib, seaborn, Streamlit native charts |
| Deployment | Streamlit Community Cloud / Hugging Face Spaces |

---

## 📌 Possible Extensions
- Add SHAP feature-importance explanations per prediction
- Combine with the "Used Bikes Prices in India" dataset for a second vehicle category
- Add a live currency/inflation adjustment toggle
- Swap Streamlit for a Flask/FastAPI + React frontend for more UI control

---

## License

This project is released under the [MIT License](LICENSE) — free to use, modify, and
distribute for personal or academic purposes.

---

## 🙌 Acknowledgements
- Dataset by [nehalbirla on Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho), sourced from CarDekho listings
