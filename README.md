# 🚗 Used Vehicle Resale Price Prediction with Depreciation Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vehicle-price-prediction-ml3.streamlit.app/)

**🔗 Live Demo:** [https://vehicle-price-prediction-ml3.streamlit.app/](https://vehicle-price-prediction-ml3.streamlit.app/)

An end-to-end regression machine learning project that predicts the resale price of used
cars and visualizes **how predicted price depreciates with age and mileage** — instead of
just returning a single point estimate like most similar projects.

Dataset: [CAR DETAILS FROM CAR DEKHO](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) (Kaggle)

---

## 🔑 What makes this project different
Most resale-price projects stop at "predict one number." This project adds a
**depreciation curve module**: fixing a vehicle's brand/fuel/transmission, it sweeps
age (0–15 yrs) and km driven (0–220,000 km) through the trained model to plot how
predicted value decays — plus a brand-vs-brand value-retention comparison.

---

## 📁 Project Structure
```
vehicle-price-prediction/
├── data/
│   ├── car_data.csv                 # raw uploaded dataset
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

### Results on this dataset (test set, sorted by R²)
| Model | R² | Adj. R² | RMSE | CV R² (mean) |
|---|---|---|---|---|
| Polynomial Regression (deg2) | 0.787 | 0.784 | 174,449 | 0.772 |
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

Full table with MAE + CV std: `reports/model_comparison.csv`

**Note:** on this particular dataset (small — ~3,500 rows after cleaning, few numeric
features), simple/kernel models (Polynomial, SVR, MLP) edge out tree ensembles. This
is a good talking point in a viva/interview — dataset size and feature richness matter
as much as model complexity. XGBoost has the best cross-validated R², suggesting it
generalizes most consistently even though its single test-split R² is slightly lower.

---

## ▶️ How to Run

### Try it live
No install needed — use the hosted app: **[vehicle-price-prediction-ml3.streamlit.app](https://vehicle-price-prediction-ml3.streamlit.app/)**

### Or run locally

#### 1. Install dependencies
```bash
pip install -r requirements.txt
```

#### 2. Run the pipeline (only needed if you want to retrain from scratch)
```bash
python src/eda_preprocessing.py     # cleans data, saves EDA plots
python src/train_models.py          # trains & compares all 12 models
python src/depreciation_analysis.py # generates depreciation curve plots
```
This regenerates everything in `reports/` and `models/`.

#### 3. Launch the app
```bash
streamlit run app.py
```
Opens at `http://localhost:8501` with three tabs:
- **Price Prediction** — input form → instant price estimate
- **Depreciation Curves** — interactive price-vs-age / price-vs-km charts + brand comparison
- **Model Comparison** — full metrics table + R² bar chart for all 12 models

---

## ☁️ Deployment

### Currently deployed on Streamlit Community Cloud
🔗 **Live app:** [https://vehicle-price-prediction-ml3.streamlit.app/](https://vehicle-price-prediction-ml3.streamlit.app/)

### Deploy your own copy

#### Option A — Streamlit Community Cloud (recommended, free)
1. Push this folder to a public GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → "New app"
3. Point it to your repo, branch, and `app.py`
4. Deploy — you'll get a public URL in ~2 minutes

#### Option B — Hugging Face Spaces
1. Create a new Space → SDK: Streamlit
2. Upload this folder's contents (including `models/` and `reports/`)
3. Space auto-builds and deploys

> ⚠️ Make sure `models/*.pkl` and `reports/model_comparison.csv` are committed to the
> repo (they're the artifacts the app loads at runtime) — don't `.gitignore` them.

---

## 🛠️ Tech Stack
- **Data/ML:** pandas, numpy, scikit-learn, XGBoost, LightGBM
- **Visualization:** matplotlib, seaborn, Streamlit native charts
- **Deployment:** Streamlit

---

## 📌 Possible Extensions
- Add SHAP feature-importance explanations per prediction
- Combine with the "Used Bikes Prices in India" dataset for a second vehicle category
- Add a live currency/inflation adjustment toggle
- Swap Streamlit for a Flask/FastAPI + React frontend for more UI control
