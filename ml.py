import xgboost as xgb
from sklearn.model_selection import train_test_split
# from init2 import target_tracts
import polars as pl

# 1. Load data
df = pl.read_csv("mastercard_data.csv")

# 2. Apply your 3-Stage Filter Logic
# Stage 1: The Crisis (Low IGS, Low Health Insurance)
# Stage 2: The Hidden Asset (High Minority/Women Biz, Low Loans)
# Stage 3: The Infrastructure (High Internet Access for Telehealth)
target_tracts = df.filter(
    (pl.col("Inclusive Growth Score") < 45) &
    (pl.col("Health Insurance Coverage Score") < pl.col("Health Insurance Coverage Score").quantile(0.2)) &
    (pl.col("Minority/Women Owned Businesses Score") > pl.col("Minority/Women Owned Businesses Score").median()) &
    (pl.col("Small Business Loans Score") < pl.col("Small Business Loans Score").quantile(0.3)) &
    (pl.col("Internet Access Score") > pl.col("Internet Access Score").median())
)
# Prepare data for the target tracts
# Goal: Predict Health Insurance Score using Economy/Infrastructure metrics
features = [
    "Small Business Loans Score", 
    "Internet Access Score", 
    "Travel Time to Work Score", 
    "Personal Income Score",
    "Commercial Diversity Score"
]
target = "Health Insurance Coverage Score"

# Drop rows with nulls in these specific columns for the model
ml_df = target_tracts.select(features + [target]).drop_nulls()

X = ml_df.select(features).to_pandas()
y = ml_df.select(target).to_pandas()

# Train a simple XGBoost Regressor
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05)
model.fit(X, y)

# Feature Importance: Which factor is the strongest predictor of health access?
importances = dict(zip(features, model.feature_importances_))
print("\nFeature Importance (The 'Levers'):")
for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
    print(f"{feat}: {imp:.4f}")


# import shap
# import matplotlib.pyplot as plt

# # 1. Calculate SHAP values
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)

# # 2. Visualize
# # This will show you if 'High' Travel Time leads to 'Low' Health Insurance
# shap.summary_plot(shap_values, X)