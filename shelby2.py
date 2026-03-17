import polars as pl
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
# # 1. Isolate Shelby County and save to CSV
# df = pl.read_csv("mastercard_data.csv")

# # Ensure your column names match the CSV exactly (Spaces and Caps)
# shelby_df = df.filter(
#     (pl.col("County") == "Shelby County") & 
#     (pl.col("State") == "Tennessee")
# )

# shelby_df.write_csv("shelby_county_data.csv")
# print(f"Shelby County isolated: {shelby_df.shape[0]} tracts saved.")
shelby_df = pl.read_csv("shelby_county_data.csv")  # Reload to ensure clean state for ML part
shelby_df = shelby_df.filter((pl.col("Inclusive Growth Score") < 45) )  # Ensure target is present for ML
print(shelby_df.shape)  # Check how many tracts we have for modeling


# 2. Localized XGBoost: What drives Health in Shelby County specifically?
features = [
    'Net Occupancy Score', 'Residential Real Estate Value Score', 
    'Acres of Park Land Score', 'Affordable Housing Score', 
    'Internet Access Score', 'Travel Time to Work Score', 
    'New Businesses Score', 'Spend Growth Score', 
    'Small Business Loans Score', 'Minority/Women Owned Businesses Score', 
    'Labor Market Engagement Index Score', 'Commercial Diversity Score', 
    'Personal Income Score', 'Spending per Capita Score', 
    'Female Above Poverty Score', 'Gini Coefficient Score', 
    'Early Education Enrollment Score'
]
target = 'Health Insurance Coverage Score'

# Prepare data (dropping nulls for the local model)
ml_df = shelby_df.select(features + [target]).drop_nulls()
X = ml_df.select(features).to_pandas()
y = ml_df.select(target).to_pandas()

# Train model
local_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.02)
local_model.fit(X, y)

# Output Importance
importances = dict(zip(features, local_model.feature_importances_))
print("\n--- Shelby County Specific Feature Importance using xgboost---")
for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
    print(f"{feat}: {imp:.4f}")

# ml_df = shelby_df.select(features + [target]).drop_nulls()
# X = ml_df.select(features).to_pandas()
# y = ml_df.select(target).to_pandas()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_scaled, y)

# 2. Lasso (Helps identify truly redundant features)
lasso = Lasso(alpha=0.1)
lasso.fit(X_scaled, y)

print("--- Linear Regression Coefficients ---")
for feat, coef in sorted(zip(features, lr.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
    print(f"{feat}: {coef:.4f}")

print("\n--- Lasso Coefficients (Feature Selection) ---")
for feat, coef in sorted(zip(features, lasso.coef_), key=lambda x: abs(x[1]), reverse=True):
    print(f"{feat}: {coef:.4f}")


ml_df = shelby_df.select(features + [target]).drop_nulls()
X = ml_df.select(features).to_pandas()
y = ml_df.select(target).to_pandas().values.ravel()

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

print("--- Random Forest Feature Importance ---")
for feat, imp in sorted(zip(features, rf.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"{feat}: {imp:.4f}")



ml_df = shelby_df.select(features + [target]).drop_nulls()
X = ml_df.select(features).to_pandas()
y = ml_df.select(target).to_pandas().values.ravel()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

