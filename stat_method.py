import polars as pl
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load and Prep
df = pl.read_csv("mastercard_data.csv")
df = df.filter(pl.col("Health Insurance Coverage Score").is_not_null() & (pl.col("Inclusive Growth Score") < 45))
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

ml_df = df.select(features + [target]).drop_nulls()
X = ml_df.select(features).to_pandas()
y = ml_df.select(target).to_pandas()

# Scale data (Essential for Linear Models)
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