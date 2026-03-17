import polars as pl
import xgboost as xgb

# 1. Load Data
df = pl.read_csv("mastercard_data.csv")
df = df.filter(pl.col("Health Insurance Coverage Score").is_not_null() & (pl.col("Inclusive Growth Score") < 45))
# 2. Define the broad levers (excluding the target and metadata)
# We focus on the 'Score' columns to see which categories drive the master IGS
lever_features = [
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

# 3. Prep and Train
ml_df = df.select(lever_features + [target]).drop_nulls()
X = ml_df.select(lever_features).to_pandas()
y = ml_df.select(target).to_pandas()

model_all = xgb.XGBRegressor(n_estimators=100)
model_all.fit(X, y)

# 4. Output Importance
importances = dict(zip(lever_features, model_all.feature_importances_))
print("\nGlobal Feature Importance (What drives the Health Insurance Coverage Score?):")
for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
    print(f"{feat}: {imp:.4f}")