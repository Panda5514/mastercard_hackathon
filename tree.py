from sklearn.ensemble import RandomForestRegressor
import polars as pl

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
y = ml_df.select(target).to_pandas().values.ravel()

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

print("--- Random Forest Feature Importance ---")
for feat, imp in sorted(zip(features, rf.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"{feat}: {imp:.4f}")