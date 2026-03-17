import polars as pl
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

df = pl.read_csv("mastercard_data.csv")
df = df.filter(pl.col("Health Insurance Coverage Score").is_not_null() & (pl.col("Inclusive Growth Score") < 45))
# ... (same features/target) ...
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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Simple Deep Learning model
mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
mlp.fit(X_scaled, y)

# NN's don't have "importance" attributes, so we use Permutation Importance
# (It breaks one column at a time to see how much the error increases)
results = permutation_importance(mlp, X_scaled, y, n_repeats=10, random_state=42)

print("--- Neural Network (Permutation Importance) ---")
for i in results.importances_mean.argsort()[::-1]:
    print(f"{features[i]}: {results.importances_mean[i]:.4f}")