import polars as pl
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # Required for MICE
from sklearn.impute import IterativeImputer

# 1. Load your raw data
df = pl.read_csv("mastercard_data.csv")

# 2. Select the numerical features you've been using for the models
features_to_impute = [
    'Net Occupancy Score', 'Residential Real Estate Value Score', 
    'Acres of Park Land Score', 'Affordable Housing Score', 
    'Internet Access Score', 'Travel Time to Work Score', 
    'New Businesses Score', 'Spend Growth Score', 
    'Small Business Loans Score', 'Minority/Women Owned Businesses Score', 
    'Labor Market Engagement Index Score', 'Commercial Diversity Score', 
    'Personal Income Score', 'Spending per Capita Score', 
    'Female Above Poverty Score', 'Gini Coefficient Score', 
    'Early Education Enrollment Score', 'Health Insurance Coverage Score'
]

# We must use Pandas for the imputer
df_pd = df.select(features_to_impute).to_pandas()

# 3. Initialize MICE (IterativeImputer)
# max_iter=10 is standard for the "chained" equations to converge
mice_imputer = IterativeImputer(max_iter=10, random_state=42)

# 4. Perform the Imputation
df_imputed_values = mice_imputer.fit_transform(df_pd)

# 5. Convert back to Polars and save
df_final = pl.from_pandas(pd.DataFrame(df_imputed_values, columns=features_to_impute))

# Join back metadata (County, State) if you need them for grouping later
metadata = df.select(['County', 'State'])
df_complete = pl.concat([metadata, df_final], how="horizontal")

df_complete.write_csv("mastercard_data_imputed.csv")
print("MICE Imputation Complete! Saved to mastercard_data_imputed.csv")