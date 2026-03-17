import polars as pl


# 1. Load the imputed data
df = pl.read_csv("mastercard_data.csv")

# 2. Isolate Shelby County, TN
# (Adjust 'County' and 'State' casing if you didn't change them to snake_case)
shelby_df = df.filter(
    (pl.col("County") == "Shelby County") & 
    (pl.col("State") == "Tennessee")
)

# 3. Neighborhood Isolation via Tract FIPS
# Note: These prefixes are common for these historical areas
# You can also filter by 'Place' if your dataset has a 'Memphis' city column
north_memphis = shelby_df.filter(pl.col("Census Tract FIPS code").cast(pl.Utf8).str.starts_with("4715700"))
orange_mound = shelby_df.filter(pl.col("Census Tract FIPS code").cast(pl.Utf8).str.starts_with("4715702"))
whitehaven = shelby_df.filter(pl.col("Census Tract FIPS code").cast(pl.Utf8).str.starts_with("47157022"))

# 4. Compare the "Levers" across neighborhoods
def get_neighborhood_stats(name, dff):
    return dff.select([
        pl.lit(name).alias("Neighborhood"),
        pl.col("Health Insurance Coverage Score").mean().alias("Health_Score"),
        pl.col("Female Above Poverty Score").mean().alias("Female_Stability"),
        pl.col("Internet Access Score").mean().alias("Internet_Readiness"),
        pl.col("Travel Time to Work Score").mean().alias("Time_Tax_Score"),
        pl.col("Minority/Women Owned Businesses Score").mean().alias("Trust_Anchor_Score")
    ])

comparison = pl.concat([
    get_neighborhood_stats("Shelby County (Avg)", shelby_df),
    get_neighborhood_stats("North Memphis", north_memphis),
    get_neighborhood_stats("Orange Mound", orange_mound),
    get_neighborhood_stats("Whitehaven", whitehaven)
])

print(comparison)