import polars as pl

# # 1. LOAD AND SAVE (Run this part once to get your CSV)
# df_original = pl.read_excel(
#     "Inclusive_Growth_Score_Data_Export_13-03-2026_070005.xlsx",
#     sheet_name="Compared to Urban-Rural",
#     engine="calamine",
#     read_options={"header_row": 1}
# )

# # Remove that empty 'N/A' column
# df_original = df_original.drop("N/A")

# # Save to CSV exactly as is (preserving spaces/caps)
# df_original.write_csv("mastercard_data.csv")


# 2. ANALYSIS (This is what you'll run most often)
df = pl.read_csv("mastercard_data.csv")

# Identify "High Leverage" tracts using EXACT column names
target_tracts = df.filter(
    # Score is low
    (pl.col("Inclusive Growth Score") < 45) & 
    
    # Health coverage is in the bottom 20%
    (pl.col("Health Insurance Coverage Score") < pl.col("Health Insurance Coverage Score").quantile(0.2)) & 
    
    # High presence of Minority/Women owned biz
    (pl.col("Minority/Women Owned Businesses Score") > pl.col("Minority/Women Owned Businesses Score").median()) &
    
    # Low access to small biz loans
    (pl.col("Small Business Loans Score") < pl.col("Small Business Loans Score").quantile(0.3))
)

# Group by County (since 'City' isn't in your list)
county_rankings = (
    target_tracts
    .group_by("County", "State")
    .len(name="tract_count")
    .sort("tract_count", descending=True)
)

print("Counties with the most high-leverage tracts:")
print(county_rankings.head(10))