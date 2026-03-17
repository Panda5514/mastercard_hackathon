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

# 3. Calculate Concentration (to beat the "Mega-County" bias)
high_leverage_counts = target_tracts.group_by("County", "State").len(name="high_leverage_tracts")
total_counts = df.group_by("County", "State").len(name="total_tracts")

final_rankings = (
    high_leverage_counts
    .join(total_counts, on=["County", "State"])
    .with_columns(
        (pl.col("high_leverage_tracts") / pl.col("total_tracts") * 100).alias("concentration_pct")
    )
    # Filter for counties with at least 10 tracts to avoid tiny sample outliers
    .filter(pl.col("total_tracts") > 10) 
    .sort("concentration_pct", descending=True)
)

print("Top Locations by Problem Concentration:")
print(final_rankings.head(10))