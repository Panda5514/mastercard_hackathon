import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Load data
df = pd.read_csv("shelby_county_data.csv")
# Filter for the specific challenge criteria
df_filtered = df[df["Inclusive Growth Score"] < 45]

# 2. All factors from your original code
features = [
    'Acres of Park Land Score', 'Affordable Housing Score', 
    'Internet Access Score', 'Travel Time to Work Score', 'Spend Growth Score', 'Minority/Women Owned Businesses Score', 
    'Labor Market Engagement Index Score',
    'Personal Income Score', 'Spending per Capita Score', 
    'Female Above Poverty Score', 'Gini Coefficient Score', 
]
target = 'Health Insurance Coverage Score'
all_cols = features + [target]

# 3. Calculate full correlation matrix
# We drop nulls to ensure the math is consistent across all pairs
corr_matrix = df_filtered[all_cols].dropna().corr()

# 4. Generate the full heatmap
plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Optional: hides the duplicate top-half
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdBu_r', center=0, 
            annot_kws={"size": 8}, cbar_kws={"shrink": .8})
plt.title("Full Correlation Matrix (IGS < 45)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("Full_Correlation_Matrix.png")
print("Heatmap saved as 'Full_Correlation_Matrix.png'")

# 5. Extract the "Hidden Chains"
# This looks for relationships between any two factors that are > 0.5
print("\n--- Significant Inter-Factor Relationships (The 'Hidden' Dominoes) ---")
relationships = corr_matrix.unstack().reset_index()
relationships.columns = ['Factor_A', 'Factor_B', 'Correlation']

# Filter out self-correlations and duplicates
relationships = relationships[relationships['Factor_A'] != relationships['Factor_B']]
# Sort by strength
significant = relationships[relationships['Correlation'].abs() > 0.5].sort_values(by='Correlation', ascending=False)

# Remove duplicates (A-B and B-A)
significant['key'] = significant.apply(lambda x: "-".join(sorted([x['Factor_A'], x['Factor_B']])), axis=1)
significant = significant.drop_duplicates('key').drop('key', axis=1)

for idx, row in significant.iterrows():
    print(f"{row['Factor_A']} <---> {row['Factor_B']}: {row['Correlation']:.4f}")