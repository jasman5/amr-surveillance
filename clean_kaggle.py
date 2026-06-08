import pandas as pd

# Load dataset
df = pd.read_csv("antimicrobial_resistance_csv.csv")

print("Dataset Loaded:", df.shape)

# Clean column names
df.columns = df.columns.str.lower().str.strip()

# Important columns
print("\nColumns:")
print(df.columns.tolist())

# Show countries
if "country" in df.columns:
    print("\nCountries:")
    print(df["country"].value_counts().head(10))

# Drug resistance columns
class_cols = [c for c in df.columns if c.startswith("class_")]

print("\nDrug Classes Found:")
print(class_cols)

# Convert wide → long format
melted = df.melt(
    id_vars=["organism", "country", "collection_year"],
    value_vars=class_cols,
    var_name="antibiotic",
    value_name="resistant"
)

# Clean names
melted["antibiotic"] = melted["antibiotic"].str.replace("class_", "")

# Resistance %
melted["resistance_pct"] = melted["resistant"] * 100

# Rename
melted = melted.rename(columns={
    "organism": "pathogen",
    "collection_year": "year"
})

# Keep useful columns
final = melted[
    ["country", "year", "pathogen", "antibiotic", "resistance_pct"]
].copy()

# Fill missing values
final["country"] = final["country"].fillna("Global")
final["year"] = final["year"].fillna(2024)

print("\nFinal Dataset Shape:", final.shape)

print("\nPreview:")
print(final.head())

final.to_csv("kaggle_india_clean.csv", index=False)

print("\nSaved: kaggle_india_clean.csv")