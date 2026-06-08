import pandas as pd

# ── Load your best file ───────────────────────────────────────────────────
df = pd.read_csv("antimicrobial_resistance_csv.csv")
print(f"Loaded: {df.shape}")

# ── Standardise column names ──────────────────────────────────────────────
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# ── Check countries ───────────────────────────────────────────────────────
print("\nCountries in dataset:")
print(df["country"].value_counts().head(10))

# ── Filter India (or keep all if no India rows) ───────────────────────────
india = df[df["country"].str.contains("India|IND", case=False, na=False)]

if len(india) == 0:
    print("\n⚠️  No India rows — using full dataset")
    india = df.copy()
    india["country"] = "Global"
else:
    print(f"\n✅ India rows: {len(india)}")

# ── Build master schema from YOUR columns ─────────────────────────────────
# This dataset uses organism instead of pathogen
# and has resistance by drug class (0/1) instead of resistance_%
# So we convert it

# Drug class columns (0 = sensitive, 1 = resistant)
class_cols = [c for c in india.columns if c.startswith("class_")]
print(f"\nDrug class columns found: {len(class_cols)}")
print(class_cols)

# Melt wide → long format
# Each row becomes: organism + drug_class + resistant(0/1)
id_cols = ["isolate_id", "organism", "country", "collection_year",
           "total_amr_genes", "total_resistance_classes"]
id_cols = [c for c in id_cols if c in india.columns]

melted = india.melt(
    id_vars=id_cols,
    value_vars=class_cols,
    var_name="antibiotic_class",
    value_name="resistant"
)

# Clean antibiotic class names
melted["antibiotic_class"] = melted["antibiotic_class"].str.replace("class_", "")

# Convert 0/1 to resistance percentage (for single isolate: 0% or 100%)
melted["resistance_pct"] = melted["resistant"] * 100

# Rename to master schema
melted = melted.rename(columns={
    "organism":         "pathogen",
    "collection_year":  "year",
    "antibiotic_class": "antibiotic",
})

# Add missing master schema columns
melted["source"]       = "Kaggle_AMR"
melted["state"]        = "National"
melted["district"]     = "Unknown"
melted["infection_type"] = "Unknown"
melted["lat"]          = 20.5937
melted["lng"]          = 78.9629

# Keep master schema columns only
keep = ["country", "state", "district", "year", "pathogen",
        "antibiotic", "resistance_pct", "infection_type",
        "source", "lat", "lng"]
keep = [c for c in keep if c in melted.columns]
final = melted[keep].dropna(subset=["pathogen", "antibiotic", "year"])

print(f"\n✅ Final rows after melt: {len(final)}")
print(f"Pathogens: {final['pathogen'].nunique()}")
print(f"Antibiotics: {final['antibiotic'].nunique()}")
print(f"Years: {sorted(final['year'].dropna().unique().astype(int).tolist())}")

final.to_csv("kaggle_india_clean.csv", index=False)
print("\n✅ Saved: kaggle_india_clean.csv")
