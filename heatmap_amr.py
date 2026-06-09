import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("processed/master_amr_icmr.csv")

# Remove missing values
df = df.dropna(
    subset=[
        "organism_name",
        "antibiotic_name",
        "is_resistant"
    ]
)

# Top organisms
top_organisms = (
    df["organism_name"]
    .value_counts()
    .head(5)
    .index
)

filtered = df[df["organism_name"].isin(top_organisms)]

# Create pivot table
pivot = pd.pivot_table(
    filtered,
    values="is_resistant",
    index="organism_name",
    columns="antibiotic_name",
    aggfunc="mean"
)

# Plot
plt.figure(figsize=(15, 7))

sns.heatmap(
    pivot,
    cmap="Reds",
    annot=False
)

plt.title("AMR Heatmap")
plt.xlabel("Antibiotics")
plt.ylabel("Organisms")

plt.tight_layout()
plt.show()