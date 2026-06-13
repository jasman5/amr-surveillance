"""
prep_atlas.py — ATLAS (Pfizer/Vivli) AMR Dataset Preprocessor
================================================================
The raw atlas_vivli_2004_2024.csv has 1,011,168 rows × 127 columns —
too large to load directly in Streamlit on every rerun.

This script extracts the India subset (17,327 isolates, 2004-2024)
and pre-aggregates it into 4 small summary CSVs for the dashboard:

  Dataset/ATLAS/atlas_yearly_trend.csv     — % resistant by species/antibiotic/year (REAL multi-year trend)
  Dataset/ATLAS/atlas_icu_comparison.csv   — ICU vs Non-ICU resistance by antibiotic
  Dataset/ATLAS/atlas_gene_prevalence.csv  — resistance gene detection counts
  Dataset/ATLAS/atlas_heatmap.csv          — species × antibiotic % resistant (2020-2024)

Run once after placing atlas_vivli_2004_2024.csv in Dataset/ATLAS/
    python prep_atlas.py
"""

import pandas as pd
import os

SRC = "Dataset/ATLAS/atlas_vivli_2004_2024.csv"
OUT = "Dataset/ATLAS"

TOP_ABX = [
    "Ciprofloxacin", "Meropenem", "Imipenem", "Amikacin", "Ceftazidime",
    "Piperacillin tazobactam", "Colistin", "Levofloxacin", "Cefepime",
    "Gentamicin", "Trimethoprim sulfa", "Tigecycline",
]

GENE_COLS = ["NDM", "OXA", "KPC", "CTX-M-1", "CTX-M-9", "TEM", "SHV", "VIM", "IMP", "CMY2"]


def main():
    if not os.path.exists(SRC):
        print(f"ERROR: {SRC} not found.")
        return

    print("Loading ATLAS dataset (this may take a minute)...")
    df = pd.read_csv(SRC, low_memory=False)
    india = df[df["Country"] == "India"].copy()
    print(f"India isolates: {len(india)}")

    top_species = india["Species"].value_counts().head(5).index.tolist()
    print(f"Top species: {top_species}")

    # ── 1. Yearly resistance trend (real multi-year data) ──────────────────
    records = []
    for ab in TOP_ABX:
        col = ab + "_I"
        if col not in india.columns:
            continue
        sub = india[india["Species"].isin(top_species)][["Species", "Year", col]].dropna(subset=[col])
        sub = sub[sub[col].isin(["Susceptible", "Intermediate", "Resistant"])]
        for (sp, yr), grp in sub.groupby(["Species", "Year"]):
            n = len(grp)
            r = (grp[col] == "Resistant").sum()
            records.append({
                "Species": sp, "Antibiotic": ab, "Year": int(yr),
                "N": n, "Resistant": int(r),
                "PercentResistant": round(100 * r / n, 1),
            })
    yearly = pd.DataFrame(records)
    yearly.to_csv(f"{OUT}/atlas_yearly_trend.csv", index=False)
    print(f"Saved atlas_yearly_trend.csv ({len(yearly)} rows)")

    # ── 2. ICU vs Non-ICU comparison ────────────────────────────────────────
    india["is_icu"] = india["Speciality"].str.contains("ICU", na=False)
    icu_records = []
    for ab in TOP_ABX:
        col = ab + "_I"
        if col not in india.columns:
            continue
        sub = india[india[col].isin(["Susceptible", "Intermediate", "Resistant"])]
        for icu, grp in sub.groupby("is_icu"):
            n = len(grp)
            r = (grp[col] == "Resistant").sum()
            icu_records.append({
                "Antibiotic": ab,
                "Setting": "ICU" if icu else "Non-ICU",
                "N": n, "Resistant": int(r),
                "PercentResistant": round(100 * r / n, 1),
            })
    icu_df = pd.DataFrame(icu_records)
    icu_df.to_csv(f"{OUT}/atlas_icu_comparison.csv", index=False)
    print(f"Saved atlas_icu_comparison.csv ({len(icu_df)} rows)")

    # ── 3. Resistance gene prevalence ───────────────────────────────────────
    gene_records = []
    for g in GENE_COLS:
        if g not in india.columns:
            continue
        n_pos = india[g].notna().sum()
        gene_records.append({"Gene": g, "Detections": int(n_pos)})
    gene_df = pd.DataFrame(gene_records).sort_values("Detections", ascending=False)
    gene_df.to_csv(f"{OUT}/atlas_gene_prevalence.csv", index=False)
    print(f"Saved atlas_gene_prevalence.csv ({len(gene_df)} rows)")

    # ── 4. Species × Antibiotic heatmap (recent years 2020-2024) ────────────
    recent = india[india["Year"] >= 2020]
    heat_records = []
    for sp in top_species:
        for ab in TOP_ABX:
            col = ab + "_I"
            if col not in recent.columns:
                continue
            sub = recent[recent["Species"] == sp][col].dropna()
            sub = sub[sub.isin(["Susceptible", "Intermediate", "Resistant"])]
            if len(sub) >= 10:
                r = (sub == "Resistant").sum()
                heat_records.append({
                    "Species": sp, "Antibiotic": ab,
                    "PercentResistant": round(100 * r / len(sub), 1),
                    "N": len(sub),
                })
    heat_df = pd.DataFrame(heat_records)
    heat_df.to_csv(f"{OUT}/atlas_heatmap.csv", index=False)
    print(f"Saved atlas_heatmap.csv ({len(heat_df)} rows)")

    print("\nDone. Dashboard Tab 7 (ATLAS Global) will now load these summaries.")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    main()
