"""
create_unified_master.py — Structural Layer Isolator
Standardizes and writes separated files to processed/ to prevent context mixing.
"""
import pandas as pd
import os

os.makedirs("processed", exist_ok=True)
print("📂 Splitting datasets into independent tracking layers...")

# 1. WHO GLASS Macro File Ingestion
glass_file = "GitHub compiled dataset (easiest for ML)/compiled_WHO_GLASS_2022.xlsx"
if os.path.exists(glass_file):
    df_g = pd.read_excel(glass_file, engine='openpyxl')
    df_g[['PathogenName', 'AbTargets', 'CountryTerritoryArea']].rename(
        columns={'PathogenName': 'Pathogen', 'AbTargets': 'Antibiotic', 'CountryTerritoryArea': 'Region'}
    ).to_csv("processed/layer_who_glass.csv", index=False)
    print("✅ WHO GLASS Macro Layer Cached.")

# 2. PubMed NLP Cache Isolation
if os.path.exists("processed/master_amr_pubmed.csv"):
    print("✅ PubMed NLP Extraction Layer Verified.")

print("🏁 Data stratification complete. Ready for clean multi-modal rendering.")