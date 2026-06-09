"""
merge_icmr.py — ICMR AMR Surveillance data pipeline
Reads the 4 .dta tables, merges them, decodes every numeric ID into
human-readable labels using Stata's embedded value-label dictionaries.

Run: python merge_icmr.py
Output: processed/master_amr_icmr.csv
"""

import pandas as pd
import pyreadstat
import os

os.makedirs("processed", exist_ok=True)

# ── 1. Load all 4 tables ──────────────────────────────────────────────────────
print("Loading Stata files...")
BASE = "ICMR Data Portal (primary source)/"
patient,  pm   = pyreadstat.read_dta(BASE + "1751622124_patient_repos.dta",        apply_value_formats=False)
hospital, hm   = pyreadstat.read_dta(BASE + "1751622145_hospital_repos.dta",       apply_value_formats=False)
sample,   sm   = pyreadstat.read_dta(BASE + "1751622161_sample_repos.dta",         apply_value_formats=False)
suscept,  supm = pyreadstat.read_dta(BASE + "1751622178_susceptibility_repos.dta", apply_value_formats=False)

print(f"  Patient: {patient.shape} | Hospital: {hospital.shape} | Sample: {sample.shape} | Suscept: {suscept.shape}")

# ── 2. Merge ──────────────────────────────────────────────────────────────────
print("\nMerging tables...")
df = suscept.merge(sample,   on=["sample_id", "patient_id", "hospital_patient_rel_id"], how="left")
df = df.merge(hospital, on=["patient_id", "hospital_patient_rel_id"],                   how="left")
df = df.merge(patient,  on="patient_id",                                                 how="left")
print(f"  Merged shape: {df.shape}")

# ── 3. Decode IDs using Stata label dictionaries ──────────────────────────────
df["organism_name"]      = df["organism_id"].map(supm.variable_value_labels["organism_id"])
df["antibiotic_name"]    = df["antibiotic_id"].map(supm.variable_value_labels["antibiotic_id"])
df["resistance"]         = df["remarks"].map(supm.variable_value_labels["remarks"]).str.capitalize()
df["panel_name"]         = df["panel_id"].map(supm.variable_value_labels["panel_id"])
df["id_method_name"]     = df["identification_method"].map(supm.variable_value_labels["identification_method"])
df["ward_name"]          = df["ward_type"].map(hm.variable_value_labels["ward_type"])
df["dept_name"]          = df["hospital_dept_id"].map(hm.variable_value_labels["hospital_dept_id"])
df["infection_type"]     = df["infection_type_id"].map(hm.variable_value_labels["infection_type_id"])
df["gender_label"]       = df["gender"].map(pm.variable_value_labels["gender"])
df["state_name"]         = df["state"].map(pm.variable_value_labels["state"])
df["location_type_name"] = df["location_type"].map(pm.variable_value_labels["location_type"])
df["sample_type_name"]   = df["sample_type_id"].map(sm.variable_value_labels["sample_type_id"])

# ── 4. Binary resistance flag for model training (Intermediate excluded) ──────
df["is_resistant"] = df["remarks"].map({1.0: 0, 2.0: None, 3.0: 1})

# ── 5. Parse dates ────────────────────────────────────────────────────────────
df["collection_date"] = pd.to_datetime(df["collection_date"], errors="coerce")
df["admission_date"]  = pd.to_datetime(df["admission_date"],  errors="coerce")
df["collection_year"] = df["collection_date"].dt.year

# ── 6. Save ───────────────────────────────────────────────────────────────────
df.to_csv("processed/master_amr_icmr.csv", index=False)

# ── 7. Summary ────────────────────────────────────────────────────────────────
print(f"\nSaved: processed/master_amr_icmr.csv  ({df.shape[0]} rows × {df.shape[1]} cols)")
print("\nOrganisms:")
print(df["organism_name"].value_counts().to_string())
print("\nTop antibiotics:")
print(df["antibiotic_name"].value_counts().head(10).to_string())
print("\nResistance breakdown:")
print(df["resistance"].value_counts().to_string())
print("\nStates:")
print(df["state_name"].value_counts().to_string())
print("\nWard types:")
print(df["ward_name"].value_counts().to_string())