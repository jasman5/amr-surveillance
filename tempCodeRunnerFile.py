"""
merge_icmr.py  —  ICMR AMR Surveillance data pipeline
Reads the 4 .dta tables, merges them, and decodes every numeric ID into
human-readable labels using the value-label dictionaries embedded in the
Stata files themselves (no hardcoded guesses).
"""

import pandas as pd
import pyreadstat
import os

os.makedirs("processed", exist_ok=True)

# ── 1. Load all 4 tables + their embedded label dictionaries ──────────────────
print("Loading Stata files...")
patient,  pm  = pyreadstat.read_dta("ICMR Data Portal (primary source)/1751622124_patient_repos.dta",   apply_value_formats=False)
hospital, hm  = pyreadstat.read_dta("ICMR Data Portal (primary source)/1751622145_hospital_repos.dta",  apply_value_formats=False)
sample,   sm  = pyreadstat.read_dta("ICMR Data Portal (primary source)/1751622161_sample_repos.dta",    apply_value_formats=False)
suscept,  supm = pyreadstat.read_dta("ICMR Data Portal (primary source)/1751622178_susceptibility_repos.dta", apply_value_formats=False)

print(f"  Patient   : {patient.shape}")
print(f"  Hospital  : {hospital.shape}")
print(f"  Sample    : {sample.shape}")
print(f"  Suscept   : {suscept.shape}")

# ── 2. Merge: susceptibility → sample → hospital → patient ───────────────────
print("\nMerging tables...")
df = suscept.merge(sample,   on=["sample_id", "patient_id", "hospital_patient_rel_id"], how="left")
df = df.merge(hospital, on=["patient_id", "hospital_patient_rel_id"],                  how="left")
df = df.merge(patient,  on="patient_id",                                                how="left")
print(f"  Merged shape: {df.shape}")

# ── 3. Pull label dictionaries straight from the Stata metadata ───────────────
# susceptibility file labels
organism_map   = supm.variable_value_labels["organism_id"]      # {1: 'Escherichia coli', ...}
antibiotic_map = supm.variable_value_labels["antibiotic_id"]    # {1: 'Amikacin', 2: 'Amikacin', ...}
remarks_map    = supm.variable_value_labels["remarks"]          # {1: 'susceptible', 2: 'Intermediate', 3: 'resistant'}
panel_map      = supm.variable_value_labels["panel_id"]
id_method_map  = supm.variable_value_labels["identification_method"]

# hospital file labels
ward_map       = hm.variable_value_labels["ward_type"]          # {1:'ICU', 2:'OPD', 3:'Ward'}
dept_map       = hm.variable_value_labels["hospital_dept_id"]
inftype_map    = hm.variable_value_labels["infection_type_id"]  # {1:'Community Acquired', ...}

# patient file labels
gender_map     = pm.variable_value_labels["gender"]             # {1:'female', 2:'male', ...}
state_map      = pm.variable_value_labels["state"]              # {1:'Andaman...', 2:'Andhra Pradesh', ...}
loc_map        = pm.variable_value_labels["location_type"]      # {1:'rural', 2:'urban'}

# sample file labels
sample_type_map = sm.variable_value_labels["sample_type_id"]

# ── 4. Decode every ID column into readable names ─────────────────────────────
# Note: antibiotic_id has duplicate names (odd IDs = disc diffusion, even = MIC)
# We keep the name but also store the original ID for the model.
df["organism_name"]      = df["organism_id"].map(organism_map)
df["antibiotic_name"]    = df["antibiotic_id"].map(antibiotic_map)
df["resistance"]         = df["remarks"].map(remarks_map).str.capitalize()   # Susceptible / Intermediate / Resistant
df["panel_name"]         = df["panel_id"].map(panel_map)
df["id_method_name"]     = df["identification_method"].map(id_method_map)
df["ward_name"]          = df["ward_type"].map(ward_map)
df["dept_name"]          = df["hospital_dept_id"].map(dept_map)
df["infection_type"]     = df["infection_type_id"].map(inftype_map)
df["gender_label"]       = df["gender"].map(gender_map)
df["state_name"]         = df["state"].map(state_map)
df["location_type_name"] = df["location_type"].map(loc_map)
df["sample_type_name"]   = df["sample_type_id"].map(sample_type_map)

# ── 5. Binary resistance flag (used by train_model.py) ───────────────────────
# 1 = Resistant, 0 = Susceptible, NaN = Intermediate (excluded from training)
df["is_resistant"] = df["remarks"].map({1.0: 0, 2.0: None, 3.0: 1})

# ── 6. Date parsing ───────────────────────────────────────────────────────────
df["collection_date"] = pd.to_datetime(df["collection_date"], errors="coerce")
df["admission_date"]  = pd.to_datetime(df["admission_date"],  errors="coerce")
df["collection_year"] = df["collection_date"].dt.year

# ── 7. Save ───────────────────────────────────────────────────────────────────
out = "processed/master_amr_icmr.csv"
df.to_csv(out, index=False)

# ── 8. Summary ────────────────────────────────────────────────────────────────
print(f"\nSaved: {out}")
print(f"Shape: {df.shape}")
print(f"\nOrganisms in data:")
print(df["organism_name"].value_counts().to_string())
print(f"\nAntibiotics tested:")
print(df["antibiotic_name"].value_counts().to_string())
print(f"\nResistance breakdown:")
print(df["resistance"].value_counts().to_string())
print(f"\nStates covered:")
print(df["state_name"].value_counts().to_string())
print(f"\nWard types:")
print(df["ward_name"].value_counts().to_string())