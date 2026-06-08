import pandas as pd

# Load all 4 ICMR tables - setting convert_categoricals=False fixes the Stata value mapping bug
print("⏳ Loading Stata files...")
patient   = pd.read_stata("ICMR Data Portal (primary source)/1751622124_patient_repos.dta",   convert_categoricals=False)
hospital  = pd.read_stata("ICMR Data Portal (primary source)/1751622145_hospital_repos.dta",  convert_categoricals=False)
sample    = pd.read_stata("ICMR Data Portal (primary source)/1751622161_sample_repos.dta",    convert_categoricals=False)
suscept   = pd.read_stata("ICMR Data Portal (primary source)/1751622178_susceptibility_repos.dta", convert_categoricals=False)

print(f" Loaded! Patient: {patient.shape} | Hospital: {hospital.shape} | Sample: {sample.shape} | Susceptibility: {suscept.shape}")

# Merge: susceptibility → sample → hospital → patient
print(" Merging data streams...")
df = suscept.merge(sample,   on=["sample_id","patient_id","hospital_patient_rel_id"], how="left")
df = df.merge(hospital, on=["patient_id","hospital_patient_rel_id"], how="left")
df = df.merge(patient,  on="patient_id", how="left")

print(f" Merged Master Dataset Shape: {df.shape}")
print(f" Generated Columns: {df.columns.tolist()}")

# Save
df.to_csv("processed/master_amr_icmr.csv", index=False)
print("\n Success! Saved: processed/master_amr_icmr.csv")