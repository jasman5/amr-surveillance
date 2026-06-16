import pickle
import pandas as pd

with open("Dataset/processed/clinical_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("Dataset/processed/model_meta.pkl", "rb") as f:
    meta = pickle.load(f)

features = meta["clinical_features"]
org_inv  = {v:k for k,v in meta["organism"].items()}
atb_inv  = {v:k for k,v in meta["antibiotic"].items()}
ward_inv = {v:k for k,v in meta["ward"].items()}
inf_inv  = {v:k for k,v in meta["infection"].items()}
dept_inv = {v:k for k,v in meta["dept"].items()}
samp_inv = {v:k for k,v in meta["sample_type"].items()}

# Test 1: Known HIGH resistance combo from your data
# Klebsiella + Ciprofloxacin in ICU should be HIGH
test1 = {
    "age": 60,
    "gender": 2.0,
    "ward_type": float(ward_inv.get(list(ward_inv.keys())[0])),
    "infection_type_id": float(inf_inv.get(list(inf_inv.keys())[0])),
    "organism_id": float(org_inv.get("Klebsiella pneumoniae", list(org_inv.values())[0])),
    "hospital_dept_id": float(dept_inv.get(list(dept_inv.keys())[0])),
    "sample_type_id": float(samp_inv.get(list(samp_inv.keys())[0])),
    "antibiotic_id": float(atb_inv.get("Ciprofloxacin", list(atb_inv.values())[0])),
}

prob1 = model.predict_proba(pd.DataFrame([test1])[features])[0][1] * 100
print(f"Klebsiella + Ciprofloxacin → {prob1:.1f}% resistance (expect HIGH >60%)")

# Test 2: Print all organism names and antibiotic names available
print("\nOrganisms in model:", list(org_inv.keys()))
print("Antibiotics in model:", list(atb_inv.keys()))
print("Wards in model:", list(ward_inv.keys()))