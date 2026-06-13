"""
train_model.py — Research-Grade Predictive Modeling Engine
Trains a RandomForestClassifier using Stratified 5-Fold Cross-Validation.
Includes antibiotic profiles directly into the feature array to prevent biological confounding.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

# UNIFIED WORKING DIRECTORY
TARGET_DIR = "Dataset/processed"
os.makedirs(TARGET_DIR, exist_ok=True)

print("🔬 Initializing Research-Grade ML Pipeline...")
baseline_path = os.path.join(TARGET_DIR, "master_amr_icmr.csv")

if not os.path.exists(baseline_path):
    raise FileNotFoundError(f"Missing baseline registry at {baseline_path}. Please run merge_icmr.py first.")

df = pd.read_csv(baseline_path)

# Filter out intermediate values to guarantee clean binary phenotypes (S vs R)
df_model = df[df["is_resistant"].notna()].copy()
df_model["is_resistant"] = df_model["is_resistant"].astype(int)

# Include antibiotic_id to make the model biologically sound
FEATURE_COLS = [
    "age",              # Continuous patient covariate
    "gender",           # Demographic factor
    "ward_type",        # Environmental exposure setting
    "infection_type_id",# Epidemiological origin
    "organism_id",      # Taxonomical feature
    "hospital_dept_id", # Care delivery setting
    "sample_type_id",   # Anatomical site matrix
    "antibiotic_id"     # Targeted biochemical challenge agent
]
TARGET_COL = "is_resistant"

X = df_model[FEATURE_COLS].copy().fillna(df_model[FEATURE_COLS].median())
y = df_model[TARGET_COL]

print(f"📊 Training Matrix Footprint: {X.shape[0]} clinical isolates across {X.shape[1]} features.")

clf = RandomForestClassifier(
    n_estimators=150,
    max_depth=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(clf, X, y, cv=cv, scoring=["accuracy", "f1", "roc_auc"])

# Fit final model weights
clf.fit(X, y)

# Compile sorted feature importances for UI visualization
importances = dict(zip(FEATURE_COLS, clf.feature_importances_))

# Save artifacts to the unified directory
with open(os.path.join(TARGET_DIR, "clinical_model.pkl"), "wb") as f: 
    pickle.dump(clf, f)

label_maps = {
    "clinical_features": FEATURE_COLS,
    "organism": dict(sorted(df_model[["organism_id","organism_name"]].dropna().drop_duplicates().values.tolist())),
    "antibiotic": dict(sorted(df_model[["antibiotic_id","antibiotic_name"]].dropna().drop_duplicates().values.tolist())),
    "ward": {1: "ICU", 2: "OPD", 3: "Ward"},
    "infection": {1: "Community Acquired", 2: "Healthcare Associated", 3: "Not Known"},
    "dept": dict(sorted(df_model[["hospital_dept_id","dept_name"]].dropna().drop_duplicates().values.tolist())),
    "sample_type": dict(sorted(df_model[["sample_type_id","sample_type_name"]].dropna().drop_duplicates().values.tolist())),
    "importances": {"clinical": importances},
    "metrics": {
        "clinical": {
            "accuracy": float(np.nanmean(cv_results["test_accuracy"])),
            "f1": float(np.nanmean(cv_results["test_f1"])),
            "roc_auc": float(np.nanmean(cv_results["test_roc_auc"]))
        }
    }
}

with open(os.path.join(TARGET_DIR, "model_meta.pkl"), "wb") as f: 
    pickle.dump(label_maps, f)
    
print(f"💾 Model optimization complete. Artifacts written safely to {TARGET_DIR}")