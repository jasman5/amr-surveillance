"""
train_model.py — Clinical Resistance Predictor (Engine 1)
Trains a Random Forest on ICMR patient-level clinical features.
Target: is_resistant (1 = Resistant, 0 = Susceptible)
Intermediate isolates excluded from training.

Run AFTER merge_icmr.py
Output: processed/clinical_model.pkl + processed/model_meta.pkl
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

warnings.filterwarnings("ignore")
os.makedirs("processed", exist_ok=True)

# ── 1. Load merged ICMR data ──────────────────────────────────────────────────
print("Loading processed/master_amr_icmr.csv ...")
df = pd.read_csv("processed/master_amr_icmr.csv")

# Keep only Susceptible (0) and Resistant (1) — drop Intermediate
df_train = df[df["is_resistant"].notna()].copy()
df_train["is_resistant"] = df_train["is_resistant"].astype(int)

print(f"Training rows : {len(df_train)}  (Resistant: {df_train['is_resistant'].sum()} | Susceptible: {(df_train['is_resistant']==0).sum()})")

# ── 2. Features and target ────────────────────────────────────────────────────
FEATURES = [
    "age",
    "gender",
    "ward_type",
    "infection_type_id",
    "organism_id",
    "hospital_dept_id",
    "sample_type_id",
    "antibiotic_id",
]

X = df_train[FEATURES].fillna(df_train[FEATURES].median())
y = df_train["is_resistant"]

# ── 3. Cross-validation ───────────────────────────────────────────────────────
print("\nRunning 5-fold stratified cross-validation ...")
clf = RandomForestClassifier(
    n_estimators=150,
    max_depth=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = cross_validate(clf, X, y, cv=cv, scoring=["accuracy", "f1", "roc_auc"])

print("\n=== CROSS-VALIDATION RESULTS ===")
print(f"  Accuracy : {np.mean(results['test_accuracy']):.3f} ± {np.std(results['test_accuracy']):.3f}")
print(f"  F1 Score : {np.mean(results['test_f1']):.3f} ± {np.std(results['test_f1']):.3f}")
print(f"  ROC-AUC  : {np.mean(results['test_roc_auc']):.3f} ± {np.std(results['test_roc_auc']):.3f}")

# ── 4. Train final model on all data ─────────────────────────────────────────
clf.fit(X, y)

print("\n=== FEATURE IMPORTANCES ===")
for feat, imp in sorted(zip(FEATURES, clf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat:<22} {imp:.4f}")

# ── 5. Save model ─────────────────────────────────────────────────────────────
with open("processed/clinical_model.pkl", "wb") as f:
    pickle.dump(clf, f)

# ── 6. Save metadata for dashboard ───────────────────────────────────────────
meta = {
    "clinical_features": FEATURES,
    "metrics": {
        "clinical": {
            "accuracy": float(np.mean(results["test_accuracy"])),
            "f1":       float(np.mean(results["test_f1"])),
            "roc_auc":  float(np.mean(results["test_roc_auc"])),
        }
    },
    "importances": {
        "clinical": dict(zip(FEATURES, clf.feature_importances_.tolist()))
    },
    "organism":    dict(df_train[["organism_id",    "organism_name"   ]].dropna().drop_duplicates().values.tolist()),
    "antibiotic":  dict(df_train[["antibiotic_id",  "antibiotic_name" ]].dropna().drop_duplicates().values.tolist()),
    "ward":        {1: "ICU", 2: "OPD", 3: "Ward"},
    "infection":   {1: "Community Acquired", 2: "Healthcare Associated", 3: "Not Known"},
    "dept":        dict(df_train[["hospital_dept_id", "dept_name"      ]].dropna().drop_duplicates().values.tolist()),
    "sample_type": dict(df_train[["sample_type_id",   "sample_type_name"]].dropna().drop_duplicates().values.tolist()),
}

with open("processed/model_meta.pkl", "wb") as f:
    pickle.dump(meta, f)

print("\nSaved: processed/clinical_model.pkl")
print("Saved: processed/model_meta.pkl")
print("\nDone. Run dashboard.py next.")