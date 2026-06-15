import pickle
import pandas as pd

with open("Dataset/processed/clinical_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("Dataset/processed/model_meta.pkl", "rb") as f:
    meta = pickle.load(f)

print("Features used:", meta["clinical_features"])
print("\nClasses:", model.classes_)
print("\nFeature importances:")
for k, v in sorted(meta["importances"]["clinical"].items(), key=lambda x: -x[1]):
    print(f"  {k}: {v:.4f}")

print("\nModel metrics (CV):")
for k, v in meta["metrics"]["clinical"].items():
    print(f"  {k}: {v:.4f}")