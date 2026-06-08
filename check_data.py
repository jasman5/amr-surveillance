import pandas as pd

files = [
    "whoglass_data_.csv.csv",
    "amr_dataset_final_prefixed.csv",
    "amr_dataset_variable_features.csv",
    "amr_summary_cleaned.csv",
    "amr_summary_dataset.csv",
    "antimicrobial_resistance_csv.csv",
    "glass_data2.csv"
]

for file in files:
    print("\n" + "=" * 60)
    print("FILE:", file)

    try:
        df = pd.read_csv(file)

        print("ROWS, COLUMNS:", df.shape)

        print("\nCOLUMNS:")
        print(df.columns.tolist())

        print("\nFIRST 5 ROWS:")
        print(df.head())

    except Exception as e:
        print("ERROR:", e)
