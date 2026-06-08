import pandas as pd
import os
import glob

print("🚀 Starting Complete Global + National AMR Data Pipeline Integration...")

# Ensure output directory exists
os.makedirs("processed", exist_ok=True)
all_datasets = []

# ---------------------------------------------------------
# 1. PROCESS ICMR CLINICAL DATA
# ---------------------------------------------------------
print("\n📦 Processing ICMR Dataset...")
try:
    icmr_df = pd.read_csv("processed/master_amr_icmr.csv")
    icmr_standard = pd.DataFrame()
    icmr_standard['pathogen'] = icmr_df['organism_id'].astype(str)
    icmr_standard['antibiotic'] = icmr_df['antibiotic_id'].astype(str)
    icmr_standard['state_region'] = icmr_df['state'].fillna('Unknown').astype(str)
    icmr_standard['source_dataset'] = 'ICMR (India)'
    all_datasets.append(icmr_standard)
    print(f"✅ Integrated ICMR data: {icmr_standard.shape}")
except Exception as e:
    print(f"⚠️ Skipping ICMR raw data merge: {e}")

# ---------------------------------------------------------
# 2. PROCESS KAGGLE GENOMIC DATA
# ---------------------------------------------------------
print("\n📦 Processing Kaggle Genomic Dataset...")
try:
    if os.path.exists("antimicrobial_resistance_csv.csv"):
        kaggle_df = pd.read_csv("antimicrobial_resistance_csv.csv")
        class_cols = [col for col in kaggle_df.columns if col.startswith('class_')]
        if class_cols and 'organism' in kaggle_df.columns:
            melted_kaggle = pd.melt(kaggle_df, id_vars=['organism', 'country'], value_vars=class_cols, var_name='antibiotic_class', value_name='is_resistant')
            resistant_only = melted_kaggle[melted_kaggle['is_resistant'] == 1].copy()
            resistant_only['antibiotic_class'] = resistant_only['antibiotic_class'].str.replace('class_', '', regex=False)
            
            kaggle_standard = pd.DataFrame()
            kaggle_standard['pathogen'] = resistant_only['organism'].astype(str)
            kaggle_standard['antibiotic'] = resistant_only['antibiotic_class'].astype(str)
            kaggle_standard['state_region'] = resistant_only['country'].fillna('Global').astype(str)
            kaggle_standard['source_dataset'] = 'Kaggle Genomic Repository'
            all_datasets.append(kaggle_standard)
            print(f"✅ Integrated Kaggle Genomic data: {kaggle_standard.shape}")
except Exception as e:
    print(f"⚠️ Skipping Kaggle merge: {e}")

# ---------------------------------------------------------
# 3. PROCESS WHO GLASS GITHUB EXCEL SHEET (Explicit Column Fix)
# ---------------------------------------------------------
print("\n📦 Processing WHO GLASS GitHub Excel Document...")
glass_ml_file = "GitHub compiled dataset (easiest for ML)/compiled_WHO_GLASS_2022.xlsx"
if os.path.exists(glass_ml_file):
    try:
        glass_ml_df = pd.read_excel(glass_ml_file, engine='openpyxl')
        
        # Using the exact columns found in your terminal scan!
        glass_ml_std = pd.DataFrame()
        glass_ml_std['pathogen'] = glass_ml_df['PathogenName'].astype(str)
        glass_ml_std['antibiotic'] = glass_ml_df['AbTargets'].astype(str)
        glass_ml_std['state_region'] = glass_ml_df['CountryTerritoryArea'].fillna('Global').astype(str)
        glass_ml_std['source_dataset'] = 'WHO GLASS (ML Excel)'
        
        all_datasets.append(glass_ml_std)
        print(f"✅ Integrated WHO GLASS ML Sheet: {glass_ml_std.shape}")
    except Exception as e:
        print(f"⚠️ Failed parsing WHO GLASS ML Excel file: {e}")
# ---------------------------------------------------------
# 4. PROCESS GLASS DASHBOARD INDIVIDUAL CSVS (Forced Skip + Filename Extraction)
# ---------------------------------------------------------
print("\n📦 Scanning GLASS Interactive Dashboard Subfolder for individual CSV profiles...")
glass_folder = "GLASS Interactive Dashboard"
if os.path.isdir(glass_folder):
    csv_files = glob.glob(os.path.join(glass_folder, "*.csv"))
    print(f"🔍 Discovered {len(csv_files)} CSV files to ingest...")
    
    for target_csv in csv_files:
        try:
            filename = os.path.basename(target_csv)
            temp_df = pd.read_csv(target_csv, skiprows=8, on_bad_lines='skip')
            
            # 1. Dynamically figure out the pathogen from the filename string
            extracted_pathogen = "Unknown Pathogen"
            if "Escherichia coli" in filename:
                extracted_pathogen = "Escherichia Coli"
            elif "Staphylococcus aureus" in filename:
                extracted_pathogen = "Staphylococcus Aureus"
            else:
                # Fallback check if it's an India general profile
                p_col = next((c for c in ['pathogen', 'Pathogen', 'PathogenName', 'Bacteria', 'organism'] if c in temp_df.columns), None)
                if p_col: extracted_pathogen = temp_df[p_col].astype(str)
            
            # 2. Match the unique antibiotic columns found in your terminal log
            a_col = next((c for c in ['AntibioticName', 'antibiotic', 'Antibiotic', 'drug', 'Drug'] if c in temp_df.columns), None)
            r_col = next((c for c in ['country', 'Country', 'region', 'state'] if c in temp_df.columns), None)
            
            # If the file has an explicit antibiotic column (like the India files)
            if a_col:
                sub_std = pd.DataFrame()
                sub_std['pathogen'] = extracted_pathogen if isinstance(extracted_pathogen, str) else extracted_pathogen
                sub_std['antibiotic'] = temp_df[a_col].astype(str)
                sub_std['state_region'] = temp_df[r_col].fillna('India').astype(str) if r_col else 'India'
                sub_std['source_dataset'] = f"WHO GLASS Dynamic ({filename[:15]})"
                all_datasets.append(sub_std)
                print(f"   ➡️ Integrated {filename}: {sub_std.shape}")
            
            # If it's an SDG file where the antibiotic target is fixed (e.g., Third-gen cephalosporins or Methicillin)
            elif "SDG-AMR-indicators" in filename:
                fixed_drug = "Third-Generation Cephalosporins" if "Escherichia coli" in filename else "Methicillin"
                sub_std = pd.DataFrame()
                sub_std['pathogen'] = extracted_pathogen
                sub_std['antibiotic'] = fixed_drug
                sub_std['state_region'] = temp_df['country'].fillna('Global Indicator').astype(str) if 'country' in temp_df.columns else 'Global Indicator'
                sub_std['source_dataset'] = f"WHO GLASS SDG ({filename[:12]})"
                all_datasets.append(sub_std)
                print(f"   ➡️ Integrated SDG Subset {filename}: {sub_std.shape}")
                
            else:
                print(f"   ⚠️ Skipping {filename} - unresolvable columns: {list(temp_df.columns)[:3]}")
        except Exception as e:
            print(f"   ❌ Failed to ingest CSV {filename}: {e}")
            
# ---------------------------------------------------------
# 5. CONCATENATE & EXPORT
# ---------------------------------------------------------
if all_datasets:
    print("\n🔄 Binding all structural layers into ultimate global registry master...")
    master_final = pd.concat(all_datasets, ignore_index=True)
    
    master_final['pathogen'] = master_final['pathogen'].str.strip().str.title()
    master_final['antibiotic'] = master_final['antibiotic'].str.strip().str.title()
    master_final['state_region'] = master_final['state_region'].str.strip()
    
    output_path = "processed/master_amr_final.csv"
    master_final.to_csv(output_path, index=False)
    
    print("\n" + "="*60)
    print(f"🎉 MISSION SUCCESS: COMPREHENSIVE AMR REGISTRY STACKED!")
    print(f"📊 Final Integrated Rows Pool: {master_final.shape[0]}")
    print(f"📂 Master CSV Location: {output_path}")
    print(f"🌍 Unique Dataset Registries Tracked:\n{master_final['source_dataset'].value_counts()}")
    print("="*60)
else:
    print("\n❌ Pipeline failed: No source records parsed effectively.")