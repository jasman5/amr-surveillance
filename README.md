# AMR Surveillance System — India
- **https://amr-surveillance.streamlit.app/** <br>

A comprehensive **Antimicrobial Resistance (AMR) surveillance platform** for India based on PubMed literature mining, genetic resistance profiles, WHO GLASS global trends, and ICMR multicenter clinical data.

## The Problem
In India, resistance patterns vary by organism, state, and hospital. 78% of the *Klebsiella pneumoniae* in a Tamil Nadu doctor's ICU are resistant to ciprofloxacin. Clinical decisions are often made arbitrarily. This system reflects that reality.

## Functionality Overview

| Component | File | Purpose |
|---|---|---|
| Data Pipeline | `pipeline/merge_icmr.py` | Decodes four ICMR Stata tables into unified clinical records |
| Data Pipeline | `pipeline/prep_atlas.py` | Pre-aggregates the 1 million-row ATLAS dataset into summaries ready for dashboards |
| Geospatial | `pipeline/choropleth_map.py` | Creates the India state resistance map (Folium) |
| ML Model | `pipeline/train_model.py` | Random Forest resistance predictor (94.6% accuracy, AUC 0.985) |
| Dashboard | `dashboard.py` | A seven-tab Streamlit interface featuring maps, heatmaps, forecasts, and a live predictor |
| Geospatial Helper | `world_map.py` | Global choropleth used by the ATLAS Global tab |
> AUC 0.983 achieved on ICMR subset (n=130). External validation on a larger cohort is recommended before clinical deployment.

## Dashboard Tabs
1. **ICMR Clinical** — Antibiogram heatmap, ward breakdown, age stratification, ICU-specific antibiogram, state distribution, antibiotic effectiveness ranking
2. **India Map** — Interactive folium map showing resistance by state, ICU vs outpatient comparison by state
3. **Forecast** — Linear regression of resistance trends per organism, with 95% confidence bands and WHO GLASS SDG indicators
4. **WHO GLASS** — Resistance trends in India from 2017–2020, global country rankings, and comparisons between India and global baselines
5. **Genomic** — Prevalence of drug classes, top 15 resistance genes, AMR burden per isolate, and clinically significant gene table (*E. coli*, n=50)
6. **Predictor** — Enter patient profile to view resistance probability gauge, full antibiotic ranking with recommendations, and consultation history log
7. **ATLAS Global** — Pfizer/Vivli 20-year trends in India (2004–2024), ICU vs non-ICU comparisons, species × antibiotic heatmap, global choropleth map

## Stack
`Python` `Pandas` `Scikit-learn` `Streamlit` `Plotly` `SciSpacy` `Biopython` `Folium` `Seaborn`

## Data Sources
- ICMR-AMRSN (restricted access — not included in repo)
- WHO GLASS 2022 compiled dataset
- Kaggle AMR Genomic dataset
- PubMed E-utilities API

## Repo Structure

```
amr-surveillance-india/
├── dashboard.py            # Streamlit app (entry point)
├── world_map.py            # imported by dashboard.py (ATLAS Global tab)
├── requirements.txt
├── pipeline/                # run once, in order, to (re)build local artifacts
│   ├── merge_icmr.py
│   ├── prep_atlas.py
│   ├── choropleth_map.py
│   └── train_model.py
└── Dataset/                 # gitignored — populate locally from your own sources
```
    ├── ICMR Data Portal (primary source)/
    ├── GitHub compiled dataset (easiest for ML)/
    ├── GLASS Interactive Dashboard/
    ├── ATLAS/
    └── processed/            # output of the pipeline scripts above
```

## Setup

```bash
pip install -r requirements.txt
python pipeline/merge_icmr.py       # build master clinical dataset (needs ICMR .dta files, not in repo)
python pipeline/prep_atlas.py       # build ATLAS summary CSVs (needs atlas_vivli_2004_2024.csv, not in repo)
python pipeline/choropleth_map.py   # build the India state map HTML
python pipeline/train_model.py      # train + save the resistance model
streamlit run dashboard.py          # launch dashboard
```

> Clinical datasets are excluded from this repository (patient privacy).
> Contact for access or use the setup scripts to pull from public sources.

