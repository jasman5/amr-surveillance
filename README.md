# AMR Surveillance System — India
link- **https://amr-surveillance.streamlit.app/** <br>
A comprehensive **Antimicrobial Resistance (AMR) surveillance platform** for India based on PubMed literature mining, genetic resistance profiles, WHO GLASS global trends, and ICMR multicenter clinical data.

## The Problem
In India, resistance patterns differ by organism, state, and hospital. 78% of the *Klebsiella pneumoniae* at a Tamil Nadu doctor's intensive care unit are resistant to ciprofloxacin. Clinical judgments are made arbitrarily. This system creates that image.

## What It Does

| Component | File | Purpose |
|---|---|---|
| Data Pipeline | `pipeline/merge_icmr.py` | Decodes 4 ICMR Stata tables into unified clinical records |
| Data Pipeline | `pipeline/prep_atlas.py` | Pre-aggregates the 1M-row ATLAS dataset into dashboard-ready summaries |
| Geospatial | `pipeline/choropleth_map.py` | Builds the India state resistance map (Folium) |
| ML Model | `pipeline/train_model.py` | Random Forest resistance predictor (94.6% accuracy, AUC 0.985) |
| Dashboard | `dashboard.py` | 7-tab Streamlit interface with maps, heatmaps, forecasting, live predictor |
| Geospatial helper | `world_map.py` | Global choropleth used by the ATLAS Global tab |
> AUC 0.983 achieved on ICMR subset (n=130). External validation on a larger cohort is recommended before clinical deployment.

## Dashboard Tabs
1. **ICMR Clinical** — Antibiogram heatmap, ward breakdown, age group stratification, ICU-specific antibiogram, state distribution, antibiotic effectiveness ranking
2. **India Map** — Interactive folium map of resistance by state, ICU vs OPD comparison by state
3. **Forecast** — Linear regression resistance trend projection per organism with 95% confidence bands and WHO GLASS SDG indicators
4. **WHO GLASS** — India resistance trends 2017–2020, global country ranking, India vs global baseline comparison
5. **Genomic** — Drug class prevalence, top 15 resistance genes, AMR burden per isolate, clinically significant gene table (*E. coli*, n=50)
6. **Predictor** — Enter patient profile → resistance probability gauge + full antibiotic ranking with recommendations + consultation history log
7. **ATLAS Global** — Pfizer/Vivli 20-year India trends (2004–2024), ICU vs non-ICU comparison, species × antibiotic heatmap, global choropleth map
   
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

