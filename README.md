# 🧬 AMR Surveillance System — India

A comprehensive **Antimicrobial Resistance (AMR) surveillance platform** for India based on PubMed literature mining, genetic resistance profiles, WHO GLASS global trends, and ICMR multicenter clinical data.

## The Problem
In India, resistance patterns differ by organism, state, and hospital. 78% of the *Klebsiella pneumoniae* at a Tamil Nadu doctor's intensive care unit are resistant to ciprofloxacin. Clinical judgments are made arbitrarily. This system creates that image.

## What It Does

| Component | File | Purpose |
|---|---|---|
| Data Pipeline | `merge_icmr.py` | Decodes 4 ICMR Stata tables into unified clinical records |
| ML Model | `train_model.py` | Random Forest resistance predictor (94.6% accuracy, AUC 0.985) |
| NLP Miner | `nlp_processor.py` | SciSpacy NER extracts pathogen/antibiotic mentions from PubMed |
| Dashboard | `dashboard.py` | 6-tab Streamlit interface with maps, heatmaps, forecasting, live predictor |
| Forecasting | `forecast_amr.py` | Linear regression resistance trend projection |

## Dashboard Tabs
1. **ICMR Clinical** — Antibiogram heatmap, ward breakdown, state distribution
2. **India Map** — Plotly geo-bubble map of resistance by state
3. **Forecast** — Resistance trend projection per organism
4. **WHO GLASS** — India vs global trends 2017–2020
5. **Genomic** — Resistance gene prevalence in *E. coli* isolates
6. **Predictor** — Enter patient details → get resistance probability + antibiotic ranking

## Stack
`Python` `Pandas` `Scikit-learn` `Streamlit` `Plotly` `SciSpacy` `Biopython` `Folium` `Seaborn`

## Data Sources
- ICMR-AMRSN (restricted access — not included in repo)
- WHO GLASS 2022 compiled dataset
- Kaggle AMR Genomic dataset
- PubMed E-utilities API

## setup

```bash
pip install -r requirements.txt
python merge_icmr.py       # build master dataset
python train_model.py      # train + save model
streamlit run dashboard.py # launch dashboard
```

> ⚠️ Clinical datasets are excluded from this repository (patient privacy).
> Contact for access or use the setup scripts to pull from public sources.


## only for pvt repo to remember

PUSH 
```bash
C:\amr_project
git status
git add .
git commit -m "Fixed dashboard layout"
git push origin main

PULL
```bash
git status
git pull origin main
