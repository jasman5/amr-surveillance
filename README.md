# 🧬 AMR Surveillance System — India

An end-to-end **Antimicrobial Resistance (AMR) surveillance platform** for India,
built on ICMR multicentre clinical data, WHO GLASS global trends, genomic resistance
profiles, and PubMed literature mining.

## The Problem
Resistance patterns in India vary by state, hospital, and organism. A doctor in Tamil Nadu
has no unified way to know that *Klebsiella pneumoniae* in their ICU is 78% resistant to
Ciprofloxacin. Clinical decisions get made blindly. This system builds that picture.

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

## Setup
```bash
pip install -r requirements.txt
python merge_icmr.py       # build master dataset
python train_model.py      # train + save model
streamlit run dashboard.py # launch dashboard
```

> ⚠️ Clinical datasets are excluded from this repository (patient privacy).
> Contact for access or use the setup scripts to pull from public sources.
