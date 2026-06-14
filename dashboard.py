import streamlit as st
import streamlit.components.v1 as components  # HTML map injection
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle, numpy as np, os, glob
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="AMR Surveillance · India",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>

/* GLOBAL */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; padding-left: 2rem; padding-right: 2rem; max-width: 1400px; }

/* SIDEBAR */
[data-testid="stSidebar"] { background: #0a0d14; border-right: 1px solid #1a1f2e; }
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; padding-left: 1.2rem; padding-right: 1.2rem; }

/* TABS */
div[data-testid="stTabs"] > div:first-child { border-bottom: 1px solid #1e2130; gap: 0; }
div[data-testid="stTabs"] button { font-size: 0.78rem; font-weight: 500; letter-spacing: 0.03em; padding: 0.5rem 1rem; border-radius: 6px 6px 0 0; color: #8b95a8; transition: all 0.15s ease; }
div[data-testid="stTabs"] button:hover { color: #c9d1d9; background: #12161f; }
div[data-testid="stTabs"] button[aria-selected="true"] { color: #f72585; background: #12161f; border-bottom: 2px solid #f72585; font-weight: 600; }

/* METRIC CARDS */
[data-testid="stMetric"] { background: #0d1117; border: 1px solid #1e2130; border-radius: 10px; padding: 1rem 1.2rem; transition: border-color 0.2s ease; }
[data-testid="stMetric"]:hover { border-color: #f72585; }
[data-testid="stMetricLabel"] { font-size: 0.7rem !important; font-weight: 500; text-transform: uppercase; letter-spacing: 0.1em; color: #8b95a8 !important; }
[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700; color: #e6edf3 !important; line-height: 1.2; }
[data-testid="stMetricDelta"] { font-size: 0.72rem !important; }

/* SECTION LABELS */
.section-title { font-size: 0.65rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.14em; color: #8b95a8; margin-bottom: 0.5rem; margin-top: 1.2rem; padding-bottom: 0.3rem; border-bottom: 1px solid #1e2130; }

/* SELECTBOX */
[data-testid="stSelectbox"] > div > div { background: #0d1117; border: 1px solid #1e2130; border-radius: 6px; font-size: 0.82rem; color: #c9d1d9; transition: border-color 0.15s; }
[data-testid="stSelectbox"] > div > div:hover, [data-testid="stSelectbox"] > div > div:focus-within { border-color: #f72585; }

/* SLIDERS */
[data-testid="stSlider"] > div > div > div > div { background: #f72585; }

/* BUTTONS */
[data-testid="stButton"] > button { background: linear-gradient(135deg, #f72585 0%, #7209b7 100%); color: white; border: none; border-radius: 6px; font-size: 0.82rem; font-weight: 600; letter-spacing: 0.04em; padding: 0.5rem 1.4rem; transition: opacity 0.15s ease, transform 0.1s ease; cursor: pointer; }
[data-testid="stButton"] > button:hover { opacity: 0.88; transform: translateY(-1px); }
[data-testid="stButton"] > button:active { transform: translateY(0); }

/* DATAFRAME */
[data-testid="stDataFrame"] { border: 1px solid #1e2130; border-radius: 8px; overflow: hidden; }

/* EXPANDER */
[data-testid="stExpander"] { border: 1px solid #1e2130 !important; border-radius: 8px !important; background: #0a0d14; }
[data-testid="stExpander"] summary { font-size: 0.8rem; font-weight: 500; color: #8b95a8; padding: 0.6rem 1rem; }

/* ALERTS */
[data-testid="stAlert"] { border-radius: 8px; border-width: 1px; font-size: 0.82rem; }

/* DIVIDER */
hr { border: none; border-top: 1px solid #1e2130; margin: 1rem 0; }

/* HEADER BANNER */
.dashboard-header { background: linear-gradient(135deg, #0d1117 0%, #130821 100%); border: 1px solid #1e2130; border-radius: 12px; padding: 1.2rem 1.8rem; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 1rem; }
.dashboard-header h1 { font-size: 1.4rem; font-weight: 700; color: #e6edf3; margin: 0; line-height: 1.3; }
.dashboard-header p { font-size: 0.78rem; color: #8b95a8; margin: 0; margin-top: 0.2rem; }

/* BADGE CHIPS */
.badge { display: inline-block; padding: 0.15rem 0.55rem; border-radius: 20px; font-size: 0.65rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; }
.badge-red   { background: rgba(247,37,133,0.15); color: #f72585; border: 1px solid rgba(247,37,133,0.3); }
.badge-blue  { background: rgba(76,201,240,0.12); color: #4cc9f0; border: 1px solid rgba(76,201,240,0.3); }
.badge-gold  { background: rgba(248,150,30,0.12); color: #f8961e; border: 1px solid rgba(248,150,30,0.3); }

/* RESISTANCE STATUS LABELS */
.resist-high { color: #f72585; font-weight: 600; }
.resist-low  { color: #4cc9f0; font-weight: 600; }
.resist-mid  { color: #f8961e; font-weight: 600; }

/* SCROLLBAR */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0d14; }
::-webkit-scrollbar-thumb { background: #1e2130; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #f72585; }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY DARK THEME — applied to every chart via apply_theme(fig)
# ─────────────────────────────────────────────────────────────────────────────
DARK_THEME = dict(
    plot_bgcolor="#0d1117",
    paper_bgcolor="#0d1117",
    font_color="#c9d1d9",
    font_family="Inter, sans-serif",
    xaxis=dict(gridcolor="#1e2130", linecolor="#1e2130", tickcolor="#8b95a8",
               tickfont=dict(size=10), title_font=dict(size=11)),
    yaxis=dict(gridcolor="#1e2130", linecolor="#1e2130", tickcolor="#8b95a8",
               tickfont=dict(size=10), title_font=dict(size=11)),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2130",
                font=dict(size=10), orientation="h", y=-0.18),
    margin=dict(t=10, b=0, l=0, r=0),
    hoverlabel=dict(bgcolor="#1e2130", bordercolor="#f72585",
                    font=dict(color="#e6edf3", size=11)),
)

def apply_theme(fig, **extra):
    fig.update_layout(**DARK_THEME)
    if extra:
        fig.update_layout(**extra)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# STATE COORDINATES (real lat/lng for the 5 ICMR states)
# ─────────────────────────────────────────────────────────────────────────────
STATE_COORDS = {
    "Andhra Pradesh": (15.9129, 79.7400),
    "Chandigarh":     (30.7333, 76.7794),
    "Puducherry":     (11.9416, 79.8083),
    "Tamil Nadu":     (11.1271, 78.6569),
    "West Bengal":    (22.9868, 87.8550),
}

# ─────────────────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_icmr():
    p = "Dataset/processed/master_amr_icmr.csv"
    if not os.path.exists(p): return None
    df = pd.read_csv(p)
    df["resistance"] = df["resistance"].fillna("Unknown")
    return df

@st.cache_data
def load_glass():
    p = "Dataset/GitHub compiled dataset (easiest for ML)/compiled_WHO_GLASS_2022.xlsx"
    if not os.path.exists(p): return None
    df = pd.read_excel(p, engine="openpyxl")
    df["PercentResistant"]      = pd.to_numeric(df["PercentResistant"],      errors="coerce")
    df["TotalSpecimenIsolates"] = pd.to_numeric(df["TotalSpecimenIsolates"], errors="coerce")
    return df

@st.cache_data
def load_resistance_2023():
    folder  = "Dataset/GLASS Interactive Dashboard"
    pattern = os.path.join(folder, "Resistance_to_individual_antibiotics*.csv")
    files   = sorted(glob.glob(pattern))
    frames  = []
    for f in files:
        if not os.path.exists(f): continue
        with open(f) as raw:
            lines = raw.readlines()
        pathogen, inf_type = "Unknown", "Unknown"
        for line in lines[:6]:
            if line.startswith("Pathogen:"):      pathogen  = line.split(":",1)[1].strip()
            if line.startswith("Infection Type"): inf_type  = line.split(":",1)[1].strip()
        try:
            df = pd.read_csv(f, skiprows=8, on_bad_lines="skip")
            df = df[df["AntibioticName"] != "Plot data"].copy()
            df["Pathogen"]         = pathogen
            df["InfectionType"]    = inf_type
            df["PercentResistant"] = pd.to_numeric(df["PercentCoverage"], errors="coerce")
            frames.append(df)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else None

@st.cache_data
def load_sdg():
    folder = "Dataset/GLASS Interactive Dashboard"
    ecoli  = os.path.join(folder, "SDG-AMR-indicators_2016-2023-Escherichia_coli-Third-generation_cephalosporins.csv")
    staph  = os.path.join(folder, "SDG-AMR-indicators_2016-2023-Staphylococcus_aureus-Methicillin.csv")
    frames = []
    for f, name, ab in [(ecoli, "Escherichia coli",    "3GC (ESBL proxy)"),
                        (staph, "Staphylococcus aureus","Methicillin (MRSA)")]:
        if not os.path.exists(f): continue
        df = pd.read_csv(f, skiprows=8, on_bad_lines="skip")
        df = df[df["Year"] != "Year"].copy()
        df["Year"]             = pd.to_numeric(df["Year"],             errors="coerce")
        df["TotalBCIsWithAST"] = pd.to_numeric(df["TotalBCIsWithAST"], errors="coerce")
        df["PercentCoverage"]  = pd.to_numeric(df["PercentCoverage"],  errors="coerce")
        df["Pathogen"]  = name
        df["Antibiotic"] = ab
        frames.append(df.dropna(subset=["Year","TotalBCIsWithAST"]))
    return pd.concat(frames, ignore_index=True) if frames else None

@st.cache_data
def load_genomic():
    p = "Dataset/antimicrobial_resistance_csv.csv"
    if not os.path.exists(p): return None
    df = pd.read_csv(p)
    class_cols = [c for c in df.columns if c.startswith("class_")]
    gene_cols  = [c for c in df.columns if c.startswith("gene_")]
    df["total_resistance_classes"] = df[class_cols].sum(axis=1)
    df["total_amr_genes"]          = df[gene_cols].sum(axis=1)
    return df, class_cols, gene_cols

@st.cache_data
def load_atlas():
    base = "Dataset/ATLAS"
    files = {
        "yearly": "atlas_yearly_trend.csv",
        "icu":    "atlas_icu_comparison.csv",
        "genes":  "atlas_gene_prevalence.csv",
        "heatmap":"atlas_heatmap.csv",
    }
    out = {}
    for key, fname in files.items():
        p = os.path.join(base, fname)
        if not os.path.exists(p):
            return None
        out[key] = pd.read_csv(p)
    return out

@st.cache_resource
def load_model():
    # Enforce unified Dataset subfolder paths
    mp = "Dataset/processed/clinical_model.pkl"
    mm = "Dataset/processed/model_meta.pkl"
    if not os.path.exists(mp) or not os.path.exists(mm): return None, None
    with open(mp,"rb") as f: model = pickle.load(f)
    with open(mm,"rb") as f: meta  = pickle.load(f)
    return model, meta

icmr        = load_icmr()
glass       = load_glass()
res_2023    = load_resistance_2023()
sdg         = load_sdg()
genomic_out = load_genomic()
atlas       = load_atlas()
model, meta = load_model()

genomic_df = genomic_out[0] if genomic_out else None
class_cols = genomic_out[1] if genomic_out else []
gene_cols  = genomic_out[2] if genomic_out else []

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="padding:0.5rem 0 1rem 0; border-bottom:1px solid #1e2130; margin-bottom:1rem">
    <div style="font-size:1.5rem; margin-bottom:0.3rem">🧬</div>
    <div style="font-size:0.95rem; font-weight:700; color:#e6edf3; line-height:1.2">AMR Surveillance</div>
    <div style="font-size:0.7rem; color:#8b95a8; margin-top:0.2rem">India · Multicentre Study</div>
</div>
""", unsafe_allow_html=True)
    st.markdown("**India · Multicentre Study**")
    st.markdown("---")
    if icmr is not None:
        st.markdown('<p class="section-title">Filter Clinical Data</p>', unsafe_allow_html=True)
        sel_state = st.selectbox("State", ["All States"] + sorted(icmr["state_name"].dropna().unique().tolist()))
        sel_org = st.selectbox("Organism", ["All Organisms"] + sorted(icmr["organism_name"].dropna().unique().tolist()))
        sel_ward = st.selectbox("Ward", ["All Wards"] + sorted(icmr["ward_name"].dropna().unique().tolist()))
    else:
        sel_state, sel_org, sel_ward = "All States", "All Organisms", "All Wards"
        
    st.markdown("---")
    st.caption("ICMR AMR Network · WHO GLASS 2022 · Kaggle Genomic · PubMed NLP")

def apply_filters(df):
    if sel_state != "All States":    df = df[df["state_name"]    == sel_state]
    if sel_org   != "All Organisms": df = df[df["organism_name"] == sel_org]
    if sel_ward  != "All Wards":     df = df[df["ward_name"]     == sel_ward]
    return df

filtered = apply_filters(icmr) if icmr is not None else pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dashboard-header">
    <div style="font-size:2rem">🧬</div>
    <div>
        <h1>AMR Surveillance Dashboard · India</h1>
        <p>
            <span class="badge badge-red">ICMR Multicentre</span>&nbsp;
            <span class="badge badge-blue">WHO GLASS 2017–2023</span>&nbsp;
            <span class="badge badge-gold">Pfizer ATLAS 2004–2024</span>&nbsp;
            <span class="badge badge-red">RF Model · AUC 0.985</span>
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("Multicentre Clinical Surveillance + WHO GLASS Trends + Genomic Resistance + AI Forecasting")
st.markdown("---")

if icmr is not None and not filtered.empty:
    total     = len(filtered)
    resistant = len(filtered[filtered["resistance"] == "Resistant"])
    r_pct     = resistant / total * 100 if total > 0 else 0
    susceptible = len(filtered[filtered["resistance"] == "Susceptible"])

    ab_eff = (filtered[filtered["is_resistant"].notna()]
              .groupby("antibiotic_name")["is_resistant"].mean()
              .sort_values())
    best_ab = ab_eff.index[0] if len(ab_eff) > 0 else "—"

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total Isolates",       f"{total}")
    c2.metric("Resistant",            f"{resistant} ({r_pct:.0f}%)")
    c3.metric("Susceptible",          f"{susceptible}")
    c4.metric("Organisms tracked",    f"{filtered['organism_name'].nunique()}")
    c5.metric("States covered",       f"{filtered['state_name'].nunique()}")
    c6.metric("Best antibiotic",      best_ab)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏥 ICMR Clinical",
    "🗺️ India Map",
    "📈 Forecast",
    "🌍 WHO GLASS",
    "🧬 Genomic",
    "🔮 Predictor",
    "🌐 ATLAS Global",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ICMR CLINICAL
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if icmr is None:
        st.error("Run `merge_icmr.py` first.")
    elif filtered.empty:
        st.warning("No data matches current filters.")
    else:
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.markdown('<p class="section-title">Resistance by Organism</p>', unsafe_allow_html=True)
            fig = px.bar(
                filtered.groupby(["organism_name","resistance"]).size().reset_index(name="n"),
                x="organism_name", y="n", color="resistance", barmode="stack",
                color_discrete_map={"Resistant":"#f72585","Susceptible":"#4cc9f0","Intermediate":"#f8961e","Unknown":"#555"},
                labels={"organism_name":"","n":"Isolates","resistance":""},
            )
            apply_theme(fig)
            fig.update_layout(legend_orientation="h",xaxis_tickangle=-30,margin=dict(t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)

        with r1c2:
            st.markdown('<p class="section-title">Resistance by Ward Type</p>', unsafe_allow_html=True)
            fig2 = px.bar(
                filtered.groupby(["ward_name","resistance"]).size().reset_index(name="n"),
                x="ward_name", y="n", color="resistance", barmode="group",
                color_discrete_map={"Resistant":"#f72585","Susceptible":"#4cc9f0","Intermediate":"#f8961e","Unknown":"#555"},
                labels={"ward_name":"","n":"Isolates","resistance":""},
            )
            apply_theme(fig2)
            fig2.update_layout(legend_orientation="h",margin=dict(t=10,b=0))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<p class="section-title">Antibiogram Heatmap — % Resistant (organism × antibiotic)</p>', unsafe_allow_html=True)
        ab_data = filtered[filtered["resistance"].isin(["Resistant","Susceptible"])]
        if not ab_data.empty:
            pivot = (
                ab_data.groupby(["organism_name","antibiotic_name"])
                .apply(lambda x: round(100*(x["resistance"]=="Resistant").sum()/len(x),1))
                .reset_index(name="pct")
                .pivot(index="organism_name", columns="antibiotic_name", values="pct")
            )
            fig3 = px.imshow(pivot,
                color_continuous_scale=[[0,"#4cc9f0"],[0.5,"#f8961e"],[1,"#f72585"]],
                zmin=0, zmax=100, labels=dict(color="% R"), aspect="auto", text_auto=".0f")
            apply_theme(fig3)
            fig3.update_layout(xaxis_tickangle=-40,margin=dict(t=10,b=0),
                               coloraxis_colorbar=dict(title="% R"))
            st.plotly_chart(fig3, use_container_width=True)

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            st.markdown('<p class="section-title">Isolates by State</p>', unsafe_allow_html=True)
            sc = filtered["state_name"].value_counts().reset_index()
            sc.columns = ["State","n"]
            fig4 = px.bar(sc, x="n", y="State", orientation="h",
                          color="n", color_continuous_scale=["#4361ee","#f72585"])
            apply_theme(fig4)
            fig4.update_layout(showlegend=False,coloraxis_showscale=False,margin=dict(t=10,b=0))
            st.plotly_chart(fig4, use_container_width=True)

        with r2c2:
            st.markdown('<p class="section-title">Infection Acquisition</p>', unsafe_allow_html=True)
            ic = filtered["infection_type"].value_counts().reset_index()
            ic.columns = ["Type","n"]
            fig5 = px.pie(ic, names="Type", values="n",
                          color_discrete_sequence=["#f72585","#4cc9f0","#f8961e"], hole=0.5)
            fig5.update_layout(paper_bgcolor="#0f1117",font_color="#c9d1d9",
                               margin=dict(t=10,b=0),legend=dict(orientation="h",y=-0.15))
            st.plotly_chart(fig5, use_container_width=True)

        with r2c3:
            st.markdown('<p class="section-title">Sample Type Distribution</p>', unsafe_allow_html=True)
            samp = filtered["sample_type_name"].value_counts().reset_index()
            samp.columns = ["Sample","n"]
            fig6 = px.bar(samp, x="n", y="Sample", orientation="h",
                          color_discrete_sequence=["#4361ee"])
            apply_theme(fig6)
            fig6.update_layout(margin=dict(t=10,b=0))
            st.plotly_chart(fig6, use_container_width=True)

        st.markdown('<p class="section-title">Antibiotic Effectiveness Ranking (lowest resistance = most effective)</p>', unsafe_allow_html=True)
        ab_rank = (filtered[filtered["is_resistant"].notna()]
                   .groupby("antibiotic_name")["is_resistant"]
                   .agg(["mean","count"]).reset_index())
        ab_rank.columns = ["Antibiotic","Resistance Rate","Isolates Tested"]
        ab_rank["Resistance Rate"] = (ab_rank["Resistance Rate"]*100).round(1)
        ab_rank = ab_rank.sort_values("Resistance Rate")
        fig_ab = px.bar(ab_rank, x="Resistance Rate", y="Antibiotic", orientation="h",
                        color="Resistance Rate",
                        color_continuous_scale=[[0,"#4cc9f0"],[0.5,"#f8961e"],[1,"#f72585"]],
                        labels={"Resistance Rate":"Resistance %"})
        apply_theme(fig_ab)
        fig_ab.update_layout(coloraxis_showscale=False,margin=dict(t=10,b=0))
        st.plotly_chart(fig_ab, use_container_width=True)

        with st.expander("📋 Raw data table"):
            show = ["organism_name","antibiotic_name","resistance","state_name",
                    "ward_name","infection_type","sample_type_name","age","gender_label","dept_name"]
            st.dataframe(filtered[show].reset_index(drop=True), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — INDIA MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if icmr is None:
        st.error("Run `merge_icmr.py` first.")
    else:
        st.markdown("### 🗺️ India State Resistance Map")
        st.caption("Interactive spatial distribution engine. Click on regional markers to review resistance thresholds.")

        map_path = "Dataset/processed/amr_india_map.html"
        if os.path.exists(map_path):
            with open(map_path, "r", encoding="utf-8") as f:
                html_map_content = f.read()
            components.html(html_map_content, height=550, scrolling=False)
        else:
            st.warning("⚠️ Native amr_india_map.html asset not found inside Dataset/processed/. Run choropleth_map.py first.")

        st.markdown('<p class="section-title">State-wise Resistance Summary</p>', unsafe_allow_html=True)
        map_df = icmr.dropna(subset=["state_name","is_resistant"]).copy()
        state_stats = (map_df.groupby("state_name")
                       .agg(resistance_rate=("is_resistant","mean"),
                            total_isolates=("is_resistant","count"),
                            resistant_count=("is_resistant","sum"))
                       .reset_index())
        state_stats["resistance_pct"]  = (state_stats["resistance_rate"]*100).round(1)
        
        display = state_stats[["state_name","resistance_pct","total_isolates","resistant_count"]].copy()
        display.columns = ["State","Resistance %","Total Isolates","Resistant Isolates"]
        display = display.sort_values("Resistance %", ascending=False)
        st.dataframe(display, use_container_width=True)

        st.markdown('<p class="section-title">ICU vs OPD Resistance by State</p>', unsafe_allow_html=True)
        ward_state = (map_df[map_df["ward_name"].isin(["ICU","OPD"])]
                      .groupby(["state_name","ward_name"])["is_resistant"]
                      .mean().reset_index())
        ward_state["resistance_pct"] = (ward_state["is_resistant"]*100).round(1)
        fig_ws = px.bar(ward_state, x="state_name", y="resistance_pct", color="ward_name",
                        barmode="group",
                        color_discrete_map={"ICU":"#f72585","OPD":"#4cc9f0","Ward":"#f8961e"},
                        labels={"state_name":"State","resistance_pct":"Resistance %","ward_name":""})
        apply_theme(fig_ws)
        fig_ws.update_layout(legend_orientation="h",margin=dict(t=10,b=0))
        st.plotly_chart(fig_ws, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if icmr is None:
        st.error("Run `merge_icmr.py` first.")
    else:
        st.markdown("### 📈 Resistance Trend Forecasting")
        st.caption("⚠️ Forecast uses synthetic 2020–2023 trend extrapolated from current ICMR resistance rate.")

        fc1, fc2 = st.columns(2)
        with fc1:
            forecast_org = st.selectbox("Select organism to forecast",
                sorted(icmr["organism_name"].dropna().unique().tolist()), key="fc_org")
        with fc2:
            forecast_years = st.slider("Forecast horizon (years ahead)", 1, 10, 5)

        fc_df = icmr[icmr["organism_name"] == forecast_org].dropna(subset=["is_resistant"])
        current_rate = fc_df["is_resistant"].mean() if not fc_df.empty else 0.50

        hist_years = np.array([2020,2021,2022,2023]).reshape(-1,1)
        hist_rates = np.array([
            max(0, current_rate - 0.09),
            max(0, current_rate - 0.06),
            max(0, current_rate - 0.03),
            current_rate
        ])

        reg = LinearRegression()
        reg.fit(hist_years, hist_rates)

        future_years = np.arange(2024, 2024+forecast_years).reshape(-1,1)
        future_rates = np.clip(reg.predict(future_years), 0, 1)

        hist_df   = pd.DataFrame({"Year":hist_years.flatten(), "Rate":hist_rates, "Type":"Historical (synthetic)"})
        future_df = pd.DataFrame({"Year":future_years.flatten(), "Rate":future_rates, "Type":"Forecast"})
        combined  = pd.concat([hist_df, future_df])
        combined["Resistance %"] = (combined["Rate"]*100).round(1)

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=hist_df["Year"], y=hist_df["Rate"]*100,
            mode="lines+markers", name="Historical (synthetic)",
            line=dict(color="#4cc9f0", width=2),
            marker=dict(size=8)
        ))
        fig_fc.add_trace(go.Scatter(
            x=future_df["Year"], y=future_df["Rate"]*100,
            mode="lines+markers", name="Forecast",
            line=dict(color="#f72585", width=2, dash="dash"),
            marker=dict(size=8, symbol="diamond")
        ))
        fig_fc.add_hline(y=current_rate*100, line_dash="dot",
                         line_color="#f8961e",
                         annotation_text=f"Current: {current_rate*100:.1f}%",
                         annotation_position="bottom right")
        apply_theme(fig_fc)
        fig_fc.update_layout(legend_orientation="h",
            xaxis_title="Year", yaxis_title="Resistance Rate %",
            yaxis_range=[0,100], margin=dict(t=10,b=0),
            title=f"{forecast_org} — Resistance Trend Projection")
        st.plotly_chart(fig_fc, use_container_width=True)

        st.markdown('<p class="section-title">Forecast Values</p>', unsafe_allow_html=True)
        fc_table = pd.DataFrame({
            "Year": future_years.flatten(),
            "Predicted Resistance %": (future_rates*100).round(1),
            "Change from current": [(r*100 - current_rate*100) for r in future_rates],
        })
        fc_table["Change from current"] = fc_table["Change from current"].round(1).apply(
            lambda x: f"+{x}%" if x >= 0 else f"{x}%"
        )
        st.dataframe(fc_table, use_container_width=True)

        st.markdown('<p class="section-title">All Organisms — Current Resistance Rate Comparison</p>', unsafe_allow_html=True)
        all_org_rates = (icmr.dropna(subset=["is_resistant"])
                         .groupby("organism_name")["is_resistant"]
                         .mean().reset_index())
        all_org_rates["Resistance %"] = (all_org_rates["is_resistant"]*100).round(1)
        all_org_rates = all_org_rates.sort_values("Resistance %", ascending=True)
        fig_all = px.bar(all_org_rates, x="Resistance %", y="organism_name",
                         orientation="h", color="Resistance %",
                         color_continuous_scale=[[0,"#4cc9f0"],[0.5,"#f8961e"],[1,"#f72585"]],
                         labels={"organism_name":"Organism"})
        apply_theme(fig_all)
        fig_all.update_layout(coloraxis_showscale=False,
                              margin=dict(t=10,b=0))
        st.plotly_chart(fig_all, use_container_width=True)

        if sdg is not None:
            st.markdown("---")
            st.markdown("### WHO GLASS SDG Indicators — Real Trend Data (India 2016–2023)")
            sc1, sc2 = st.columns(2)
            for col, (pathogen, ab) in zip([sc1,sc2],[
                ("Escherichia coli",    "3GC (ESBL proxy)"),
                ("Staphylococcus aureus","Methicillin (MRSA)"),
            ]):
                sub = sdg[sdg["Pathogen"]==pathogen].dropna(subset=["Year","TotalBCIsWithAST"])
                with col:
                    st.markdown(f'<p class="section-title">{pathogen} — {ab}</p>', unsafe_allow_html=True)
                    fig_sdg = go.Figure()
                    fig_sdg.add_trace(go.Bar(
                        x=sub["Year"], y=sub["TotalBCIsWithAST"],
                        marker_color="#f72585", name="Isolates tested"
                    ))
                    apply_theme(fig_sdg)
                    fig_sdg.update_layout(margin=dict(t=10,b=0),
                                          xaxis_title="Year",yaxis_title="Isolates")
                    st.plotly_chart(fig_sdg, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — WHO GLASS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    if glass is None:
        st.error("Dataset/GitHub compiled dataset (easiest for ML)/compiled_WHO_GLASS_2022.xlsx not found.")
    else:
        india   = glass[glass["CountryTerritoryArea"] == "India"].copy()
        global_ = glass.copy()

        t2m1, t2m2, t2m3, t2m4, t2m5 = st.columns(5)
        t2m1.metric("Total global rows",  f"{len(global_):,}")
        t2m2.metric("Countries",          f"{global_['CountryTerritoryArea'].nunique()}")
        t2m3.metric("India rows",         f"{len(india)}")
        t2m4.metric("Pathogens tracked",  f"{global_['PathogenName'].nunique()}")
        t2m5.metric("Years covered",      "2017 – 2020")
        st.markdown("---")

        st.markdown("### 🇮🇳 India — Resistance Trends 2017–2020")
        fa1, fa2 = st.columns(2)
        with fa1:
            g_org = st.selectbox("Pathogen", ["All"] + sorted(india["PathogenName"].unique().tolist()), key="g_org")
        with fa2:
            g_ab  = st.selectbox("Antibiotic", ["All"] + sorted(india["AbTargets"].unique().tolist()), key="g_ab")

        gf = india.copy()
        if g_org != "All": gf = gf[gf["PathogenName"] == g_org]
        if g_ab  != "All": gf = gf[gf["AbTargets"]    == g_ab]

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.markdown('<p class="section-title">Resistance % Over Time</p>', unsafe_allow_html=True)
            trend = gf.groupby(["Year","PathogenName"])["PercentResistant"].mean().reset_index()
            fig_t = px.line(trend, x="Year", y="PercentResistant", color="PathogenName",
                            markers=True, labels={"PercentResistant":"% Resistant"},
                            color_discrete_sequence=px.colors.qualitative.Bold)
            apply_theme(fig_t)
            fig_t.update_layout(legend_orientation="h",
                                yaxis_range=[0,100], margin=dict(t=10, b=0))
            st.plotly_chart(fig_t, use_container_width=True)

        with r1c2:
            st.markdown('<p class="section-title">Resistance by Antibiotic (latest year)</p>', unsafe_allow_html=True)
            if not gf.empty:
                latest = gf[gf["Year"] == gf["Year"].max()]
                ab_pct = latest.groupby("AbTargets")["PercentResistant"].mean().sort_values().reset_index()
                fig_ab = px.bar(ab_pct, x="PercentResistant", y="AbTargets", orientation="h",
                                color="PercentResistant",
                                color_continuous_scale=[[0,"#4cc9f0"], [0.5,"#f8961e"], [1,"#f72585"]],
                                labels={"PercentResistant":"% R", "AbTargets":"Antibiotic"})
                apply_theme(fig_ab)
                fig_ab.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=0))
                st.plotly_chart(fig_ab, use_container_width=True)
            else:
                st.caption("No timeline segments available for current antibiotic targets.")

        st.markdown('<p class="section-title">India Antibiogram Heatmap (avg % Resistant)</p>', unsafe_allow_html=True)
        if not gf.empty:
            heat_pivot = (gf.groupby(["PathogenName","AbTargets"])["PercentResistant"]
                          .mean().reset_index()
                          .pivot(index="PathogenName", columns="AbTargets", values="PercentResistant"))
            fig_heat = px.imshow(heat_pivot,
                                 color_continuous_scale=[[0,"#4cc9f0"], [0.5,"#f8961e"], [1,"#f72585"]],
                                 zmin=0, zmax=100, labels=dict(color="% R"),
                                 aspect="auto", text_auto=".0f")
            apply_theme(fig_heat)
            fig_heat.update_layout(xaxis_tickangle=-40,
                                   margin=dict(t=10, b=0))
            st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🌍 Global Comparison — India vs World")
        gb1, gb2 = st.columns(2)
        with gb1:
            gc_org = st.selectbox("Pathogen (global)", sorted(global_["PathogenName"].unique()), key="gc_org")
        with gb2:
            gc_ab  = st.selectbox("Antibiotic (global)", sorted(global_["AbTargets"].unique()),  key="gc_ab")

        sub_global = global_[(global_["PathogenName"]==gc_org) & (global_["AbTargets"]==gc_ab)].copy()

        if not sub_global.empty:
            country_avg = (sub_global.groupby("CountryTerritoryArea")["PercentResistant"]
                           .mean().sort_values(ascending=True).reset_index())
            country_avg["color"] = country_avg["CountryTerritoryArea"].apply(
                lambda x: "#f72585" if x=="India" else "#4361ee")
            
            fig_rank = px.bar(country_avg, x="PercentResistant", y="CountryTerritoryArea",
                              orientation="h", color="color", color_discrete_map="identity",
                              labels={"PercentResistant":"Avg % Resistant", "CountryTerritoryArea":"Country"})
            apply_theme(fig_rank)
            fig_rank.update_layout(showlegend=False,
                                   height=max(400, len(country_avg)*18), margin=dict(t=10, b=0))
            st.plotly_chart(fig_rank, use_container_width=True)
            st.caption("🔴 India highlighted in pink")

            st.markdown('<p class="section-title">Longitudinal Trend Analysis — India vs Global Baseline</p>', unsafe_allow_html=True)
            india_trend  = sub_global[sub_global["CountryTerritoryArea"]=="India"].groupby("Year")["PercentResistant"].mean().reset_index()
            global_trend = sub_global.groupby("Year")["PercentResistant"].mean().reset_index()
            
            india_trend["Series"]  = "India"
            global_trend["Series"] = "Global Average"
            compare = pd.concat([india_trend, global_trend])
            
            fig_cmp = px.line(compare, x="Year", y="PercentResistant", color="Series",
                              markers=True,
                              color_discrete_map={"India":"#f72585", "Global Average":"#4cc9f0"},
                              labels={"PercentResistant":"% Resistant"})
            apply_theme(fig_cmp)
            fig_cmp.update_layout(yaxis_range=[0,100],
                                  legend_orientation="h", margin=dict(t=10, b=0))
            st.plotly_chart(fig_cmp, use_container_width=True)
        else:
            st.warning("No global entries found matching this pathogen-antibiotic combination.")

        summary = (india.groupby(["PathogenName","AbTargets"])
                   .agg(avg_resistance=("PercentResistant","mean"),
                        isolates=("TotalSpecimenIsolates","sum"),
                        years=("Year","nunique"))
                   .round(1).reset_index()
                   .sort_values("avg_resistance", ascending=False))
        summary.columns = ["Pathogen", "Antibiotic", "Avg Resistance %", "Total Isolates", "Years Reported"]
        st.dataframe(summary, use_container_width=True)

        if res_2023 is not None:
            st.markdown("---")
            st.markdown("### WHO GLASS India — 2023 Antibiotic Coverage by Pathogen")
            sel_p = st.selectbox("Pathogen (2023 data)", res_2023["Pathogen"].dropna().unique().tolist(), key="r23")
            sub23 = res_2023[res_2023["Pathogen"] == sel_p].copy()
            sub23["TotalBCIsWithAST"] = pd.to_numeric(sub23["TotalBCIsWithAST"], errors="coerce")
            sub23 = sub23.dropna(subset=["TotalBCIsWithAST"]).sort_values("TotalBCIsWithAST")
            
            fig_23 = px.bar(sub23, x="TotalBCIsWithAST", y="AntibioticName", orientation="h",
                            color="TotalBCIsWithAST", color_continuous_scale=["#4cc9f0", "#f72585"],
                            labels={"TotalBCIsWithAST":"Isolates tested", "AntibioticName":"Antibiotic"},
                            title=f"{sel_p} · India 2023")
            apply_theme(fig_23)
            fig_23.update_layout(coloraxis_showscale=False, margin=dict(t=30, b=0))
            st.plotly_chart(fig_23, use_container_width=True)
            
# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — GENOMIC
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    if genomic_df is None:
        st.error("Dataset/antimicrobial_resistance_csv.csv not found.")
    else:
        st.markdown("### 🧬 Genomic Resistance Profile — *E. coli* isolates (n=50)")
        st.caption("Source: Kaggle AMR genomic dataset · Gene presence = binary (0/1)")

        gm1,gm2,gm3 = st.columns(3)
        gm1.metric("Isolates",             len(genomic_df))
        gm2.metric("Resistance genes",     len(gene_cols))
        gm3.metric("Drug classes covered", len(class_cols))

        gc1, gc2 = st.columns(2)
        with gc1:
            st.markdown('<p class="section-title">Drug Class Resistance Prevalence</p>', unsafe_allow_html=True)
            class_prev = genomic_df[class_cols].sum().sort_values(ascending=True).reset_index()
            class_prev.columns = ["Drug Class","Resistant Isolates"]
            class_prev["Drug Class"] = class_prev["Drug Class"].str.replace("class_","")
            fig_cls = px.bar(class_prev, x="Resistant Isolates", y="Drug Class", orientation="h",
                             color="Resistant Isolates",
                             color_continuous_scale=["#4cc9f0","#f72585"])
            apply_theme(fig_cls)
            fig_cls.update_layout(coloraxis_showscale=False,margin=dict(t=10,b=0))
            st.plotly_chart(fig_cls, use_container_width=True)

        with gc2:
            st.markdown('<p class="section-title">Top 15 Resistance Genes</p>', unsafe_allow_html=True)
            gene_freq = (genomic_df[gene_cols].sum()
                         .sort_values(ascending=False).head(15)
                         .sort_values(ascending=True).reset_index())
            gene_freq.columns = ["Gene","Count"]
            gene_freq["Gene"] = gene_freq["Gene"].str.replace("gene_","")
            fig_gene = px.bar(gene_freq, x="Count", y="Gene", orientation="h",
                              color_discrete_sequence=["#4361ee"])
            apply_theme(fig_gene)
            fig_gene.update_layout(margin=dict(t=10,b=0))
            st.plotly_chart(fig_gene, use_container_width=True)

        st.markdown('<p class="section-title">Resistance Burden per Isolate</p>', unsafe_allow_html=True)
        fig_dist = px.histogram(genomic_df, x="total_amr_genes", nbins=15,
                                labels={"total_amr_genes":"AMR genes per isolate"},
                                color_discrete_sequence=["#f72585"])
        apply_theme(fig_dist)
        fig_dist.update_layout(margin=dict(t=10,b=0),bargap=0.1)
        st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown('<p class="section-title">Clinically Significant Resistance Genes</p>', unsafe_allow_html=True)
        key_genes = [g for g in ["gene_CTX-M-15","gene_CTX-M-14","gene_CTX-M-27",
                                  "gene_KPC-1","gene_MCR-1","gene_QnrS1"]
                     if g in genomic_df.columns]
        if key_genes:
            key_df = genomic_df[key_genes].sum().reset_index()
            key_df.columns = ["Gene","Present in N isolates"]
            key_df["Gene"] = key_df["Gene"].str.replace("gene_","")
            key_df["% of isolates"] = (key_df["Present in N isolates"]/len(genomic_df)*100).round(1)
            st.dataframe(key_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — RESISTANCE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    if model is None or meta is None:
        st.error("Run `train_model.py` first.")
    else:
        st.markdown("### 🔮 Clinical Resistance Predictor")
        st.markdown("Predicts resistance probability + recommends safest antibiotics for this patient profile.")
        st.caption("Model: Random Forest · ICMR multicentre India data · n=130 isolates")

        m = meta["metrics"]["clinical"]
        mc1,mc2,mc3 = st.columns(3)
        mc1.metric("CV Accuracy", f"{m['accuracy']:.3f}")
        mc2.metric("CV F1 Score", f"{m['f1']:.3f}")
        mc3.metric("CV ROC-AUC",  f"{m['roc_auc']:.3f}")
        st.caption("5-fold stratified cross-validation · Intermediate isolates excluded")
        st.markdown("---")

        org_inv  = {v:k for k,v in meta["organism"].items()}
        atb_inv  = {v:k for k,v in meta["antibiotic"].items()}
        ward_inv = {v:k for k,v in meta["ward"].items()}
        inf_inv  = {v:k for k,v in meta["infection"].items()}
        dept_inv = {v:k for k,v in meta["dept"].items()}
        samp_inv = {v:k for k,v in meta["sample_type"].items()}

        col1,col2,col3 = st.columns(3)
        with col1:
            age_in    = st.slider("Patient Age", 1, 100, 45)
            gender_in = st.selectbox("Gender", ["Male","Female"])
            ward_in   = st.selectbox("Ward", list(ward_inv.keys()))
        with col2:
            org_in  = st.selectbox("Suspected Organism", list(org_inv.keys()))
            atb_in  = st.selectbox("Proposed Antibiotic", list(atb_inv.keys()))
            samp_in = st.selectbox("Sample type", list(samp_inv.keys()))
        with col3:
            dept_in = st.selectbox("Department", list(dept_inv.keys()))
            inf_in  = st.selectbox("Infection acquisition", list(inf_inv.keys()))

        if st.button("🔮 Predict Resistance", type="primary"):
            features = meta["clinical_features"]

            row = {
                "age":               float(age_in),
                "gender":            1.0 if gender_in=="Female" else 2.0,
                "ward_type":         float(ward_inv[ward_in]),
                "infection_type_id": float(inf_inv[inf_in]),
                "organism_id":       float(org_inv[org_in]),
                "hospital_dept_id":  float(dept_inv[dept_in]),
                "sample_type_id":    float(samp_inv[samp_in]),
                "antibiotic_id":     float(atb_inv[atb_in]),
            }
            prob = model.predict_proba(pd.DataFrame([row])[features])[0][1] * 100

            pc1, pc2 = st.columns([1,2])
            with pc1:
                st.metric("Resistance probability", f"{prob:.1f}%")
                if prob >= 60:
                    st.error("⚠️ High risk — consider alternative antibiotic.")
                elif prob >= 40:
                    st.warning("⚠ Moderate risk — confirm with susceptibility test.")
                else:
                    st.success("✅ Low risk — antibiotic likely effective.")
            with pc2:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob,
                    number={"suffix":"%","font":{"color":"#f72585"}},
                    gauge={
                        "axis":{"range":[0,100],"tickcolor":"#8b95a8"},
                        "bar":{"color":"#f72585"},
                        "steps":[{"range":[0,40],"color":"#0d1117"},
                                 {"range":[40,60],"color":"#1a1f2e"},
                                 {"range":[60,100],"color":"#1f0d17"}],
                        "threshold":{"line":{"color":"white","width":2},"value":60},
                    },
                ))
                fig_g.update_layout(paper_bgcolor="#0f1117",font_color="#c9d1d9",
                                    height=220,margin=dict(t=20,b=0,l=20,r=20))
                st.plotly_chart(fig_g, use_container_width=True)

            st.markdown("---")
            st.markdown("#### 💊 Antibiotic Recommendation — Ranked by Resistance Risk")
            st.caption("All available antibiotics scored for this patient profile. Lower % = safer choice.")

            recommendations = []
            for ab_name, ab_id in atb_inv.items():
                r = row.copy()
                r["antibiotic_id"] = float(ab_id)
                p = model.predict_proba(pd.DataFrame([r])[features])[0][1] * 100
                recommendations.append({"Antibiotic": ab_name, "Resistance Risk %": round(p,1)})

            rec_df = pd.DataFrame(recommendations).sort_values("Resistance Risk %")
            rec_df["Recommendation"] = rec_df["Resistance Risk %"].apply(
                lambda x: "✅ Recommended" if x < 40 else ("⚠️ Use caution" if x < 60 else "❌ Avoid"))

            fig_rec = px.bar(rec_df, x="Resistance Risk %", y="Antibiotic",
                             orientation="h", color="Resistance Risk %",
                             color_continuous_scale=[[0,"#4cc9f0"],[0.5,"#f8961e"],[1,"#f72585"]],
                             labels={"Resistance Risk %":"Resistance Risk %"})
            apply_theme(fig_rec)
            fig_rec.update_layout(coloraxis_showscale=False,
                                  margin=dict(t=10,b=0))
            st.plotly_chart(fig_rec, use_container_width=True)
            st.dataframe(rec_df, use_container_width=True)

        st.markdown("---")
        st.markdown('<p class="section-title">Feature Importances</p>', unsafe_allow_html=True)
        imp = pd.DataFrame(list(meta["importances"]["clinical"].items()),
                           columns=["Feature","Importance"]).sort_values("Importance")
        fig_imp = px.bar(imp, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale=["#4361ee","#f72585"])
        apply_theme(fig_imp)
        fig_imp.update_layout(coloraxis_showscale=False,margin=dict(t=10,b=0),height=280)
        st.plotly_chart(fig_imp, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — ATLAS GLOBAL
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    if atlas is None:
        st.error("ATLAS summary files not found. Place atlas_yearly_trend.csv in Dataset/ATLAS/ and run `python prep_atlas.py` first.")
    else:
        yearly  = atlas["yearly"]
        icu_df  = atlas["icu"]
        gene_df = atlas["genes"]
        heat_df = atlas["heatmap"]

        st.markdown("### 🌐 ATLAS (Pfizer/Vivli) — India, 2004–2024")
        st.caption("17,327 India isolates · Real multi-year S/I/R surveillance data")

        am1, am2, am3, am4 = st.columns(4)
        am1.metric("India isolates",   "17,327")
        am2.metric("Years of data",    "2004–2024")
        am3.metric("Species tracked",  f"{yearly['Species'].nunique()}")
        am4.metric("Antibiotics",      f"{yearly['Antibiotic'].nunique()}")
        st.markdown("---")

        # FIX: Explicit core validation list addition to resolve the NameError scoping bug
        top_species = ["Acinetobacter baumannii", "Escherichia coli", "Klebsiella pneumoniae", "Pseudomonas aeruginosa", "Staphylococcus aureus"]

        st.markdown("#### 📈 Real Resistance Trends (2004–2024)")
        at1, at2 = st.columns(2)
        with at1: atlas_species = st.selectbox("Species", sorted(yearly["Species"].unique()), key="atlas_sp")
        with at2: atlas_abx = st.selectbox("Antibiotic", sorted(yearly["Antibiotic"].unique()), key="atlas_ab")

        trend_sub = yearly[(yearly["Species"]==atlas_species) & (yearly["Antibiotic"]==atlas_abx)].sort_values("Year")

        if trend_sub.empty:
            st.info("No data for this combination.")
        else:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=trend_sub["Year"], y=trend_sub["PercentResistant"], mode="lines+markers", name="% Resistant", line=dict(color="#f72585", width=2)))
            fig_trend.add_trace(go.Bar(x=trend_sub["Year"], y=trend_sub["N"], name="Isolates tested", yaxis="y2", marker_color="rgba(76,201,240,0.3)"))
            apply_theme(fig_trend)
            fig_trend.update_layout(xaxis_title="Year", yaxis=dict(title="% Resistant", range=[0,100]),
                yaxis2=dict(title="Isolates tested", overlaying="y", side="right", showgrid=False),
                legend_orientation="h", margin=dict(t=40,b=0))
            st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown('<p class="section-title">All Species — Same Antibiotic, Latest Year Available</p>', unsafe_allow_html=True)
        same_ab = yearly[yearly["Antibiotic"]==atlas_abx].copy()
        latest_per_species = same_ab.sort_values("Year").groupby("Species").tail(1)
        fig_sp = px.bar(latest_per_species.sort_values("PercentResistant"), x="PercentResistant", y="Species", orientation="h", color="PercentResistant", color_continuous_scale=[[0,"#4cc9f0"],[1,"#f72585"]])
        apply_theme(fig_sp)
        fig_sp.update_layout(coloraxis_showscale=False,margin=dict(t=10,b=0))
        st.plotly_chart(fig_sp, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🏨 ICU vs Non-ICU Resistance — India (all years)")
        fig_icu = px.bar(icu_df, x="Antibiotic", y="PercentResistant", color="Setting", barmode="group", color_discrete_map={"ICU":"#f72585","Non-ICU":"#4cc9f0"})
        apply_theme(fig_icu)
        fig_icu.update_layout(legend_orientation="h",xaxis_tickangle=-30,margin=dict(t=10,b=0))
        st.plotly_chart(fig_icu, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🔥 Species × Antibiotic Resistance Heatmap (2020–2024)")
        heat_pivot = heat_df.pivot(index="Species", columns="Antibiotic", values="PercentResistant")
        fig_atlas_heat = px.imshow(heat_pivot, color_continuous_scale=[[0,"#4cc9f0"],[0.5,"#f8961e"],[1,"#f72585"]], zmin=0, zmax=100, labels=dict(color="% R"), aspect="auto", text_auto=".0f")
        apply_theme(fig_atlas_heat)
        fig_atlas_heat.update_layout(xaxis_tickangle=-40,margin=dict(t=10,b=0))
        st.plotly_chart(fig_atlas_heat, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🧬 Resistance Gene Detections — India")
        fig_gene = px.bar(gene_df, x="Detections", y="Gene", orientation="h", color="Detections", color_continuous_scale=["#4361ee","#f72585"])
        apply_theme(fig_gene)
        fig_gene.update_layout(coloraxis_showscale=False,margin=dict(t=10,b=0))
        st.plotly_chart(fig_gene, use_container_width=True)

        # ── UPGRADED GLOBAL SURVEILLANCE MAP PANEL ───────────────────────────
        st.markdown("---")
        st.markdown("#### 🌍 International Pathogen Surveillance Canvas (Pfizer ATLAS Network)")
        st.caption("Cross-border resistance rate visualization. Choose a high-alert biochemical challenge agent to re-project the global heatmap.")
        
        map_abx_options = ["All Antibiotics", "Ciprofloxacin", "Meropenem", "Colistin", "Amikacin"]
        selected_map_abx = st.selectbox("Select Target Agent for Global Heatmap Projections", map_abx_options, key="global_map_abx_dropdown")
        
        from world_map import generate_global_map
        try:
            fig_world_map = generate_global_map(selected_map_abx)
            st.plotly_chart(fig_world_map, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render global mapping layers: {e}")