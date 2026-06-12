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
  .block-container{padding-top:1.4rem;padding-bottom:1rem}
  .section-title{font-size:.7rem;text-transform:uppercase;letter-spacing:.12em;
                 color:#8b95a8;margin-bottom:.3rem;margin-top:1rem}
  div[data-testid="stTabs"] button{font-size:.82rem}
</style>
""", unsafe_allow_html=True)

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

@st.cache_resource
def load_model():
    mp, mm = "Dataset/processed/clinical_model.pkl", "Dataset/processed/model_meta.pkl"
    if not os.path.exists(mp) or not os.path.exists(mm): return None, None
    with open(mp,"rb") as f: model = pickle.load(f)
    with open(mm,"rb") as f: meta  = pickle.load(f)
    return model, meta

icmr        = load_icmr()
glass       = load_glass()
res_2023    = load_resistance_2023()
sdg         = load_sdg()
genomic_out = load_genomic()
model, meta = load_model()

genomic_df = genomic_out[0] if genomic_out else None
class_cols = genomic_out[1] if genomic_out else []
gene_cols  = genomic_out[2] if genomic_out else []

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 AMR Surveillance")
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
st.markdown("## 🧬 AMR Surveillance Dashboard · India")
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏥 ICMR Clinical",
    "🗺️ India Map",
    "📈 Forecast",
    "🌍 WHO GLASS",
    "🧬 Genomic",
    "🔮 Predictor",
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
            fig.update_layout(plot_bgcolor="#0f1117",paper_bgcolor="#0f1117",font_color="#c9d1d9",
                              legend_orientation="h",xaxis_tickangle=-30,margin=dict(t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)

        with r1c2:
            st.markdown('<p class="section-title">Resistance by Ward Type</p>', unsafe_allow_html=True)
            fig2 = px.bar(
                filtered.groupby(["ward_name","resistance"]).size().reset_index(name="n"),
                x="ward_name", y="n", color="resistance", barmode="group",
                color_discrete_map={"Resistant":"#f72585","Susceptible":"#4cc9f0","Intermediate":"#f8961e","Unknown":"#555"},
                labels={"ward_name":"","n":"Isolates","resistance":""},
            )
            fig2.update_layout(plot_bgcolor="#0f1117",paper_bgcolor="#0f1117",font_color="#c9d1d9",
                               legend_orientation="h",margin=dict(t=10,b=0))
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
            fig3.update_layout(plot_bgcolor="#0f1117",paper_bgcolor="#0f1117",font_color="#c9d1d9",
                               xaxis_tickangle=-40,margin=dict(t=10,b=0),
                               coloraxis_colorbar=dict(title="% R"))
            st.plotly_chart(fig3, use_container_width=True)

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            st.markdown('<p class="section-title">Isolates by State</p>', unsafe_allow_html=True)
            sc = filtered["state_name"].value_counts().reset_index()
            sc.columns = ["State","n"]
            fig4 = px.bar(sc, x="n", y="State", orientation="h",
                          color="n", color_continuous_scale=["#4361ee","#f72585"])
            fig4.update_layout(plot_bgcolor="#0f1117",paper_bgcolor="#0f1117",font_color="#c9d1d9",
                               showlegend=False,coloraxis_showscale=False,margin=dict(t=10,b=0))
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
            fig6.update_layout(plot_bgcolor="#0f1117",paper_bgcolor="#0f1117",font_color="#c9d1d9",
                               margin=dict(t=10,b=0))
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
        fig_ab.update_layout(plot_bgcolor="#0f1117",paper_bgcolor="#0f1117",font_color="#c9d1d9",
                             coloraxis_showscale=False,margin=dict(t=10,b=0))
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

        # CRITICAL REPOSITORY PATH ALIGNMENT FIX
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
        fig_ws.update_layout(plot_bgcolor="#0f1117",paper_bgcolor="#0f1117",
                             font_color="#c9d1d9",legend_orientation="h",margin=dict(t=10,b=0))
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
        fig_fc.update_layout(
            plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
            font_color="#c9d1d9", legend_orientation="h",
            xaxis_title="Year", yaxis_title="Resistance Rate %",
            yaxis_range=[0,100], margin=dict(t=10,b=0),
            title=f"{forecast_org} — Resistance Trend Projection"
        )
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
        fig_all.update_layout(plot_bgcolor="#0f1117",paper_bgcolor="#0f1117",
                              font_color="#c9d1d9",coloraxis_showscale=False,
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
                    fig_sdg.update_layout(plot_bgcolor="#0f1117",paper_bgcolor="#0f1117",
                                          font_color="#c9d1d9",margin=dict(t=10,b=0),
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
            fig_t.update_layout(plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                                font_color="#c9d1d9", legend_orientation="h",
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
                fig_ab.update_layout(plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                                     font_color="#c9d1d9", coloraxis_showscale=False, margin=dict(t=10, b=0))
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
            fig_heat.update_layout(plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                                   font_color="#c9d1d9", xaxis_tickangle=-40,
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
            fig_rank.update_layout(plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                                   font_color="#c9d1d9", showlegend=False,
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
            fig_cmp.update_layout(plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                                  font_color="#c9d1d9", yaxis_range=[0,100],
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
            fig_23.update_layout(plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                                 font_color="#c9d1d9", coloraxis_showscale=False, margin=dict(t=30, b=0))
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
            fig_cls.update_layout(plot_bgcolor="#0f1117",paper_bgcolor="#0f1117",font_color="#c9d1d9",
                                  coloraxis_showscale=False,margin=dict(t=10,b=0))
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
            fig_gene.update_layout(plot_bgcolor="#0f1117",paper_bgcolor="#0f1117",font_color="#c9d1d9",
                                   margin=dict(t=10,b=0))
            st.plotly_chart(fig_gene, use_container_width=True)

        st.markdown('<p class="section-title">Resistance Burden per Isolate</p>', unsafe_allow_html=True)
        fig_dist = px.histogram(genomic_df, x="total_amr_genes", nbins=15,
                                labels={"total_amr_genes":"AMR genes per isolate"},
                                color_discrete_sequence=["#f72585"])
        fig_dist.update_layout(plot_bgcolor="#0f1117",paper_bgcolor="#0f1117",font_color="#c9d1d9",
                               margin=dict(t=10,b=0),bargap=0.1)
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

            # ── Antibiotic Recommendation Engine ─────────────────────────────
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
            fig_rec.update_layout(plot_bgcolor="#0f1117",paper_bgcolor="#0f1117",
                                  font_color="#c9d1d9",coloraxis_showscale=False,
                                  margin=dict(t=10,b=0))
            st.plotly_chart(fig_rec, use_container_width=True)
            st.dataframe(rec_df, use_container_width=True)

        st.markdown("---")
        st.markdown('<p class="section-title">Feature Importances</p>', unsafe_allow_html=True)
        imp = pd.DataFrame(list(meta["importances"]["clinical"].items()),
                           columns=["Feature","Importance"]).sort_values("Importance")
        fig_imp = px.bar(imp, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale=["#4361ee","#f72585"])
        fig_imp.update_layout(plot_bgcolor="#0f1117",paper_bgcolor="#0f1117",font_color="#c9d1d9",
                              coloraxis_showscale=False,margin=dict(t=10,b=0),height=280)
        st.plotly_chart(fig_imp, use_container_width=True)