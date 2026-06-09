import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle, numpy as np, os, glob

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
# LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_icmr():
    p = "processed/master_amr_icmr.csv"
    if not os.path.exists(p): return None
    df = pd.read_csv(p)
    df["resistance"] = df["resistance"].fillna("Unknown")
    return df

@st.cache_data
def load_glass():
    p = "GitHub compiled dataset (easiest for ML)/compiled_WHO_GLASS_2022.xlsx"
    if not os.path.exists(p): return None
    df = pd.read_excel(p, engine="openpyxl")
    return df[df["CountryTerritoryArea"].str.contains("India", case=False, na=False)].copy()

@st.cache_data
def load_resistance_2023():
    """Load all 3 Resistance_to_individual_antibiotics CSVs from GLASS Dashboard folder."""
    folder = "."
    pattern = os.path.join(folder, "Resistance_to_individual_antibiotics*.csv")
    files = sorted(glob.glob(pattern))
    frames = []
    for f in files:
        # Extract pathogen from raw header line 3
        with open(f) as raw:
            lines = raw.readlines()
        pathogen = "Unknown"
        inf_type = "Unknown"
        for line in lines[:6]:
            if line.startswith("Pathogen:"): pathogen = line.split(":",1)[1].strip()
            if line.startswith("Infection Type"): inf_type = line.split(":",1)[1].strip()
        try:
            df = pd.read_csv(f, skiprows=8, on_bad_lines="skip")
            df = df[df["AntibioticName"] != "Plot data"].copy()
            df["Pathogen"] = pathogen
            df["InfectionType"] = inf_type
            df["PercentResistant"] = pd.to_numeric(df["PercentCoverage"], errors="coerce")
            frames.append(df)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else None

@st.cache_data
def load_sdg():
    """Load both SDG trend CSVs."""
    folder = "."
    ecoli = os.path.join(folder, "SDG-AMR-indicators_2016-2023-Escherichia_coli-Third-generation_cephalosporins.csv")
    staph = os.path.join(folder, "SDG-AMR-indicators_2016-2023-Staphylococcus_aureus-Methicillin.csv")
    frames = []
    for f, name, ab in [(ecoli, "Escherichia coli", "3GC (ESBL proxy)"),
                        (staph, "Staphylococcus aureus", "Methicillin (MRSA)")]:
        if not os.path.exists(f): continue
        df = pd.read_csv(f, skiprows=8, on_bad_lines="skip")
        df = df[df["Year"] != "Year"].copy()
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df["TotalBCIsWithAST"] = pd.to_numeric(df["TotalBCIsWithAST"], errors="coerce")
        df["PercentCoverage"] = pd.to_numeric(df["PercentCoverage"], errors="coerce")
        df["Pathogen"] = name
        df["Antibiotic"] = ab
        frames.append(df.dropna(subset=["Year","TotalBCIsWithAST"]))
    return pd.concat(frames, ignore_index=True) if frames else None

@st.cache_data
def load_genomic():
    """Load Kaggle genomic dataset."""
    p = "antimicrobial_resistance_csv.csv"
    if not os.path.exists(p): return None
    df = pd.read_csv(p)
    # drug class resistance columns
    class_cols = [c for c in df.columns if c.startswith("class_")]
    gene_cols  = [c for c in df.columns if c.startswith("gene_")]
    df["total_resistance_classes"] = df[class_cols].sum(axis=1)
    df["total_amr_genes"]          = df[gene_cols].sum(axis=1)
    return df, class_cols, gene_cols

@st.cache_resource
def load_model():
    mp, mm = "processed/clinical_model.pkl", "processed/model_meta.pkl"
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

genomic_df   = genomic_out[0] if genomic_out else None
class_cols   = genomic_out[1] if genomic_out else []
gene_cols    = genomic_out[2] if genomic_out else []

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 AMR Surveillance")
    st.markdown("**India · Multicentre Study**")
    st.markdown("---")
    if icmr is not None:
        st.markdown('<p class="section-title">Filter Clinical Data (Tab 1)</p>', unsafe_allow_html=True)
        sel_state = st.selectbox("State", ["All States"]  + sorted(icmr["state_name"].dropna().unique().tolist()))
        sel_org   = st.selectbox("Organism", ["All Organisms"] + sorted(icmr["organism_name"].dropna().unique().tolist()))
        sel_ward  = st.selectbox("Ward",  ["All Wards"]   + sorted(icmr["ward_name"].dropna().unique().tolist()))
    st.markdown("---")
    st.caption("Data: ICMR AMR Network · WHO GLASS 2022 · WHO GLASS Dashboard 2023 · Kaggle Genomic")

def apply_filters(df):
    if sel_state != "All States":    df = df[df["state_name"]    == sel_state]
    if sel_org   != "All Organisms": df = df[df["organism_name"] == sel_org]
    if sel_ward  != "All Wards":     df = df[df["ward_name"]     == sel_ward]
    return df

filtered = apply_filters(icmr) if icmr is not None else None

# ─────────────────────────────────────────────────────────────────────────────
# HEADER METRICS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## AMR Surveillance Dashboard · India")
st.markdown("Multicentre Clinical Surveillance + WHO GLASS Trends + Genomic Resistance Analysis")
st.markdown("---")

if icmr is not None and filtered is not None:
    total = len(filtered)
    resistant = len(filtered[filtered["resistance"] == "Resistant"])
    r_pct = resistant / total * 100 if total > 0 else 0
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Clinical isolates",   f"{total}")
    c2.metric("Resistant",           f"{resistant} ({r_pct:.0f}%)")
    c3.metric("Organisms",           f"{filtered['organism_name'].nunique()}")
    c4.metric("States",              f"{filtered['state_name'].nunique()}")
    c5.metric("GLASS India records", f"{len(glass)}" if glass is not None else "—")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏥 ICMR Clinical",
    "🌍 WHO GLASS India",
    "🧬 Genomic Resistance",
    "🔮 Resistance Predictor",
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
                labels={"organism_name":"Organism","n":"Isolates","resistance":""},
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
                labels={"ward_name":"Ward","n":"Isolates","resistance":""},
            )
            fig2.update_layout(plot_bgcolor="#0f1117",paper_bgcolor="#0f1117",font_color="#c9d1d9",
                               legend_orientation="h",margin=dict(t=10,b=0))
            st.plotly_chart(fig2, use_container_width=True)

        # Antibiogram heatmap
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
                zmin=0, zmax=100, labels=dict(color="% R"), aspect="auto")
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
            st.markdown('<p class="section-title">Sample Type</p>', unsafe_allow_html=True)
            samp = filtered["sample_type_name"].value_counts().reset_index()
            samp.columns = ["Sample","n"]
            fig6 = px.bar(samp, x="n", y="Sample", orientation="h",
                          color_discrete_sequence=["#4361ee"])
            fig6.update_layout(plot_bgcolor="#0f1117",paper_bgcolor="#0f1117",font_color="#c9d1d9",
                               margin=dict(t=10,b=0))
            st.plotly_chart(fig6, use_container_width=True)

        with st.expander("Raw data"):
            show = ["organism_name","antibiotic_name","resistance","state_name",
                    "ward_name","infection_type","sample_type_name","age","gender_label","dept_name"]
            st.dataframe(filtered[show].reset_index(drop=True), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — WHO GLASS INDIA
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    if glass is None:
        st.error("GLASS xlsx not found.")
    else:
        st.markdown("### WHO GLASS India — Resistance Trends 2017–2020")
        gc1, gc2 = st.columns([1,2])
        with gc1:
            g_org = st.selectbox("Pathogen", ["All"] + sorted(glass["PathogenName"].unique().tolist()))
            g_ab  = st.selectbox("Antibiotic", ["All"] + sorted(glass["AbTargets"].unique().tolist()))
        gf = glass.copy()
        if g_org != "All": gf = gf[gf["PathogenName"] == g_org]
        if g_ab  != "All": gf = gf[gf["AbTargets"]    == g_ab]

        gr1, gr2 = st.columns(2)
        with gr1:
            st.markdown('<p class="section-title">Resistance % Over Time</p>', unsafe_allow_html=True)
            trend = gf.groupby(["Year","PathogenName"])["PercentResistant"].mean().reset_index()
            fig_t = px.line(trend, x="Year", y="PercentResistant", color="PathogenName",
                            markers=True, labels={"PercentResistant":"% Resistant"},
                            color_discrete_sequence=px.colors.qualitative.Bold)
            fig_t.update_layout(plot_bgcolor="#0f1117",paper_bgcolor="#0f1117",font_color="#c9d1d9",
                                legend_orientation="h",yaxis_range=[0,100],margin=dict(t=10,b=0))
            st.plotly_chart(fig_t, use_container_width=True)

        with gr2:
            st.markdown(f'<p class="section-title">Resistance by Antibiotic ({int(gf["Year"].max())})</p>', unsafe_allow_html=True)
            latest = gf[gf["Year"]==gf["Year"].max()]
            ab_pct = latest.groupby("AbTargets")["PercentResistant"].mean().sort_values().reset_index()
            fig_ab = px.bar(ab_pct, x="PercentResistant", y="AbTargets", orientation="h",
                            color="PercentResistant",
                            color_continuous_scale=[[0,"#4cc9f0"],[0.5,"#f8961e"],[1,"#f72585"]],
                            labels={"PercentResistant":"% R","AbTargets":"Antibiotic"})
            fig_ab.update_layout(plot_bgcolor="#0f1117",paper_bgcolor="#0f1117",font_color="#c9d1d9",
                                 coloraxis_showscale=False,margin=dict(t=10,b=0))
            st.plotly_chart(fig_ab, use_container_width=True)

        st.markdown("---")

        if res_2023 is not None:
            st.markdown("### WHO GLASS India — 2023 Resistance by Antibiotic")
            pathogens_2023 = res_2023["Pathogen"].unique().tolist()
            sel_p = st.selectbox("Pathogen (2023)", pathogens_2023)
            sub23 = res_2023[res_2023["Pathogen"] == sel_p].copy()
            sub23 = sub23[sub23["TotalBCIsWithAST"] != "-"].copy()
            sub23["TotalBCIsWithAST"] = pd.to_numeric(sub23["TotalBCIsWithAST"], errors="coerce")
            sub23 = sub23.dropna(subset=["TotalBCIsWithAST"])

            fig_23 = px.bar(
                sub23.sort_values("TotalBCIsWithAST"),
                x="TotalBCIsWithAST", y="AntibioticName", orientation="h",
                color="TotalBCIsWithAST",
                color_continuous_scale=["#4cc9f0","#f72585"],
                labels={"TotalBCIsWithAST":"Isolates with AST","AntibioticName":"Antibiotic"},
                title=f"{sel_p} · {sub23['InfectionType'].iloc[0] if len(sub23)>0 else ''} · India 2023",
            )
            fig_23.update_layout(plot_bgcolor="#0f1117",paper_bgcolor="#0f1117",font_color="#c9d1d9",
                                 coloraxis_showscale=False,margin=dict(t=30,b=0))
            st.plotly_chart(fig_23, use_container_width=True)

        st.markdown("---")

        if sdg is not None:
            st.markdown("### SDG AMR Indicators — India 2016–2023")
            sc1, sc2 = st.columns(2)
            for col, (pathogen, ab) in zip([sc1, sc2], [
                ("Escherichia coli", "3GC (ESBL proxy)"),
                ("Staphylococcus aureus", "Methicillin (MRSA)"),
            ]):
                sub = sdg[sdg["Pathogen"] == pathogen].dropna(subset=["Year","TotalBCIsWithAST"])
                with col:
                    st.markdown(f'<p class="section-title">{pathogen} — {ab} Isolates Over Time</p>', unsafe_allow_html=True)
                    fig_sdg = px.bar(sub, x="Year", y="TotalBCIsWithAST",
                                     color_discrete_sequence=["#f72585"],
                                     labels={"TotalBCIsWithAST":"Isolates tested","Year":"Year"})
                    fig_sdg.update_layout(plot_bgcolor="#0f1117",paper_bgcolor="#0f1117",
                                          font_color="#c9d1d9",margin=dict(t=10,b=0))
                    st.plotly_chart(fig_sdg, use_container_width=True)

        st.markdown("---")
        st.markdown('<p class="section-title">India Resistance Summary Table</p>', unsafe_allow_html=True)
        summary = (
            gf.groupby(["PathogenName","AbTargets"])
            .agg(avg_resistance=("PercentResistant","mean"),
                 isolates=("TotalSpecimenIsolates","sum"),
                 years=("Year","nunique"))
            .round(1).reset_index()
            .sort_values("avg_resistance", ascending=False)
        )
        summary.columns = ["Pathogen","Antibiotic","Avg Resistance %","Total Isolates","Years Reported"]
        st.dataframe(summary, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — GENOMIC RESISTANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    if genomic_df is None:
        st.error("antimicrobial_resistance_csv.csv not found in project root.")
    else:
        st.markdown("### Genomic Resistance Profile — *E. coli* isolates (n=50)")
        st.caption("Source: Kaggle AMR genomic dataset · All isolates are Escherichia coli · Gene presence = binary (0/1)")

        gm1, gm2, gm3 = st.columns(3)
        gm1.metric("Isolates",             len(genomic_df))
        gm2.metric("Resistance genes",     len(gene_cols))
        gm3.metric("Drug classes covered", len(class_cols))

        gc1, gc2 = st.columns(2)
        with gc1:
            st.markdown('<p class="section-title">Drug Class Resistance Prevalence</p>', unsafe_allow_html=True)
            class_prev = (
                genomic_df[class_cols].sum()
                .sort_values(ascending=True)
                .reset_index()
            )
            class_prev.columns = ["Drug Class","Resistant Isolates"]
            class_prev["Drug Class"] = class_prev["Drug Class"].str.replace("class_","")
            fig_cls = px.bar(class_prev, x="Resistant Isolates", y="Drug Class", orientation="h",
                             color="Resistant Isolates",
                             color_continuous_scale=["#4cc9f0","#f72585"])
            fig_cls.update_layout(plot_bgcolor="#0f1117",paper_bgcolor="#0f1117",font_color="#c9d1d9",
                                  coloraxis_showscale=False,margin=dict(t=10,b=0))
            st.plotly_chart(fig_cls, use_container_width=True)

        with gc2:
            st.markdown('<p class="section-title">Top 15 Resistance Genes by Frequency</p>', unsafe_allow_html=True)
            gene_freq = (
                genomic_df[gene_cols].sum()
                .sort_values(ascending=False)
                .head(15)
                .sort_values(ascending=True)
                .reset_index()
            )
            gene_freq.columns = ["Gene","Count"]
            gene_freq["Gene"] = gene_freq["Gene"].str.replace("gene_","")
            fig_gene = px.bar(gene_freq, x="Count", y="Gene", orientation="h",
                              color_discrete_sequence=["#4361ee"])
            fig_gene.update_layout(plot_bgcolor="#0f1117",paper_bgcolor="#0f1117",font_color="#c9d1d9",
                                   margin=dict(t=10,b=0))
            st.plotly_chart(fig_gene, use_container_width=True)

        st.markdown('<p class="section-title">Resistance Burden Distribution (genes per isolate)</p>', unsafe_allow_html=True)
        fig_dist = px.histogram(
            genomic_df, x="total_amr_genes", nbins=15,
            labels={"total_amr_genes":"AMR genes per isolate","count":"Isolates"},
            color_discrete_sequence=["#f72585"],
        )
        fig_dist.update_layout(plot_bgcolor="#0f1117",paper_bgcolor="#0f1117",font_color="#c9d1d9",
                               margin=dict(t=10,b=0), bargap=0.1)
        st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown('<p class="section-title">Clinically Significant Resistance Genes</p>', unsafe_allow_html=True)
        key_genes = ["gene_CTX-M-15","gene_CTX-M-14","gene_CTX-M-27","gene_KPC-1","gene_MCR-1","gene_QnrS1"]
        key_genes = [g for g in key_genes if g in genomic_df.columns]
        if key_genes:
            key_df = genomic_df[key_genes].sum().reset_index()
            key_df.columns = ["Gene","Present in N isolates"]
            key_df["Gene"] = key_df["Gene"].str.replace("gene_","")
            key_df["% of isolates"] = (key_df["Present in N isolates"] / len(genomic_df) * 100).round(1)
            st.dataframe(key_df, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — RESISTANCE PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    if model is None or meta is None:
        st.error("Run `train_model.py` first.")
    else:
        st.markdown("### Clinical Resistance Predictor")
        st.markdown(
            "Predicts antibiotic resistance probability for a patient case. "
            "Model: Random Forest trained on ICMR multicentre India data (n=130 isolates)."
        )

        m = meta["metrics"]["clinical"]
        mc1,mc2,mc3 = st.columns(3)
        mc1.metric("CV Accuracy", f"{m['accuracy']:.3f}")
        mc2.metric("CV F1 Score", f"{m['f1']:.3f}")
        mc3.metric("CV ROC-AUC", f"{m['roc_auc']:.3f}")
        st.caption("5-fold stratified cross-validation · Intermediate isolates excluded from training")

        st.markdown("---")
        st.markdown("#### Patient details")

        org_inv  = {v:k for k,v in meta["organism"].items()}
        atb_inv  = {v:k for k,v in meta["antibiotic"].items()}
        ward_inv = {v:k for k,v in meta["ward"].items()}
        inf_inv  = {v:k for k,v in meta["infection"].items()}
        dept_inv = {v:k for k,v in meta["dept"].items()}
        samp_inv = {v:k for k,v in meta["sample_type"].items()}

        col1,col2,col3 = st.columns(3)
        with col1:
            age_in    = st.slider("Age", 1, 100, 45)
            gender_in = st.selectbox("Gender", ["Male","Female"])
            ward_in   = st.selectbox("Ward", list(ward_inv.keys()))
        with col2:
            org_in  = st.selectbox("Organism", list(org_inv.keys()))
            atb_in  = st.selectbox("Antibiotic", list(atb_inv.keys()))
            samp_in = st.selectbox("Sample type", list(samp_inv.keys()))
        with col3:
            dept_in = st.selectbox("Department", list(dept_inv.keys()))
            inf_in  = st.selectbox("Infection acquisition", list(inf_inv.keys()))

        if st.button("Predict", type="primary"):
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
                if prob >= 60:   st.error("⚠ High risk — consider alternative antibiotic.")
                elif prob >= 40: st.warning("Moderate risk — confirm with susceptibility testing.")
                else:            st.success("Low risk — antibiotic likely effective.")
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
        st.markdown('<p class="section-title">Feature Importances</p>', unsafe_allow_html=True)
        imp = pd.DataFrame(list(meta["importances"]["clinical"].items()),
                           columns=["Feature","Importance"]).sort_values("Importance")
        fig_imp = px.bar(imp, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale=["#4361ee","#f72585"])
        fig_imp.update_layout(plot_bgcolor="#0f1117",paper_bgcolor="#0f1117",font_color="#c9d1d9",
                              coloraxis_showscale=False,margin=dict(t=10,b=0),height=280)
        st.plotly_chart(fig_imp, use_container_width=True)