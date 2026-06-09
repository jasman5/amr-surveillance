import streamlit as st
import pandas as pd
import plotly.express as px

# Page config
st.set_page_config(
    page_title="AMR Dashboard",
    layout="wide"
)

# Title
st.title("Antimicrobial Resistance Dashboard")

# Load data
df = pd.read_csv("processed/master_amr_icmr.csv")

# Clean data
df = df.dropna(
    subset=[
        "organism_name",
        "antibiotic_name",
        "state_name",
        "is_resistant"
    ]
)

# Sidebar filters
st.sidebar.header("Filters")

organisms = st.sidebar.multiselect(
    "Select Organism",
    options=df["organism_name"].unique(),
    default=df["organism_name"].unique()[:3]
)

states = st.sidebar.multiselect(
    "Select State",
    options=df["state_name"].unique(),
    default=df["state_name"].unique()
)

filtered = df[
    (df["organism_name"].isin(organisms)) &
    (df["state_name"].isin(states))
]

# KPIs
total_samples = len(filtered)

avg_resistance = filtered["is_resistant"].mean()

top_antibiotic = (
    filtered.groupby("antibiotic_name")["is_resistant"]
    .mean()
    .sort_values()
    .index[0]
)

col1, col2, col3 = st.columns(3)

col1.metric("Total Samples", total_samples)

col2.metric(
    "Average Resistance",
    f"{avg_resistance:.2f}"
)

col3.metric(
    "Most Effective Antibiotic",
    top_antibiotic
)

# Resistance by state
state_chart = (
    filtered.groupby("state_name")["is_resistant"]
    .mean()
    .reset_index()
)

# Include all selected states
all_states = pd.DataFrame({
    "state_name": states
})

state_chart = all_states.merge(
    state_chart,
    on="state_name",
    how="left"
)

# Fill missing values
state_chart["is_resistant"] = (
    state_chart["is_resistant"]
    .fillna(0)
)

# Resistance by state
state_chart = (
    filtered.groupby("state_name")["is_resistant"]
    .mean()
    .reset_index()
)

# Include all selected states
all_states = pd.DataFrame({
    "state_name": states
})

state_chart = all_states.merge(
    state_chart,
    on="state_name",
    how="left"
)

# Fill missing values
state_chart["is_resistant"] = (
    state_chart["is_resistant"]
    .fillna(0)
)

fig1 = px.bar(
    state_chart,
    x="state_name",
    y="is_resistant",
    color="is_resistant",
    title="Resistance by State"
)

st.plotly_chart(fig1, use_container_width=True)

# Organism chart
org_chart = (
    filtered.groupby("organism_name")["is_resistant"]
    .mean()
    .reset_index()
)

fig2 = px.pie(
    org_chart,
    names="organism_name",
    values="is_resistant",
    title="Resistance Share"
)

st.plotly_chart(fig2, use_container_width=True)

# Antibiotic effectiveness
ab_chart = (
    filtered.groupby("antibiotic_name")["is_resistant"]
    .mean()
    .sort_values()
    .reset_index()
)

fig3 = px.bar(
    ab_chart,
    x="antibiotic_name",
    y="is_resistant",
    color="is_resistant",
    title="Antibiotic Resistance"
)

st.plotly_chart(fig3, use_container_width=True)

# Raw data
st.subheader("Dataset Preview")

st.dataframe(filtered.head(50))