import streamlit as st
import pandas as pd
import plotly.express as px

# Load cleaned dataset
df = pd.read_csv("kaggle_india_clean.csv")

st.title("AMR Surveillance Dashboard")

st.write("Dataset Shape:", df.shape)

# Sidebar filters
pathogen = st.sidebar.selectbox(
    "Select Pathogen",
    df["pathogen"].unique()
)

# Filter data
filtered = df[df["pathogen"] == pathogen]

# Bar chart
fig = px.bar(
    filtered,
    x="antibiotic",
    y="resistance_pct",
    color="antibiotic",
    title=f"Resistance Profile: {pathogen}"
)

st.plotly_chart(fig)

# Show table
st.subheader("Dataset Preview")
st.dataframe(filtered.head(20))