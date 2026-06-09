import pandas as pd
import folium

# Load dataset
df = pd.read_csv("processed/master_amr_icmr.csv")

# Remove missing values
df = df.dropna(subset=["state_name", "is_resistant"])

# Calculate resistance rate per state
state_resistance = (
    df.groupby("state_name")["is_resistant"]
    .mean()
    .reset_index()
)

print(state_resistance)

# Create India base map
india_map = folium.Map(
    location=[22.5, 78.9],
    zoom_start=5
)

# Add circles for each state
for _, row in state_resistance.iterrows():

    folium.CircleMarker(
        location=[22.5, 78.9],  # temporary center
        radius=row["is_resistant"] * 20,
        popup=f"{row['state_name']} : {row['is_resistant']:.2f}",
        color="red",
        fill=True,
        fill_opacity=0.6
    ).add_to(india_map)

# Save map
india_map.save("amr_india_map.html")

print("Map saved as amr_india_map.html")
