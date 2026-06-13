"""
choropleth_map.py — Geospatial Mapping Engine
Aggregates state-level resistance metrics from ICMR clinical isolates
and projects them onto a spatial visualization layer using Folium.
"""

import pandas as pd
import folium
import os

# Define absolute coordinate map for the reporting clinical centers
STATE_COORDS = {
    "Andhra Pradesh": (15.9129, 79.7400),
    "Chandigarh":     (30.7333, 76.7794),
    "Puducherry":     (11.9416, 79.8083),
    "Tamil Nadu":     (11.1271, 78.6569),
    "West Bengal":    (22.9868, 87.8550),
}

def generate_amr_map():
    p = "Dataset/processed/master_amr_icmr.csv"
    if not os.path.exists(p):
        print("⚠️ Processed ICMR file not found. Run pipelines first.")
        return False
        
    df = pd.read_csv(p)
    df = df.dropna(subset=["state_name", "is_resistant"])
    
    # Calculate volume and mean resistance rate per state
    state_stats = df.groupby("state_name").agg(
        resistance_rate=("is_resistant", "mean"),
        isolate_count=("is_resistant", "count")
    ).reset_index()
    
    # Initialize base map centered over India
    india_map = folium.Map(
        location=[22.5937, 78.9629],
        zoom_start=5,
        tiles="CartoDB dark_matter"  # Clean aesthetic matching your dashboard theme
    )
    
    # Render proportional spatial overlays
    for _, row in state_stats.iterrows():
        state = row["state_name"]
        rate = row["resistance_rate"]
        count = row["isolate_count"]
        
        # Fallback to map center if state coords are missing
        lat, lng = STATE_COORDS.get(state, (22.5937, 78.9629))
        
        # Dynamic styling parameters based on resistance metrics
        percentage = rate * 100
        marker_color = "#f72585" if percentage >= 50 else "#4cc9f0"
        radius_size = max(8, min(25, int(count * 0.5)))  # Scale radius with data density
        
        popup_text = f"""
        <div style='font-family: sans-serif; font-size: 12px; color: #333;'>
            <strong>State:</strong> {state}<br>
            <strong>Resistance Rate:</strong> {percentage:.1f}%<br>
            <strong>Total Isolates:</strong> {count}
        </div>
        """
        
        folium.CircleMarker(
            location=[lat, lng],
            radius=radius_size,
            popup=folium.Popup(popup_text, max_width=200),
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.6,
            weight=1.5
        ).add_to(india_map)
        
    os.makedirs("Dataset/processed", exist_ok=True)
    india_map.save("Dataset/processed/amr_india_map.html")
    print("✅ Geospatial map layer updated and saved successfully.")
    return True

if __name__ == "__main__":
    generate_amr_map()