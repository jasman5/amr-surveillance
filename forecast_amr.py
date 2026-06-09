import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("processed/master_amr_icmr.csv")

# Remove missing values
df = df.dropna(subset=["is_resistant"])

# Select organism
organism = "Escherichia coli"

filtered = df[df["organism_name"] == organism]

# Current resistance rate
current_rate = filtered["is_resistant"].mean()

print("Current Resistance Rate:", current_rate)

# Create synthetic historical trend
years = np.array([2020, 2021, 2022, 2023]).reshape(-1, 1)

rates = np.array([
    current_rate - 0.09,
    current_rate - 0.06,
    current_rate - 0.03,
    current_rate
])

# Train regression model
model = LinearRegression()
model.fit(years, rates)

# Predict future years
future_years = np.array([2024, 2025, 2026, 2027, 2028]).reshape(-1, 1)

future_rates = model.predict(future_years)

# Print predictions
print("\nFuture Predictions:")

for year, rate in zip(future_years.flatten(), future_rates):
    print(year, "->", round(rate, 3))

# Plot
plt.figure(figsize=(10, 5))

plt.plot(
    years,
    rates,
    marker='o',
    label='Historical Trend'
)

plt.plot(
    future_years,
    future_rates,
    marker='o',
    linestyle='--',
    label='Predicted Trend'
)

plt.xlabel("Year")
plt.ylabel("Resistance Rate")
plt.title(f"AMR Trend Projection - {organism}")

plt.legend()

plt.grid(True)

plt.show()