import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Load datasets
ds_T = xr.open_dataset("2m_temperature_2000.nc")
ds_Td = xr.open_dataset("2m_dewpoint_2000.nc")

# Rename variables for clarity
ds_T = ds_T.rename({"var167": "T"})     # 2m temperature
ds_Td = ds_Td.rename({"var168": "Td"})  # 2m dewpoint

# Merge datasets
ds = xr.merge([ds_T, ds_Td])

# Convert to Celsius
T_C = ds["T"] - 273.15
Td_C = ds["Td"] - 273.15

# Calculate RH using Magnus formula
RH = 100 * (np.exp((17.625 * Td_C) / (243.04 + Td_C)) /
            np.exp((17.625 * T_C) / (243.04 + T_C)))

# Compute spatial average at each time step
mean_RH_time = RH.mean(dim=["lat", "lon"])

# Plot
plt.figure(figsize=(12, 6))
mean_RH_time.plot()
plt.title("Mean Relative Humidity vs Time (2020)")
plt.ylabel("RH (%)")
plt.xlabel("Time")
plt.grid(True)
plt.tight_layout()
plt.savefig("rh_vs_time.png", dpi=300)
plt.show()

