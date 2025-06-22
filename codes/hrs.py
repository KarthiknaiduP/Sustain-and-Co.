import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load temperature and dewpoint datasets ---
ds_T = xr.open_dataset("2m_temperature_2000.nc").rename({"var167": "T"})
ds_Td = xr.open_dataset("2m_dewpoint_2000.nc").rename({"var168": "Td"})

# --- 2. Calculate Relative Humidity using Magnus formula ---
T_C = ds_T["T"] - 273.15
Td_C = ds_Td["Td"] - 273.15
RH = 100 * (np.exp((17.625 * Td_C) / (243.04 + Td_C)) /
            np.exp((17.625 * T_C) / (243.04 + T_C)))

# --- 3. Define representative locations (adjust as needed) ---
locations = {
    'North (Jammu)': {'lat': 34.0, 'lon': 77.5},
    'East (Kolkata)': {'lat': 22.6, 'lon': 88.4},
    'South (Chennai)': {'lat': 13.1, 'lon': 80.3},
    'West (Ahmedabad)': {'lat': 23.0, 'lon': 72.6}
}

rh_bins = np.arange(0, 101, 1)  # 1% bins for RH
hours_in_year = 8760  # For hourly data

# --- 4. Plotting ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axes = axes.flatten()

for ax, (region, coords) in zip(axes, locations.items()):
    # Extract RH at the location
    rh_loc = RH.sel(lat=coords['lat'], lon=coords['lon'], method='nearest')
    rh_values = rh_loc.values.flatten()
    num_timesteps = len(rh_values)
    hours_per_timestep = hours_in_year / num_timesteps

    # Histogram
    hist, bin_edges = np.histogram(rh_values, bins=rh_bins)
    hist_hours = hist * hours_per_timestep

    # Plot as bar
    ax.bar(bin_edges[:-1], hist_hours, width=1, color='skyblue', edgecolor='k', alpha=0.7)
    ax.set_title(region, fontsize=13)
    ax.set_xlabel('Relative Humidity (%)')
    ax.set_ylabel('Hours per Year')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 500)

plt.suptitle('Relative Humidity Distribution (Hours per Year) at Key Indian Locations (2020)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

