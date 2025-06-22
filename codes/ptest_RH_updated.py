import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import linregress

# Load temperature and dewpoint datasets
ds_T = xr.open_dataset("2m_temperature_2000.nc")
ds_Td = xr.open_dataset("2m_dewpoint_2000.nc")

# Rename variables for clarity
ds_T = ds_T.rename({"var167": "T"})     # 2m temperature
ds_Td = ds_Td.rename({"var168": "Td"})  # 2m dewpoint

# Merge datasets on time, lat, lon
ds = xr.merge([ds_T, ds_Td])

# Compute relative humidity using the Magnus formula
Td_C = ds["Td"] - 273.15  # Convert K to °C
T_C = ds["T"] - 273.15    # Convert K to °C

# Compute RH with numpy's exp function
RH = 100 * (np.exp((17.625 * Td_C) / (243.04 + Td_C)) /
            np.exp((17.625 * T_C) / (243.04 + T_C)))

# Compute mean RH over time
mean_RH = RH.mean(dim="time")

# Trend Calculation: Perform linear regression for each grid point
lons = ds["lon"].values
lats = ds["lat"].values

# Initialize arrays to store the trend and p-value for each grid point
trend = np.full((len(lats), len(lons)), np.nan)
p_value = np.full((len(lats), len(lons)), np.nan)

# Loop over each grid point and calculate the trend using linear regression
for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        # Get time series of RH at this grid point
        time_series = RH[:, i, j].values
        
        # Perform linear regression (slope, intercept, r-value, p-value, std_err)
        _, _, _, p_value[i, j], _ = linregress(np.arange(len(time_series)), time_series)
        
        # Store the trend (slope) if p-value is significant
        if p_value[i, j] < 0.05:
            trend[i, j] = np.sign(np.polyfit(np.arange(len(time_series)), time_series, 1)[0])

# Flatten the lat, lon, and trend arrays for easier plotting
lons_flat = np.tile(lons, len(lats))  # Repeat lons to match trend size
lats_flat = np.repeat(lats, len(lons))  # Repeat lats to match trend size
trend_flat = trend.flatten()  # Flatten the trend array

# Plotting
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

# Add coastlines and gridlines
ax.coastlines()
ax.gridlines(draw_labels=True)

# Plot the mean RH contours
contour = ax.contourf(lons, lats, mean_RH, levels=np.linspace(0, 100, 21), cmap="Spectral_r")

# Overlay the trend (increasing or decreasing) on the map
ax.scatter(lons_flat[trend_flat == 1], lats_flat[trend_flat == 1], color="green", s=1, label="Increasing Trend", transform=ccrs.PlateCarree())
ax.scatter(lons_flat[trend_flat == -1], lats_flat[trend_flat == -1], color="red", s=1, label="Decreasing Trend", transform=ccrs.PlateCarree())

# Add color bar
cbar = plt.colorbar(contour, ax=ax, orientation="vertical")
cbar.set_label("Relative Humidity (%)")

# Add title and labels
plt.title("Mean Relative Humidity at 2m (%) with Trend Overlays (2020)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Add legend for trend markers
plt.legend()

# Show plot and save the image
plt.tight_layout()
plt.savefig("mean_rh_trend_overlay_2000.png", dpi=300)
plt.show()

