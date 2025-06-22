import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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

# Compute 98th percentile RH over time
percentile98_RH = RH.quantile(0.98, dim="time")

# Prepare lats and lons
lons = ds["lon"].values
lats = ds["lat"].values

# Create the plot
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

# Define optimized levels for high humidity (98th percentile focus)
levels = np.linspace(70, 100, 31)  # Tight spacing between 70 and 100

# Contour plot with high-contrast colormap
cf = plt.contourf(lons, lats, percentile98_RH, levels=levels,
                  cmap="plasma", extend="max")

# Add features
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.STATES, linewidth=0.5)

# Add colorbar and title
cbar = plt.colorbar(cf, orientation="horizontal", pad=0.05, aspect=50)
cbar.set_label("98th Percentile Relative Humidity (%)")
plt.title("98th Percentile Relative Humidity Map (Year 2000)")

plt.tight_layout()
plt.show()
