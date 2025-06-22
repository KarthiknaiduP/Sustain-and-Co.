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

# Compute mean RH over time
mean_RH = RH.mean(dim="time")

# Create a map projection with Cartopy
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

# Add coastlines and gridlines
ax.coastlines()
ax.gridlines(draw_labels=True)

# Plot contours with clear levels
contour = ax.contourf(ds["lon"], ds["lat"], mean_RH, levels=np.linspace(0, 100, 21), cmap="Spectral_r")

# Add color bar
cbar = plt.colorbar(contour, ax=ax, orientation="vertical")
cbar.set_label("Relative Humidity (%)")

# Add title and labels
plt.title("Mean Relative Humidity at 2m (%) - 2020")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Show plot and save the image
plt.tight_layout()
plt.savefig("mean_rh_2000_cartopy.png", dpi=300)
plt.show()

