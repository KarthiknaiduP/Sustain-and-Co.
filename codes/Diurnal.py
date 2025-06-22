import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load and merge datasets (same as reference)
ds_T = xr.open_dataset("2m_temperature_2000.nc").rename({"var167": "T"})
ds_Td = xr.open_dataset("2m_dewpoint_2000.nc").rename({"var168": "Td"})
ds = xr.merge([ds_T, ds_Td])

# Calculate RH using Magnus formula (same as reference)
Td_C = ds["Td"] - 273.15
T_C = ds["T"] - 273.15
RH = 100 * (np.exp((17.625 * Td_C)/(243.04 + Td_C)) / 
           np.exp((17.625 * T_C)/(243.04 + T_C)))

# Calculate daily RH range and annual mean
daily_max = RH.resample(time='D').max()
daily_min = RH.resample(time='D').min()
mean_diurnal_range = (daily_max - daily_min).mean(dim='time')

# Plot configuration
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

# Use divergent colormap for range visualization
levels = np.linspace(0, 40, 21)  # 0-40% range with 2% increments
cf = ax.contourf(ds.lon, ds.lat, mean_diurnal_range, 
                levels=levels, cmap='viridis', extend='both')

# Add map features
ax.coastlines(linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.3)
ax.add_feature(cfeature.STATES, linewidth=0.2)

# Colorbar and labels
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.05, aspect=50)
cbar.set_label('Mean Daily Humidity Range (%)')
plt.title("Annual Mean Diurnal Humidity Range (2020)")
plt.tight_layout()
plt.show()

