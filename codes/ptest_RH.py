import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# === 1. Load Temperature and Dewpoint Data for 2000 ===
ds_T = xr.open_dataset("2m_temperature_2000.nc").rename({"var167": "T"})   # Temperature
ds_Td = xr.open_dataset("2m_dewpoint_2000.nc").rename({"var168": "Td"})    # Dewpoint

# === 2. Merge and Compute Relative Humidity (RH) ===
ds = xr.merge([ds_T, ds_Td])
T_C = ds["T"] - 273.15    # Convert Kelvin to Celsius
Td_C = ds["Td"] - 273.15

# Magnus formula for RH
RH = 100 * (np.exp((17.625 * Td_C) / (243.04 + Td_C)) /
            np.exp((17.625 * T_C) / (243.04 + T_C)))

# === 3. Prepare Time for Trend Calculation ===
# Convert time to numeric days starting from 0
time = ds.time.dt.dayofyear
time = time - time[0]  # Start from 0

# === 4. Compute Slope and P-Value using Linear Regression ===
slope = np.zeros((len(ds.lat), len(ds.lon)))
pvalue = np.zeros_like(slope)

for i in range(len(ds.lat)):
    for j in range(len(ds.lon)):
        y = RH[:, i, j]
        if np.all(np.isnan(y)):
            slope[i, j] = np.nan
            pvalue[i, j] = np.nan
        else:
            res = linregress(time, y)
            slope[i, j] = res.slope        # Trend (% per day)
            pvalue[i, j] = res.pvalue      # Significance of trend

# Convert to DataArray for visualization
slope_da = xr.DataArray(slope, coords=[ds.lat, ds.lon], dims=["lat", "lon"])
pvalue_da = xr.DataArray(pvalue, coords=[ds.lat, ds.lon], dims=["lat", "lon"])

# === 5. Mask Non-Significant Trends (p >= 0.05) ===
sig_slope = slope_da.where(pvalue_da < 0.05)

# === 6. Plot Statistically Significant RH Trends ===
fig = plt.figure(figsize=(10, 6))
ax = plt.subplot(111, projection=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines(draw_labels=True)

# Define good levels for % change per day (slope)
cf = ax.contourf(ds.lon, ds.lat, sig_slope, levels=np.linspace(-0.2, 0.2, 21), 
                 cmap="RdBu_r", extend='both')

# Colorbar and annotations
plt.colorbar(cf, ax=ax, orientation="vertical", label="RH Trend (% per day)")
plt.title("Statistically Significant RH Trend at 2m (2020)")  # Year changed to 2020
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.tight_layout()
plt.savefig("significant_rh_trend_2020.png", dpi=300)  # Filename changed to 2020
plt.show()

