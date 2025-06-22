import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import rioxarray

# --- Load and preprocess data ---
ds_T = xr.open_dataset("2m_temperature_2000.nc").rename({"var167": "T"})
ds_Td = xr.open_dataset("2m_dewpoint_2000.nc").rename({"var168": "Td"})
ds = xr.merge([ds_T, ds_Td])

# Convert to Celsius and compute RH
Td_C = ds["Td"] - 273.15
T_C = ds["T"] - 273.15
RH = 100 * (np.exp((17.625 * Td_C) / (243.04 + Td_C)) /
            np.exp((17.625 * T_C) / (243.04 + T_C)))
percentile98_RH = RH.quantile(0.98, dim="time")

lons = ds["lon"].values
lats = ds["lat"].values

# Load landcover
landcover = rioxarray.open_rasterio("landcover_resampled_0.1deg.tif").squeeze()

# --- Coordinates for landcover symbols ---
transform = landcover.rio.transform()
ny, nx = landcover.shape
x_coords = np.arange(nx) * transform.a + transform.c + transform.a / 2
y_coords = np.arange(ny) * transform.e + transform.f + transform.e / 2
xx, yy = np.meshgrid(x_coords, y_coords)
xx_flat = xx.flatten()
yy_flat = yy.flatten()
values_flat = landcover.values.flatten()

# --- Landcover Class Info ---
class_codes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
class_labels = [
    "Tree cover", "Shrubland", "Grassland", "Cropland", "Built-up",
    "Bare/sparse", "Snow/Ice", "Water bodies", "Wetland", "Mangroves", "Moss/Lichen"
]
symbols = ['P', 'X', '.', 'v', 's', '^', '*', 'o', 'D', '>', '<']
colors = [
    '#006400', '#FFBB22', '#FFFF4C', '#F096FF', '#FA0000',
    '#B4B4B4', '#F0F0F0', '#0064C8', '#0096A0', '#00CF75', '#FAE6A0'
]

fig, ax = plt.subplots(figsize=(12, 10))

# --- Darker RH background using inferno colormap ---
levels = np.linspace(70, 100, 16)
cf = ax.contourf(
    lons, lats, percentile98_RH,
    levels=levels, cmap='inferno', alpha=0.5, extend='both'
)

# --- Bigger, visible landcover symbols ---
for cls, label, sym, col in zip(class_codes, class_labels, symbols, colors):
    mask = values_flat == cls
    if np.any(mask):
        ax.scatter(
            xx_flat[mask], yy_flat[mask],
            marker=sym, c=col, label=label,
            s=5, alpha=0.4, edgecolors='black', linewidths=0.2
        )

# --- Axes, colorbar, legend ---
cbar_rh = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.07, aspect=40)
cbar_rh.set_label('98th Percentile Relative Humidity (%)', fontsize=12)

ax.set_xlabel('Longitude', fontsize=13)
ax.set_ylabel('Latitude', fontsize=13)
ax.set_xlim(65, 98)
ax.set_ylim(5, 38)
ax.set_aspect('auto')
ax.set_title('RH Extreme (98th %) with Land Cover Symbols', fontsize=15)
ax.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)

# --- Legend with larger symbols ---
ax.legend(
    title='Land Cover Classes', bbox_to_anchor=(1.01, 1), loc='upper left',
    fontsize=9, markerscale=2.5, frameon=False
)

plt.tight_layout()
plt.show()
plt.savefig("RH_Landcover_TightDarkBiggerSymbols.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("Updated plot saved as RH_Landcover_TightDarkBiggerSymbols.png")

