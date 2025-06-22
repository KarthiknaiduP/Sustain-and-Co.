import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from metpy.calc import relative_humidity_from_dewpoint
from metpy.units import units

# Weight parameters for soiling index
ALPHA = 0.4  # PM10 weight
BETA = 0.2   # Wind speed weight
GAMMA = 0.1  # RH weight
DELTA = 0.2  # Dry days weight
EPSILON = 0.1  # Rain weight

def calculate_soiling_index(ds_meteo, ds_precip):
    """Calculate soiling index from meteorological data"""
    # Calculate RH from T and Td
    t2m = ds_meteo['t2m'] * units.K
    d2m = ds_meteo['d2m'] * units.K
    rh = relative_humidity_from_dewpoint(t2m, d2m) * 100
    rh = rh.metpy.dequantify()

    # Wind speed calculation
    ws = np.sqrt(ds_meteo['u10']**2 + ds_meteo['v10']**2)

    # Align precipitation and convert to mm
    precip = ds_precip['tp'] * 1000
    precip = precip.reindex_like(ds_meteo.time, method='nearest')

    # Dry periods calculation
    daily_precip = precip.resample(time='1D').sum()
    dry_days = (daily_precip < 0.1).astype(int)
    dry_periods = dry_days.rolling(time=3, min_periods=1).sum() >= 3
    dry_periods_3h = dry_periods.reindex_like(precip.time, method='ffill')

    # Normalization function
    def normalize(data):
        data_min = data.min(dim='time', skipna=True)
        data_max = data.max(dim='time', skipna=True)
        return (data - data_min) / (data_max - data_min)

    # Normalize parameters
    pm10_norm = normalize(ds_meteo['pm10'])
    ws_norm = normalize(ws)
    rh_norm = normalize(rh)
    dry_days_norm = normalize(dry_periods_3h.astype(float))
    rain_norm = normalize(precip)

    # Composite Soiling Index
    SI = (
        ALPHA * pm10_norm +
        BETA * (1 - ws_norm) +
        GAMMA * rh_norm +
        DELTA * dry_days_norm -
        EPSILON * rain_norm
    )

    return SI.resample(time='1MS').mean()

def generate_om_risk_map():
    print("Loading datasets...")
    chunks = {'time': 4}
    
    # Load datasets with consistent naming
    ds_meteo = xr.open_dataset("data_sfc.nc", chunks=chunks).rename({'valid_time': 'time'})
    ds_precip = xr.open_dataset("total_precip_2020.nc", chunks=chunks).rename({'valid_time': 'time'})
    ds_T = xr.open_dataset("2m_temperature_2000.nc", chunks=chunks).rename({"var167": "T"})
    ds_Td = xr.open_dataset("2m_dewpoint_2000.nc", chunks=chunks).rename({"var168": "Td"})
    ds_deltaT = xr.open_dataset("max_diurnal_cell_temperature_difference.nc", chunks=chunks)

    print("Calculating Soiling Index...")
    SI = calculate_soiling_index(ds_meteo, ds_precip)

    print("Calculating RH and condensation days...")
    ds = xr.merge([ds_T, ds_Td])
    Td_C = ds["Td"] - 273.15
    T_C = ds["T"] - 273.15
    RH = 100 * (np.exp((17.625 * Td_C) / (243.04 + Td_C)) /
                np.exp((17.625 * T_C) / (243.04 + T_C)))
    condensation_days = (RH > 90).sum(dim="time")

    print("Processing max ΔT...")
    deltaT_varname = list(ds_deltaT.data_vars)[0]
    deltaT_max = ds_deltaT[deltaT_varname]

    print("Scoring metrics...")
    # Scoring functions
    score = lambda data, thresholds: xr.apply_ufunc(
        lambda x: np.digitize(x, thresholds),
        data,
        vectorize=True
    )
    
    SI_score = score(SI, [0.3, 0.5, 0.7])
    RH_score = score(condensation_days, [90, 120, 150])
    T_score = score(deltaT_max, [30, 40, 50])

    print("Calculating composite O&M risk...")
    final_risk = 0.4 * SI_score + 0.3 * RH_score + 0.3 * T_score
    zones = score(final_risk, [1.0, 2.0, 2.5])

    print("Preparing 2D map...")
    # Reduce to 2D spatial data
    zones_2d = zones.mean(dim='time').squeeze()
    
    # Aggressive dimension reduction
    while zones_2d.ndim > 2:
        # Remove non-spatial dimensions
        for dim in list(zones_2d.dims):
            if dim not in ['lat', 'lon', 'latitude', 'longitude']:
                if dim in zones_2d.dims:
                    zones_2d = zones_2d.isel({dim: 0})
        zones_2d = zones_2d.squeeze()
    
    zones_2d = zones_2d.compute()
    
    # Extract coordinates
    lons = zones_2d.coords.get('lon', zones_2d.coords.get('longitude', np.arange(zones_2d.shape[1])))
    lats = zones_2d.coords.get('lat', zones_2d.coords.get('latitude', np.arange(zones_2d.shape[0])))
    
    # Ensure 1D coordinate arrays
    if lons.ndim > 1: lons = lons.values.ravel()
    if lats.ndim > 1: lats = lats.values.ravel()
    
    lon2d, lat2d = np.meshgrid(lons, lats)
    z_values = zones_2d.values
    
    # Final dimension check
    if z_values.ndim != 2:
        z_values = z_values.squeeze()[:len(lats), :len(lons)]

    print("Generating risk map...")
    plt.figure(figsize=(12, 8))
    levels = [-0.5, 0.5, 1.5, 2.5, 3.5]
    colors = ['green', 'yellow', 'orange', 'red']
    
    # Plot with safety checks
    if z_values.size == 0:
        plt.text(0.5, 0.5, "NO DATA AVAILABLE\nCheck calculations", 
                 ha='center', va='center', fontsize=16)
    else:
        contour = plt.contourf(lon2d, lat2d, z_values, levels=levels, colors=colors)
        cbar = plt.colorbar(contour, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(['Low', 'Moderate', 'High', 'Very High'])
        plt.title("Composite O&M Risk Map", fontsize=14)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("O_and_M_Risk_Map.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Operation completed successfully!")

if __name__ == "__main__":
    generate_om_risk_map()
