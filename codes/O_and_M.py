import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import warnings
warnings.filterwarnings('ignore')

# Updated thresholds and weights for better variation capture
SOILING_THRESHOLDS = [0.05, 0.15, 0.3, 0.5]    # More sensitive to lower soiling risks
RH_THRESHOLDS = [15, 30, 60, 90]                # Earlier detection of humidity risks
TEMP_THRESHOLDS = [15, 20, 25, 30]              # More sensitive to thermal stress
WEIGHTS = [0.35, 0.35, 0.3]                     # Balanced weights for SI, RH, and ΔT

def calculate_rh(T, Td):
    """Calculate relative humidity from temperature and dewpoint using Magnus formula"""
    try:
        # Ensure valid temperature ranges
        T = np.clip(T, -50, 60)
        Td = np.clip(Td, -60, 50)
        
        # Magnus formula for relative humidity
        alpha = 17.625
        beta = 243.04
        
        numerator = np.exp((alpha * Td) / (beta + Td))
        denominator = np.exp((alpha * T) / (beta + T))
        
        rh = 100 * (numerator / denominator)
        return np.clip(rh, 0, 100)  # Ensure RH is between 0-100%
    except Exception as e:
        print(f"Error in RH calculation: {e}")
        return np.full_like(T, 50)  # Return neutral RH if calculation fails

def score_data(data, thresholds, reverse=False):
    """
    Score data based on thresholds with improved handling
    reverse=True for cases where lower values mean higher risk
    """
    score = xr.zeros_like(data, dtype=float)
    
    if reverse:
        # For variables where lower values mean higher risk (e.g., wind speed)
        for i, thresh in enumerate(reversed(thresholds)):
            score = score.where(data >= thresh, len(thresholds) - i)
    else:
        # For variables where higher values mean higher risk
        for i, thresh in enumerate(thresholds):
            score = score.where(data <= thresh, i + 1)
    
    return score

def robust_normalize(data, percentile_range=(5, 95)):
    """Robust normalization using percentiles to handle outliers"""
    valid_data = data.where(np.isfinite(data))
    p_low, p_high = np.percentile(valid_data.values[~np.isnan(valid_data.values)], percentile_range)
    
    # Avoid division by zero
    if p_high - p_low == 0:
        return xr.zeros_like(data)
    
    normalized = (data.clip(p_low, p_high) - p_low) / (p_high - p_low)
    return normalized

def standardize_coords(ds):
    """Standardize coordinate names to latitude and longitude"""
    coord_map = {}
    
    # Handle various coordinate naming conventions
    for coord in ds.coords:
        if coord.lower() in ['lat', 'latitude', 'y']:
            coord_map[coord] = 'latitude'
        elif coord.lower() in ['lon', 'longitude', 'long', 'x']:
            coord_map[coord] = 'longitude'
    
    return ds.rename(coord_map)

def calculate_soiling_index(ds_meteo, ds_precip):
    """Calculate enhanced soiling index with better normalization"""
    print("  - Processing PM10 data...")
    pm10 = ds_meteo['pm10']
    
    print("  - Calculating wind speed...")
    wind_speed = np.sqrt(ds_meteo['u10']**2 + ds_meteo['v10']**2)
    
    print("  - Processing precipitation data...")
    precip = ds_precip['tp'] * 1000  # Convert m to mm
    
    # Robust normalization for each component
    pm10_norm = robust_normalize(pm10)
    wind_norm = robust_normalize(wind_speed)
    precip_norm = robust_normalize(precip)
    
    # Enhanced soiling index formula
    # Higher PM10 increases soiling, higher wind decreases it, rain helps cleaning
    SI = pm10_norm - 0.4 * wind_norm - 0.3 * precip_norm
    
    # Monthly averaging for temporal stability
    SI_monthly = SI.resample(time='1MS').mean()
    SI_annual = SI_monthly.mean(dim='time')
    
    return SI_annual

def calculate_humidity_risk(ds_temp, ds_dew):
    """Calculate humidity-based risk with improved methodology"""
    print("  - Converting temperature units...")
    T = ds_temp['T'] - 273.15  # Convert K to °C
    Td = ds_dew['Td'] - 273.15  # Convert K to °C
    
    print("  - Calculating relative humidity...")
    RH = calculate_rh(T, Td)
    
    # Calculate daily averages
    RH_daily = RH.resample(time='1D').mean()
    
    # Count high humidity days (RH > 85% for corrosion risk)
    high_humidity_days = (RH_daily > 85).sum(dim='time')
    
    # Also consider average RH levels
    avg_RH = RH_daily.mean(dim='time')
    
    # Combine both metrics
    humidity_risk = 0.6 * high_humidity_days + 0.4 * avg_RH
    
    return humidity_risk

def interpolate_to_common_grid(datasets, target_coords):
    """Interpolate all datasets to a common grid"""
    interpolated = []
    target_lat, target_lon = target_coords
    
    for i, ds in enumerate(datasets):
        print(f"  - Interpolating dataset {i+1}...")
        
        # Ensure coordinate names are standardized
        if 'lat' in ds.dims:
            ds = ds.rename({'lat': 'latitude'})
        if 'lon' in ds.dims:
            ds = ds.rename({'lon': 'longitude'})
        
        # Interpolate to target grid
        ds_interp = ds.interp(
            latitude=target_lat, 
            longitude=target_lon, 
            method='linear',
            kwargs={'fill_value': np.nan}
        )
        
        interpolated.append(ds_interp)
    
    return interpolated

def generate_risk_map():
    """Main function to generate the O&M risk map"""
    print("🚀 Starting O&M Risk Map Generation...")
    print("=" * 50)
    
    # Load datasets with error handling
    print("📂 Loading datasets...")
    try:
        ds_meteo = standardize_coords(
            xr.open_dataset("data_sfc.nc").rename({'valid_time': 'time'})
        )
        ds_precip = standardize_coords(
            xr.open_dataset("total_precip_2020.nc").rename({'valid_time': 'time'})
        )
        ds_temp = standardize_coords(
            xr.open_dataset("2m_temperature_2000.nc").rename({'var167': 'T'})
        )
        ds_dew = standardize_coords(
            xr.open_dataset("2m_dewpoint_2000.nc").rename({'var168': 'Td'})
        )
        ds_deltaT = standardize_coords(
            xr.open_dataset("max_diurnal_cell_temperature_difference.nc")
        )
        print("✅ All datasets loaded successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
        print("Please ensure all data files are in the correct directory.")
        return
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return
    
    # Calculate individual risk components
    print("\n🧮 Calculating risk components...")
    
    print("1️⃣ Soiling Index...")
    SI_annual = calculate_soiling_index(ds_meteo, ds_precip)
    SI_score = score_data(SI_annual, SOILING_THRESHOLDS)
    
    print("2️⃣ Humidity Risk...")
    humidity_risk = calculate_humidity_risk(ds_temp, ds_dew)
    RH_score = score_data(humidity_risk, RH_THRESHOLDS)
    
    print("3️⃣ Temperature Variation...")
    deltaT = ds_deltaT[list(ds_deltaT.data_vars)[0]]
    T_score = score_data(deltaT, TEMP_THRESHOLDS)
    
    # Define target grid (use the highest resolution dataset)
    print("\n🗺️ Setting up common grid...")
    target_lat = SI_score.latitude
    target_lon = SI_score.longitude
    
    # Interpolate all scores to common grid
    print("🔄 Interpolating to common grid...")
    datasets = [RH_score, T_score]
    RH_interp, T_interp = interpolate_to_common_grid(
        datasets, (target_lat, target_lon)
    )
    
    # Combine risk scores
    print("\n⚖️ Combining risk scores...")
    combined_risk = (
        WEIGHTS[0] * SI_score +
        WEIGHTS[1] * RH_interp +
        WEIGHTS[2] * T_interp
    )
    
    # Final normalization
    print("📊 Final normalization...")
    final_risk = robust_normalize(combined_risk, percentile_range=(2, 98))
    
    # Quality checks
    print("\n🔍 Quality checks:")
    print(f"  - Data range: {np.nanmin(final_risk.values):.3f} to {np.nanmax(final_risk.values):.3f}")
    print(f"  - Valid points: {np.sum(~np.isnan(final_risk.values)):,}")
    print(f"  - Grid shape: {final_risk.shape}")
    
    # Generate the plot
    print("\n🎨 Generating visualization...")
    create_enhanced_plot(final_risk)
    
    print("\n✅ O&M Risk Map generation completed successfully!")

def create_enhanced_plot(risk_data):
    """Create an enhanced visualization of the risk map"""
    
    # Set up the figure with high DPI for quality
    fig = plt.figure(figsize=(14, 10), dpi=300)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Enhanced background styling
    ax.add_feature(cfeature.OCEAN, facecolor='#f0f8ff', zorder=0)
    ax.add_feature(cfeature.LAND, facecolor='white', zorder=1)
    
    # Geographic features with subtle styling
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='#666666', alpha=0.7, zorder=4)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='#333333', zorder=4)
    ax.add_feature(cfeature.STATES.with_scale('50m'), 
                   linewidth=0.3, edgecolor='#999999', alpha=0.5, zorder=4)
    
    # Enhanced gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                      alpha=0.3, linestyle='--', zorder=3)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {'size': 11, 'color': '#444444'}
    gl.ylabel_style = {'size': 11, 'color': '#444444'}
    
    # Use a sophisticated colormap for better variation display
    from matplotlib.colors import LinearSegmentedColormap
    
    # Custom colormap for risk visualization
    colors = ['#2E8B57', '#90EE90', '#FFFF00', '#FFA500', '#FF4500', '#8B0000']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('risk_map', colors, N=n_bins)
    cmap.set_bad('white', 1.0)
    
    # High-resolution contour plot
    levels = np.linspace(0, 1, 60)  # 60 levels for ultra-smooth gradients
    
    contour = ax.contourf(
        risk_data.longitude, 
        risk_data.latitude, 
        risk_data.values,
        levels=levels,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        extend='both',
        zorder=2
    )
    
    # Enhanced colorbar
    cbar = plt.colorbar(contour, ax=ax, orientation='vertical', 
                        shrink=0.7, pad=0.02, aspect=30)
    cbar.set_label("O&M Risk Score (Normalized 0-1)", fontsize=13, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Add risk level annotations
    risk_levels = [
        (0.0, 0.2, "Very Low Risk", '#2E8B57'),
        (0.2, 0.4, "Low Risk", '#90EE90'),
        (0.4, 0.6, "Moderate Risk", '#FFFF00'),
        (0.6, 0.8, "High Risk", '#FFA500'),
        (0.8, 1.0, "Very High Risk", '#8B0000')
    ]
    
    # Add a legend for risk levels
    from matplotlib.patches import Rectangle
    legend_elements = []
    for low, high, label, color in risk_levels:
        legend_elements.append(Rectangle((0, 0), 1, 1, facecolor=color, 
                                       edgecolor='black', linewidth=0.5, label=label))
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98),
              fontsize=10, framealpha=0.9, fancybox=True, shadow=True)
    
    # Enhanced title and subtitle
    ax.set_title("Operations & Maintenance Risk Assessment Map", 
                fontsize=18, fontweight='bold', pad=20)
    ax.text(0.5, 0.02, "Risk factors: Soiling Index (35%) + Humidity Risk (35%) + Temperature Variation (30%)",
            transform=ax.transAxes, ha='center', fontsize=11, 
            style='italic', color='#555555')
    
    # Set extent if needed (adjust based on your data coverage)
    # ax.set_global()  # Uncomment if you want global coverage
    
    plt.tight_layout()
    
    # Save with high quality
    output_filename = "Enhanced_OM_Risk_Map_2025.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"💾 Map saved as: {output_filename}")

if __name__ == "__main__":
    generate_risk_map()
