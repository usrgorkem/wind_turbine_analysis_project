# Wind Turbine SCADA Data Analysis

A complete, professional exploratory data analysis (EDA) of a 3,600 kW wind turbine's
10-minute SCADA measurements covering the full year 2018.

---

## Dataset Information

| Property        | Value                                      |
|-----------------|--------------------------------------------|
| File            | `T1.csv`                                   |
| Rows            | 50,530                                     |
| Columns         | 5                                          |
| Interval        | 10 minutes                                 |
| Period          | 01 Jan 2018 – 31 Dec 2018                 |
| Rated capacity  | 3,600 kW                                   |
| Cut-in speed    | ≈ 3.5 m/s                                 |
| Cut-out speed   | ≈ 25.0 m/s                                |
| Source note     | Publicly available wind turbine SCADA dataset (Kaggle) |

### Columns

| Column                          | Unit  | Description                                      |
|---------------------------------|-------|--------------------------------------------------|
| `Date/Time`                     | —     | Timestamp: `DD MM YYYY HH:MM`                    |
| `LV ActivePower (kW)`           | kW    | Actual electrical power delivered to the grid    |
| `Wind Speed (m/s)`              | m/s   | Wind speed at hub height                         |
| `Theoretical_Power_Curve (KWh)` | kWh   | Manufacturer's expected output per reading       |
| `Wind Direction (°)`            | °     | Wind bearing (0 = North, clockwise)              |

---

## Tools Used

| Library      | Purpose                                      |
|--------------|----------------------------------------------|
| `pandas`     | Data loading, cleaning, grouping             |
| `numpy`      | Numerical operations, binning                |
| `matplotlib` | All plotting (histograms, scatter, polar)    |
| `seaborn`    | Heatmaps, violin plots, styling              |

Python version: 3.9+

---

## How to Run

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn
```

### 2. Place the dataset

## 2. Place the dataset

The script automatically searches for `T1.csv` in the following locations:

- Same directory as the script  
- Current working directory  
- Downloads folder  

Alternatively, you can specify a custom path:

```bash
python wind_turbine_analysis.py --data "C:\path\to\T1.csv"
```

### 3. Run the analysis

```bash
python wind_turbine_analysis.py
```

All 9 PNG plots will be saved to the current working directory and displayed
interactively. Console output covers all sections (statistics, insights, conclusion).

---

## Key Insights

- **Wind speed is the dominant driver** of power output (Pearson r ≈ 0.94), consistent
  with the theoretical cubic relationship between wind speed and extractable power.

- **Mean capacity factor ≈ 26–30%** — typical for an onshore wind turbine, meaning
  the turbine delivers roughly 26–30% of what it would produce if running at full
  power 24/7.

- **Overall efficiency ≈ 85–90%** of the theoretical power curve — losses arise from
  turbulence, yaw misalignment, blade soiling, and curtailment events.

- **Curtailment events** (wind speed > cut-in but power ≤ 0 kW) occur in a small
  but measurable fraction of readings, pointing to grid constraints or planned
  maintenance windows.

- **Seasonal variation** is pronounced: winter/spring months outperform summer/autumn,
  driven by stronger regional wind resources during the colder half of the year.

---

## Output Files

| File                                    | Description                                  |
|-----------------------------------------|----------------------------------------------|
| `wind_turbine_analysis.py`              | Main analysis script                         |
| `plot1_wind_speed_distribution.png`     | Histogram + seasonal violin of wind speed    |
| `plot2_power_distribution_monthly_cf.png` | Power histogram + monthly capacity factor  |
| `plot3_power_curve.png`                 | Scatter + theoretical & actual power curves  |
| `plot4_correlation_heatmap.png`         | Pearson correlation heatmap (lower triangle) |
| `plot5_power_heatmap_hour_month.png`    | Mean power by hour-of-day and month          |
| `plot6_wind_rose.png`                   | Polar wind rose (36 sectors, 3 speed bands)  |
| `plot7_actual_vs_theoretical.png`       | Actual vs theoretical power scatter          |
| `plot8_monthly_energy_curtailment.png`  | Monthly MWh bars + curtailment count         |
| `plot9_feature_correlations.png`        | Feature correlation ranking bar chart        |
| `README.md`                             | This file                                    |
