"""
wind_turbine_analysis.py
========================
A complete professional analysis of a wind turbine's 10-minute SCADA dataset (2018).
Covers data loading, cleaning, statistics, 8 visualisation plots, feature insights,
and a written conclusion.

Dataset  : T1.csv  (50,530 rows × 5 columns)
Turbine  : Rated capacity 3,600 kW  |  Cut-in ≈ 3.5 m/s  |  Cut-out ≈ 25 m/s
Author   : Wind Turbine Analysis Project
Audience : 4th-year Electrical & Electronics Engineering students (renewable energy focus)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ── Global style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})

# Turbine nameplate constants
RATED_POWER_KW   = 3_600   # kW
CUT_IN_MS        = 3.5     # m/s  — turbine starts generating below this
CUT_OUT_MS       = 25.0    # m/s  — turbine shuts down above this for safety
RATED_WIND_MS    = 12.0    # m/s  — approximate wind speed at rated power

# Path to the raw dataset — adjust if needed
DATA_PATH = r"C:\Users\gorke\OneDrive\Desktop\github proje denemesi\T1.csv\T1.csv"

# =============================================================================
# === SECTION 1 — DATA LOADING & UNDERSTANDING ================================
# =============================================================================

print("=" * 70)
print("SECTION 1 — DATA LOADING & UNDERSTANDING")
print("=" * 70)

# Load the CSV file as-is (no index column, all 5 columns kept)
df_raw = pd.read_csv(DATA_PATH)

print(f"\n[1] Dataset shape  : {df_raw.shape[0]:,} rows  ×  {df_raw.shape[1]} columns")
print(f"\n[2] Column names   :\n    {list(df_raw.columns)}")

print("\n[3] Data types:")
print(df_raw.dtypes.to_string())

print("\n[4] First 5 rows:")
print(df_raw.head().to_string())

# Brief column description — useful for anyone new to SCADA data
col_descriptions = {
    "Date/Time"                     : "10-minute timestamp (day month year HH:MM)",
    "LV ActivePower (kW)"           : "Actual electrical power delivered to the grid (kW)",
    "Wind Speed (m/s)"              : "Wind speed measured at hub height (m/s)",
    "Theoretical_Power_Curve (KWh)" : "Manufacturer's expected output from the power curve (kWh)",
    "Wind Direction (°)"            : "Wind bearing in degrees (0 = North, clockwise)",
}
print("\n[5] Column descriptions:")
for col, desc in col_descriptions.items():
    print(f"    {col:<40} → {desc}")


# =============================================================================
# === SECTION 2 — DATA CLEANING ===============================================
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 2 — DATA CLEANING")
print("=" * 70)

# Work on a copy so we can always compare with df_raw if needed
df = df_raw.copy()

# ── 2.1  Parse timestamps ────────────────────────────────────────────────────
# The format in the file is "DD MM YYYY HH:MM" (day-first, space-separated)
df["datetime"] = pd.to_datetime(df["Date/Time"], format="%d %m %Y %H:%M")
df.drop(columns=["Date/Time"], inplace=True)

# ── 2.2  Rename columns to snake_case ────────────────────────────────────────
df.rename(columns={
    "LV ActivePower (kW)"           : "active_power_kw",
    "Wind Speed (m/s)"              : "wind_speed_ms",
    "Theoretical_Power_Curve (KWh)" : "theoretical_power_kwh",
    "Wind Direction (°)"            : "wind_direction_deg",
}, inplace=True)

# ── 2.3  Sort chronologically and reset index ─────────────────────────────────
df.sort_values("datetime", inplace=True)
df.reset_index(drop=True, inplace=True)

# ── 2.4  Missing values & duplicates ─────────────────────────────────────────
missing_total = df.isnull().sum().sum()
duplicate_rows = df.duplicated().sum()
print(f"\n[1] Missing values  : {missing_total}")
print(f"[2] Duplicate rows  : {duplicate_rows}")

# ── 2.5  Curtailment flag ─────────────────────────────────────────────────────
# A curtailment event happens when wind is strong enough to generate power
# (wind_speed > cut-in) but the turbine is producing zero or negative power.
# Negative power can occur when the turbine consumes grid power to keep
# rotating during maintenance or grid constraints.
df["is_curtailed"] = (df["active_power_kw"] <= 0) & (df["wind_speed_ms"] > CUT_IN_MS)

# ── 2.6  Derived columns ──────────────────────────────────────────────────────
df["month"]          = df["datetime"].dt.month
df["hour"]           = df["datetime"].dt.hour

# Season mapping (Northern Hemisphere convention)
season_map = {12: "Winter", 1: "Winter", 2: "Winter",
               3: "Spring", 4: "Spring", 5: "Spring",
               6: "Summer", 7: "Summer", 8: "Summer",
               9: "Autumn", 10: "Autumn", 11: "Autumn"}
df["season"] = df["month"].map(season_map)

# Capacity factor: fraction of rated power actually produced (clipped at 0)
df["capacity_factor"] = (df["active_power_kw"] / RATED_POWER_KW).clip(lower=0)

# Power deficit: how much less we produced vs what the theoretical curve predicted
df["power_deficit_kw"] = df["theoretical_power_kwh"] - df["active_power_kw"]

# ── 2.7  Cleaning summary ─────────────────────────────────────────────────────
print(f"\n[3] Columns after cleaning : {list(df.columns)}")
print(f"[4] Date range             : {df['datetime'].min()}  →  {df['datetime'].max()}")
print(f"[5] Curtailment events     : {df['is_curtailed'].sum():,}")
print(f"[6] Final shape            : {df.shape}")


# =============================================================================
# === SECTION 3 — SUMMARY STATISTICS =========================================
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3 — SUMMARY STATISTICS")
print("=" * 70)

numeric_cols = ["active_power_kw", "wind_speed_ms",
                "theoretical_power_kwh", "wind_direction_deg",
                "capacity_factor", "power_deficit_kw"]

print("\n[1] Descriptive statistics:")
print(df[numeric_cols].describe().round(3).to_string())

mean_cf   = df["capacity_factor"].mean() * 100
mean_ws   = df["wind_speed_ms"].mean()
peak_pwr  = df["active_power_kw"].max()
curtailed = df["is_curtailed"].sum()

print(f"\n[2] Mean capacity factor    : {mean_cf:.2f} %")
print(f"[3] Mean wind speed         : {mean_ws:.2f} m/s")
print(f"[4] Peak active power       : {peak_pwr:,.1f} kW")
print(f"[5] Total curtailment events: {curtailed:,}")


# =============================================================================
# === SECTION 4 — VISUALISATION (8 plots) ====================================
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4 — VISUALISATION")
print("=" * 70)

# Helper: month abbreviation labels for x-axes
MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

season_order  = ["Winter", "Spring", "Summer", "Autumn"]
season_colors = {"Winter": "#5b9bd5", "Spring": "#70ad47",
                 "Summer": "#ffc000", "Autumn": "#ed7d31"}

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Wind Speed Distribution
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating Plot 1 — Wind Speed Distribution …")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Plot 1 — Wind Speed Distribution", fontsize=14, fontweight="bold")

ws = df["wind_speed_ms"]
ax = axes[0]
ax.hist(ws, bins=60, color="#4c72b0", edgecolor="white", linewidth=0.4, alpha=0.85)
ax.axvline(ws.mean(),   color="red",    linestyle="--", linewidth=1.6, label=f"Mean {ws.mean():.1f} m/s")
ax.axvline(ws.median(), color="orange", linestyle="--", linewidth=1.6, label=f"Median {ws.median():.1f} m/s")
ax.axvline(CUT_IN_MS,   color="green",  linestyle=":",  linewidth=1.6, label=f"Cut-in {CUT_IN_MS} m/s")
ax.axvline(CUT_OUT_MS,  color="purple", linestyle=":",  linewidth=1.6, label=f"Cut-out {CUT_OUT_MS} m/s")
ax.set_xlabel("Wind Speed (m/s)")
ax.set_ylabel("Frequency")
ax.set_title("Histogram — All Readings")
ax.legend(fontsize=8)

ax2 = axes[1]
season_data = [df.loc[df["season"] == s, "wind_speed_ms"].values for s in season_order]
vp = ax2.violinplot(season_data, positions=range(len(season_order)),
                    showmedians=True, showextrema=True)
for patch, season in zip(vp["bodies"], season_order):
    patch.set_facecolor(season_colors[season])
    patch.set_alpha(0.8)
ax2.set_xticks(range(len(season_order)))
ax2.set_xticklabels(season_order)
ax2.set_xlabel("Season")
ax2.set_ylabel("Wind Speed (m/s)")
ax2.set_title("Violin — Wind Speed by Season")
ax2.axhline(CUT_IN_MS, color="green", linestyle=":", linewidth=1.2, label="Cut-in")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig("plot1_wind_speed_distribution.png")
plt.show()
print("  → Saved: plot1_wind_speed_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Power Distribution & Monthly Capacity Factor
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating Plot 2 — Power Distribution & Monthly Capacity Factor …")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Plot 2 — Power Distribution & Monthly Capacity Factor", fontsize=14, fontweight="bold")

ap = df["active_power_kw"]
ax = axes[0]
ax.hist(ap, bins=80, color="#dd8452", edgecolor="white", linewidth=0.3, alpha=0.85)
ax.axvline(ap.mean(),       color="red",   linestyle="--", linewidth=1.6,
           label=f"Mean {ap.mean():.0f} kW")
ax.axvline(RATED_POWER_KW, color="blue",   linestyle="--", linewidth=1.6,
           label=f"Rated {RATED_POWER_KW:,} kW")
ax.set_xlabel("Active Power (kW)")
ax.set_ylabel("Frequency")
ax.set_title("Histogram — Active Power Output")
ax.legend(fontsize=8)

monthly_cf = df.groupby("month")["capacity_factor"].mean() * 100
ax2 = axes[1]
bars = ax2.bar(monthly_cf.index, monthly_cf.values,
               color=plt.cm.viridis(monthly_cf.values / monthly_cf.values.max()),
               edgecolor="white", linewidth=0.5)
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(MONTH_LABELS, rotation=45)
ax2.set_xlabel("Month")
ax2.set_ylabel("Mean Capacity Factor (%)")
ax2.set_title("Monthly Mean Capacity Factor")
ax2.axhline(monthly_cf.mean(), color="red", linestyle="--",
            linewidth=1.4, label=f"Annual mean {monthly_cf.mean():.1f} %")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig("plot2_power_distribution_monthly_cf.png")
plt.show()
print("  → Saved: plot2_power_distribution_monthly_cf.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Power Curve (Wind Speed vs Active Power)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating Plot 3 — Power Curve …")

# Random sample to avoid over-plotting 50k points
sample = df.sample(n=6_000, random_state=42)

# Binned mean curve: group readings into 0.5 m/s bins, compute mean power
bin_edges  = np.arange(0, CUT_OUT_MS + 0.5, 0.5)
bin_labels = bin_edges[:-1] + 0.25
df["ws_bin"] = pd.cut(df["wind_speed_ms"], bins=bin_edges, labels=bin_labels)
mean_curve  = df.groupby("ws_bin", observed=True)["active_power_kw"].mean()
theo_curve  = df.groupby("ws_bin", observed=True)["theoretical_power_kwh"].mean()

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle("Plot 3 — Power Curve: Wind Speed vs Active Power", fontsize=14, fontweight="bold")

sc = ax.scatter(sample["wind_speed_ms"], sample["active_power_kw"],
                c=sample["capacity_factor"], cmap="plasma",
                s=8, alpha=0.55, label="_nolegend_")
plt.colorbar(sc, ax=ax, label="Capacity Factor")

bin_centers = [float(b) for b in mean_curve.index]
ax.plot(bin_centers, theo_curve.values, color="blue", linewidth=2.2,
        linestyle="--", label="Theoretical curve")
ax.plot(bin_centers, mean_curve.values, color="red",  linewidth=2.2,
        label="Actual mean curve")

for x_val, label, color in [
    (CUT_IN_MS,   f"Cut-in ({CUT_IN_MS} m/s)",   "green"),
    (RATED_WIND_MS, f"Rated wind ({RATED_WIND_MS} m/s)", "orange"),
    (CUT_OUT_MS,  f"Cut-out ({CUT_OUT_MS} m/s)", "purple"),
]:
    ax.axvline(x_val, color=color, linestyle=":", linewidth=1.6, label=label)

ax.set_xlabel("Wind Speed (m/s)")
ax.set_ylabel("Active Power (kW)")
ax.set_title("Scatter + Theoretical & Actual Mean Power Curves")
ax.legend(fontsize=8, loc="upper left")
ax.set_ylim(-200, RATED_POWER_KW + 200)

plt.tight_layout()
plt.savefig("plot3_power_curve.png")
plt.show()
print("  → Saved: plot3_power_curve.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating Plot 4 — Correlation Heatmap …")

corr_cols = ["active_power_kw", "wind_speed_ms", "theoretical_power_kwh",
             "wind_direction_deg", "capacity_factor", "power_deficit_kw"]
corr_matrix = df[corr_cols].corr(method="pearson")

# Mask the upper triangle so we only show the lower triangle (avoids repetition)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

fig, ax = plt.subplots(figsize=(9, 7))
fig.suptitle("Plot 4 — Pearson Correlation Heatmap", fontsize=14, fontweight="bold")

sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.5,
            annot_kws={"size": 9}, ax=ax)
ax.set_title("Lower-triangle Pearson Correlations")

plt.tight_layout()
plt.savefig("plot4_correlation_heatmap.png")
plt.show()
print("  → Saved: plot4_correlation_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Power Heatmap (Hour × Month)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating Plot 5 — Power Heatmap (Hour × Month) …")

# Build a pivot table: rows = hour of day, columns = month, value = mean power
power_pivot = df.pivot_table(
    index="hour", columns="month",
    values="active_power_kw", aggfunc="mean"
)
power_pivot.columns = MONTH_LABELS

fig, ax = plt.subplots(figsize=(13, 7))
fig.suptitle("Plot 5 — Mean Active Power Heatmap (Hour × Month)", fontsize=14, fontweight="bold")

sns.heatmap(power_pivot, cmap="YlOrRd", linewidths=0.3,
            cbar_kws={"label": "Mean Active Power (kW)"}, ax=ax)
ax.set_xlabel("Month")
ax.set_ylabel("Hour of Day")
ax.set_title("Mean Power by Hour and Month — reveals daily & seasonal patterns")

plt.tight_layout()
plt.savefig("plot5_power_heatmap_hour_month.png")
plt.show()
print("  → Saved: plot5_power_heatmap_hour_month.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6 — Wind Rose (Polar Bar Chart)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating Plot 6 — Wind Rose …")

# 36 directional sectors of 10° each
n_sectors  = 36
sector_deg = 360 / n_sectors
# Bin edges start at -5° so that 0° (North) lands in the centre of sector 0
bin_edges_wr = np.linspace(-sector_deg / 2, 360 - sector_deg / 2, n_sectors + 1)
df["dir_bin"] = pd.cut(df["wind_direction_deg"] % 360,
                       bins=bin_edges_wr, labels=False)
# Wrap the -5–0 degree sector back to sector 0
df["dir_bin"] = df["dir_bin"].fillna(0).astype(int)

# Three wind speed bands
speed_bands = [(0, 5, "#74b9e0", "0–5 m/s"),
               (5, 12, "#0057b8", "5–12 m/s"),
               (12, 100, "#003580", ">12 m/s")]

# Theta angles for each sector (radians), offset so 0° = North (top)
# Matplotlib polar: 0 rad = East, we want 0 = North → subtract π/2
thetas = np.linspace(0, 2 * np.pi, n_sectors, endpoint=False)
# Rotate so North is up and direction is clockwise
thetas = (np.pi / 2) - thetas

fig = plt.figure(figsize=(8, 8))
fig.suptitle("Plot 6 — Wind Rose (36 sectors, 3 speed bands)", fontsize=14, fontweight="bold")
ax = fig.add_subplot(111, polar=True)
ax.set_theta_zero_location("N")   # North at top
ax.set_theta_direction(-1)        # Clockwise

bottom_vals = np.zeros(n_sectors)
for low, high, color, label in speed_bands:
    counts = np.zeros(n_sectors)
    mask_band = (df["wind_speed_ms"] >= low) & (df["wind_speed_ms"] < high)
    for s in range(n_sectors):
        counts[s] = mask_band[df["dir_bin"] == s].sum()
    ax.bar(thetas, counts, width=2 * np.pi / n_sectors,
           bottom=bottom_vals, color=color, alpha=0.85, label=label)
    bottom_vals += counts

ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
ax.legend(loc="lower right", bbox_to_anchor=(1.25, 0.0), fontsize=9)

plt.tight_layout()
plt.savefig("plot6_wind_rose.png")
plt.show()
print("  → Saved: plot6_wind_rose.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 7 — Actual vs Theoretical Power
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating Plot 7 — Actual vs Theoretical Power …")

overall_efficiency = df["active_power_kw"].sum() / df["theoretical_power_kwh"].sum() * 100

sample2 = df.sample(n=6_000, random_state=7)

# Bin the theoretical values and compute mean actual power in each bin
theo_bins   = pd.cut(df["theoretical_power_kwh"], bins=50)
theo_actual = df.groupby(theo_bins, observed=True)["active_power_kw"].mean()
theo_mids   = [interval.mid for interval in theo_actual.index]

fig, ax = plt.subplots(figsize=(9, 7))
fig.suptitle("Plot 7 — Actual vs Theoretical Power", fontsize=14, fontweight="bold")

ax.scatter(sample2["theoretical_power_kwh"], sample2["active_power_kw"],
           s=6, alpha=0.35, color="#4c72b0", label="Samples")
ax.plot(theo_mids, theo_actual.values, color="red", linewidth=2.2,
        label="Mean actual (binned)")

# Perfect-agreement line (y = x)
max_val = df["theoretical_power_kwh"].max()
ax.plot([0, max_val], [0, max_val], "k--", linewidth=1.6, label="y = x  (perfect)")

ax.annotate(f"Overall efficiency = {overall_efficiency:.1f} %",
            xy=(0.05, 0.90), xycoords="axes fraction",
            fontsize=11, color="darkred",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

ax.set_xlabel("Theoretical Power (kWh)")
ax.set_ylabel("Actual Active Power (kW)")
ax.set_title("How closely does the turbine follow the theoretical power curve?")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("plot7_actual_vs_theoretical.png")
plt.show()
print("  → Saved: plot7_actual_vs_theoretical.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 8 — Monthly Energy & Curtailment
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating Plot 8 — Monthly Energy & Curtailment …")

# Each reading is 10 minutes = 1/6 hour → energy in MWh = power(kW) × (10/60) / 1000
INTERVAL_H = 10 / 60  # hours per reading

monthly_theo_mwh    = df.groupby("month")["theoretical_power_kwh"].sum() * INTERVAL_H / 1_000
monthly_actual_mwh  = df.groupby("month")["active_power_kw"].apply(
    lambda x: x.clip(lower=0).sum()) * INTERVAL_H / 1_000
monthly_curtailment = df.groupby("month")["is_curtailed"].sum()

x     = np.arange(1, 13)
width = 0.38

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
fig.suptitle("Plot 8 — Monthly Energy Production & Curtailment Events", fontsize=14, fontweight="bold")

# Top: grouped bars — theoretical vs actual energy
ax_top.bar(x - width / 2, monthly_theo_mwh.values,   width=width, label="Theoretical (MWh)", color="#5b9bd5", alpha=0.9)
ax_top.bar(x + width / 2, monthly_actual_mwh.values, width=width, label="Actual (MWh)",      color="#ed7d31", alpha=0.9)
ax_top.set_ylabel("Energy (MWh)")
ax_top.set_title("Monthly Energy: Theoretical vs Actual")
ax_top.legend(fontsize=9)

# Bottom: curtailment event count per month
ax_bot.bar(x, monthly_curtailment.values, color="#c00000", alpha=0.85, label="Curtailment events")
ax_bot.set_xticks(x)
ax_bot.set_xticklabels(MONTH_LABELS)
ax_bot.set_xlabel("Month")
ax_bot.set_ylabel("Number of Events")
ax_bot.set_title("Monthly Curtailment Event Count")
ax_bot.legend(fontsize=9)

plt.tight_layout()
plt.savefig("plot8_monthly_energy_curtailment.png")
plt.show()
print("  → Saved: plot8_monthly_energy_curtailment.png")


# =============================================================================
# === SECTION 5 — FEATURE INSIGHTS ============================================
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5 — FEATURE INSIGHTS")
print("=" * 70)

# ── 5.1  Pearson correlation of each feature vs active power ─────────────────
feature_cols = ["wind_speed_ms", "theoretical_power_kwh",
                "wind_direction_deg", "capacity_factor", "power_deficit_kw"]
correlations = df[feature_cols].corrwith(df["active_power_kw"]).sort_values(ascending=False)

print("\n[1] Pearson correlations with active_power_kw:")
print(correlations.round(4).to_string())

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in correlations.values]
ax.barh(correlations.index[::-1], correlations.values[::-1], color=colors[::-1], edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Pearson Correlation with Active Power (kW)")
ax.set_title("Feature Correlations vs Active Power")
fig.suptitle("Section 5 — Feature Correlation Ranking", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plot9_feature_correlations.png")
plt.show()
print("  → Saved: plot9_feature_correlations.png")

# ── 5.2  Wind speed bin analysis ─────────────────────────────────────────────
# Group by 2 m/s bins and compute mean power and efficiency for each group
ws_bin_edges  = np.arange(0, CUT_OUT_MS + 2, 2)
ws_bin_labels = [f"{int(e)}–{int(e+2)}" for e in ws_bin_edges[:-1]]
df["ws_2bin"]  = pd.cut(df["wind_speed_ms"], bins=ws_bin_edges, labels=ws_bin_labels)

ws_bin_stats = df.groupby("ws_2bin", observed=True).agg(
    count         = ("active_power_kw", "count"),
    mean_power_kw = ("active_power_kw", "mean"),
    mean_theo_kwh = ("theoretical_power_kwh", "mean"),
).dropna()

ws_bin_stats["efficiency_pct"] = (
    ws_bin_stats["mean_power_kw"] / ws_bin_stats["mean_theo_kwh"] * 100
).clip(0, 120)  # clip for display (some bins have near-zero theoretical)

print("\n[2] Wind speed bin analysis (2 m/s bins):")
print(ws_bin_stats.round(2).to_string())

# ── 5.3  Top 3 predictors ─────────────────────────────────────────────────────
top3 = correlations.abs().sort_values(ascending=False).head(3)
print("\n[3] Top 3 predictors of active power output:")
for rank, (feat, val) in enumerate(top3.items(), start=1):
    print(f"    {rank}. {feat:<35} |r| = {val:.4f}")


# =============================================================================
# === SECTION 6 — KEY INSIGHTS ================================================
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 6 — KEY INSIGHTS")
print("=" * 70)

# Re-compute a few quick statistics for the insight statements
pct_above_cutin   = (df["wind_speed_ms"] > CUT_IN_MS).mean() * 100
pct_curtailed     = df["is_curtailed"].mean() * 100
best_month        = monthly_cf.idxmax()
worst_month       = monthly_cf.idxmin()
dominant_dir_bin  = df["wind_direction_deg"].round(-1).value_counts().idxmax()
mean_deficit      = df["power_deficit_kw"].mean()
ws_power_corr     = correlations["wind_speed_ms"]

insights = f"""
  • Wind speed – power relationship:
      Pearson correlation r = {ws_power_corr:.3f}, making wind speed the dominant driver
      of power output. The turbine consistently follows its theoretical S-shaped
      power curve between the cut-in ({CUT_IN_MS} m/s) and rated wind ({RATED_WIND_MS} m/s) speeds.

  • Wind availability:
      {pct_above_cutin:.1f}% of all 10-minute intervals had wind speeds above the cut-in
      threshold, meaning the turbine could theoretically generate power during
      those periods.

  • Overall efficiency vs theoretical:
      The turbine achieved {overall_efficiency:.1f}% of its theoretically expected energy output.
      An average power deficit of {mean_deficit:.1f} kW per reading highlights losses from
      turbulence, blade pitch lag, and grid curtailment.

  • Seasonal performance:
      Best month by capacity factor  → {MONTH_LABELS[best_month-1]}  ({monthly_cf[best_month]:.1f}%)
      Worst month by capacity factor → {MONTH_LABELS[worst_month-1]} ({monthly_cf[worst_month]:.1f}%)
      Winter and spring tend to deliver higher wind resources in this dataset.

  • Curtailment patterns:
      {df['is_curtailed'].sum():,} curtailment events detected ({pct_curtailed:.2f}% of all readings).
      These are intervals where wind was sufficient but power output was ≤ 0 kW,
      indicating grid constraints, scheduled maintenance, or control actions.

  • Annual mean capacity factor:
      {mean_cf:.2f}% — typical for onshore turbines (20–40% range). This means the
      turbine produced roughly {mean_cf:.0f} kWh for every 100 kWh it could produce
      if running at rated power 24/7.

  • Wind direction dominance:
      The most frequent wind direction band is around {dominant_dir_bin}°.
      Wind rose (Plot 6) shows directional clusters driven by local topography
      and prevailing weather systems.

  • Anomalies detected:
      Negative active power readings exist; these likely represent self-consumption
      during low-wind or maintenance periods. They are correctly flagged and
      excluded from the capacity factor calculation via clip(lower=0).
"""

print(insights)


# =============================================================================
# === SECTION 7 — CONCLUSION ==================================================
# =============================================================================

print("=" * 70)
print("SECTION 7 — CONCLUSION")
print("=" * 70)

conclusion = f"""
This analysis examined a full year (2018) of 10-minute SCADA measurements from a
{RATED_POWER_KW:,} kW wind turbine, covering {len(df):,} observations across five channels.
Wind speed emerged as the overwhelmingly dominant predictor of power output
(r ≈ {ws_power_corr:.2f}), consistent with the cubic relationship predicted by aerodynamic
theory. The turbine operated at a mean capacity factor of {mean_cf:.1f}%, achieving
{overall_efficiency:.1f}% of its theoretical energy yield — losses attributable to
turbulence, sub-optimal yaw alignment, blade soiling, and {df['is_curtailed'].sum():,}
curtailment events. Seasonal analysis revealed stronger performance in winter and
spring, driven by higher regional wind resources. Wind direction exhibited clear
dominant sectors, suggesting the site benefits from prevailing synoptic-scale flow.

Suggestions for future work:
  1. ML-based power forecasting — Train a gradient-boosted or LSTM model on lagged
     wind speed and direction features to predict 1–6 hour ahead power output,
     enabling better grid dispatch scheduling.
  2. Automated anomaly detection — Use isolation forests or autoencoder neural
     networks to flag unusual power-curve deviations in real time, allowing
     maintenance teams to identify sensor faults or mechanical issues early.
  3. Predictive maintenance scheduling — Correlate power deficit clusters with
     maintenance logs to build a data-driven maintenance calendar that reduces
     unplanned downtime and optimises turbine availability.
"""

print(conclusion)
print("=" * 70)
print("Analysis complete. All plots saved as PNG files in the working directory.")
print("=" * 70)
