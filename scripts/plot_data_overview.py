"""
plot_data_overview.py
Generate overview figures for the project datasets.

Produces:
  - Station data availability timeline (colored by mean growing season length)
  - Climate indices time series (ONI, NAO, AMO)
  - Ocean heat content time series by basin
  - Station map (requires cartopy -- skipped if not installed)

Usage:
    python scripts/plot_data_overview.py

Figures are saved to figures/.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Clean up default style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})

# Station names for labeling
STATION_NAMES = {
    "USW00014739": "Boston",
    "USW00094728": "New York",
    "USW00013739": "Raleigh",
    "USW00012839": "Miami",
    "USW00003927": "Nashville",
    "USW00014820": "Chicago",
    "USW00014733": "Burlington",
    "USW00093820": "Atlanta",
    "USW00013874": "Pittsburgh",
    "USW00012960": "New Orleans",
    "USW00014764": "Portland ME",
    "USW00093721": "Richmond",
}


def plot_station_availability():
    """Horizontal bar chart showing how many years of data each station has."""
    gs = pd.read_csv(DATA_DIR / "example_data.csv")

    stations = gs.groupby("station_id").agg(
        min_year=("year", "min"),
        max_year=("year", "max"),
        count=("year", "count"),
        mean_gsl=("growing_season_length", "mean"),
    ).reset_index()
    stations["label"] = stations["station_id"].map(STATION_NAMES).fillna(stations["station_id"])
    stations = stations.sort_values("mean_gsl", ascending=True)

    colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, len(stations)))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (_, row) in enumerate(stations.iterrows()):
        ax.barh(i, row["max_year"] - row["min_year"],
                left=row["min_year"], height=0.65, color=colors[i],
                edgecolor="white", linewidth=0.5)
        ax.text(row["max_year"] + 1, i, f'{row["count"]} yr',
                va="center", fontsize=8, color="#666666")

    ax.set_yticks(range(len(stations)))
    ax.set_yticklabels(stations["label"], fontsize=9)
    ax.set_xlabel("Year")
    ax.set_title("Station Data Availability (colored by mean GSL)", fontsize=12)
    ax.set_xlim(1860, 2035)

    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.RdYlBu_r,
        norm=plt.Normalize(stations["mean_gsl"].min(), stations["mean_gsl"].max())
    )
    cbar = fig.colorbar(sm, ax=ax, pad=0.12, aspect=30, shrink=0.8)
    cbar.set_label("Mean GSL (days)", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "station_availability.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved figures/station_availability.png")


def plot_climate_indices():
    """Time series of climate teleconnection indices from 1950 onward."""
    idx = pd.read_csv(DATA_DIR / "climate_indices.csv")
    idx = idx[idx["year"] >= 1950]

    fig, ax = plt.subplots(figsize=(8, 3.5))

    for col, label, color in [
        ("oni_annual", "ENSO (ONI)", "#e74c3c"),
        ("nao_annual", "NAO", "#2d6a9f"),
        ("amo_annual", "AMO", "#e67e22"),
    ]:
        if col in idx.columns:
            ax.plot(idx["year"], idx[col], label=label, color=color,
                    linewidth=1.2, alpha=0.85)

    ax.axhline(0, color="#aaaaaa", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Year")
    ax.set_ylabel("Index Value")
    ax.set_title("Climate Teleconnection Indices (annual means)", fontsize=12)
    ax.legend(fontsize=9, ncol=3, loc="upper left")
    ax.set_xlim(1950, 2026)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "climate_indices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved figures/climate_indices.png")


def plot_ohc():
    """Time series of ocean heat content by basin."""
    ohc = pd.read_csv(DATA_DIR / "argo_ohc.csv")

    fig, ax = plt.subplots(figsize=(8, 3.5))

    for col, label, color in [
        ("ohc700_north_atlantic", "North Atlantic 0-700m", "#c0392b"),
        ("ohc700_pacific", "Pacific 0-700m", "#16a085"),
        ("ohc700_world", "World 0-700m", "#2c3e50"),
    ]:
        if col in ohc.columns:
            ax.plot(ohc["year"], ohc[col], label=label, color=color, linewidth=1.8)

    # Uncertainty band for North Atlantic
    if "ohc700_north_atlantic_se" in ohc.columns:
        ax.fill_between(
            ohc["year"],
            ohc["ohc700_north_atlantic"] - 2 * ohc["ohc700_north_atlantic_se"],
            ohc["ohc700_north_atlantic"] + 2 * ohc["ohc700_north_atlantic_se"],
            color="#c0392b", alpha=0.12,
        )

    ax.axhline(0, color="#aaaaaa", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Year")
    ax.set_ylabel("OHC Anomaly (10$^{22}$ J)")
    ax.set_title("Ocean Heat Content (NOAA/NCEI Levitus)", fontsize=12)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "ohc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved figures/ohc.png")


def plot_station_map():
    """Map of station locations. Requires cartopy (optional)."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.patheffects as pe
    except ImportError:
        print("Skipping station map (install cartopy: pip install cartopy)")
        return

    gs = pd.read_csv(DATA_DIR / "example_data.csv")

    known_coords = {
        "USW00014739": (42.36, -71.01, "Boston"),
        "USW00094728": (40.78, -73.97, "New York"),
        "USW00013739": (35.89, -78.78, "Raleigh"),
        "USW00012839": (25.79, -80.32, "Miami"),
        "USW00003927": (36.12, -86.69, "Nashville"),
        "USW00014820": (41.98, -87.91, "Chicago"),
        "USW00014733": (44.47, -73.15, "Burlington"),
        "USW00093820": (33.63, -84.44, "Atlanta"),
        "USW00013874": (40.49, -80.23, "Pittsburgh"),
        "USW00012960": (29.98, -90.25, "New Orleans"),
        "USW00014764": (43.64, -70.31, "Portland ME"),
        "USW00093721": (37.51, -77.32, "Richmond"),
    }

    proj = ccrs.LambertConformal(central_longitude=-82, central_latitude=37)
    fig, ax = plt.subplots(figsize=(8, 6.5), subplot_kw={"projection": proj})
    ax.set_extent([-98, -65, 24, 48], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="#f5f5f2", edgecolor="none")
    ax.add_feature(cfeature.OCEAN, facecolor="#dce9f2")
    ax.add_feature(cfeature.LAKES, facecolor="#dce9f2", edgecolor="#999999", linewidth=0.4)
    ax.add_feature(cfeature.STATES, edgecolor="#aaaaaa", linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, edgecolor="#666666", linewidth=0.8)
    ax.add_feature(cfeature.COASTLINE, edgecolor="#888888", linewidth=0.6)

    label_offsets = {
        "Boston":      (6, -8, "left", "top"),
        "New York":    (6, -8, "left", "top"),
        "Raleigh":     (6, 4, "left", "bottom"),
        "Miami":       (6, 4, "left", "bottom"),
        "Nashville":   (-6, 4, "right", "bottom"),
        "Chicago":     (-6, 4, "right", "bottom"),
        "Burlington":  (6, 4, "left", "bottom"),
        "Atlanta":     (-6, -4, "right", "top"),
        "Pittsburgh":  (-6, -6, "right", "top"),
        "New Orleans": (6, 4, "left", "bottom"),
        "Portland ME": (6, -6, "left", "top"),
        "Richmond":    (6, 4, "left", "bottom"),
    }

    for sid, (lat, lon, name) in known_coords.items():
        row = gs[gs["station_id"] == sid]
        n = len(row)
        ax.scatter(lon, lat, s=max(50, n / 2), c="#2d6a9f", edgecolors="white",
                   linewidth=0.8, zorder=5, transform=ccrs.PlateCarree())
        dx, dy, ha, va = label_offsets.get(name, (6, 4, "left", "bottom"))
        txt = ax.annotate(
            f"{name} ({n} yr)",
            xy=(lon, lat), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            fontsize=8, ha=ha, va=va,
            xytext=(dx, dy), textcoords="offset points", zorder=6,
            color="#333333",
        )
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])

    ax.set_title("Sample Station Locations", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "station_map.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved figures/station_map.png")


if __name__ == "__main__":
    plot_station_availability()
    plot_climate_indices()
    plot_ohc()
    plot_station_map()
    print("\nDone!")
