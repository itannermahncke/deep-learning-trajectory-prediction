import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def plot_aircraft_trajectory(
    csv_path: str | pd.DataFrame,
    arrow_spacing=10,
    altitude_col="geoaltitude",
    figsize=(10, 8),
):
    """
    Plot aircraft trajectory from CSV data.

    Parameters
    ----------
    csv_path : str
        Path to CSV file
    arrow_spacing : int
        Plot heading arrows every N samples
    altitude_col : str
        Column used for altitude colormap
    figsize : tuple
        Figure size
    """

    # Load data
    if type(csv_path) == str:
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(csv_path)

    # Drop missing required values
    df = df.dropna(subset=["lat", "lon", "heading", altitude_col])

    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    heading = df["heading"].to_numpy()
    altitude = df[altitude_col].to_numpy()

    fig, ax = plt.subplots(figsize=figsize)

    # ---- Create colormapped trajectory line ----
    points = np.array([lon, lat]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    line = LineCollection(
        segments, cmap="viridis", norm=plt.Normalize(altitude.min(), altitude.max())
    )
    line.set_array(altitude[:-1])
    line.set_linewidth(2.5)

    ax.add_collection(line)

    # ---- Add heading arrows ----
    arrow_scale = 0.002  # tune based on geographic scale

    for i in range(0, len(df), arrow_spacing):
        theta = np.deg2rad(90 - heading[i])
        # convert compass heading to plot angle

        dx = arrow_scale * np.cos(theta)
        dy = arrow_scale * np.sin(theta)

        ax.arrow(
            lon[i],
            lat[i],
            dx,
            dy,
            head_width=0.0008,
            head_length=0.0008,
            fc="black",
            ec="black",
            alpha=0.7,
            length_includes_head=True,
        )

    # ---- Labels / formatting ----
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Aircraft Trajectory")

    ax.autoscale()
    ax.set_aspect("equal", adjustable="datalim")

    cbar = plt.colorbar(line, ax=ax)
    cbar.set_label(f"{altitude_col} (m)")

    plt.tight_layout()
    plt.show()


def plot_flight_time_distribution(flight_dfs, time_col="time", figsize=(10, 6)):
    """
    Plot distribution of total flight times across multiple flights.

    Parameters
    ----------
    flight_dfs : list of pandas.DataFrame
        Each dataframe contains flight data with a time column
    time_col : str
        Column name for timestamps (seconds)
    figsize : tuple
        Figure size
    """

    # ---- Compute total flight times ----
    flight_times = []

    for df in flight_dfs:
        if len(df) < 2:
            continue

        t_start = df[time_col].min()
        t_end = df[time_col].max()
        flight_times.append(t_end - t_start)

    flight_times = np.array(flight_times) / 60

    # ---- Stats ----
    mean_val = np.mean(flight_times)
    median_val = np.median(flight_times)
    min_val = np.min(flight_times)
    max_val = np.max(flight_times)

    # ---- Plot ----
    plt.figure(figsize=figsize)
    plt.hist(flight_times, bins=20)

    # vertical markers
    plt.axvline(mean_val, linestyle="--", linewidth=2, label=f"Mean: {mean_val:.1f}min")
    plt.axvline(
        median_val, linestyle="-.", linewidth=2, label=f"Median: {median_val:.1f}min"
    )
    plt.axvline(min_val, linestyle=":", linewidth=2, label=f"Min: {min_val:.1f}min")
    plt.axvline(max_val, linestyle=":", linewidth=2, label=f"Max: {max_val:.1f}min")

    # labels
    plt.xlabel("Flight Time (minutes)")
    plt.ylabel("Count")
    plt.title("Distribution of Flight Durations")
    plt.legend()

    plt.tight_layout()
    plt.show()
