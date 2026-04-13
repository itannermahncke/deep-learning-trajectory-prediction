import pandas as pd

csv = "states_2021-05-17-00.csv"
df = pd.read_csv(f"data/raw/{csv}")

df = df.reset_index().rename(columns={"index": "global_index"})
print("Unique icao24:", df["icao24"].nunique())

flight_rows = []

for icao24, group in df.groupby("icao24", sort=False):
    group = group.copy()
    indices = group["global_index"].to_list()
    states = group["onground"].to_list()

    in_flight = False
    start_idx = None

    for i, state in enumerate(states):
        if pd.isna(state):
            continue

        if not in_flight and state is False:
            in_flight = True
            start_idx = indices[i]

        elif in_flight and state is True:
            end_idx = indices[i]
            flight_rows.append({"icao24": icao24, "start": start_idx, "end": end_idx})
            in_flight = False
            start_idx = None

flights_df = pd.DataFrame(flight_rows)

print(flights_df.head())
print("Number of detected flights:", len(flights_df))

flights_df.to_csv(f"data/flights/{csv}-flight-segments.csv", index=True)
