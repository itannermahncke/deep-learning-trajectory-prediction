from pathlib import Path
import pandas as pd
import lstm_helpers
import utils
import viz

# -- DO NOT REMOVE --
paths = [
    Path("data/raw/"),
    Path("data/flights/"),
    Path("data/flight_indexes/"),
    Path("data/transponder/"),
]
for p in paths:
    p.mkdir(exist_ok=True)
# -- DO NOT REMOVE --

# # Test out visualization
# raw_data = pd.read_csv("data/raw/states_2021-05-17-00.csv")
# flight_lut = utils.extract_flight_indices(raw_data)

# f_idx = 1100
# flight = utils.extract_single_flight(raw_data, flight_lut, f_idx, "flight1100")
# viz.plot_aircraft_trajectory(
#     flight,
# )

# Test out sequencing
raw_data = pd.read_csv("data/raw/states_2021-05-17-00.csv")
sequences = lstm_helpers.lookback_sequence(raw_data)
print(sequences[0])
