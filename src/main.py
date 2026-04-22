from pathlib import Path
import pandas as pd
import lstm_helpers
import src.utils as utils
import src.viz as viz

# -- DO NOT REMOVE --
paths = [
    Path("data/raw/"),
    Path("data/flights/"),
    Path("data/flight_indexes/"),
    Path("data/transponder/"),
]
for p in paths:
    p.mkdir(exist_ok=True)


raw_data = pd.read_csv("data/raw/states_2021-05-17-00.csv")
flight_idx_table = utils.extract_flight_indices(raw_data, None, "test.csv")
# -- DO NOT REMOVE --


viz.plot_flight_time_distribution(utils.extract_all_flights(raw_data, flight_idx_table))
# # Test out visualization
# raw_data = pd.read_csv("data/raw/states_2021-05-17-00.csv")
# flight_lut = utils.extract_flight_indices(raw_data)

# f_idx = 1100
# flight = utils.extract_single_flight(raw_data, flight_lut, f_idx, "flight1100")
# viz.plot_aircraft_trajectory(
#     flight,
# )

# Test out sequencing
# sequences = lstm_helpers.lookback_sequence(raw_data)

# # longest flight
# longest_flight_idx = utils.extract_longest_flight(
#     flight_idx_table,
# )
# print(longest_flight_idx)
# viz.plot_aircraft_trajectory(
#     utils.extract_single_flight(
#         raw_data,
#         flight_idx_table,
#         longest_flight_idx,
#     )
# )

# viz.plot_aircraft_trajectory(
#     utils.extract_single_flight(
#         raw_data,
#         flight_idx_table,
#         600,
#     )
# )
