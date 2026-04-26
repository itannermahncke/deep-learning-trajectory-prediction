"""
Helper functions related to preprocessing for the LSTM.
"""

import pandas as pd
import utils
import numpy as np


def lookback_sequence(raw_data, lookback_size=100, columns=None):
    """
    Given a full raw dataset, drop excess columns and produce a sequence of lookback sections that do not contain flight-to-flight overlap.

    Args:
        raw_data

    Returns:
        a list of lookback_size-sized arrays containing sections of single-flight data. Arrays referencing the same flight are offset from the next in sequence are offset by one timestep.
    """
    print(raw_data)
    # this object will be returned
    sequences = []
    # grab sorted flights
    all_flights: list[pd.DataFrame] = utils.extract_all_flights(
        raw_data,
        utils.extract_flight_indices(raw_data),
    )
    print(all_flights)
    for flight in all_flights:
        flight = flight[columns]

        # skip flights that are too short
        if len(flight) < lookback_size + 1:
            continue

        for i in range(len(flight) - lookback_size):
            seq = flight.iloc[i : i + lookback_size + 1].to_numpy()
            sequences.append(seq)
    # return
    print(sequences)
    return np.stack(sequences)
