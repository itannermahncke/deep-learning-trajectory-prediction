"""
Helper functions related to preprocessing for the LSTM.
"""

import pandas as pd
import utils


def lookback_sequence(raw_data: pd.DataFrame, lookback_size=100):
    """
    Given a full raw dataset, drop excess columns and produce a sequence of lookback sections that do not contain flight-to-flight overlap.

    Args:
        raw_data

    Returns:
        a list of lookback_size-sized arrays containing sections of single-flight data. Arrays referencing the same flight are offset from the next in sequence are offset by one timestep.
    """
    # this object will be returned
    sequences = []

    # grab sorted flights
    all_flights: list[pd.DataFrame] = utils.extract_all_flights(
        raw_data,
        utils.extract_flight_indices(raw_data),
    )

    # sequence each flight
    for flight in all_flights:
        # toss irrelevant data
        flight = flight[
            [
                "time",
                "lat",
                "lon",
                "velocity",
                "heading",
                "baroaltitude",
                "geoaltitude",
            ]
        ]
        # add a rightward sequence for each datapoint if possible; otherwise go to next flight
        for i in range(len(flight)):
            if (len(flight) - i + 1) >= lookback_size:
                sequences.append(flight.iloc[i : lookback_size + 1].to_numpy())
            else:
                break

    # return
    return sequences
