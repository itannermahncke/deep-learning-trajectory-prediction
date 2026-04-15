import pandas as pd

csv = "states_2021-05-17-00.csv"
df = pd.read_csv(f"data/raw/{csv}")


def extract_flight_indices(all_flight_df, csv_name=None):
    """
    Given a 24-hour dataset, extract the start and end indices for each unique flight in the dataset.
    Flights begin and end at the first and last instances of onground=False.

    Args:
        all_flight_df (DataFrame): 24-hour dataset, sourced from OpenSky.
        csv_name (str): Name of desired CSV output. If left as None, no file is created.
    """
    # prep for index extraction
    df = all_flight_df.reset_index().rename(columns={"index": "global_index"})
    flight_rows = []

    # loop through each unique transponder code - may contain multiple flights per code
    for icao24, group in df.groupby("icao24", sort=False):
        # extract global index and onground state for each row of the transponder
        group = group.copy()
        indices = group["global_index"].to_list()
        states = group["onground"].to_list()

        # tracker bools
        in_flight = False
        start_idx = None

        # for each state, determine if it is a starting or ending index, if either
        for i, state in enumerate(states):
            if pd.isna(state):
                continue

            # first instance of onground being false
            if not in_flight and state is False:
                in_flight = True
                start_idx = indices[i]

            # first instance of onground being true
            elif in_flight and state is True:
                # save the prior index
                end_idx = indices[i - 1]
                flight_rows.append(
                    {"icao24": icao24, "start": start_idx, "end": end_idx}
                )
                in_flight = False
                start_idx = None

    # flight index dataset
    flight_index_df = pd.DataFrame(flight_rows).reset_index()

    # save to CSV if desired
    if csv_name is not None:
        flight_index_df.to_csv(f"data/flight_indexes/{csv_name}", index=True)

    # return the flight index dataset
    return flight_index_df


def extract_single_flight(
    all_flight_df, flight_index_df, flight_index=0, csv_name=None
):
    """
    Given a 24-hour OpenSky dataset and a flight index DataFrame associated with it, extract all rows associated with a single flight.
    """
    # extract the row of data in the indexing DF associated w/ desired flight
    flight_info = flight_index_df.iloc[flight_index].to_dict()

    # slice the 24-hour dataset by the desired flight's start and end values
    data_segment = all_flight_df.iloc[flight_info["start"] : flight_info["end"] + 1]

    # filter by desired flight's transponder code to get full, uncluttered flight
    single_flight = data_segment[data_segment["icao24"] == flight_info["icao24"]]

    # save to CSV if desired
    if csv_name is not None:
        single_flight.to_csv(f"data/flights/{csv_name}.csv", index=True)

    # return the full flight as a DF
    return single_flight


def extract_all_flights(
    all_flight_df,
    flight_index_df,
):
    """
    Extract each unique flight in a raw dataset as its own DataFrame. Return a list of flights.
    """
    flights = []
    flight_count = len(flight_index_df)
    for flight_idx in range(0, flight_count):
        flights.append(
            extract_single_flight(
                all_flight_df,
                flight_index_df,
                flight_idx,
            )
        )
    return flights


def extract_single_transponder(all_flight_df, icao24_code, csv_name=None):
    """
    Given a 24-hour OpenSky dataset and a transponder code, extract all rows associated with a single transponder.
    """
    # filter by desired flight's transponder code to get all flights
    single_code = all_flight_df[all_flight_df["icao24"] == icao24_code]

    # save to CSV if desired
    if csv_name is not None:
        single_code.to_csv(f"data/transponder/{icao24_code}_{csv_name}", index=True)

    # return the full dataset as a DF
    return single_code


csv_name = "states_2021-05-17-00.csv"
full_flight_df = pd.read_csv(f"data/raw/{csv_name}")
flight_index_df = extract_flight_indices(full_flight_df, f"flight_indexes_{csv_name}")
flight_index_df = pd.read_csv(
    f"data/flight_indexes/flight_indexes_states_2021-05-17-00.csv"
)
flight_index_df = pd.read_csv(
    f"data/flight_indexes/flight_indexes_states_2021-05-17-00.csv"
)

desired_flight = 1049
single_flight = extract_single_flight(
    full_flight_df,
    flight_index_df,
    desired_flight,
    f"flight{desired_flight}_{csv_name}",
)

# code = "0d0997"
# single_code = extract_single_transponder(
#     full_flight_df,
#     code,
#     f"code{code}_{csv_name}",
# )
