"""
Functions for processing data and calculating errors for dashboard.py.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler

# Initiate scalers
SCALER_DATA = StandardScaler()


def read_csv(file_path: str) -> pd.DataFrame:
    """
    Clean and read a CSV file and convert it to a DataFrame.

    Args:
        file_path: a string representing the path to the CSV file.

    Returns:
        a DataFrame containing the cleaned data.
    """
    # Read the data file
    df = pd.read_csv(file_path)
    df = pd.DataFrame(df)
    df = df.map(lambda x: x.replace(",", "") if isinstance(x, str) else x)
    # remove any columns that contain mostly NA
    df = df.dropna(thresh=len(df) - 10, axis=1)
    # remove any rows that contain a null value
    df = df.replace(["Null", "Null value"], np.nan)
    df = df.dropna(axis=0)
    return df


def process_time_columns(df, timestamp):
    """
    Split the timestamp column into time and day-of-week columns

    Args:
        df: a DataFrame containing the timestamp column.
        timestamp: a string representing the name of the timestamp column.

    Returns:
        df: a DataFrame containing the time and day-of-week columns.
    """
    df[timestamp] = pd.to_datetime(df[timestamp])
    df = pd.concat(
        [
            pd.get_dummies(df[timestamp].dt.day_name()).astype(int),
            df,
        ],
        axis=1,
    )
    df.insert(
        0,
        "Time",
        df[timestamp].dt.hour * 60 + df[timestamp].dt.minute,
    )
    df.drop([timestamp], axis=1, inplace=True)
    return df


def get_data(processed_data, look_back, train_size):
    """
    Split a dataset into x and y test batches

    Args:
        processsed_data: a DataFrame to split into train and test data.
        look_back: an integer representing the size of the prediction window.
        train_size: an integer representing the number of data-points to
        allocate to the training batch.

    Returns:
        processed_data: a TensorDataset containing the x and y test data.
    """

    # MinMaxScaler the prediction variable
    processed_data = SCALER_DATA.fit_transform(processed_data)
    sequenced_data = []

    # Create all possible sequences of length seq_len
    for index in range(len(processed_data) - look_back):
        sequenced_data.append(processed_data[index : index + look_back])
    sequenced_data = np.array(sequenced_data)

    # Divide the dataset into x and y data
    x_test = sequenced_data[train_size:, :-1]
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_test = sequenced_data[train_size:, -1, :]
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    # Wrap the data in a TensorDataset
    processed_data = TensorDataset(x_test, y_test)
    return processed_data


def calculate_error(predicted_values, actual_values, labels, indexes, alpha=5 / 11):
    """
    Calculate the error between the predicted and actual values.

    Args:
        predicted_values: a list of the predicted values.
        actual_values: a list of the actual values.

    Returns:
        errors: a DataFrame of the errors.
    """
    errors = {"em_avg": [], "mean_abs_err": []}
    for label in labels:
        errors[f"{label}_err"] = []
    for i, row in enumerate(predicted_values):
        # Add the mean absolute error to the errors dictionary
        current_error = sum(  # ignore time, day, and outdoor air variables
            abs(actual_values[i][j] - row[j]) for j in indexes
        ) / (len(indexes))
        errors["mean_abs_err"].append(current_error)

        # Add the exponential moving average to the errors dictionary
        if i == 0:
            errors["em_avg"].append(current_error)
        else:
            current_ema = (alpha * errors["mean_abs_err"][i]) + (
                (1 - alpha) * errors["em_avg"][i - 1]
            )
            errors["em_avg"].append(current_ema)

        # Add the error for each variable to the errors dictionary
        for j, label in enumerate(labels):
            errors[f"{label}_err"].append(abs(actual_values[i][j] - row[j]))
    errors = pd.DataFrame(errors)
    return errors


def compare_setpoint(df, variables):
    """
    Compare the setpoint to the actual values.

    Args:
        df: a DataFrame containing the setpoint and actual values.
        variables: a list of the variables that have a setpoint to compare to

    Returns:
        df: a DataFrame containing the setpoint and actual values.
    """
    anomalies = pd.DataFrame()
    for variable in variables:
        anomalies[variable] = df[variable]
        anomalies[f"{variable}Spt"] = df[f"{variable}Spt"]
        anomalies[f"{variable}_setpoint_diff"] = abs(
            df[variable] - df[f"{variable}Spt"]
        )
        # Calculate mean and standard deviation for the setpoint difference column
        mean_diff = anomalies[f"{variable}_setpoint_diff"].mean()
        stdev_diff = anomalies[f"{variable}_setpoint_diff"].std()

        # Calculate z-scores and determine anomalies
        anomalies[f"{variable}_z_score"] = (
            anomalies[f"{variable}_setpoint_diff"] - mean_diff
        ) / stdev_diff
        anomalies[f"{variable}_anomaly"] = anomalies[f"{variable}_z_score"].abs() > 1
    return anomalies


def rescale_data(values, labels):
    """
    Switch data from min-max scale back to original values.

    Args:
        values: a list of the scaled values to rescale.

    Returns:
        rescaled_data: a DataFrame of the rescaled values.
    """
    values = np.array(values)
    rescaled_data = SCALER_DATA.inverse_transform(values)
    rescaled_data = pd.DataFrame(rescaled_data, columns=labels)
    rescaled_data = rescaled_data.round(2)
    return rescaled_data
