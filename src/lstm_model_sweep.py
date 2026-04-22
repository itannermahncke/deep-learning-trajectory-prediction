"""
Create an lstm model from a variable in a timeseries
"""

import pandas as pd
import numpy as np

from src.lstm.lstm_pipeline_module import LSTMPipeline

import wandb

default_config = {
    "epochs": 100,
    "patience": 15,  # Number of epochs early stop will wait for for improvement in validation loss
    "delta": 0.0005,  # Minimum number validation loss needs to improve by
    "dataset": "",
    "variables": [],
    "train_size": 0.7,
    "input": 0,
    "output": 0,
    "repeat_times": 1,
    "name": "",
}

# Enter in whatever parameters. Multiple values in one parameter will make it do it a sweep.

sweep_config = {
    "method": "random",  # "grid" will do all combinations while "random" will do random combinations. Set the count variable below for number of random runs.
    "metric": {"name": "validation_loss", "goal": "minimize"},
    "parameters": {
        "batch_size": {"values": [64]},
        "look_back": {"values": [20, 30, 50]},
        "hidden": {"values": [192, 256, 320]},
        "layer": {"values": [1]},
        "dropout": {"values": [0.05, 0.1, 0.15]},
        "learning_rate": {"values": [0.0002, 0.0003, 0.0004]},
        "lambda_l1": {"values": [0.0]},
        "lambda_l2": {"values": [1e-6]},
    },
}

# Names of the csvs the model will train on
datasets = ["states_2021-05-17-00"]

# Name of the timestamp column
timestamp = "Time.stamp"


def train(config):
    def train_inner(config=None):

        # Initialize the run.
        # Set mode to "disabled" if wandb does not need to be run. Helpful when only needed to test things

        run = wandb.init(config=config, group="initial-sweep", mode="online")

        # Set config to whatever parameters are chosen for this run based on the sweep_config
        config = wandb.config

        # Add in the values from default_config
        config.update(default_config)

        # Set run name
        run_name = f'lstm-{config["look_back"]}, {config["hidden"]}, {config["layer"]}, {config["learning_rate"]}, {config["dropout"]}'
        run.name = run_name

        # Get dataset
        timeseries = pd.read_csv(config["dataset"])
        df = pd.DataFrame(timeseries)
        print(df)

        # Remove any commas in the data
        df = df.applymap(lambda x: x.replace(",", "") if isinstance(x, str) else x)
        # # Remove any columns that contain mostly NA
        # df = df.dropna(thresh=len(df) - 10, axis=1)
        # Remove any rows that contain a null value
        df = df.replace(["Null", "Null value"], np.nan)
        df = df.dropna(axis=0)

        # # Add the days of the week columns
        # df[timestamp] = pd.to_datetime(df[timestamp])
        # df = pd.concat(
        #     [pd.get_dummies(df[timestamp].dt.day_name()).astype(int), df], axis=1
        # )
        # # Insert time of day column
        # df.insert(0, "Time", df[timestamp].dt.hour * 60 + df[timestamp].dt.minute)

        # # Remove the old timestamp column
        # df.drop([timestamp], axis=1, inplace=True)

        # Get the features the model will be training on
        variables = [
            "lat",
            "lon",
            "velocity",
            "heading",
            "baroaltitude",
            "geoaltitude",
        ]

        # Updating the values in the config that were previously undefined.
        # This allows it so that the number of features, inputs, and outputs
        # do not need to be manually entered
        new_values = {
            "variables": variables,
            "input": len(variables),
            "output": len(variables),
            "name": run_name,
        }
        config.update(new_values, allow_val_change=True)
        # Run model pipeline
        pipeline = LSTMPipeline(config, df)
        model = pipeline.run()

        wandb.finish()

    return train_inner


count = 50  # Number of runs the sweep agent goes for

# Count gets updated to number of all combinations if grid sweep is done
if sweep_config["method"] == "grid":
    count = 1
    for parameter, values in sweep_config["parameters"].items():
        count *= len(values["values"])

for dataset in datasets:
    # Set sweep ID. New sweep ID for each set of data
    sweep_id = wandb.sweep(sweep_config, project="aircraft-trajectory-lstm")
    # Set dataset in config
    default_config["dataset"] = f"data/raw/{dataset}.csv"
    # Run the sweep
    wandb.agent(sweep_id, function=train(sweep_config), count=count)
