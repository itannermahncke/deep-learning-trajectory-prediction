"""
Create a dashboard to visualize the results of the LSTM model for anomaly detection.
"""

import statistics
from datetime import timedelta
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import torch
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader
from lstm_model_class import LSTMModel
from anomaly_detection_helpers import (
    read_csv,
    process_time_columns,
    get_data,
    calculate_error,
    compare_setpoint,
    rescale_data,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Configure paths
BUILD_FOLDER = "anomaly_detection/plots"
MODEL_FOLDER = "anomaly_detection/saved_models"
DATA_FOLDER = "data"
DATA_NAME = "ahu_mac_1_oa"
TIME_STAMP = "Time.stamp"

# Define model parameters
LOOK_BACK = 96
BATCH_SIZE = 64
dims = {
    "input": 0,
    "hidden": 192,
    "layer": 1,
    "dropout": 0.5,
    "output": 0,
    "repeat_times": 2,
}

indexes = range(8, 38)

# Get dataset
DATA_PATH = f"{DATA_FOLDER}/{DATA_NAME}.csv"

df = read_csv(DATA_PATH)

processed_data = df.copy()

# Add time columns
processed_data = process_time_columns(processed_data, TIME_STAMP)

# Set train size
train_size = int(processed_data.shape[0] * 0.7)

# Get variables
variables = processed_data.columns.tolist()
dims["input"] = len(variables)
dims["output"] = len(variables)

# Configure data for testing
processed_data = get_data(processed_data, LOOK_BACK, train_size)

# Create loader
data_loader = DataLoader(processed_data, batch_size=BATCH_SIZE, shuffle=False)

# Load model
model = LSTMModel(dims)
model.load_state_dict(torch.load(f"{MODEL_FOLDER}/{DATA_NAME}.pth"))
model = model.to(DEVICE)
model.eval()

# Feed data through model
predicted_values = []
actual_values = []
with torch.no_grad():
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        prediction = model(x_batch)  # pylint: disable=not-callable
        predicted_values += prediction.tolist()
        actual_values += y_batch.tolist()

errors = calculate_error(predicted_values, actual_values, variables, indexes)
indexes = [i + 2 for i in indexes]
# Add timestamp column
errors["timestamp"] = pd.to_datetime(
    df[TIME_STAMP].iloc[train_size + LOOK_BACK - 1 :]
).reset_index(drop=True)
# Set threshold to MAE standard deviation
threshold = statistics.stdev(errors["mean_abs_err"])
print(threshold)
errors["anomaly"] = [
    errors["mean_abs_err"][i] > errors["em_avg"][i] + threshold
    for i in range(len(predicted_values))
]

# Rescale predicted and actual values
predicted_values = rescale_data(predicted_values, variables)
predicted_values["error"] = errors["mean_abs_err"]
predicted_values["ema"] = errors["em_avg"]
predicted_values["anomaly"] = errors["anomaly"]

actual_values = rescale_data(actual_values, variables)
actual_values["error"] = errors["mean_abs_err"]
actual_values["ema"] = errors["em_avg"]
actual_values["anomaly"] = errors["anomaly"]

setpoint_anomalies = compare_setpoint(df, ["SaStpr", "SaTmp"])
setpoint_anomalies["timestamp"] = pd.to_datetime(df[TIME_STAMP])

setpoint_anomalies.to_csv(f"{BUILD_FOLDER}/setpoint_anomalies.csv", index=False)
predicted_values.to_csv(f"{BUILD_FOLDER}/predicted_values.csv", index=False)
actual_values.to_csv(f"{BUILD_FOLDER}/actual_values.csv", index=False)
errors.to_csv(f"{BUILD_FOLDER}/errors.csv", index=False)

################################################################################
# Initialize Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div(
    children=[
        html.H1(children="LSTM Model Dashboard"),
        dcc.Dropdown(
            id="dropdown",
            options=[
                {"label": errors.columns[i][:-4], "value": errors.columns[i][:-4]}
                for i in indexes
            ],
            value=errors.columns[indexes[0]][:-4],
        ),
        dcc.Graph(id="plot_3"),
        dcc.Graph(id="plot_1"),
        dcc.Graph(id="plot_2"),
    ]
)


@app.callback(
    Output("plot_1", "figure"),
    [
        Input("dropdown", "value"),  # for choosing variable with dropdown
        Input("plot_2", "clickData"),  # for choosing variable with click on plot 2
        Input("plot_2", "figure"),  # for focusing graph with click on plot 2
    ],
)
def change_variable(variable, variable_click_data, plot_2_figure_data):
    """
    Create two subplots: one to display the anomaly calculation with the average
    error and the EMA, and the other to display the predicted and actual values
    of the selected variable.
    """
    if variable_click_data is not None:
        variable = variable_click_data["points"][0]["x"]
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Anomaly Calculation", f"Predictions for {variable}"],
    )
    # Graph first subplot
    figure.add_trace(
        go.Scatter(
            x=errors["timestamp"],
            y=errors["mean_abs_err"],
            mode="lines",
            name="Average Error",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=errors["timestamp"],
            y=errors["em_avg"] + threshold,
            mode="lines",
            name="EMA",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=errors["timestamp"],
            y=errors[f"{variable}_err"],
            mode="lines",
            name=f"{variable} Error",
            line={"color": "green", "width": 0.75},
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=errors["timestamp"][list(errors["anomaly"])],
            y=errors["mean_abs_err"][list(errors["anomaly"])],
            mode="markers",
            name="Anomaly",
            marker={"color": "red", "size": 10},
        ),
        row=1,
        col=1,
    )
    # Graph second subplot
    figure.add_trace(
        go.Scatter(
            x=errors["timestamp"],
            y=actual_values[variable],
            mode="lines",
            name="Actual",
            line={"color": "green"},
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=errors["timestamp"],
            y=predicted_values[variable],
            mode="lines",
            name="Predicted",
            line={"color": "blue"},
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=errors["timestamp"][list(errors["anomaly"])],
            y=actual_values[variable][list(errors["anomaly"])],
            mode="markers",
            name="Anomaly",
            marker={"color": "red", "size": 10},
        ),
        row=2,
        col=1,
    )

    # Focus graph
    if plot_2_figure_data is not None:
        clicked_timestamp = pd.to_datetime(
            plot_2_figure_data["layout"]["title"]["text"][18:]
        )
        min_x = clicked_timestamp - timedelta(days=3)
        max_x = clicked_timestamp + timedelta(days=3)
    figure.update_layout(height=1000)
    figure.update_xaxes(range=[min_x, max_x])
    return figure


@app.callback(
    Output("plot_2", "figure"),
    [Input("plot_1", "clickData")],
)
def display_variable_distribution(click_data):
    if click_data is not None:
        clicked_timestamp = click_data["points"][0]["x"]
        timestamp_errors = errors.iloc[
            errors[errors["timestamp"] == clicked_timestamp].index[0]
        ]
    else:
        clicked_timestamp = errors["timestamp"].iloc[-1]
        timestamp_errors = errors.iloc[-1]
    figure = {
        "data": [
            go.Bar(
                x=[errors.columns[i][:-4] for i in indexes],
                y=[timestamp_errors.values[i] for i in indexes],
                name="Average Error",
            )
        ],
        "layout": go.Layout(
            title=f"Variable Error at {clicked_timestamp}",
        ),
    }
    return figure


@app.callback(
    [Output("plot_3", "figure"), Output("plot_3", "style")],
    [Input("dropdown", "value")],
)
def plot_setpoint_anomalies(variable):
    if variable in ["SaStpr", "SaTmp"]:
        figure = {
            "data": [
                go.Scatter(
                    x=setpoint_anomalies["timestamp"],
                    y=setpoint_anomalies[f"{variable}"],
                    mode="lines",
                    name="Actual",
                    line={"color": "green"},
                ),
                go.Scatter(
                    x=setpoint_anomalies["timestamp"],
                    y=setpoint_anomalies[f"{variable}Spt"],
                    mode="lines",
                    name="Setpoint",
                    line={"color": "blue"},
                ),
                go.Scatter(
                    x=setpoint_anomalies["timestamp"][
                        list(setpoint_anomalies[f"{variable}_anomaly"])
                    ],
                    y=setpoint_anomalies[f"{variable}"][
                        list(setpoint_anomalies[f"{variable}_anomaly"])
                    ],
                    mode="markers",
                    name="Anomaly",
                    marker={"color": "red", "size": 10},
                ),
            ],
            "layout": go.Layout(
                title=f"Setpoint Error for {variable}",
                xaxis={
                    "range": [
                        setpoint_anomalies["timestamp"].iloc[-576],
                        setpoint_anomalies["timestamp"].iloc[-1],
                    ]
                },
            ),
        }
        style = {"display": "block"}
    else:
        figure = {
            "data": [],
            "layout": go.Layout(
                title="",
                xaxis={"visible": False},
                yaxis={"visible": False},
            ),
        }
        style = {"display": "none"}
    return figure, style


# Run app
if __name__ == "__main__":
    app.run()
