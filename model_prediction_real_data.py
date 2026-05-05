import os
import re

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# from src.simple_lstm_model import SimpleLSTMModel as LSTMModel
from src.simple_bilstm import SimpleBiLSTM as LSTMModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "0.1842-lstm-30, 192, 1, 0.0002"

VARIABLES = [
    "lat",
    "lon",
    "velocity",
    "heading",
    "baroaltitude",
    "geoaltitude",
]


def get_model_config_from_filename(model_path):
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    match = re.search(
        r"lstm-(\d+),\s*(\d+),\s*(\d+),\s*([0-9.]+)",
        model_name,
    )
    if match is None:
        raise ValueError(
            f"Could not parse model parameters from filename: {model_name}"
        )

    return {
        "look_back": int(match.group(1)),
        "hidden": int(match.group(2)),
        "layer": int(match.group(3)),
        "learning_rate": float(match.group(4)),
    }


def clean_filename_text(text):
    return str(text).replace(" ", "").replace(",", "_").replace("/", "-")


def fit_scaler_on_training_features(raw_csv_path, variables):
    raw_df = pd.read_csv(raw_csv_path)
    raw_df = raw_df[variables].dropna().reset_index(drop=True)

    scaler = StandardScaler()
    scaler.fit(raw_df)
    return scaler


def extract_flight_data(raw_csv_path, icao24, start, end, variables):
    raw_df = pd.read_csv(raw_csv_path)

    flight_df = raw_df.iloc[start : end + 1].copy()
    flight_df = flight_df[flight_df["icao24"] == icao24].copy()
    flight_df = flight_df[variables].dropna().reset_index(drop=True)

    if flight_df.empty:
        raise ValueError("Flight extraction returned empty dataframe")

    return flight_df


def build_input_target_sequences(df_scaled, variables, look_back):
    data = df_scaled[variables].to_numpy(dtype=np.float32)

    windows = []
    for i in range(len(data) - look_back + 1):
        windows.append(data[i : i + look_back])

    windows = np.array(windows, dtype=np.float32)

    x_seq = windows[:, :-1, :]
    y_seq = windows[:, -1, :]

    return x_seq, y_seq


def load_trained_model(model_path, model_config, repeat_times=1):
    dims = {
        "input": len(VARIABLES),
        "hidden": model_config["hidden"],
        "layer": model_config["layer"],
        "output": len(VARIABLES),
    }

    model = LSTMModel(dims)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    return model


def run_model_predictions(model, data_loader):
    predictions = []
    actual_values = []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            output = model(x_batch)

            predictions.append(output.cpu().numpy())
            actual_values.append(y_batch.cpu().numpy())

    return np.vstack(predictions), np.vstack(actual_values)


def save_prediction_plots(
    actual, predicted, variables, model_path, flight_info, save_dir="plots"
):
    os.makedirs(save_dir, exist_ok=True)

    raw_model_name = os.path.splitext(os.path.basename(model_path))[0]
    model_name = clean_filename_text(raw_model_name)

    flight_id = clean_filename_text(
        f'{flight_info["icao24"]}_{flight_info["start"]}_{flight_info["end"]}'
    )

    for i, var in enumerate(variables):
        plt.figure(figsize=(12, 5))
        plt.plot(actual[:, i], label="Actual")
        plt.plot(predicted[:, i], label="Predicted")

        plt.title(
            f"{var.capitalize()}: Actual vs Predicted\n"
            f"Flight {flight_info['icao24']}"
        )
        plt.xlabel("Timestep")
        plt.ylabel(var)
        plt.legend()
        plt.tight_layout()

        filename = f"{model_name}_{var}_{flight_id}.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=300)
        plt.close()

        print(f"Saved: {filename}")


def main():
    model_path = f"best_models/{MODEL_NAME}.pth"
    raw_csv_path = "data/raw/states_2021-05-17-00.csv"

    flight_info = {
        "icao24": "a390ff",
        "start": 3021,
        "end": 1756220,
    }

    model_config = get_model_config_from_filename(model_path)
    print(model_config)

    scaler = fit_scaler_on_training_features(raw_csv_path, VARIABLES)

    flight_df = extract_flight_data(
        raw_csv_path,
        flight_info["icao24"],
        flight_info["start"],
        flight_info["end"],
        VARIABLES,
    )

    scaled_flight_df = pd.DataFrame(
        scaler.transform(flight_df),
        columns=VARIABLES,
    )

    x_seq, y_seq = build_input_target_sequences(
        scaled_flight_df,
        VARIABLES,
        model_config["look_back"],
    )

    dataset = TensorDataset(
        torch.tensor(x_seq, dtype=torch.float32),
        torch.tensor(y_seq, dtype=torch.float32),
    )
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model = load_trained_model(model_path, model_config, repeat_times=1)

    predicted_scaled, actual_scaled = run_model_predictions(model, data_loader)

    predicted = scaler.inverse_transform(predicted_scaled)
    actual = scaler.inverse_transform(actual_scaled)

    save_prediction_plots(
        actual,
        predicted,
        VARIABLES,
        model_path,
        flight_info,
    )


if __name__ == "__main__":
    main()
