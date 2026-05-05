import os
import re

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# from src.simple_lstm_model import SimpleLSTMModel as LSTMModel
from src.simple_bilstm import SimpleBiLSTM as LSTMModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "0.0224-bilstm-25, 192, 1, 0.0004003177913925228"

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


def fit_scaler_on_features(raw_csv_path, variables):
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


def load_trained_model(model_path, model_config, repeat_times=1):
    dims = {
        "input": len(VARIABLES),
        "hidden": model_config["hidden"],
        "layer": model_config["layer"],
        "output": len(VARIABLES),
        "repeat_times": repeat_times,
    }

    model = LSTMModel(dims)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    return model


def run_hybrid_rollout(
    model,
    scaled_flight_df,
    variables,
    look_back,
    teacher_forcing_ratio=0.5,
):
    if not (0.0 <= teacher_forcing_ratio <= 1.0):
        raise ValueError("teacher_forcing_ratio must be between 0.0 and 1.0")

    data = scaled_flight_df[variables].to_numpy(dtype=np.float32)

    input_window_size = look_back

    if len(data) < input_window_size + 1:
        raise ValueError("Flight is too short for rollout.")

    total_prediction_steps = len(data) - input_window_size
    actual_scaled = data[input_window_size:]

    switch_index = int(total_prediction_steps * teacher_forcing_ratio)
    switch_index = max(0, min(switch_index, total_prediction_steps))

    if switch_index < input_window_size:
        raise ValueError(
            "switch_index is too early. Need at least input_window_size "
            "previous predictions before switching to fully predicted windows."
        )

    predicted_scaled = []

    for step in range(total_prediction_steps):
        if step < switch_index:
            input_window = data[step : step + input_window_size].copy()
        else:
            input_window = np.array(
                predicted_scaled[-input_window_size:],
                dtype=np.float32,
            )

        if input_window.shape != (input_window_size, len(variables)):
            raise ValueError(
                f"Input window has wrong shape: {input_window.shape}, "
                f"expected {(input_window_size, len(variables))}"
            )

        model_input = torch.tensor(
            input_window[np.newaxis, :, :],
            dtype=torch.float32,
            device=DEVICE,
        )

        with torch.no_grad():
            prediction = model(model_input).cpu().numpy()[0]

        current_state = input_window[-1]
        prediction = current_state + prediction

        predicted_scaled.append(prediction)
    return actual_scaled, predicted_scaled, switch_index


def save_hybrid_rollout_plots(
    actual,
    predicted,
    variables,
    model_path,
    flight_info,
    switch_index,
    teacher_forcing_ratio,
    save_dir="plots_hybrid",
):
    os.makedirs(save_dir, exist_ok=True)

    raw_model_name = os.path.splitext(os.path.basename(model_path))[0]
    model_name = clean_filename_text(raw_model_name)

    flight_id = clean_filename_text(
        f'{flight_info["icao24"]}_{flight_info["start"]}_{flight_info["end"]}'
    )

    ratio_text = str(teacher_forcing_ratio).replace(".", "p")

    for i, variable in enumerate(variables):
        plt.figure(figsize=(12, 5))
        plt.plot(actual[:, i], label="Actual")
        plt.plot(predicted[:, i], label="Predicted")
        plt.axvline(
            switch_index,
            linestyle="--",
            label="Switch to own predictions",
        )

        plt.title(
            f"{variable.capitalize()}: Actual vs Predicted\n"
            f"Flight {flight_info['icao24']}"
        )
        plt.xlabel("Prediction Timestep")
        plt.ylabel(variable)
        plt.legend()
        plt.tight_layout()

        filename = f"{model_name}_{variable}_{flight_id}.png"
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

    teacher_forcing_ratio = 0.70

    model_config = get_model_config_from_filename(model_path)
    print(model_config)

    scaler = fit_scaler_on_features(raw_csv_path, VARIABLES)

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

    model = load_trained_model(model_path, model_config, repeat_times=1)

    actual_scaled, predicted_scaled, switch_index = run_hybrid_rollout(
        model=model,
        scaled_flight_df=scaled_flight_df,
        variables=VARIABLES,
        look_back=model_config["look_back"],
        teacher_forcing_ratio=teacher_forcing_ratio,
    )

    actual = scaler.inverse_transform(actual_scaled)
    predicted = scaler.inverse_transform(predicted_scaled)

    save_hybrid_rollout_plots(
        actual=actual,
        predicted=predicted,
        variables=VARIABLES,
        model_path=model_path,
        flight_info=flight_info,
        switch_index=switch_index,
        teacher_forcing_ratio=teacher_forcing_ratio,
    )


if __name__ == "__main__":
    main()
