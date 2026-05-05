"""
This module contains the LSTMPipeline class, used to train and test an LSTM/BiLSTM model.

This version trains the model to predict DELTAS instead of absolute next states:
    target = next_state - current_state

For plotting/evaluation, it converts model output back into an absolute state:
    predicted_state = current_state + predicted_delta
"""

import torch
from torch import nn, optim

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simple_bilstm import SimpleBiLSTM as LSTMModel
from early_stopper import EarlyStopper

import wandb
import lstm_helpers as lstm_helpers


class LSTMPipeline:
    def __init__(self, hyperparameters: dict, timeseries: pd.DataFrame):
        self._hyperparameters = hyperparameters
        self._timeseries = timeseries

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. PyTorch is not seeing the GPU.")

        self._device = torch.device("cuda:0")
        self._model = LSTMModel(self._hyperparameters).to(self._device)
        self._scaler = StandardScaler()

        self._split_data = self._split()
        self._loaders = self._make_loaders()

    def run(self) -> LSTMModel:
        train_pred, val_pred = self.train_and_test()
        train_pred, val_pred, y_train, y_test = self._rescale_data(train_pred, val_pred)
        self._plot(train_pred, val_pred, y_train, y_test)
        return self._model

    def train_and_test(self) -> tuple:
        early_stopper = EarlyStopper(
            patience=self._hyperparameters["patience"],
            min_delta=self._hyperparameters["delta"],
        )

        lowest_validation_loss = float("inf")
        best_model = None

        for epoch in range(self._hyperparameters["epochs"]):
            self._model.train()
            training_loss = self._train_model()

            self._model.eval()
            with torch.no_grad():
                validation_loss = self._test_model()

            self._log(training_loss, validation_loss, epoch)

            if validation_loss < lowest_validation_loss:
                lowest_validation_loss = validation_loss
                best_model = {
                    key: value.detach().cpu().clone()
                    for key, value in self._model.state_dict().items()
                }

            if early_stopper.early_stop(validation_loss):
                break

        return self._get_results_best_model(best_model, lowest_validation_loss)

    def _split(self):
        """
        Creates train/test data.

        Input:
            x_seq = past states

        Target:
            y_delta = next_state - current_state

        Also stores:
            y_train_state / y_test_state = actual next_state
        so plots can compare predicted absolute state against actual absolute state.
        """
        data_raw = self._timeseries.copy()
        data_raw[self._hyperparameters["variables"]] = self._scaler.fit_transform(
            self._timeseries[self._hyperparameters["variables"]]
        )

        look_back = self._hyperparameters["look_back"]

        sequences = lstm_helpers.lookback_sequence(
            data_raw,
            lookback_size=look_back,
            columns=self._hyperparameters["variables"],
        )

        # Each sequence should be shaped:
        # [look_back + 1, num_features]
        #
        # x_seq: all points except final next-state target
        # current_state: last point inside x_seq
        # next_state: final point after x_seq
        x_seq = sequences[:, :-1, :]
        current_state = sequences[:, -2, :]
        next_state = sequences[:, -1, :]

        y_delta = next_state - current_state

        train_ratio = self._hyperparameters["train_size"]
        split_index = int(len(x_seq) * train_ratio)

        split_data = {
            "x_train": torch.tensor(x_seq[:split_index], dtype=torch.float),
            "y_train": torch.tensor(y_delta[:split_index], dtype=torch.float),
            "y_train_state": torch.tensor(next_state[:split_index], dtype=torch.float),
            "x_test": torch.tensor(x_seq[split_index:], dtype=torch.float),
            "y_test": torch.tensor(y_delta[split_index:], dtype=torch.float),
            "y_test_state": torch.tensor(next_state[split_index:], dtype=torch.float),
        }

        return split_data

    def _make_loaders(self):
        train = torch.utils.data.TensorDataset(
            self._split_data["x_train"],
            self._split_data["y_train"],
        )
        test = torch.utils.data.TensorDataset(
            self._split_data["x_test"],
            self._split_data["y_test"],
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=train,
            batch_size=self._hyperparameters["batch_size"],
            shuffle=False,
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=test,
            batch_size=self._hyperparameters["batch_size"],
            shuffle=False,
        )

        criterion = nn.L1Loss(reduction="mean")
        optimizer = optim.Adam(
            self._model.parameters(),
            lr=self._hyperparameters["learning_rate"],
        )

        return {
            "train_loader": train_loader,
            "test_loader": test_loader,
            "criterion": criterion,
            "optimizer": optimizer,
        }

    def _train_model(self):
        batch_losses = []

        for x_batch, y_batch in self._loaders["train_loader"]:
            x_batch = x_batch.to(self._device)
            y_batch = y_batch.to(self._device)

            loss = self._train_step(x_batch, y_batch)
            batch_losses.append(loss)

        return torch.mean(torch.stack(batch_losses))

    def _train_step(self, x_val, y_val):
        predicted_delta = self._model(x_val)
        loss = self._loaders["criterion"](predicted_delta, y_val)

        loss.backward()
        self._loaders["optimizer"].step()
        self._loaders["optimizer"].zero_grad()

        return loss

    def _test_model(self):
        batch_losses = []

        for x_val, y_val in self._loaders["test_loader"]:
            x_val = x_val.to(self._device)
            y_val = y_val.to(self._device)

            predicted_delta = self._model(x_val)
            loss = self._loaders["criterion"](predicted_delta, y_val)
            batch_losses.append(loss)

        return torch.mean(torch.stack(batch_losses))

    def _log(self, training_loss, validation_loss, epoch):
        wandb.log(
            {
                "training_loss": training_loss.item(),
                "validation_loss": validation_loss.item(),
                "epoch": epoch,
            },
            step=epoch,
        )

        print(
            f"[{epoch + 1}] Training loss: {training_loss.item():.4f}\t"
            f"Validation loss: {validation_loss.item():.4f}"
        )

    def _get_results_best_model(self, best_model, validation_loss):
        """
        Loads the best model and gets absolute-state predictions for plotting.

        The model output is delta:
            predicted_delta

        Convert to absolute state:
            predicted_state = current_state + predicted_delta
        """
        predictions = {"train_loader": [], "test_loader": []}
        best_model_name = f'{validation_loss:.4f}-{self._hyperparameters["name"]}'

        self._model.load_state_dict(best_model)

        torch.save(
            self._model.state_dict(),
            f"best_models/{best_model_name}.pth",
        )

        self._model.eval()

        for phase in ["train_loader", "test_loader"]:
            with torch.no_grad():
                for x_batch, _ in self._loaders[phase]:
                    x_batch = x_batch.to(self._device)

                    predicted_delta = self._model(x_batch)
                    current_state = x_batch[:, -1, :]
                    predicted_state = current_state + predicted_delta

                    yhat = predicted_state.detach().cpu().numpy().tolist()
                    predictions[phase] += yhat

        return predictions["train_loader"], predictions["test_loader"]

    def _rescale_data(self, train_pred, val_pred):
        train_pred = self._scaler.inverse_transform(np.array(train_pred))
        val_pred = self._scaler.inverse_transform(np.array(val_pred))

        y_train = self._scaler.inverse_transform(
            self._split_data["y_train_state"].detach().cpu().numpy()
        )
        y_test = self._scaler.inverse_transform(
            self._split_data["y_test_state"].detach().cpu().numpy()
        )

        return train_pred, val_pred, y_train, y_test

    def _plot(self, train_pred, val_pred, y_train, y_test):
        def create_graph(i, pred, actual, variable, phase):
            figure, axis = plt.subplots()
            x_axis = range(len(pred[:, i]))

            axis.plot(x_axis, pred[:, i])
            axis.plot(x_axis, actual[:, i], alpha=0.3)
            axis.legend(["Predicted", "Actual"])
            axis.set_title(f"{phase} Validation {variable}")

            return figure

        images = []

        for i, variable in enumerate(self._hyperparameters["variables"]):
            train_figure = create_graph(i, train_pred, y_train, variable, "Train")
            test_figure = create_graph(i, val_pred, y_test, variable, "Test")

            images.append(wandb.Image(train_figure, caption=f"{variable} - Train"))
            images.append(wandb.Image(test_figure, caption=f"{variable} - Test"))

            plt.close(train_figure)
            plt.close(test_figure)

        wandb.log({"Prediction Plots": images})
