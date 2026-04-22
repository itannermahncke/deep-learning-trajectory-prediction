"""
This module contains the LSTMPipeline class, which is used to train and test an LSTM model.
"""

import torch
from torch import nn, optim

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from src.lstm.lstm_model_class import LSTMModel
from src.lstm.early_stopper import EarlyStopper

import wandb

import lstm_helpers


class LSTMPipeline:
    """
    A class to represent the LSTM pipeline.

    Attributes:
        hyperparameters (dict): Configuration parameters for the model and training.
        timeseries (pd.DataFrame): The raw time series data.
        device (str): The device to use for training.
        model (LSTMModel): The LSTM model.
        scaler (StandardScaler): The scaler for the data.
        split_data (dict): A dictionary containing the training and testing sets.
        loaders (dict): A dictionary containing the data loaders, loss, and optimizer for training.

    Methods:
        run(): Trains the model and logs the results.
        train_and_test(): Trains and tests the model.
    """

    def __init__(self, hyperparameters: dict, timeseries: pd.DataFrame):
        """
        Initializes the LSTMPipeline with the given hyperparameters and time series data.

        Args:
            hyperparameters (dict): Configuration parameters for the model and training.
            timeseries (pd.DataFrame): The raw time series data.
        """
        self._hyperparameters = hyperparameters
        self._timeseries = timeseries
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = LSTMModel(self._hyperparameters).to(self._device)
        self._scaler = StandardScaler()

        # Preprocess data
        self._split_data = self._split()
        self._loaders = self._make_loaders()

    def run(self) -> LSTMModel:
        """
        Trains the model and logs the results

        Returns:
            LSTMModel: The trained model.
        """
        train_pred, val_pred = self.train_and_test()
        train_pred, val_pred, y_train, y_test = self._rescale_data(train_pred, val_pred)
        self._plot(train_pred, val_pred, y_train, y_test)
        return self._model

    def train_and_test(self) -> tuple:
        """
        Trains and evaluates the model.

        Returns:
            tuple: A tuple containing the predictions for the training and testing sets.
        """

        # Logs gradient
        wandb.watch(self._model, self._loaders["criterion"], log="all", log_freq=1)

        # Create early stop
        early_stopper = EarlyStopper(
            patience=self._hyperparameters["patience"],
            min_delta=self._hyperparameters["delta"],
        )

        lowest_validation_loss = float("inf")

        for epoch in range(self._hyperparameters["epochs"]):
            # Sets model to train mode
            self._model.train()
            # Begin training model
            training_loss = self._train_model()

            # Sets model to evaluation mode
            self._model.eval()
            # With no grad so gradient does not get computed during evaluation
            with torch.no_grad():
                # Begin evaluating model
                validation_loss = self._test_model()

            # Log training loss, validation loss, and epoch in wandb
            self._log(training_loss, validation_loss, epoch)

            # Ensures that the best model is saved at the end of training
            # best_model stores the state of the best model
            if validation_loss < lowest_validation_loss:
                lowest_validation_loss = validation_loss
                best_model = self._model.state_dict()

            # If there has been no improvement of value delta for patience number of epochs, end run
            if early_stopper.early_stop(validation_loss):
                break
            print(early_stopper.counter)

        return self._get_results_best_model(best_model, lowest_validation_loss)

    def _split(self):
        """
        Splits the time series data into training and testing sets.

        Returns:
            dict: A dictionary containing the training and testing sets.
        """
        # Scale dataset
        data_raw = self._timeseries.copy()
        data_raw[self._hyperparameters["variables"]] = self._scaler.fit_transform(
            self._timeseries[self._hyperparameters["variables"]]
        )
        # Creates sequences based on look back
        # Sequences will look like an array where each index contains the previous
        # look_back amount of data points
        look_back = self._hyperparameters["look_back"]
        # sequences = []
        # for index in range(len(data_raw) - look_back):
        #     sequences.append(data_raw[index : index + look_back])
        # sequences = np.array(sequences)
        sequences = lstm_helpers.lookback_sequence(
            data_raw,
            lookback_size=look_back,
            columns=self._hyperparameters["variables"],
        )

        # Split sequences into input and expected output
        # x_seq is the input and y_seq is the output
        # Each element of x_seq contains the previous look_back amount of data points
        # Each element of y_seq will contain the value of what the next timestep should be
        x_seq, y_seq = sequences[:-1, :-1, :], sequences[1:, -1, :]

        # Determine split index
        train_ratio = self._hyperparameters["train_size"]
        split_index = int(len(x_seq) * train_ratio)

        # Splitting the dataset into training and testing sets and converting to PyTorch tensors
        # x is input, y is expected output
        split_data = {
            "x_train": torch.tensor(x_seq[:split_index], dtype=torch.float),
            "y_train": torch.tensor(y_seq[:split_index], dtype=torch.float),
            "x_test": torch.tensor(x_seq[split_index:], dtype=torch.float),
            "y_test": torch.tensor(y_seq[split_index:], dtype=torch.float),
        }
        return split_data

    def _make_loaders(self):
        """
        Creates the training and testing data loaders, loss, and optimizer.

        Returns:
            dict: A dictionary containing the data loaders, loss, and optimizer.
        """

        # Combine the train input and expected output into respective TensorDatasets
        train = torch.utils.data.TensorDataset(
            self._split_data["x_train"], self._split_data["y_train"]
        )
        test = torch.utils.data.TensorDataset(
            self._split_data["x_test"], self._split_data["y_test"]
        )

        # Create loaders for train and test data with batches of batch_size
        # Shuffle is set to false because this is a timeseries
        train_loader = torch.utils.data.DataLoader(
            dataset=train, batch_size=self._hyperparameters["batch_size"], shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test, batch_size=self._hyperparameters["batch_size"], shuffle=False
        )

        # Make the loss and optimizer
        criterion = nn.L1Loss(reduction="mean")
        optimizer = optim.Adam(
            self._model.parameters(), lr=self._hyperparameters["learning_rate"]
        )

        return {
            "train_loader": train_loader,
            "test_loader": test_loader,
            "criterion": criterion,
            "optimizer": optimizer,
        }

    def _train_model(self):
        """
        Trains the model.

        Returns:
            torch.Tensor: The training loss.
        """
        # Array for storing the losses of each batch
        batch_losses = []
        # Train model
        for x_batch, y_batch in self._loaders["train_loader"]:
            x_batch = x_batch.to(self._device)
            y_batch = y_batch.to(self._device)
            loss = self._train_step(x_batch, y_batch)
            batch_losses.append(loss)
        # Average the losses of the batches
        training_loss = torch.mean(torch.stack(batch_losses))
        return training_loss

    def _train_step(self, x_val, y_val):
        """
        Trains the model for a single step.

        Args:
            x_val (torch.Tensor): The input data.
            y_val (torch.Tensor): The target data.

        Returns:
            torch.Tensor: The loss.
        """
        y_pred = self._model(x_val)  # pylint: disable=not-callable
        loss = self._loaders["criterion"](y_val, y_pred)
        # Compute L1 and L2 regularization terms
        l1_reg = sum(param.abs().sum() for param in self._model.parameters())
        l2_reg = sum(param.pow(2).sum() for param in self._model.parameters())
        # Add L1 and L2 regularization to the loss
        loss = (
            loss
            + self._hyperparameters["lambda_l1"] * l1_reg
            + self._hyperparameters["lambda_l2"] * l2_reg
        )
        loss.backward()
        self._loaders["optimizer"].step()
        self._loaders["optimizer"].zero_grad()
        return loss

    def _test_model(self):
        """
        Tests the model.

        Returns:
            torch.Tensor: The validation loss.
        """
        # Array for storing the losses of each batch
        batch_losses = []
        for x_val, y_val in self._loaders["test_loader"]:
            x_val = x_val.to(self._device)
            y_val = y_val.to(self._device)
            y_pred = self._model(x_val)  # pylint: disable=not-callable
            # print(f"{x_val}, {y_val}, {y_pred}")
            loss = self._loaders["criterion"](y_val, y_pred)
            batch_losses.append(loss)
        # Average the losses of the batches
        validation_loss = torch.mean(torch.stack(batch_losses))
        return validation_loss

    def _log(self, training_loss, validation_loss, epoch):
        """
        Logs the training and validation losses.

        Args:
            training_loss (torch.Tensor): The training loss.
            validation_loss (torch.Tensor): The validation loss.
            epoch (int): The current epoch.
        """
        wandb.log(
            {
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "epoch": epoch,
            },
            step=epoch,
        )
        print(
            f"[{epoch+1}] Training loss: {training_loss:.4f}\t \
                Validation loss: {validation_loss:.4f}"
        )

    def _get_results_best_model(self, best_model, validation_loss):
        """
        Gets the results from the best model.

        Args:
            best_model (dict): The best model.
            validation_loss (torch.Tensor): The validation loss.

        Returns:
            tuple: A tuple containing the predictions for the training and testing sets.
        """
        predictions = {"train_loader": [], "test_loader": []}
        best_model_name = f'{validation_loss:.4f}-{self._hyperparameters["name"]}'
        # Load the state of the best model
        self._model.load_state_dict(best_model)

        # Save the state of the best model to folder
        torch.save(
            self._model.state_dict(),
            f"best_models/{best_model_name}.pth",
        )

        # Set model to evaluation mode
        self._model.eval()

        # Get the predictions the model makes on both the training and test data
        for phase in ["train_loader", "test_loader"]:
            with torch.no_grad():
                for x_batch, _ in self._loaders[phase]:
                    x_batch = x_batch.to(self._device)
                    yhat = self._model(x_batch).tolist()  # pylint: disable=not-callable
                    predictions[phase] += yhat
        return predictions["train_loader"], predictions["test_loader"]

    def _rescale_data(self, train_pred, val_pred):
        """
        Rescales the data.

        Args:
            train_pred (list): The predictions for the training set.
            val_pred (list): The predictions for the testing set.

        Returns:
            tuple: A tuple containing the predictions for the training and testing sets.
        """
        # Unscales the data
        train_pred = self._scaler.inverse_transform(np.array(train_pred))
        val_pred = self._scaler.inverse_transform(np.array(val_pred))
        y_train = self._scaler.inverse_transform(self._split_data["y_train"])
        y_test = self._scaler.inverse_transform(self._split_data["y_test"])
        return train_pred, val_pred, y_train, y_test

    def _plot(self, train_pred, val_pred, y_train, y_test):
        """
        Plots the predictions in wandb.

        Args:
            train_pred (np.ndarray): The predictions for the training set.
            val_pred (np.ndarray): The predictions for the testing set.
            y_train (np.ndarray): The target values for the training set.
            y_test (np.ndarray): The target values for the testing set.
        """

        # Creates matplot graph of actual data and model prediction
        def create_graph(i, pred, actual, variable, phase):
            print(variable)
            print(i)
            figure, axis = plt.subplots()
            x_axis = range(len(pred[:, i]))
            axis.plot(x_axis, pred[:, i])
            axis.plot(x_axis, actual[:, i], alpha=0.3)
            axis.legend(["Predicted", "Actual"])
            axis.set_title(f"{phase} Validation {variable}")
            return figure

        # Creates plots for each feature
        # Each feature will have a plot that shows its predictions for
        # the train data and predictions for the test data
        for i, variable in enumerate(self._hyperparameters["variables"]):
            train_figure = create_graph(i, train_pred, y_train, variable, "Train")
            test_figure = create_graph(i, val_pred, y_test, variable, "Test")
            table = wandb.Table(columns=["plot"])
            table.add_data(wandb.Image(train_figure))
            table.add_data(wandb.Image(test_figure))

            wandb.log({f"{variable} Plots": table})
