class EarlyStopper:
    """
    A class to stop training if the validation loss does not improve after a set
    number of epochs.

    Attributes:
        patience: an integer representing the number of epochs to wait before
        stopping training.
        min_delta: a float representing the minimum change in validation loss to
        be considered an improvement.
        counter: an integer representing the number of epochs since the last
        improvement.
        min_validation_loss: a float representing the lowest validation

    Methods:
        early_stop: a method to determine if the model should stop training.
    """

    def __init__(self, patience=1, min_delta=0):
        """
        Initiate the EarlyStopper class.

        Args:
            patience: an integer representing the number of epochs to wait before
            stopping training.
            min_delta: a float representing the minimum change in validation loss
            to be considered an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        """
        Determine if the model should stop training.

        Args:
            validation_loss: a float representing the current validation loss.
        """
        print(self.min_validation_loss - validation_loss)
        if self.min_validation_loss - validation_loss > self.min_delta:
            self.counter = 0
            self.min_validation_loss = validation_loss
        else:
            self.counter += 1
        if self.counter >= self.patience:
            return True
        return False