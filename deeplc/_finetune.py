from __future__ import annotations

import copy
import logging

import torch
from torch.utils.data import DataLoader

LOGGER = logging.getLogger(__name__)


class DeepLCFineTuner:
    """
    Class for fine-tuning a DeepLC model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to fine-tune.
    train_data : torch.utils.data.Dataset
        Dataset containing the training data.
    device : str, optional, default='cpu'
        The device on which to run the model ('cpu' or 'cuda').
    learning_rate : float, optional, default=0.001
        The learning rate for the optimizer.
    epochs : int, optional, default=10
        Number of training epochs.
    batch_size : int, optional, default=256
        Batch size for training.
    validation_data : torch.utils.data.Dataset or None, optional
        If provided, used directly for validation. Otherwise, a fraction of
        `train_data` will be held out.
    validation_split : float, optional, default=0.1
        Fraction of `train_data` to reserve for validation when
        `validation_data` is None.
    patience : int, optional, default=5
        Number of epochs with no improvement on validation loss before stopping.
    """

    def __init__(
        self,
        model,
        train_data,
        device="cpu",
        learning_rate=0.001,
        epochs=10,
        batch_size=256,
        validation_data=None,
        validation_split=0.1,
        patience=5,
    ):
        self.model = model.to(device)
        self.train_data = train_data
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_data = validation_data
        self.validation_split = validation_split
        self.patience = patience

    def _freeze_layers(self, unfreeze_keywords="33_1"):
        """
        Freezes all layers except those that contain the unfreeze_keyword
        in their name.
        """

        for name, param in self.model.named_parameters():
            param.requires_grad = unfreeze_keywords in name

    def prepare_data(self, data, shuffle=True):
        return DataLoader(data, batch_size=self.batch_size, shuffle=shuffle)

    def fine_tune(self):
        LOGGER.debug("Starting fine-tuning...")
        if self.validation_data is None:
            # Split the training data into training and validation sets
            val_size = int(len(self.train_data) * self.validation_split)
            train_size = len(self.train_data) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                self.train_data, [train_size, val_size]
            )
        else:
            train_dataset = self.train_data
            val_dataset = self.validation_data
        train_loader = self.prepare_data(train_dataset)
        val_loader = self.prepare_data(val_dataset, shuffle=False)

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
        )
        loss_fn = torch.nn.L1Loss()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            running_loss = 0.0
            self.model.train()
            for batch in train_loader:
                batch_X, batch_X_sum, batch_X_global, batch_X_hc, target = batch

                target = target.view(-1, 1)

                optimizer.zero_grad()
                outputs = self.model(batch_X, batch_X_sum, batch_X_global, batch_X_hc)
                loss = loss_fn(outputs, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch_X, batch_X_sum, batch_X_global, batch_X_hc, target = batch
                    target = target.view(-1, 1)
                    outputs = self.model(batch_X, batch_X_sum, batch_X_global, batch_X_hc)
                    val_loss += loss_fn(outputs, target).item()
            avg_val_loss = val_loss / len(val_loader)

            LOGGER.debug(
                f"Epoch {epoch + 1}/{self.epochs}, "
                f"Loss: {avg_loss:.4f}, "
                f"Validation Loss: {avg_val_loss:.4f}"
            )
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    LOGGER.debug(f"Early stopping triggered {epoch + 1}")
                    break
        self.model.load_state_dict(best_model_wts)
        return self.model
