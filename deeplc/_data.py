import torch
from torch.utils.data import Dataset


class DeepLCDataset(Dataset):
    """
    Custom Dataset class for DeepLC used for loading features from peptide sequences.

    Parameters
    ----------
    X : ndarray
        Feature matrix for input data.
    X_sum : ndarray
        Feature matrix for sum of input data.
    X_global : ndarray
        Feature matrix for global input data.
    X_hc : ndarray
        Feature matrix for high-order context features.
    target : ndarray, optional
        The target retention times. Default is None.
    """

    def __init__(self, X, X_sum, X_global, X_hc, target=None):
        self.X = torch.from_numpy(X).float()
        self.X_sum = torch.from_numpy(X_sum).float()
        self.X_global = torch.from_numpy(X_global).float()
        self.X_hc = torch.from_numpy(X_hc).float()

        if target is not None:
            self.target = torch.from_numpy(target).float()  # Add target values if provided
        else:
            self.target = None  # If no target is provided, set it to None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.target is not None:
            # Return both features and target during training
            return (
                self.X[idx],
                self.X_sum[idx],
                self.X_global[idx],
                self.X_hc[idx],
                self.target[idx],
            )
        else:
            # Return only features during prediction
            return (self.X[idx], self.X_sum[idx], self.X_global[idx], self.X_hc[idx])
