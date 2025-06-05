import torch
from psm_utils.psm_list import PSMList
from torch.utils.data import Dataset

from deeplc._features import encode_peptidoform


class DeepLCDataset(Dataset):
    """Custom Dataset class for DeepLC used for loading features from peptide sequences."""

    def __init__(self, psm_list: PSMList, add_ccs_features: bool = False):
        self.psm_list = psm_list
        self.add_ccs_features = add_ccs_features
        
        self._targets = self._get_targets(psm_list)
    
    @staticmethod
    def _get_targets(psm_list: PSMList) -> torch.Tensor | None:
        retention_times = [psm.retention_time for psm in psm_list]
        if None not in retention_times:
            return torch.tensor(retention_times, dtype=torch.float32)
        else:
            return None

    def __len__(self):
        return len(self.psm_list)

    def __getitem__(self, idx) -> tuple:
        if not isinstance(idx, int):
            raise TypeError(f"Index must be an integer, got {type(idx)} instead.")
        features = encode_peptidoform(
            self.psm_list[idx].peptidoform,
            add_ccs_features=self.add_ccs_features
        )
        feature_tuples = (
            torch.from_numpy(features["matrix"]).to(dtype=torch.float32),
            torch.from_numpy(features["matrix_sum"]).to(dtype=torch.float32),
            torch.from_numpy(features["matrix_global"]).to(dtype=torch.float32),
            torch.from_numpy(features["matrix_hc"]).to(dtype=torch.float32),
        )
        targets = self._targets[idx] if self._targets is not None else torch.full_like(
            feature_tuples[0], fill_value=float('nan'), dtype=torch.float32
        )
        return feature_tuples, targets
