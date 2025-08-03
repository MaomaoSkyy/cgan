import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset


class MotorFFTDataset(Dataset):
    """Dataset loader for pre-computed FFT numpy files.

    Parameters
    ----------
    data_dir : str
        Directory containing ``.npy`` spectrogram files.
    label_file : str, optional
        Path to an external label mapping file. The file should contain two
        comma separated columns: ``filename,label``. If not provided, labels
        are inferred from the filename by extracting the last number before the
        extension.
    """

    def __init__(self, data_dir, label_file: str | None = None):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
        self.file_list.sort()  # 保证一致性

        # Optional external label mapping
        self.label_map = {}
        if label_file and os.path.exists(label_file):
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        self.label_map[parts[0]] = int(parts[1])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)
        data = np.load(file_path)

        # Ensure channel-first orientation: [channels, seq_len]
        if data.ndim == 1:
            data = data[np.newaxis, :]
        elif data.shape[0] > data.shape[1]:
            data = data.T

        spec_tensor = torch.from_numpy(data).float().unsqueeze(0)  # [1, C, L]

        # Resolve label either from external map or filename pattern
        if file_name in self.label_map:
            label = self.label_map[file_name]
        else:
            match = re.search(r"(\d+)(?=\.npy$)", file_name)
            label = int(match.group(1)) if match else 0
        label_tensor = torch.tensor(label, dtype=torch.long)

        return spec_tensor, label_tensor
