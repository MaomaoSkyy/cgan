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

        # Pre-compute labels for all files and determine class count
        self.labels = []
        for fname in self.file_list:
            if fname in self.label_map:
                label = self.label_map[fname]
            else:
                match = re.search(r"(\d+)(?=\.npy$)", fname)
                label = int(match.group(1)) if match else 0
            self.labels.append(label)
        self.num_classes = len(set(self.labels))

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

        # Use pre-computed labels
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return spec_tensor, label_tensor
