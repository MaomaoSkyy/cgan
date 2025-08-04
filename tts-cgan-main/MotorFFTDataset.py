import os
import numpy as np
import torch
from torch.utils.data import Dataset


class MotorFFTDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
        self.file_list.sort()  # 保证一致性

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = np.load(file_path)  # shape: [channels, freq_bins]

        # 自动从文件名中提取标签
        file_name = self.file_list[idx]
        try:
            label = int(file_name.split("_label_")[-1].split(".")[0])
        except:
            label = 0  # fallback if格式不对

        # 转为 Tensor
        spec_tensor = torch.FloatTensor(data).unsqueeze(0)  # shape: [1, C, F] (假设需要额外维度)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return spec_tensor, label_tensor
