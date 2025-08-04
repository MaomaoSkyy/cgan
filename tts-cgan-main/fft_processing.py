import pandas as pd
import numpy as np
import os
from scipy.fft import fft
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ==== 配置参数 ====
CSV_PATH = r"E:\Downloads\tts-cgan-main\tts-cgan-main\data\Motor_fault_train.csv"
SAVE_DIR = r"E:\Downloads\tts-cgan-main\tts-cgan-main\data\processed_fft"
SELECTED_COLS = ['wind_speed', 'wind_dir', 'env_temp', 'power_W', 'temp_drv', 'temp_nondrv']
LABEL_COL = 'label'
WIN_SIZE = 256
STEP_SIZE = 128
FS = 1  # 采样频率（Hz），如有实际值请替换

# ==== 确保保存目录存在 ====
os.makedirs(SAVE_DIR, exist_ok=True)

# ==== 加载数据 ====
df = pd.read_csv(CSV_PATH)
print(f"数据加载成功，包含 {len(df)} 条记录。")

# 检查是否包含所有需要的列
for col in SELECTED_COLS + [LABEL_COL]:
    if col not in df.columns:
        raise ValueError(f"列 '{col}' 不在 CSV 文件中，请检查列名。")

data = df[SELECTED_COLS].values
labels = df[LABEL_COL].values

# ==== 滑动窗口划分 ====
def sliding_windows(data, labels, win_size, step_size):
    X_win, y_win = [], []
    for start in range(0, len(data) - win_size + 1, step_size):
        end = start + win_size
        X_win.append(data[start:end])
        y_win.append(int(np.round(labels[start:end].mean())))  # 取主标签
    return np.array(X_win), np.array(y_win)

X_windows, y_windows = sliding_windows(data, labels, WIN_SIZE, STEP_SIZE)
print(f"滑窗完成，生成 {len(X_windows)} 个样本窗口。")

# ==== 执行傅里叶变换 ====
def process_fft_batch(X_batch):
    fft_batch = []
    for x in X_batch:
        channel_ffts = []
        for ch in range(x.shape[1]):
            signal = x[:, ch]
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)  # 每通道标准化
            freq_domain = np.abs(fft(signal))[:len(signal)//2]  # 取一半频率
            channel_ffts.append(freq_domain)
        fft_batch.append(np.stack(channel_ffts))  # shape: [channels, freq_bins]
    return np.stack(fft_batch)  # shape: [samples, channels, freq_bins]

fft_data = process_fft_batch(X_windows)
print(f"FFT变换完成，结果 shape: {fft_data.shape}")  # e.g., [197, 6, 128]

# ==== 全局归一化 ====
scaler = MinMaxScaler()
fft_flat = fft_data.reshape(-1, fft_data.shape[-1])
fft_scaled = scaler.fit_transform(fft_flat)
fft_data_norm = fft_scaled.reshape(fft_data.shape)
print("归一化完成。")

# ==== 保存每个样本 ====
print(f"开始保存 {len(fft_data_norm)} 个样本到：{SAVE_DIR}")
for i in range(len(fft_data_norm)):
    filename = f"fft_sample_{i}_label_{y_windows[i]}.npy"
    path = os.path.join(SAVE_DIR, filename)
    np.save(path, fft_data_norm[i])
    print(f"已保存：{filename}")

print("✅ 全部FFT样本保存完毕。")

# ==== 可选：可视化第一个样本的频谱图 ====
try:
    import matplotlib.pyplot as plt
    plt.imshow(fft_data_norm[0], aspect='auto', cmap='viridis')
    plt.title(f"FFT of sample 0, label {y_windows[0]}")
    plt.xlabel("Frequency bins")
    plt.ylabel("Channels")
    plt.colorbar()
    plt.show()
except:
    pass
