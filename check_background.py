import os
import h5py
import numpy as np

# 获取文件路径
background_path = os.getenv('DATA_DIR') + "/background.h5"

# 假设采样率为2048 Hz
sample_rate = 2048  
segment_duration = 3600  # 每段时间长度为4秒
samples_per_segment = int(segment_duration * sample_rate)  # 每段包含的样本数量 (8192)

# 创建保存数据的主目录
output_dir = "segmented_data"
os.makedirs(output_dir, exist_ok=True)

# 打开 HDF5 文件
with h5py.File(background_path, 'r') as f:
    # 读取 'H1' 和 'L1' 数据集
    h1_data = f['H1'][:]
    l1_data = f['L1'][:]

    # 获取数据集的总样本数
    total_samples_h1 = h1_data.shape[0]
    total_samples_l1 = l1_data.shape[0]

    # 初始化列表以存储切分后的数据段
    h1_segments = []
    l1_segments = []

    # 按时间长度切分 'H1' 和 'L1' 数据集，并保存到新的小 HDF5 文件中
    for i in range(0, total_samples_h1, samples_per_segment):
        # 切分 'H1' 数据集
        segment_h1 = h1_data[i:i + samples_per_segment]
        # 切分 'L1' 数据集
        segment_l1 = l1_data[i:i + samples_per_segment]

        # 确保仅处理完整的一段
        if segment_h1.shape[0] == samples_per_segment and segment_l1.shape[0] == samples_per_segment:
            # 创建用于保存的文件夹，文件夹名包含当前时间段
            segment_folder = os.path.join(output_dir, f"{i/sample_rate:.1f}_{(i + samples_per_segment)/sample_rate:.1f}_background")
            os.makedirs(segment_folder, exist_ok=True)

            # 保存切分数据段到新的 HDF5 文件
            new_background_path = os.path.join(segment_folder, 'background.h5')
            with h5py.File(new_background_path, 'w') as new_f:
                new_f.create_dataset('H1', data=segment_h1)
                new_f.create_dataset('L1', data=segment_l1)
            
            print(f"Created {new_background_path} with 'H1' and 'L1' datasets.")

print("Data segmentation and saving completed.")
