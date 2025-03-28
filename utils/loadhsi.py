import torch
import numpy as np
import scipy.io as scio


def loadhsi(case):
    if case == 'muffle':
        file = './dataset/muffle_dataset_130_90.mat'
        data = scio.loadmat(file)
        image = data['Y']  # (130, 90, 64)
        lidar = data['MPN']
        label = data['label']
        label = label.astype(np.float32).transpose(2, 0, 1)
        M_true = data['M']
        M_init = data['M1']

    P = label.shape[0]
    band = image.shape[2]
    col = image.shape[0]
    row = image.shape[1]
    ldr_dim = lidar.shape[2]

    image = image.astype(np.float32)  # 数据类型转换
    label = label.astype(np.float32)
    lidar = lidar.astype(np.float32)

    M_init = torch.from_numpy(M_init).unsqueeze(2).unsqueeze(3).float()

    return image, lidar, label, M_init, M_true, P, band, col, row, ldr_dim
