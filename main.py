import os
import numpy as np
import torch
import scipy.io as scio
import torch.utils
import torch.utils.data as Data
from torch import nn
from time import time
from tqdm import tqdm
from model.Multi import Multi
from utils.loadhsi import loadhsi
from utils.evaluation import evaluate
from utils.loss import MinVolumn, SparseLoss

output_path = './result_out/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('training on', device)

cases = ['muffle']
case = cases[0]
epochs = 40


# ---------------------权重设置---------------------s
def weights_init(m):
    classname = m.__class__.__name__  # 根据m的具体输入来设置对应的值
    if hasattr(m, 'weight') and m.weight is not None:
        if classname.find('Conv') != -1:
            nn.init.kaiming_uniform_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias.data, 0)


def train(lr=1e-2, lamda=8e-4, delta=0.5, lamda_sad=10, batch_size=256):

    Image, Lidar, A_true, M_init, M_true, P, band, col, row, ldr_dim = loadhsi(case)

    Image = torch.tensor(Image).to(device)
    Lidar = torch.tensor(Lidar).to(device)
    M_init = M_init.to(device)

    X = torch.reshape(Image, (col * row, band)).unsqueeze(2).unsqueeze(3)
    Y = torch.reshape(Lidar, (col * row, ldr_dim)).unsqueeze(2).unsqueeze(3)

    Label_train = Data.TensorDataset(X, Y)
    label_train_loader = Data.DataLoader(Label_train, batch_size=batch_size, shuffle=True)

    # -----------set model-----------
    model = Multi(band, P, ldr_dim).to(device)
    model.apply(weights_init)
    model.Decoder[0].weight.data = M_init.clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -----------loss funtion-----------
    criterionSparse = SparseLoss(lamda)
    criterionVolumn = MinVolumn(band, P, delta)

    # -----------train-----------
    tic = time()
    losses = []
    for epoch in tqdm(range(epochs)):
        model.train()
        for step, traindata in enumerate(label_train_loader):
            x, y = traindata
            abun, output, M_hat, abun_hsi, abun_ldr = model(x, y)

            output = torch.reshape(output, (output.shape[0], band))
            x = torch.reshape(x, (output.shape[0], band))

            # 1
            loss_rec = ((output - x) ** 2).sum() / y.shape[0]

            # 2
            Sparse = criterionSparse(abun)

            # 3
            M_hat = M_hat.squeeze(-1)
            em_bar = M_hat.mean(dim=0, keepdim=True)
            aa = (M_hat * em_bar).sum(dim=2)
            em_bar_norm = em_bar.square().sum(dim=2).sqrt()
            em_tensor_norm = M_hat.square().sum(dim=2).sqrt()
            sad = torch.acos(aa / (em_bar_norm + 1e-5) / (em_tensor_norm + 1e-5))
            loss_sad = sad.sum() / y.shape[0] / P

            # 4
            Volumn = criterionVolumn(model.Decoder[0].weight)

            loss = loss_rec + Sparse + Volumn + lamda_sad * loss_sad
            optimizer.zero_grad()  # 清空过往梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 根据梯度更新网络参数

        losses.append(loss.detach().cpu().numpy())

    toc = time()

    scio.savemat(output_path + 'loss.mat', {'loss': losses})

    # -----------result-----------
    model.eval()
    with torch.no_grad():
        abun, output, M_hat, abun_hsi, abun_ldr = model(X, Y)

        EM_hat = torch.reshape(M_hat, (band, P)).cpu().data.numpy()
        Y_hat = output.squeeze(-1).squeeze(-1).cpu().data.numpy()
        A_hat = abun.squeeze(-1).squeeze(-1).cpu().data.numpy().T
        A_true = A_true.reshape(P, col * row)

        Image = Image.reshape(col * row, band).cpu().numpy()

        dev = np.zeros([P, P])
        for i in range(P):
            for j in range(P):
                dev[i, j] = np.mean((A_hat[i, :] - A_true[j, :]) ** 2)
        pos = np.argmin(dev, axis=0)  # 求最小值的索引
        A_hat = A_hat[pos, :]
        EM_hat = EM_hat[:, pos]

        armse_y, asad_y, armse_em, sad_err, asad_em, armse_a, class_rmse, armse = evaluate(EM_hat, M_true,
                                                                                                        A_hat, A_true,
                                                                                                        Image, Y_hat)

        scio.savemat(output_path + 'results.mat', {'EM': EM_hat.T,
                                                   'A': A_hat,
                                                   'Y_hat': Y_hat})

    return armse_y, asad_y, armse_em, sad_err, asad_em, armse_a, class_rmse, armse, toc - tic


if __name__ == '__main__':
    armse_y, asad_y, armse_em, sad_err, asad_em, armse_a, class_rmse, armse, tim = train()

    print('*' * 70)
    print('time elapsed:', tim)
    print('RESULTS:')
    print('aRMSE_Y:', armse_y)
    print('aSAD_Y:', asad_y)

    print('RMSE_em', armse_em)
    print('SAD', sad_err)
    print('aSAD', asad_em)

    print('RMSE_a:', armse_a)
    print('RMSE:', class_rmse)
    print('aRMSE:', armse)

