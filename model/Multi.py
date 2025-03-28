import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ChannelAttention(nn.Module):
    def __init__(self, kernel_size):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.LeakyReLU()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)

        m = self.max_pool(x)
        m = self.conv(m.squeeze(-1).transpose(-1, -2))
        m = m.transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y + m)

        return y.expand_as(x)


class Multi(nn.Module):
    def __init__(self, band, P, ldr_dim):
        super().__init__()
        self.band = band
        self.P = P
        self.ldr_dim = ldr_dim
        self.softmax = nn.Softmax(dim=1)

        # ---------------attention---------------
        self.ChannelAttention = ChannelAttention(1)

        # ---------------encoder of HSI---------------
        self.AE_first_block1 = nn.Sequential(
            nn.Conv2d(self.band, 32 * self.P, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32 * self.P),
            nn.Dropout(),
            nn.LeakyReLU(0.01)
        )

        self.AE_first_block2 = nn.Sequential(
            nn.Conv2d(32 * self.P, 16 * self.P, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16 * self.P),
            nn.LeakyReLU(0.01)
        )

        self.AE_first_block3 = nn.Sequential(
            nn.Conv2d(16 * self.P, 4 * self.P, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * self.P),
            nn.LeakyReLU(0.01)
        )

        self.AE_first_block4 = nn.Sequential(
            nn.Conv2d(4 * self.P, self.P, kernel_size=3, stride=1, padding=1),
        )

        self.AE_second_block1 = nn.Sequential(
            nn.Conv2d(self.ldr_dim, self.P // 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.P // 2),
            nn.Dropout(),
            nn.LeakyReLU(0.01)
        )

        self.AE_second_block2 = nn.Sequential(
            nn.Conv2d(self.P // 2, self.P, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.P),
            nn.LeakyReLU(0.01)
        )

        # ---------------decoder---------------

        self.Decoder = nn.Sequential(
            nn.Conv2d(self.P, self.band, kernel_size=1, stride=1, bias=False),
        )

        # ---------------fusion---------------
        self.LFIc = nn.Sequential(
            nn.Conv2d(self.P * 2, self.P, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.P),
            nn.LeakyReLU(0.01)
        )

        self.GFIc = nn.Sequential(
            nn.Conv2d(self.P, self.P, kernel_size=1, stride=1),
        )

        self.FC = nn.Sequential(
            nn.Linear(self.P, self.P),
            nn.BatchNorm1d(self.P),
            nn.LeakyReLU(0.01)
        )

        # ---------------Abun---------------
        self.GetAbun = nn.Sequential(
            nn.Conv2d(self.P * 3, self.P, kernel_size=3, stride=1, padding=1)
        )

        self.GetEM = nn.Sequential(
            nn.Conv2d(self.P * 3, self.P, kernel_size=3, stride=1, padding=1)
        )

    def first_encoder(self, input):
        x = self.AE_first_block1(input)
        x = self.AE_first_block2(x)
        x = self.AE_first_block3(x)
        x = self.AE_first_block4(x)
        first_a = x
        return first_a

    def second_encoder(self, input):
        x = self.AE_second_block1(input)
        x = self.AE_second_block2(x)
        second_a = self.ChannelAttention(x)
        return second_a

    def LFI(self, f_hsi, f_ldr):
        H = torch.cat((f_hsi, f_ldr), dim=1)
        feature = self.LFIc(H)
        return feature

    def GFI(self, f_hsi, f_ldr):
        F_conv_a = self.GFIc(f_hsi)
        F_conv_a = F_conv_a.squeeze(-1).squeeze(-1)
        F_attention_a = F.softmax(torch.mm(F_conv_a.T, F_conv_a), dim=1)

        F_conv_b = self.GFIc(f_ldr)
        F_conv_b = F_conv_b.squeeze(-1).squeeze(-1)
        F_attention_b = F.softmax(torch.mm(F_conv_b.T, F_conv_b), dim=1)

        H1 = torch.mm(F_conv_b, F_attention_a)
        f_hsi = f_hsi.squeeze(-1).squeeze(-1)
        H1 = self.FC(H1)
        f_GAFM_a = H1 + f_hsi

        H2 = torch.mm(F_conv_a, F_attention_b)
        H2 = self.FC(H2)
        f_ldr = f_ldr.squeeze(-1).squeeze(-1)
        f_GAFM_b = H2 + f_ldr

        feature = torch.cat((f_GAFM_a, f_GAFM_b), dim=1)
        feature = feature.unsqueeze(2).unsqueeze(3)

        return feature

    def abun(self, fea):
        abun = self.GetAbun(fea)
        a = F.softmax(abun, dim=1)
        return a

    def forward(self, x, y):
        feature_hsi = self.first_encoder(x)
        feature_ldr = self.second_encoder(y)

        # ------ CMF -----
        feature_LFI = self.LFI(feature_hsi, feature_ldr)
        feature_GFI = self.GFI(feature_hsi, feature_ldr)
        feature = torch.cat((feature_LFI, feature_GFI), dim=1)

        abun = self.abun(feature)

        Y = self.Decoder(abun)
        EM_hat = self.Decoder[0].weight.data

        return abun, Y, EM_hat, feature_hsi, feature_ldr
