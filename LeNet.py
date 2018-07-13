import torch.nn as nn
import torch.nn.functional as F
from SBP_utils import Conv2d_SBP, Linear_SBP, SBP_layer

class LeNet_GS(nn.Module):
    def __init__(self):
        super(LeNet_GS, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fcbn1 = nn.BatchNorm1d(120)
        self.fc2   = nn.Linear(120, 84)
        self.fcbn2 = nn.BatchNorm1d(84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fcbn1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.fcbn2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out

    def group_sparse(self):
        bn_weight_list = [self.bn1.weight, self.bn2.weight, self.fcbn1.weight, self.fcbn2.weight]
        return bn_weight_list

class LeNet(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1   = nn.Linear(800, 500)
        self.fc2   = nn.Linear(500, 10)


    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out = self.conv2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)

        return out


class LeNet_SBP(nn.Module):
    def __init__(self):
        super(LeNet_SBP, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)

        self.sbp_1 = SBP_layer(20)
        self.sbp_2 = SBP_layer(50)
        self.sbp_3 = SBP_layer(800)
        self.sbp_4 = SBP_layer(500)

    def forward(self, x):
        if self.training:

            out = self.conv1(x)
            out,kl1 = self.sbp_1(out)
            out = F.relu(out)
            out = F.max_pool2d(out, 2)
            out = self.conv2(out)
            out,kl2 = self.sbp_2(out)
            out = F.relu(out)
            out = F.max_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            out, kl3 = self.sbp_3(out)
            out = self.fc1(out)
            out, kl4 = self.sbp_4(out)
            out = F.relu(out)
            out = self.fc2(out)


            kl_sum = (0.3*kl1+0.3*kl2+0.2*kl3+0.2*kl4)
            return out,kl_sum
        else:
            out = self.conv1(x)
            out = self.sbp_1(out)
            out = F.relu(out)
            out = F.max_pool2d(out, 2)
            out= self.conv2(out)
            out = self.sbp_2(out)

            out = F.relu(out)
            out = F.max_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            out = self.sbp_3(out)
            out = self.fc1(out)
            out = self.sbp_4(out)
            out = F.relu(out)
            out = self.fc2(out)


            return out

    def layerwise_sparsity(self):
        return [self.sbp_1.layer_sparsity(), self.sbp_2.layer_sparsity(), self.sbp_3.layer_sparsity(),
                self.sbp_4.layer_sparsity()]

    def display_snr(self):
        return [self.sbp_1.display_snr(), self.sbp_2.display_snr(), self.sbp_3.display_snr(), self.sbp_4.display_snr()]