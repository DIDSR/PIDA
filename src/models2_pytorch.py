import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import torch.nn.functional as F
import torch.utils.data as utils


class CNNT_3D(nn.Module):
    def __init__(self,Dropout,dropout_conv,dropout_fc):
        super(CNNT_3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3),padding=1)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=(3, 3, 3),padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3),padding=1)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3),padding=1)
        self.conv5 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3),padding=1)
        self.conv6 = nn.Conv3d(128, 128, kernel_size=(3, 3, 3),padding=1)
        self.Dropout=Dropout
        self.dropout_conv=nn.Dropout(p=dropout_conv)
        self.dropout_fc=nn.Dropout(p=dropout_fc)
        self.pool = nn.MaxPool3d(2,2)
        self.avgpool = nn.AvgPool3d(2)
        self.fc1 = nn.Linear(128 * 3 * 3 * 1, 128)
        self.fc_out = nn.Linear(128, 2)

    def forward(self, x):
        if self.Dropout=='yes':
            out = F.leaky_relu(self.conv1(x))
            # print(out.shape)
            out = self.dropout_conv(out)
            out = self.pool(F.leaky_relu(self.conv2(out)))
            out = self.dropout_conv(out)
            # print(out.shape)
            out = F.leaky_relu(self.conv3(out))
            out = self.dropout_conv(out)
            # print(out.shape)
            out = self.pool(F.leaky_relu(self.conv4(out)))
            out = self.dropout_conv(out)
            # print(out.shape)
            out = F.leaky_relu(self.conv5(out))
            out = self.dropout_conv(out)
            # print(out.shape)
            out = self.pool(F.leaky_relu(self.conv6(out)))
            out = self.dropout_conv(out)
            # print(out.shape)
            out = self.avgpool(out)
            # print(out.shape, 'avg')
            out = out.view(-1, 128 * 3 * 3 * 1)
            out = self.dropout_fc(self.fc1(out))
            out = self.dropout_fc(self.fc_out(out))
        else:
            out = F.leaky_relu(self.conv1(x))
            out = self.pool(F.leaky_relu(self.conv2(out)))
            out = F.leaky_relu(self.conv3(out))
            out = self.pool(F.leaky_relu(self.conv4(out)))
            out = F.leaky_relu(self.conv5(out))
            out = self.pool(F.leaky_relu(self.conv6(out)))
            out = self.avgpool(out)
            out = out.view(-1, 128 * 3 * 3 * 1)
            out = self.fc1(out)
            out = self.fc_out(out)
        return out   # F.log_softmax(out, dim=1)