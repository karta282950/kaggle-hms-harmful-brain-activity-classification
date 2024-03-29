import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchtoolbox.tools import mixup_data, mixup_criterion
import sys, os
sys.path.append(os.path.join(sys.path[0],'bar','sub','dir'))
from Kaggle_KL_Div.kaggle_kl_div import score
from omegaconf import DictConfig
import hydra
import pandas as pd
import gc

'''
To Do:
* loss cant decrease
* 

Done:
* Create resnet
* create main to check

Ref:
* https://www.researchgate.net/publication/359051366_GWNET_Detecting_Gravitational_Waves_using_Hierarchical_and_Residual_Learning_based_1D_CNNs
'''
class ResNet_1D_Block(nn.Module):
    def __init__(
            self, 
            cfg: DictConfig, 
            in_channels, 
            out_channels, 
            kernel_size,
            stride,
            padding,
            downsampling):
        super(ResNet_1D_Block, self).__init__()
        self.cfg = cfg
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=0.0, inplace=False)
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        self.mp = nn.MaxPool1d(
            kernel_size=2, stride=2, padding=0)
        self.downsampling = downsampling
    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out = self.mp(out)
        identity = self.downsampling(x)

        out+=identity
        return out

class EEGNet(nn.Module):
    def __init__(
            self,
            cfg: DictConfig,
            kernels,
            in_channels=20,
            fixed_kernel_size=17,
            num_classes=6):
        super(EEGNet, self).__init__()
        self.cfg = cfg
        self.kernels = kernels
        self.planes = 24
        self.parallel_conv = nn.ModuleList()
        self.in_channels = in_channels
        
        for _, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.planes,
                kernel_size=(kernel_size),
                stride=1,
                padding=0,
                bias=False,)
            self.parallel_conv.append(sep_conv)

        self.bn1 = nn.BatchNorm1d(num_features=self.planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv1d(
            in_channels=self.planes,
            out_channels=self.planes,
            kernel_size=fixed_kernel_size,
            stride=2,
            padding=2,
            bias=False,)
        self.block = self._make_resnet_layer(
            kernel_size=fixed_kernel_size,
            stride=1,
            padding=fixed_kernel_size//2)
        self.bn2 = nn.BatchNorm1d(num_features=self.planes)
        self.avgpool = nn.AvgPool1d(
            kernel_size=6,
            stride=6,
            padding=2)
        self.rnn = nn.GRU(
            input_size=self.in_channels,
            hidden_size=128,
            num_layers=1,
            bidirectional=True)
        self.fc = nn.Linear(
            in_features=424, out_features=num_classes)
        self.rnn1 = nn.GRU(
            input_size=156,
            hidden_size=156,
            num_layers=1,
            bidirectional=True)

    def _make_resnet_layer(
            self,
            kernel_size,
            stride,
            blocks=9,
            padding=0):
        layers = []
        downsample = None
        base_width = self.planes

        for i in range(blocks):
            downsampling = nn.Sequential(
                    nn.MaxPool1d(
                        kernel_size=2,
                        stride=2,
                        padding=0))
            layers.append(
                ResNet_1D_Block(
                    self.cfg, 
                    in_channels=self.planes,
                    out_channels=self.planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    downsampling=downsampling))

        return nn.Sequential(*layers)

    def forward(self, x):
        out_sep = []

        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)

        out = torch.cat(out_sep, dim=2)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)  

        out = self.block(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.avgpool(out)  
        
        out = out.reshape(out.shape[0], -1)  

        rnn_out, _ = self.rnn(x.permute(0,2, 1))
        new_rnn_h = rnn_out[:, -1, :]  

        new_out = torch.cat([out, new_rnn_h], dim=1)  
        result = self.fc(new_out)

        return result
