'''
Author: myyao
Date: 2021-09-07 14:28:26
Description: 
'''
import torch
from torch import nn
import os
import time
from tqdm import trange
import pandas as pd
import config
from config import measure


def make_layers(in_channels,out_channels,kernel_size, stride, padding):
    conv = nn.Sequential(#(1, 96, 11, 4, 2)
        nn.Conv2d(in_channels,out_channels,kernel_size, stride, padding),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels,out_channels,kernel_size=1, stride=1, padding=0),#1x1卷积,整合多个feature map的特征
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels,out_channels,kernel_size=1, stride=1, padding=0),#1x1卷积,整合多个feature map的特征
        nn.ReLU(inplace=True)
    )

    return conv


class NinNet(nn.Module):
    def __init__(self):
        super(NinNet, self).__init__()
        self.cov1 = make_layers(1, 96, 11, 4, 2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.cov2 = make_layers(96, 256, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.cov3 = make_layers(256, 384, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.cov4 = make_layers(384, 10, kernel_size=3, stride=1, padding=1)

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6, stride=1)
        )

    def forward(self, img):
        feature = self.cov1(img)
        feature = self.pool1(feature)
        feature = self.cov2(feature)
        feature = self.pool2(feature)
        feature = self.cov3(feature)
        feature = self.pool3(feature)
        feature = self.cov4(feature)
        output = self.gap(feature)
        output = output.view(img.shape[0], -1)  # [batch,10,1,1]-->[batch,10]

        return output


if __name__ == '__main__':

    path = "tmp.pt"
    loop_num = config.loop_num
    net = NinNet()
    layer_name = []
    for name, layer in net.named_children():
        if name.startswith("cov") or name == "gap":
            for _name, _layer in layer.named_children():
                layer_name.append("{}-{}-{}".format(name, _name, str(_layer).split("(")[0]))
        else:
            layer_name.append("{}-{}".format(name, str(layer).split("(")[0]))

    s = [0 for i in range(len(layer_name))]
    t = [0 for i in range(len(layer_name))]

    # for lay in layer_name:
    #     print(lay)
    # exit(0)

    for _ in trange(loop_num):
        X = torch.rand(1, 1, 224, 224)
        num_layer = 0
        for name, layer in net.named_children():
            if name.startswith("cov") or name == "gap":
                for _name, _layer in layer.named_children():
                    X, _t, _s = measure(X, _layer)
                    t[num_layer] += _t
                    s[num_layer] += _s
                    num_layer += 1
            else:
                X, _t, _s = measure(X, layer)
                t[num_layer] += _t
                s[num_layer] += _s
                num_layer += 1

    for i in range(len(t)):
        t[i] = t[i] / loop_num
        s[i] = s[i] / loop_num
    df = pd.DataFrame([t, s], index = ["exec time", "size (b)"], columns=layer_name)
    # df = pd.DataFrame([t], index = ["exec time"], columns=layer_name)
    df.to_csv("nin.csv")