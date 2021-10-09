'''
Author: myyao
Date: 2021-06-17 15:45:50
Description: 
'''
import torch
from torch import nn
import os
import time
from tqdm import trange
import pandas as pd
import torchvision
import config
from config import measure

if __name__ == '__main__':
    # torchvision.models.mobilenet_v3_small()
    # torchvision.models.resnet18()
    path = "tmp.pt"
    loop_num = config.loop_num
    net = torchvision.models.resnet18(pretrained=False)
    layer_name = []
    for name, layer in net.named_children():
        if name.startswith("layer"):
            for _name, _layer in layer.named_children():
                for tmp0, tmp1 in _layer.named_children():
                    if tmp0 == "downsample":
                        for tmp2, tmp3 in tmp1.named_children():
                            layer_name.append("{}-{}-{}-{}-{}".format(name, _name, tmp0, tmp2, str(tmp3).split("(")[0]))
                    else:
                        layer_name.append("{}-{}-{}-{}".format(name, _name, tmp0, str(tmp1).split("(")[0]))
        else:
            layer_name.append("{}-{}".format(name, str(layer).split("(")[0]))
    # for lay in layer_name:
    #     print(lay)
    # exit(0)
    # record time and size
    t = [0 for _ in range(len(layer_name))]
    s = [0 for _ in range(len(layer_name))]

    for _ in trange(loop_num):
        X = torch.rand(1, 3, 224, 224)
        num_layer = 0
        for name, layer in net.named_children():
            if name.startswith("layer"):
                down_sample_data = X
                for _name, _layer in layer.named_children():
                    for tmp0, tmp1 in _layer.named_children():
                        if tmp0 == "downsample":
                            for tmp2, tmp3 in tmp1.named_children():
                                down_sample_data, _t, _s = measure(down_sample_data, tmp3)
                                t[num_layer] += _t
                                s[num_layer] += _s
                                num_layer += 1

                        else:
                            X, _t, _s = measure(X, tmp1)
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
    df.to_csv("restnet18.csv")