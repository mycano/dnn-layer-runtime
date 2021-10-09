import torch
from torch import nn
import os
import time
from tqdm import trange
import pandas as pd
import config
from torchvision import models


if __name__ == '__main__':
    net  = models.vgg16()
    layer_name = []
    for name1, net1 in net.named_children():
        if name1 == "avgpool":
            layer_name.append(name1)
        else:
            for name2, net2 in net1.named_children():
                layer_name.append("{}-{}-{}".format(name1, name2, config.str_split(net2)))
    cal_time = [0 for _ in range(len(layer_name))]
    size = [0 for _ in range(len(layer_name))]
    for _ in trange(config.loop_num):
        x = torch.rand((1, 3, 224, 224))
        num_layer = 0
        for name1, net1 in net.named_children():
            if name1 == "avgpool":
                x, _t, _s = config.measure(x, net1)
                cal_time[num_layer] += _t
                size[num_layer] += _s
                num_layer += 1
            else:
                for name2, net2 in net1.named_children():
                    x, _t, _s = config.measure(x, net2)
                    cal_time[num_layer] += _t
                    size[num_layer] += _s
                    num_layer += 1
    for idx in range(len(layer_name)):
        cal_time[idx] /= config.loop_num
        size[idx] /= config.loop_num
    df = pd.DataFrame([cal_time, size], index = ["exe time per layer", "size (b)"], columns=layer_name)
    df.to_csv("vgg16.csv")

