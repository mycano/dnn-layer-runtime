'''
Author: myyao
Date: 2021-06-17 18:55:27
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
    net = torchvision.models.mobilenet.mobilenet_v3_small(pretrained=False)
    layer_name = []
    for _name, _net in net.named_children():
        if _name == "features":
            for _name1, _net1 in _net.named_children():
                if _name1 == "0" or _name1 == "12":
                    layer_name.append("{}-{}-{}".format(_name, _name1, str(_net1).split("(")[0]))
                else:
                    for _name2, _net2 in _net1.named_children():
                        for _name3, _net3 in _net2.named_children():
                            if str(_net3).startswith("ConvBNActivation"):
                                layer_name.append("{}-{}-{}-{}-{}".format(_name, _name1, _name2, _name3, str(_net3).split("(")[0]))
                            else:
                                for _name4, _net4 in _net3.named_children():
                                    layer_name.append("{}-{}-{}-{}-{}-{}".format(_name, _name1, _name2, _name3, _name4, str(_net4).split("(")[0]))
        elif _name == "avgpool":
            layer_name.append(_name)
        else:
            for _name1, _net1 in _net.named_children():
                layer_name.append("{}-{}-{}".format(_name, _name1, str(_net1).split("(")[0]))
    # for lay in layer_name:
    #     print(lay)
    # exit(0)
    # for l in layer_name:
    #     print(l)
    t = [0 for _ in range(len(layer_name))]
    s = [0 for _ in range(len(layer_name))]
    
    for _ in trange(config.loop_num):
        x = torch.rand(1, 3, 224, 224)
        num_layer = 0
        for _name, _net in net.named_children():
            if _name == "features":
                for _name1, _net1 in _net.named_children():
                    if _name1 == "0" or _name1 == "12":
                        x, _t, _s = measure(x, _net1)
                        t[num_layer] += _t
                        s[num_layer] += _s
                        num_layer += 1
                    else:
                        for _name2, _net2 in _net1.named_children():
                            for _name3, _net3 in _net2.named_children():
                                if str(_net3).startswith("ConvBNActivation"):
                                    x, _t, _s = measure(x, _net3)
                                    t[num_layer] += _t
                                    s[num_layer] += _s
                                    num_layer += 1
                                else:
                                    for _name4, _net4 in _net3.named_children():
                                        # print("{}-{}-{}-{}-{}-{}-{}".format(_name, _name1, _name2, _name3, _name4, _net4, x.shape))
                                        x, _t, _s = measure(x, _net4)
                                        t[num_layer] += _t
                                        s[num_layer] += _s
                                        num_layer += 1
            elif _name == "avgpool":
                x, _t, _s = measure(x, _net)
                t[num_layer] += _t
                s[num_layer] += _s
                num_layer += 1
            else:
                for _name1, _net1 in _net.named_children():
                    x, _t, _s = measure(x, _net1)
                    t[num_layer] += _t
                    s[num_layer] += _s
                    num_layer += 1
    for i in range(len(t)):
        t[i] = t[i] / config.loop_num
        s[i] = s[i] / config.loop_num
    df = pd.DataFrame([t, s], index = ["exec time", "size (b)"], columns=layer_name)
    df.to_csv("mobileNet_large.csv")