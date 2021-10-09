import torch
from torch import nn
import os
import time
from tqdm import trange
import pandas as pd
from yolo_v4_tiny_body import BasicConv

from nets_CSPdarknet53_tiny import darknet53_tiny
from nets_attention import cbam_block, eca_block, se_block

import config
from config import measure

attention_block = [se_block, cbam_block, eca_block]

# -------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   Conv2d + BatchNormalization + LeakyReLU
# -------------------------------------------------#

class Yolo_head(nn.Module):
    def __init__(self, filters_list, in_filters):
        super(Yolo_head, self).__init__()
        self.basicConv = BasicConv(in_filters, filters_list[0], 3)
        self.conv2d = nn.Conv2d(filters_list[0], filters_list[1], 1)

    def forward(self, x):
        out = self.basicConv(x)
        out = self.conv2d(out)
        return out


class Upsample_net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample_net, self).__init__()

        self.basicConv =  BasicConv(in_channels, out_channels, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')


    def forward(self, x, ):
        x = self.basicConv(x)
        x = self.upsample(x)
        return x


if __name__ == '__main__':

    path = "tmp.pt"
    loop_num = config.loop_num
    num_anchors = 1
    num_classes = 1
    phi = 0
    x1, x2 = None, None
    feat1_att, feat2_att, upsample_att = None, None, None
    start_time, end_time = 0, 0
    filesize = 0
    if 1 <= phi and phi <= 3:
        feat1_att = attention_block[phi - 1](256)
        feat2_att = attention_block[phi - 1](512)
        upsample_att = attention_block[phi - 1](128)
    layer_name = []
    back_bone_net = darknet53_tiny(None)
    conv_for_P5_net = BasicConv(512, 256, 1)
    yolo_p5 = Yolo_head([512, num_anchors * (5 + num_classes)], 256)
    upsample = Upsample_net(256, 128)
    yolo_p4 = Yolo_head([256, num_anchors * (5 + num_classes)], 384)

    for name, layer in back_bone_net.named_children():
        if name.startswith("conv"):
            for _name, _layer in layer.named_children():
                layer_name.append("CSPDarkNet-{}-{}-{}".format(name, _name, config.str_split(_layer)))
        elif name.startswith("resblock_body"):
           for _name, _layer in layer.named_children():
                if _name == "maxpool":
                    layer_name.append("CSPDarkNet-{}-{}-{}".format(name, _name, config.str_split(_layer)))
                for tmp0, tmp1 in _layer.named_children():
                    layer_name.append("CAPDarkNet-{}-{}-{}-{}".format(name, _name, tmp0, config.str_split(tmp1)))
    for name, layer in conv_for_P5_net.named_children():
        layer_name.append("conv_P5-{}-{}".format(name, config.str_split(layer)))
    for name, layer in yolo_p5.named_children():
        if name == "basicConv":
            for _name, _layer in layer.named_children():
                layer_name.append("yolo_p5-{}-{}-{}".format(name, _name, config.str_split(_layer)))
        else:
            layer_name.append("yolo_p5-{}-{}".format(name, config.str_split(layer)))
    for name, layer in upsample.named_children():
        if name == "basicConv":
            for _name, _layer in layer.named_children():
                layer_name.append("upsample-{}-{}-{}".format(name, _name, config.str_split(_layer)))
        else:
            layer_name.append("upsample-{}-{}".format(name, config.str_split(layer)))
    for name, layer in yolo_p4.named_children():
        if name == "basicConv":
            for _name, _layer in layer.named_children():
                layer_name.append("yolo_p4-{}-{}-{}".format(name, _name, config.str_split(_layer)))
        else:
            layer_name.append("yolo_p4-{}-{}".format(name, config.str_split(layer)))

    t = [0 for _ in range(len(layer_name))]
    s = [0 for _ in range(len(layer_name))]


    for _ in trange(loop_num):
        X = torch.rand(1, 3, 224, 224)
        feat1, feat2 = None, None
        route, route1 = None, None
        num_layer = 0
        for name, layer in back_bone_net.named_children():
            if name.startswith("conv"):
                for _name, _layer in layer.named_children():
                    X, _t, _s = measure(X, _layer)
                    t[num_layer] += _t
                    s[num_layer] += _s
                    num_layer += 1
            elif name.startswith("resblock_body"):
                # name == "resblock_body_.."
                for _name, _layer in layer.named_children():
                    # _name == "conv1"
                    if _name == "maxpool":
                        X, _t, _s = measure(X, _layer)
                        t[num_layer] += _t
                        s[num_layer] += _s
                        num_layer += 1
                    else:
                        for tmp0, tmp1 in _layer.named_children():
                            X, _t, _s = measure(X, tmp1)
                            t[num_layer] += _t
                            s[num_layer] += _s
                            num_layer += 1
                        if _name == "conv1":
                            route = X
                            X = torch.split(X, route.shape[1] // 2, dim=1)[1]
                        elif _name == "conv2":
                            route1 = X
                        elif _name == "conv3":
                            X = torch.cat([X, route1], dim=1)
                        elif _name == "conv4":
                            feat1 = X
                            X = torch.cat([route, X], dim=1)
        feat2 = X
        if 1 <= phi and phi <= 3:
            feat1 = feat1_att(feat1)
            feat2 = feat2_att(feat2)
        # out: feat1, feat2, X
        # body of conv for P5 net
        # p5 = conv_for_P5_net(x2)
        for name, layer in conv_for_P5_net.named_children():
            feat2, _t, _s = measure(feat2, layer)
            t[num_layer] += _t
            s[num_layer] += _s
            num_layer += 1
        p5 = feat2
        # body of yolo p5
        # out0 = yolo_p5(p5)
        out0 = p5
        for name, layer in yolo_p5.named_children():
            if name == "basicConv":
                for _name, _layer in layer.named_children():
                    out0, _t, _s = measure(out0, _layer)
                    t[num_layer] += _t
                    s[num_layer] += _s
                    num_layer += 1
            else:
                out0, _t, _s = measure(out0, layer)
                t[num_layer] += _t
                s[num_layer] += _s
                num_layer += 1
        # body of upsample
        # p5_upsample = upsample(p5)
        for name, layer in upsample.named_children():
            if name == "basicConv":
                for _name, _layer in layer.named_children():
                    p5, _t, _s = measure(p5, _layer)
                    t[num_layer] += _t
                    s[num_layer] += _s
                    num_layer += 1
            else:
                p5, _t, _s = measure(p5, layer)
                t[num_layer] += _t
                s[num_layer] += _s
                num_layer += 1
        p5_upsample = p5
        if 1 <= phi and phi <= 3:
            p5_upsample = upsample_att(p5_upsample)
        p4 = torch.cat([p5_upsample, feat1], axis=1)
        # body of yolo p4
        # out1 = yolo_p4(p4)
        for name, layer in yolo_p4.named_children():
            if name == "basicConv":
                for _name, _layer in layer.named_children():
                    p4, _t, _s = measure(p4, _layer)
                    t[num_layer] += _t
                    s[num_layer] += _s
                    num_layer += 1
            else:
                p4, _t, _s = measure(p4, layer)
                t[num_layer] += _t
                s[num_layer] += _s
                num_layer += 1
        out1 = p4
        # output is: out0, out1
    for i in range(len(t)):
        t[i] = t[i] / loop_num
        s[i] = s[i] / loop_num
    df = pd.DataFrame([t, s], index = ["exec time", "size (b)"], columns=layer_name)
    # print(df)
    df.to_csv("yolo_v4_tiny.csv")