import torch
from torch import nn
import os
import time
from tqdm import trange
import pandas as pd
import config
from torchvision import models

if __name__ == "__main__":
    net = models.inception_v3()
    x = torch.rand((1, 3, 224, 224))
    print(net)
    start_time = time.time()
    net(x)
    end_time = time.time()
    print(end_time-start_time)