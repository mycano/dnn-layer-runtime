'''
Author: myyao
Date: 2021-06-20 17:30:27
Description: 
'''
import time
import torch
import os
loop_num = 200

def measure(x, net, save_size=False):
    path = "tmp.pt"
    start_time = time.time()
    try:
        x = net(x)
    except:
        x = torch.reshape(x, (1, -1))
        x = net(x)
    end_time = time.time()
    size = 0
    if save_size:
        torch.save(x, path)
        time.sleep(0.01)
        size = os.path.getsize(path)
        try:
            os.remove(path)
        except:
            pass
    return x, end_time-start_time, size

def str_split(net):
    return str(net).split("(")[0]