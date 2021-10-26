# Introduction
This project is used to measure the DNN inference time among different layer, e.g., conv, pool, etc.
Then, for different models, its measure file is different, due to the architecture.
Some DNN artectures contain multiple DNN block, e.g., mobileNet, yet the layers in others are in sequence.

In this project, we have implemented some DNN models running time of different layers as follows,
* VGG16
* network in network
* mobileNet V3
* ResNet18

The model architecture is generated from "torchvision.models", a Python package.

After running, the outfile will be generated in current folder.

# Requirement package
* Python 3.6 or other lastest version
* torch
* torchvision
* tqdm
* pandas
* numpy

# How to run
for different models, e.g., nin, use:
```python
python test_nin.py
```

for all models, use:
```python
python script.py
```

# File architecture

different model layer running time:
* test_mobileNet.py
* test_nin.py
* test_resNet18.py

scripy.py to run all the model file

config.py: loop_num for measurement