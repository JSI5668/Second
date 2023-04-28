from torchviz import make_dot
from torch.autograd import Variable
from tqdm import tqdm
import network
import os
import random
import argparse
import numpy as np

from torch.utils import data

from torchsummary import summary
import torch
import torch.nn as nn


from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

# model = nn.Sequential()
# model.add_module('W0', nn.Linear(8, 16))
# model.add_module('tanh', nn.Tanh())
PATH = 'C:/checkpoint_void/pytorch/segmentation/camvid_original_model/model.pt'
model = torch.load(PATH)
# Variable을 통하여 Input 생성
# x = Variable(torch.randn(1, 8))
x = torch.zeros(1,3,256,256)
# 앞에서 생성한 model에 Input을 x로 입력한 뒤 (model(x))  graph.png 로 이미지를 출력합니다.
make_dot(model(x), params=dict(model.named_parameters())).render("graph_1", format="png")