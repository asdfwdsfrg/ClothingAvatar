import numpy as np
from torch.utils.data.dataloader import DataLoader
import cv2
import os
import torch


from lib.datasets.light_stage.multi_view_dataset import Dataset
from lib.networks.body_model import BodyModel

data_root = 'data/zju_mocap/CoreView_387'
human = 'CoreView_387'
ann_file = 'data/zju_mocap/CoreView_387/annots.npy'
split = 'test'



a = torch.ones(128, 24, 3)
b = torch.rand(128, 24, 3)
c = torch.mul(a, b)
print(c)
print(c.shape)
       