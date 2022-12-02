from multiprocessing import allow_connection_pickling
import numpy as np
from torch.utils.data.dataloader import DataLoader
import cv2
import os
import torch


from lib.datasets.light_stage.multi_view_dataset import Dataset
from lib.networks.body_model import BodyModel

exp_name = 'transformd_nodes_T_with_view2'
metric_pth = os.path.join('data/result/StructureNeRF/{}/metrics.npy'.format(exp_name))
m = np.load(metric_pth, allow_pickle=True).item()
print(np.mean(m['psnr']))
print(np.mean(m['ssim']))
print(np.mean(m['mse']))


       