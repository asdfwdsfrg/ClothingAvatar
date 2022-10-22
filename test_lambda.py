import os
import time

import torch
from torch import nn

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
torch.cuda.set_device(0)
x = torch.rand([15, 5, 6], device = 'cuda:0')
mask = x.ge(0.5)
print(mask)
print(x[mask])