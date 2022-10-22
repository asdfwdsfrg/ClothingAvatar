import time

import torch.nn as nn
import numpy as np
import torch
from torch.utils.cpp_extension import load
input = torch.randn(5 , 3,4, 6)
b = torch.randn(5, 1,1, 6)
print((input - b).shape)

