import torch
from torch.distributions import Normal, kl_divergence
import numpy as np
from torch.utils.cpp_extension import load
import sys
import os

wpts = torch.rand(1, 4, 1024, 4, device = "cuda:1")
mask = torch.rand(128, 4, 1024, device = "cuda:1").ge(0.5)
wpts_hit = torch.zeros(128, 4, 800, 4, device = "cuda:1")
cuda_module = load(name="blend_feature",
                  sources=["cuda_module/kernel/blend_feature.cpp", "cuda_module/kernel/blend_feature.cu"],
                   verbose=True)

n = 128
s_max = 800
cuda_module.launch_pts_hit(wpts, mask, wpts_hit, 128, 4, 1024, s_max)
torch.set_printoptions(threshold=sys.maxsize)
torch.cuda.synchronize(device = "cuda:1")
print(wpts_hit.shape)
with open('wpts_hit.txt', 'w') as f:
    f.write(str(wpts_hit))

