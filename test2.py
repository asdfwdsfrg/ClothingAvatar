from torch.distributions import Normal, kl_divergence
import torch
import os
import numpy as np
from lib.networks.embedder import *
xyz = torch.ones(128, 3)
t = torch.ones(128, 1)



