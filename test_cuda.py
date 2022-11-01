import torch
from torch.distributions import Normal, kl_divergence

normal_pred = Normal(1, 2)
normal_t = Normal(0, 1)
kl_loss1 = -0.5 * torch.sum(1 + torch.log(torch.tensor(2)) - 1 ** 2 - 2)
kl_loss2 = kl_divergence(normal_pred, normal_t)
print(kl_loss1, kl_loss2)
