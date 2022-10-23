import torch


a = torch.ones(5,6)
a[0] = a[0] * 2
print(a)
print(a.mean())
