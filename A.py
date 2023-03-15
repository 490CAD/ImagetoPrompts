import torch

a = torch.Tensor([1,2,3]).unsqueeze(0)
print(a)
print(a[0])
print(a[0,0])