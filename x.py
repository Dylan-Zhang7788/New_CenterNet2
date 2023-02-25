import torch 
A=torch.randn(3,3,4)
print(A.shape)
torch.cat(A,[0],dim=2)
print(A.shape)
