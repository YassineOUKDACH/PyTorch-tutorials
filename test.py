import torch 

# creat matrix with pytorch  !!

x = torch.empty(5, 3)
print(x)
# y = torch.rand(5,3)
# print(y)

z = torch.tensor([5.5, 3])
print(z)

# create a tensor form exesting tensor 

d = x.new_ones(5, 3, dtype = torch.double)
print(d)