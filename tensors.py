# tensors are specialized data structure that are very similar to arrays and matrics.

import torch 
import numpy as np 

#------------------Intialise tensor--------------------- ---

# "tensors can be created directly form data"

data = [[1, 2], [3, 4]]

x_data = torch.tensor(data)
print(x_data)

#----------------From a numpy array------------------ 

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

#---------------- From another tensor --------------- 

x_ones = torch.ones_like(x_data)
print(f'Ones tensor: \n {x_ones} \n')

x_rand = torch.rand_like(x_data, dtype = torch.float)
print(f'Random tensor \n {x_rand} \n')

#-------------with radnom or constant values--------- 

shape = (2, 3, )
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

#-------------Attributes of a Tensor--------------- 
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

#------------Operations on Tensors------------------
import torch
tensor = torch.ones(4, 4)
print(f'first row {tensor[0]} \n')
print(f'First column {tensor[:, 0]}')
print(f'Last coloumn {tensor[..., -1]}')
tensor[..., -1] = 0
tensor[:, 2] = -1
print(tensor)
tensors_concat = torch.cat([tensor, tensor, tensor], dim = 1)
print("concatenate tensors", tensors_concat) 
#---------------Arithmetic Operations---------------------
new_tensor = torch.randn(2, 3)
print('new_tesnor', new_tensor)
print('TRanspose of new tensor', new_tensor.T)

y1 = new_tensor @ new_tensor.T
print('Y1', y1)
y2 = new_tensor.matmul(new_tensor.T)
print('Y2', y2)

y3 = torch.rand_like(y1)
print('Y3', y3)
print('Y 3 will have the same value  \n', torch.matmul(new_tensor, new_tensor.T, out = y3))
# compute the element wise product 
z1 = torch.ones(3, 3)
z1[:, 1] = -1
print(z1)
z2 = z1 * z1
print('Wise element product', z2)
print('mul', z1.mul(z1))
z3 = torch.rand_like(z2)
print(torch.mul(z1, z1, out = z3))
z4 =  z1.sum()
print('item', z4.item(), type(z4))
print('z4', z4)
z4.add_(5)
print(z4)
#--------------Bridge with numpy---------------
t = torch.ones(5)
print(f't: {t}')
n = t.numpy()
print(f'n: {n}')
t.add_(2)
print(f"t: {t}")
print(f"n: {n}")
#NumPy array to Tensor 
n_new = np.ones(5)
t_new = torch.from_numpy(n_new)
