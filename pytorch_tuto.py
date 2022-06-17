from tkinter.tix import DirTree
from numpy import dtype
import torch 
import math 
from torchvision import datasets

# -*- coding: utf-8 -*-

#-------------- with numpy ---------------
import numpy as np
import math

# Create random input and output data
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# Randomly initialize weights
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    # y = a + b x + c x^2 + d x^3
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')
dtype = torch.float32
# #---------------- with pytorch-------------
# #specifiy the divices 

print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
device = torch.device('cpu')
# # ceate a random input and output data 

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype = dtype)
y = torch.sin(x)

#Randomly intialise wrights 
a = torch.randn((), device=device, dtype = dtype)
b = torch.randn((), device=device, dtype = dtype)
c = torch.randn((), device=device, dtype = dtype)
d = torch.randn((), device=device, dtype = dtype)
print(a, b, c, d)
#learning rate 

learning_rate = 1e-6 

for i in range(2000):
    # Forward pass: compute predicted y
        y_pred = a + b * x + c * x ** 2 + d * x ** 3
        
        #compute and print loss

        loss = (y_pred - y).pow(2).sum().item()
        
        if i%100 == 99:
            print(i, loss)
            
# backup to compute gradients of a, b, c, d with the respect of loss 
        grad_y_pred = 2.0 * (y_pred - y)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_a * x).sum()
        grad_c = (grad_y_pred * x**2).sum()
        grad_d = (grad_y_pred * x**3 ).sum()
            # Update weights using gradient descent
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d
            
print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')