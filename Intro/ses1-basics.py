# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 23:53:18 2019

@author: Ernest (Khashayar) Namdar
"""
import numpy as np
arr=[[6, 8], [7, 3]]
print(type(arr))
np.array(arr)
print(type(arr))
arr = np.array(arr)
print(type(arr))

import torch
torch.Tensor(arr)
print(type(arr))
arr = torch.Tensor(arr)
print(type(arr))


np.ones((7, 7))
torch.ones((7, 7))

np.random.rand(7, 7)
torch.rand(7, 7)

# seeds
np.random.seed(42)
np.random.rand(3, 3)

torch.manual_seed(7)
torch.rand(3, 3)

# GPU seed
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(7)

# convert from numpy
torch_tensor = torch.from_numpy(np.array([[6, 8], [7, 3]]))
print(torch_tensor)

# not all types are convertible
# double, float, int64, int32, and uint8 are ok

"""
Nothing>double tensor
Int64> LongTensor
Int32> IntTensor
Uint8> ByteTensor
Float64> Double
Float32> FloatTensor
Np.double> DoubleTensor
"""

# Torch to numpy
# Name_of_tensor.numpy()
torch_tensor.numpy()

# Tensors can be on CPU and GPU
# By default, everything is on CPU
"""
if torch.cuda.is_available():
    Name_of_tensor.cuda()
To convert back:
Name_of_tensor.cpu()
"""

a = torch.ones(3, 3)
a.size()
a.view(9)

# a + b is ok
# c = torch.add(a, b) is also ok
# c.add_(a) is in-place addition  (c is changed)
# c = add(a) (c is not changed)
# a.sub(b)  subtraction
# torch.mul(a, b)   or .mul_

# a.mean(dim=0)   good for a vector
# a.mean(dim=1)  error for a vector
"""
a.size()->3,11
dim =1 means we are working on 11 elements (the results are just 3)
a.std(dim=0)
"""

# Variables and Gradients
from torch.autograd import Variable

# Variable wraps a tensor
my_var = Variable(torch.ones(3, 3), requires_grad=True)

# variables and tensors are same in operations like + and *
# difference is in gradient
# .backward() and .grad

# super(…) inherits all the attributes of the class
"""
Pytorch save models in pickle files
Model.save(model.state_dict(), ‘k_model.pk1’)
"""

# CPU to GPU
# to use GPU model & variable have to be on GPU
"""
if torch.cuda.is_available():  # this way runs both on cpu and gpu
    model.cuda()

if torch.cuda.is_available():
    inputs = Variable(...).cuda()
else:
    inputs = Variable(...)
"""
