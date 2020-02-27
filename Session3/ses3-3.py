# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:01:40 2020

@author: Ernest Namdar

Ref:
    https://github.com/onnx/tutorials/blob/master/tutorials/PytorchTensorflowMnist.ipynb
"""

from torch.autograd import Variable
import torch
from models import CNN


# Load the trained model
PyTorch_model = CNN()
PyTorch_model.load_state_dict(torch.load("./CNN_State.pt"))

# Export the trained model to ONNX
dummy_input = Variable(torch.randn(1, 1, 28, 28)) # one single-channel 28x28 picture will be the input to the model
torch.onnx.export(PyTorch_model, dummy_input, "./CNN.onnx")