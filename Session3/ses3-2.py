# -*- coding: utf-8 -*-
"""
Created on Feb 23 23:22:40 2020

@author: Ernest (Khashayar) Namdar
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from unet_model import UNet
from PIL import Image
import torchvision.transforms.functional as TF


def random_seed(seed_value):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value) # gpu vars
    torch.backends.cudnn.deterministic = True  #needed
    torch.backends.cudnn.benchmark = False


def train(iters, inp, gt, model, optimizer, criterion):
    if torch.cuda.is_available():
        inp = Variable(inp.cuda())
        gt = Variable(gt.cuda())
    else:
        inp = Variable(inp)
        gt = Variable(gt)
    for it in range(iters):
        # print("Iteration #", it+1, "is now started!")

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        output = model(inp)
        # print(torch.max(output))
        # print(torch.min(output))

        fig = plt.figure()
        plt.imshow(output[0].detach().cpu().permute(1,2,0).numpy())
        # plt.show
        writer.add_figure("Output", fig, it+1)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(output.view(-1), gt.view(-1))
        print("Itereation", it+1, "Loss is:", loss.item())
        writer.add_scalar("Itereation_Loss", loss.item(), it+1)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()


if __name__ == "__main__":
    seed = 21
    random_seed(seed)
    writer = SummaryWriter()

    image = Image.open('./ht2-c2.jpg')
    out = TF.to_tensor(image)
    out = out.reshape(1,3,640,640)
    inp = torch.rand(1,3,640,640)

    fig = plt.figure()
    plt.imshow(out[0].permute(1,2,0).numpy())
    # plt.show
    writer.add_figure("Ground Truth", fig)

    fig = plt.figure()
    plt.imshow(inp[0].permute(1,2,0).numpy())
    writer.add_figure("Input", fig)

    num_iter = 500
    writer.add_scalar("Number_of_Iterations", num_iter)

    model = UNet(3, 3)
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.MSELoss()

    learning_rate = 0.1
    writer.add_scalar("Learning_Rate", learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    train(num_iter, inp, out, model, optimizer, criterion)
    writer.close()

