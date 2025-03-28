import json
import os
import time
import sys

import logging

import numpy as np

import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset, ConcatDataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate


root = '/home/kxj200023/data/'

# LOCAL_IMAGE_LIST_PATH = 'metas/intra_test/train_label.json'

DEST = '/home/kxj200023/data/CCMNIST1/'

transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

original_dataset_tr = MNIST(root, train=True, download=True, transform=transform)
original_dataset_te = MNIST(root, train=False, download=True, transform=transform)

data = ConcatDataset([original_dataset_tr, original_dataset_te])
original_images = torch.cat([img for img, _ in data])
original_labels = torch.cat([torch.tensor(label).unsqueeze(0) for _, label in data])

shuffle = torch.randperm(len(original_images))

original_images = original_images[shuffle]
original_labels = original_labels[shuffle]

datasets = []
dict = {}
num = 1
environments = [0,1,2]



def torch_bernoulli_(p, size):
    return (torch.rand(size) < p).float()


def torch_xor_(a, b):
    return (a - b).abs()

def colored_dataset(images, labels, env):

    x = torch.zeros(len(images), 1, 224, 224)

    environement_color = -1
    if env == 0:
        environement_color = 0.1
    elif env == 1:
        environement_color = 0.3
    elif env == 2:
        environement_color = 0.5


    # Assign a binary label based on the digit
    fake_labels = (labels < 5).float()


    # Assign a color based on the label; flip the color with probability e
    colors = torch_xor_(fake_labels, torch_bernoulli_(environement_color, len(fake_labels)))

    x = torch.squeeze(images, dim=1)
    ones = torch.ones_like(x)
    x = torch.logical_xor(x, ones).float()

    zeros = torch.zeros_like(x)
    bg1 = torch.logical_xor(x, zeros).float()
    bg2 = bg1
    ones = torch.ones_like(bg2)
    x = torch.logical_xor(bg1,ones)
    for i in range(x.size(0)):
        if colors[i] == 1.0:
            x[i,:,:][x[i,:,:] == 0.0] = 1.0
        else:
            bg2[i,:,:] = 0.0

    if env == 0:
        x = torch.stack([x, bg2, bg2], dim=1)
    elif env == 1:
        x = torch.stack([bg2, x, bg2], dim=1)
    elif env == 2:
        x = torch.stack([bg2, bg2, x], dim=1)


    # Apply the color to the image by zeroing out the other color channel
    # x[torch.tensor(range(len(x))), (1 - colors).long(), :, :] *= 0
    # x[torch.tensor(range(len(x))), :, :, :][x[torch.tensor(range(len(x))), :, :, :] == 0] = 1


    x = x.float()  # .div_(255.0)
    y = fake_labels.view(-1).long()

    return [x,y,colors]

for i in range(len(environments)):
    images = original_images[i::len(environments)]
    labels = original_labels[i::len(environments)]
    datasets.append(colored_dataset(images, labels, environments[i]))

for i in range(len(environments)):
    outpath = DEST+str(i)+'/'
    x = datasets[i][0]
    y = datasets[i][1]
    z = datasets[i][2]

    # torch.save((x, y, z), "/data/kxj200023/CCMNIST1/" + str(i) + "_224.pt")

    for j in range(y.size(0)):
        image_tensor = x[j]
        label = y[j].item()
        color = z[j].item()
        img_name = str(num)+'.png'
        dict[img_name]=[i,label,color]
        torchvision.utils.save_image(image_tensor,outpath+str(label)+'/'+img_name)
        num += 1


with open(DEST+'data.json', 'w') as fp:
    json.dump(dict, fp)