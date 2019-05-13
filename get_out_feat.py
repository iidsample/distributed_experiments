import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np

def main():
    val_range = np.arange(0,1,0.1)
    data_dict = dict()
    max_val = 0;
    min_val = 0;
    # initialize data dict 
    for i in val_range:
        data_dict[i] = 0

    valdir = os.path.join('/home/ubuntu/data/imagenet', 'validation') # path to data folder

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

    
    model = models.resnet18(pretrained=True)

    model_conv_output = nn.Sequential(*list(model.children()))[:-1]
    device = "cuda:0"
    model_conv_output.to(device)
    tota_elements = 0
    for m in model_conv_output.parameters():
        m.requires_grad = False

    for i, (input, target) in enumerate(val_loader):
        input = input.to(device)
        target = target.to(device)

        output = model_conv_output(input)
        tota_elements += output.reshape(-1).size(0)
        for val_test in val_range:
            # puts one where condition is matched
            # import ipdb; ipdb.set_trace()
            num_ones = torch.where(output <= val_test, torch.tensor(1, device=device), torch.tensor(0, device=device))
            elem_less = num_ones.nonzero().size(0)
            data_dict[val_test] += elem_less
            max_batch = torch.max(output)
        if max_batch > max_val:
            max_val = max_batch
        min_batch = torch.min(output)
        if min_batch < min_val:
            min_val = min_batch
        
    print ("Max val = {}".format(max_val))
    print ("Min val = {}".format(min_val))
    print ("Num stats = {}".format(data_dict))
    print ("Total num elements = {}".format(tota_elements))
   

if __name__ == "__main__":
    main()

            
        


    


