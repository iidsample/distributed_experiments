import torch
import torchvision
import torch.nn as nn

model = torchvision.resnet18(pretrained=True)

model_conv_output = nn.sequential(*list(model.children()))[:-2]
device = "cuda:0"
model_conv_output.to(device)

for model_conv_output.parameters():
    model_conv_output.requires_grad = False


