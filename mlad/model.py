import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.nn.init import kaiming_uniform_
import torch
import torchvision.models.video as video_models
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
from .models.v1 import VideoTransformer as Model1


def weights_init(m):
    if isinstance(m, nn.Linear):
        kaiming_uniform_(m.weight.data)


def build_model(version, num_clips, num_classes, feature_dim, hidden_dim, num_layers):
    if version == 'v1':
        model = Model1(num_clips, num_classes, feature_dim, hidden_dim, num_layers)
    return model


if __name__ == '__main__':
    model = build_model('v1', 128, 65, 2048, 128, 5)
    model.cuda()

    model.eval()
    features = Variable(torch.rand(2, 128, 2048)).cuda()
    activations = model(features)

    for key in activations:
        print(key, ' ----> ', activations[key].shape)
    
    total_params = sum(p.numel() for p in model.parameters()) 
    print(total_params)
