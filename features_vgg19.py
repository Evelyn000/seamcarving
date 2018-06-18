# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 17:58:21 2018

@author: mayiping
"""

import torch
import matplotlib.image as mpimg
import numpy as np
from torchvision import models
from torch.autograd import Variable
import torchvision.transforms as transforms

class vggmodel():
    def __init__(self, model):
        self.model = model
        self.model.eval()
        img = mpimg.imread('coast.jpg')
        self.image = self.image_for_pytorch(img)

    def show(self):
        x = self.image
        for index, layer in enumerate(self.model):
            print(index,layer)
            #print(layer.weight)
            print(x)  # print every layer value
            x = layer(x)
            
    def image_for_pytorch(self, img):
        transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]  
            transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                 std=(0.229, 0.224, 0.225))
        ])
            
        imgres = transform(img)
        imgres = Variable(torch.unsqueeze(imgres, dim=0), requires_grad=True)
        return imgres

if __name__ == '__main__':
    # here extract features directly
    pretrained_model = models.vgg19(pretrained=True).features 
    model = vggmodel(pretrained_model)
    model.show()
