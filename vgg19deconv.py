import torch.nn as nn
import torchvision.models as models

import numpy as np

vgg19_pretrained = models.vgg19(pretrained=True)
cfg=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

class VGG19_conv(torch.nn.Module):
    def __init__(self, n_classes):
        super(VGG19_conv, self).__init__()
        self.features=make_layers()
        self.feature_outputs = [0]*len(self.features)
        self.pool_indices = dict()
        self._initialize_weights()


    def _initialize_weights(self):
        # initializing weights using ImageNet-trained model from PyTorch
        for i, layer in enumerate(vgg19_pretrained.features):
            if isinstance(layer, nn.Conv2d):
                self.features[i].weight.data = layer.weight.data
                self.features[i].bias.data = layer.bias.data
'''
    def get_conv_layer_indices(self):
        return [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
'''
    def forward_features(self, x):
        output = x
        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                output, indices = layer(output)
                self.feature_outputs[i] = output
                self.pool_indices[i] = indices
            else:
                output = layer(output)
                self.feature_outputs[i] = output
        return output

    def make_layers():
        layers=[]
        in_channels=3
        for v in cfg:
            if v == 'M':
                layers+=[nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)]
            else:
                layers+=[nn.Conv2d(in_channels, v, kernel_size=3, padding=1), nn.ReLU()]
                in_channels=v
        return nn.Sequential(*layers)


class VGG19_deconv(torch.nn.Module):
    def __init__(self):
        super(VGG19_deconv, self).__init__()

        #!!!!todo
        self.conv2DeconvIdx = {0:17, 2:16, 5:14, 7:13, 10:11, 12:10, 14:9, 17:7, 19:6, 21:5, 24:3, 26:2, 28:1}
        self.conv2DeconvBiasIdx = {0:16, 2:14, 5:13, 7:11, 10:10, 12:9, 14:7, 17:6, 19:5, 21:3, 24:2, 26:1, 28:0}
        self.unpool2PoolIdx = {15:4, 12:9, 8:16, 4:23, 0:30}

        
        self.deconv_features = make_layers()
        self.deconv_first_layers = make_layers(first_layer=True)
        self._initialize_weights()


    def make_layers(first_layer=False):
        layers=[]
        for v in range(len(cfg), -1, -1):
            if cfg[v] == 'M':
                layers+=[nn.MaxUnpool2d(2, stride=2)]
            elif v!=0:
                if cfg[v-1]=='M':
                    if first_layer:
                        layers+=[nn.ConvTranspose2d(1, cfg[v-2], 3, padding=1)]
                    else:
                        layers+=[nn.ConvTranspose2d(cfg[v], cfg[v-2], 3, padding=1)]
                else:
                    layers+=[nn.ConvTranspose2d(cfg[v], cfg[v], 3, padding=1)]
        if first_layer:
            return nn.ModuleList(*layers)
        else:
            return nn.Sequential(*layers)



    def _initialize_weights(self):
        # initializing weights using ImageNet-trained model from PyTorch
        for i, layer in enumerate(vgg19_pretrained.features):
            if isinstance(layer, nn.Conv2d):
                self.deconv_features[self.conv2DeconvIdx[i]].weight.data = layer.weight.data
                biasIdx = self.conv2DeconvBiasIdx[i]
                if biasIdx > 0:
                    self.deconv_features[biasIdx].bias.data = layer.bias.data
                

    def forward(self, x, layer_number, map_number, pool_indices):
        start_idx = self.conv2DeconvIdx[layer_number]
        if not isinstance(self.deconv_first_layers[start_idx], torch.nn.ConvTranspose2d):
            raise ValueError('Layer '+str(layer_number)+' is not of type Conv2d')
        # set weight and bias
        self.deconv_first_layers[start_idx].weight.data = self.deconv_features[start_idx].weight[map_number].data[None, :, :, :]
        self.deconv_first_layers[start_idx].bias.data = self.deconv_features[start_idx].bias.data        
        # first layer will be single channeled, since we're picking a particular filter
        output = self.deconv_first_layers[start_idx](x)

        # transpose conv through the rest of the network
        for i in range(start_idx+1, len(self.deconv_features)):
            if isinstance(self.deconv_features[i], torch.nn.MaxUnpool2d):
                output = self.deconv_features[i](output, pool_indices[self.unpool2PoolIdx[i]])
            else:
                output = self.deconv_features[i](output)
        return output
