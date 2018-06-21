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

    def get_conv_layer_indices(self):
        return [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]

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
        self.conv2DeconvIdx = {0:20, 2:19, 5:17, 7:16, 10:14, 12:13, 14:12, 16:11, 19:9, 21:8, 23:7, 25:6, 28:4, 30:3, 32:2, 34:1}
        self.conv2DeconvBiasIdx = {0:19, 2:17, 5:16, 7:14, 10:13, 12:12, 14:11, 16:9, 19:8, 21:7, 23:6, 25:4, 28:3, 30:2, 32:1, 34:0}
        self.unpool2PoolIdx = {18:4, 15:9, 10:18, 5:27, 0:36}

        
        self.deconv_features = make_layers()
        self.deconv_first_layers = make_layers(first_layer=True)
        self._initialize_weights()


    def make_layers(first_layer=False):
        layers=[]
        for v in range(len(cfg)-1, -1, -1):
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
