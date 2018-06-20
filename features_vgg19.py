import torch
import matplotlib.image as mpimg
import numpy as np
from torchvision import models
from torch.autograd import Variable
import torchvision.transforms as transforms
import cv2

class vggmodel():
    def __init__(self, model, img):
        self.model = model
        self.model.eval()
        self.image = self.image_for_pytorch(img)

    def show(self):
        x = self.image
        for index, layer in enumerate(self.model):
            print(index,layer)
            print(x)  # print every layer value
            x = layer(x)
            
    def extract_firstlayer(self):
        x = self.image
        cnt = 0
        for index, layer in enumerate(self.model):
            print(index, layer)
            if cnt == 7:
                #print(x)
                return x
            x = layer(x)
            cnt = cnt + 1
            
    def extract_secondlayer(self):
        x = self.image
        cnt = 0
        for index, layer  in enumerate(self.model):
            print(index,layer)
            if cnt == 10:
                return x
            x = layer(x)
            cnt = cnt + 1
    
    def extract_thirdlayer(self):
        x = self.image
        cnt = 0
        for index, layer  in enumerate(self.model):
            print(index,layer)
            if cnt == 19:
                return x
            x = layer(x)
            cnt = cnt + 1
    
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
    
    img = mpimg.imread('coast.jpg')
    # here extract features directly
    pretrained_model = models.vgg19(pretrained=True).features 
    model = vggmodel(pretrained_model, img)
    model.show()
    
    firstlayer = model.extract_firstlayer()
    secondlayer = model.extract_secondlayer()
    thirdlayer = model.extract_thirdlayer()
    print('firstlayer shape', firstlayer.shape)
    print('secondlayer shape', secondlayer.shape)
    print('thirdlayer shape', thirdlayer.shape)
    
    featuretensor1 = (firstlayer.squeeze(0)).data.numpy()
    featuretensor2 = (secondlayer.squeeze(0)).data.numpy()
    featuretensor3 = (thirdlayer.squeeze(0)).data.numpy()
    
    channel, height, width = featuretensor1.shape
    featuremap1 = np.zeros((height, width))
    for i in range (0, channel):
        featuremap1 += abs(featuretensor1[i]*featuretensor1[i])
    
    channel, height, width = featuretensor2.shape
    featuremap2 = np.zeros((height, width))
    for i in range (0, channel):
        featuremap2 += abs(featuretensor2[i]*featuretensor2[i])
    
    channel, height, width = featuretensor3.shape
    featuremap3 = np.zeros((height, width))
    for i in range (0, channel):
        featuremap3 += abs(featuretensor3[i]*featuretensor3[i])
    
    height, width = img.shape[:2]    
    np.resize(featuremap1, (height, width))
    np.resize(featuremap2, (height, width))
    np.resize(featuremap3, (height, width))
    
    featuremap = featuremap1 + featuremap2 + featuremap3
    
    b = featuremap
    g = featuremap
    r = featuremap
    img =cv2.merge([b,g,r])
    cv2.imwrite('res.jpg', img)
    
    
