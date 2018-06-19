import torch
import matplotlib.image as mpimg
import numpy as np
from torchvision import models
from torch.autograd import Variable
import torchvision.transforms as transforms
import cv2

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
            
    def extract_firstrelu(self):
        x = self.image
        cnt = 0
        for index, layer in enumerate(self.model):
            print(index, layer)
            if cnt == 1:
                #print(x)
                return x
            x = layer(x)
            cnt = cnt + 1
            
    def extract_secondrelu(self):
        x = self.image
        cnt = 0
        for index, layer  in enumerate(self.model):
            print(index,layer)
            if cnt == 3:
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
    # here extract features directly
    pretrained_model = models.vgg19(pretrained=True).features 
    model = vggmodel(pretrained_model)
    model.show()
    firstrelu = model.extract_firstrelu()
    a = firstrelu.squeeze(0)
    b = a.data.numpy()
    channel, height, width = b.shape
    acmp = np.zeros((height, width))
    for i in range (0, channel):
        acmp += abs(b[i])
    
    print('acmp', acmp)
    
    B = acmp
    G = acmp
    R = acmp
    img =cv2.merge([B,G,R])
    cv2.imwrite('acmp_res.jpg', img)
    
    
