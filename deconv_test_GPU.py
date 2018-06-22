from vgg19deconvtest import *
import numpy as np
import cv2
import sys
import torch

def decon_img(raw_img):
    img = (raw_img-raw_img.min())/(raw_img.max()-raw_img.min())*255
    #img = img.astype(np.uint16)
    ret = np.zeros((1, 224, 224), dtype=np.float64)
    ret[0] = (img[0]+img[1]+img[2])/3.0
    ret = (ret-ret.min())/(ret.max()-ret.min())*255
    ret= ret.transpose(1,2,0)
    return ret

def energy_vgg(img, layer):
	m, n, c = img.shape
	img = cv2.resize(img, (224, 224))

	img_var = torch.autograd.Variable(torch.FloatTensor(img.transpose(2,0,1)[np.newaxis,:,:,:].astype(float)).cuda())


	conv = VGG19_conv()
	conv.cuda()
	conv_layer_indices = conv.get_conv_layer_indices()

	conv_out = conv(img_var)

	deconv=VGG19_deconv()
	deconv.cuda()
	if layer in conv_layer_indices:
		print('layer:', layer, conv.feature_outputs[layer].shape)
		ret = deconv(conv.feature_outputs[layer], layer, conv.pool_indices)
		img = decon_img(ret.data.cpu().numpy()[0])
		img = cv2.resize(img, (n, m))
		return img
		'''
		n_maps = conv.feature_outputs[layer].data.cpu().numpy().shape[1]


		raw_map = np.zeros((3, 224, 224), dtype=np.float64)
		for map_idx in range(n_maps):
			ret = deconv(conv.feature_outputs[layer][0][map_idx][None,None,:,:], layer, map_idx, conv.pool_indices)
			raw_map += ret.data.cpu().numpy()[0]
		img = decon_img(raw_map)
		img = cv2.resize(img, (n, m))
		return img
		'''

	else:
		return -1


if __name__ == '__main__':
	filename = sys.argv[1]
	img = cv2.imread(filename).astype(np.int16)
	
	for layer in [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]:
		energy_map = energy_vgg(img, layer)
		energy_map = energy_map.astype(np.uint16)
		filename='./deconvtest/layer'+str(layer)+'.jpg'
		cv2.imwrite(filename, energy_map)
