from vgg19deconv import *
import numpy as np
import cv2
import sys
import torch

def decon_img(layer_output):
    raw_img = layer_output.transpose(1,2,0)
    img = (raw_img-raw_img.min())/(raw_img.max()-raw_img.min())*255
    img = img.astype(np.uint16)
    return img

if __name__ == '__main__':
	filename = sys.argv[1]
	img = cv2.imread(filename).astype(np.int16)
	m, n, c = img.shape
	img = cv2.resize(img, (224, 224))

	img_var = torch.autograd.Variable(torch.FloatTensor(img.transpose(2,0,1)[np.newaxis,:,:,:].astype(float)).cuda())


	conv = VGG19_conv()
	conv.cuda()
	conv_layer_indices = conv.get_conv_layer_indices()

	conv_out = conv(img_var)

	deconv=VGG19_deconv()
	deconv.cuda()
	for layer in conv_layer_indices:
		n_maps = conv.feature_outputs[layer].data.cpu().numpy().shape[1]


		raw_map = np.zeros((3, 224, 224), dtype=np.float64)
		for map_idx in range(n_maps):
			ret = deconv(conv.feature_outputs[layer][0][map_idx][None,None,:,:], layer, map_idx, conv.pool_indices)
			raw_map += ret.data.cpu().numpy()[0]
		#decon = deconv(conv.feature_outputs[layer][0][map_idx][None,None,:,:], layer, map_idx, conv.pool_indices)
		img = decon_img(raw_map)
		img=cv2.resize(img, (n, m))
		filename='./deconv/deconvlayer'+str(layer)+'.jpg'
		print(filename, img.shape)
		print(img)
		cv2.imwrite(filename, img)


'''
	layer=28

	n_maps = conv.feature_outputs[layer].data.numpy().shape[1]
	map_idx = 0
		#for map_idx in range(n_maps):
		#	decon = vgg16_d(conv.feature_outputs[layer][0][map_idx][None,None,:,:], conv_layer, map_idx, conv.pool_indices)
	decon = deconv(conv.feature_outputs[layer][0][map_idx][None,None,:,:], layer, map_idx, conv.pool_indices)
	#print(layer, ' ', decon.shape, '\n')
	img = decon_img(decon)
	'''