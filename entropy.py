# -*- coding: utf-8 -*-
"""
Created on Sat May 26 13:30:28 2018

@author: mayiping
"""
import sys
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import Image
img = mpimg.imread('coast.jpg') # 读取和代码处于同一目录下的 coast.jpg
# 此时 img 就已经是一个 np.array 了，可以对它进行任意处理
print(img.shape) #(400, 600, 3)

plt.subplot(231)
plt.imshow(img) # 显示图片
plt.title('coast')
plt.axis('off') # 不显示坐标轴

def Image_transpose(img): # im is a image
    return img

#le := local entropy
def Energy_function_without_le(X): # X is img(400,600,3)
    return M # M is a matrix

def Energy_funciton_with_le(X): # X is img(400,600,3)
    return M

