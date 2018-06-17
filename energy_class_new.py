# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 19:37:00 2018

@author: mayiping
"""

import math
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import cv2
#import Image
#img = mpimg.imread('coast.jpg') # 读取和代码处于同一目录下的 coast.jpg
# 此时 img 就已经是一个 np.array 了，可以对它进行任意处理
# print(img.shape) #(400, 600, 3)
'''
plt.subplot(231)
plt.imshow(img) # 显示图片
plt.title('coast')
plt.axis('off') # 不显示坐标轴
'''
class ENERGY:
    def __init__(self, type):
        #kernel for local entropy computation
        self.kernel_9_9 = np.ones((9,9)).astype(np.float64)
        
        #kernel for forward energy computation
        self.kernel_x = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]], dtype=np.float64)
        self.kernel_y_left = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float64)
        self.kernel_y_right = np.array([[0., 0., 0.], [1., 0., 0.], [0., -1., 0.]], dtype=np.float64)
        
        self.energy_type = type
        
    def compute_energy(self, img):
        if self.energy_type == 0:
            return self.without_le(img)
        elif self.energy_type == 1:
            return self.with_le(img)
        elif self.energy_type == 2:
            return self.forward(img)
        
    def without_le(self, img):
        height, width = img.shape[:2]
        B,G,R = cv2.split(img)
        
        M = np.zeros((height,width))
        
        Kernel = np.zeros((9,3,3))
        for i in range (0, 9):
            tmp = np.zeros((3,3))
            tmp[int(i/3), i%3] = -1
            Kernel[i] = np.array([[0,0,0],[0,1,0],[0,0,0]])+tmp

        for i in range (4, 8):
            Kernel[i] = Kernel[i+1]
        
        Rres = np.zeros((height, width))
        Gres = np.zeros((height, width))
        Bres = np.zeros((height, width))
        for i in range (0, 8):
            res = cv2.filter2D(R,-1,kernel=Kernel[i],anchor=(-1,-1))
            res = abs(res)
            Rres += res
        Rres = Rres/8
        
        for i in range (0, 8):
            res = cv2.filter2D(G,-1,kernel=Kernel[i],anchor=(-1,-1))
            res = abs(res)
            Gres += res
        Gres = Gres/8
        
        for i in range (0, 8):
            res = cv2.filter2D(B,-1,kernel=Kernel[i],anchor=(-1,-1))
            res = abs(res)
            Bres +=res
        Bres = Bres/8
        
        M = (Rres + Gres + Bres)/3
            
        return M # M is a matrix
    
    def with_le(self, img):
        height, width = img.shape[:2]
        B,G,R = cv2.split(img)
        Gray = R*0.3 + G*0.59 + B*0.11
        kernel99= np.ones((9,9))
        basef = cv2.filter2D(Gray,-1,kernel=kernel99,anchor=(-1, -1))
        
        
        H = np.zeros((height,width))
        for i in range(4, height - 4):
            for j in range (4, width - 4):
                p = np.zeros((9,9))
                s = 0
                for m in range (i-4,i+5):
                    for n in range (j-4,j+5):
                        p[m-(i-4),n-(j-4)] = Gray[m,n] / basef[i,j]
                        p[m-(i-4),n-(j-4)] = -p[m-(i-4),n-(j-4)]*math.log(p[m-(i-4),n-(j-4)])
                        s += p[m-(i-4),n-(j-4)]
                H[i,j] = s
        
        M = self.without_le(img)
        return H + M
    
    def forward(self,img):
        energy_map = self.with_le(img)
        return self.forward_energy_map(energy_map, img)
    
    def forward_energy_map(self, energy_map,img):
        mat_x = self.neighbourmat_forward(self.kernel_x, img)
        mat_y_left = self.neighbourmat_forward(self.kernel_y_left, img)
        mat_y_right = self.neighbourmat_forward(self.kernel_y_right, img)
        
        m,n = energy_map.shape
        F = np.copy(energy_map)
        
        for i in range (1,m):
            for j in range (0,n):
                if j == 0:
                    e_right = F[i-1,j+1]+mat_x[i-1,j+1]+mat_y_right[i-1,j+1]
                    e_up = F[i-1,j]+mat_x[i-1,j]
                    F[i,j]=energy_map[i,j]+min(e_right,e_up)
                elif j == n-1:
                    e_left = F[i-1,j-1]+mat_x[i-1,j-1]+mat_y_left[i-1,j-1]
                    e_up = F[i-1,j]+mat_x[i-1,j]
                    F[i,j]=energy_map[i,j]+min(e_left,e_up)
                else:
                    e_left = F[i-1,j-1]+mat_x[i-1,j-1]+mat_y_left[i-1,j-1]
                    e_right = F[i-1,j+1]+mat_x[i-1,j+1]+mat_y_right[i-1,j+1]
                    e_up = F[i-1,j]+mat_x[i-1,j]
                    F[i,j] = energy_map[i,j]+min(e_left,e_right,e_up)
        return F       


    def neighbourmat_forward(self,kernel,img):
        B,G,R = cv2.split(img)
        res = np.absolute(cv2.filter2D(B,-1,kernel=kernel,anchor=(-1, -1)))+\
              np.absolute(cv2.filter2D(G,-1,kernel=kernel,anchor=(-1, -1)))+\
              np.absolute(cv2.filter2D(R,-1,kernel=kernel,anchor=(-1, -1)))
        return res
