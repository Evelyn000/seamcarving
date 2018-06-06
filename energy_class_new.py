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
img = mpimg.imread('coast.jpg') # 读取和代码处于同一目录下的 coast.jpg
# 此时 img 就已经是一个 np.array 了，可以对它进行任意处理
print(img.shape) #(400, 600, 3)

plt.subplot(231)
plt.imshow(img) # 显示图片
plt.title('coast')
plt.axis('off') # 不显示坐标轴

class ENERGY:
    def __init__(self):
        #kernel for local entropy computation
        self.kernel_9_9 = np.ones((9,9)).astype(np.float64)
        
        #kernel for forward energy computation
        self.kernel_x = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]], dtype=np.float64)
        self.kernel_y_left = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float64)
        self.kernel_y_right = np.array([[0., 0., 0.], [1., 0., 0.], [0., -1., 0.]], dtype=np.float64)
        
        self.energy_type = {"without_le","with_le","forward"}
        
    def compute_energy(self, img):
        if self.energy_type == "without_le":
            return self.without_le(img)
        elif self.energy_type == "with_le":
            return self.with_le(img)
        elif self.energy_type == "forward":
            return self.forward(img)
        
    def without_le(self, img):
        height, width = img.shape[:2]
        B = (img[0:height, 0:width, 0]).reshape((height,width))
        G = (img[0:height, 0:width, 1]).reshape((height,width))
        R = (img[0:height, 0:width, 2]).reshape((height,width))
        M = np.zeros((height,width))
        xlist = [-1,0,1,1,1,0,-1,-1]
        ylist = [-1,-1,-1,0,1,1,1,0]
        val = np.zeros(8)
        
        Rpad = np.pad(R,((1,1),(1,1)),'constant',constant_values=(0,0))
        Gpad = np.pad(G,((1,1),(1,1)),'constant',constant_values=(0,0))
        Bpad = np.pad(B,((1,1),(1,1)),'constant',constant_values=(0,0))
        
        for i in range (1,height + 1):
            for j in range (1,width + 1):
                s = 0
                for d in range (0,8):
                    val[d]=(abs(Rpad[i,j]-Rpad[i+xlist[d],j+ylist[d]])+abs(Gpad[i,j]-Gpad[i+xlist[d],j+ylist[d]])+abs(Bpad[i,j]-Bpad[i+xlist[d],j+ylist[d]]))/3
                    s += val[d]
                if (i==1 and j==1) or (i==1 and j==width) or (i==height and j==1) or (i==height and j==width):
                        normalize = 3
                elif (i>1 and i<height+1 and j>1 and j<width+1):
                        normalize = 8
                else:
                        normalize = 5
                M[i-1,j-1] = s/normalize            
            
        return M # M is a matrix
    
    def with_le(self, img):
        height, width = img.shape[:2]
        B,G,R = cv2.split(img)
        Gray = R*0.3 + G*0.59 + B*0.11
       
        N = self.neighbourmat_le(Gray, self.kernel_9_9)
        H = np.zeros((height,width))
        for i in range(4, height - 4):
            for j in range (4, width - 4):
                p = np.zeros((9,9))
                s = 0
                for m in range (i-4,i+5):
                    for n in range (j-4,j+5):
                        p[m,n] = Gray[m,n] / N[i,j]
                        p[m,n] = -p[m,n]*math.log(p[m,n])
                        s += p[m,n]
                H[i,j] = s
        
        M = self.energy_map_without_le(self)
        return H + M
    
    def forward(self,img):
        energy_map = self.with_le(img)
        return self.forward_energy_map(self, energy_map)
    
    def forward_energy_map(self, energy_map):
        mat_x = self.neighbourmat_forward(self.kernel_x)
        mat_y_left = self.neighbourmat_forward(self.kernel_y_left)
        mat_y_right = self.neighbourmat_forward(self.kernel_y_right)
        
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

    def neighbourmat_le(Gray,kernel):
        res = cv2.filter2D(Gray,-1,kernel=kernel,anchor=(-1, -1))
        return res
    def neighbourmat_forward(self,kernel):
        B,G,R = cv2.split(self.img_out)
        res = np.absolute(cv2.filter2D(B,-1,kernel=kernel,anchor=(-1, -1)))+\
              np.absolute(cv2.filter2D(G,-1,kernel=kernel,anchor=(-1, -1)))+\
              np.absolute(cv2.filter2D(R,-1,kernel=kernel,anchor=(-1, -1)))
        return res