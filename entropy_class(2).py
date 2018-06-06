# -*- coding: utf-8 -*-
"""
Created on Mon May 28 18:47:22 2018

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

class ENTROPY:
    def __init__(self, filename, height_out, width_out):
        self.filename = filename
        self.height_out = height_out
        self.width_out = width_out
        self.img_in = mpimg.imread(filename).astype(np.float64)
        self.img_out = np.copy(self.img_in)
        self.height_in, self.width_in = self.img_in.shape[:2]
        
        #kernel for local entropy computation
        self.kernel_9_9 = np.ones((9,9)).astype(np.float64)
        
        #kernel for forward energy computation
        self.kernel_x = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]], dtype=np.float64)
        self.kernel_y_left = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float64)
        self.kernel_y_right = np.array([[0., 0., 0.], [1., 0., 0.], [0., -1., 0.]], dtype=np.float64)
        
        B,G,R = cv2.split(self.out_image)
        self.Gray = R*0.3 + G*0.59 + B*0.11
        
    #le := local entropy
    def energy_map_without_le(self):
        B = (self.img_in[0:self.height_in, 0:self.width_in, 0]).reshape((self.height_in,self.width_in))
        G = (self.img_in[0:self.height_in, 0:self.width_in, 1]).reshape((self.height_in,self.width_in))
        R = (self.img_in[0:self.height_in, 0:self.width_in, 2]).reshape((self.height_in,self.width_in))
        M = np.zeros((self.height_in,self.width_in))
        xlist = [-1,0,1,1,1,0,-1,-1]
        ylist = [-1,-1,-1,0,1,1,1,0]
        val = np.zeros(8)
        
        Rpad = np.pad(R,((1,1),(1,1)),'constant',constant_values=(0,0))
        Gpad = np.pad(G,((1,1),(1,1)),'constant',constant_values=(0,0))
        Bpad = np.pad(B,((1,1),(1,1)),'constant',constant_values=(0,0))
        
        for i in range (1,self.height_in + 1):
            for j in range (1,self.width_in + 1):
                s = 0
                for d in range (0,8):
                    val[d]=(abs(Rpad[i,j]-Rpad[i+xlist[d],j+ylist[d]])+abs(Gpad[i,j]-Gpad[i+xlist[d],j+ylist[d]])+abs(Bpad[i,j]-Bpad[i+xlist[d],j+ylist[d]]))/3
                    s += val[d]
                if (i==1 and j==1) or (i==1 and j==self.width_in) or (i==self.height_in and j==1) or (i==self.height_in and j==self.width_in):
                        normalize = 3
                elif (i>1 and i<self.height_in+1 and j>1 and j<self.width_in+1):
                        normalize = 8
                else:
                        normalize = 5
                M[i-1,j-1] = s/normalize            
            
        return M # M is a matrix
        
    def Entropy(t):
        return -t * math.log(t)
    
    def neighbourmat_le(self,kernel):
        res = cv2.filter2D(self.Gray,-1,kernel=kernel,anchor=(-1, -1))
        return res
    
    def energy_map_with_le(self):
        N = self.neighbourmat_le(self, self.kernel_9_9)
        H = np.zeros((self.height_in,self.width_in))
        for i in range(4, self.height_in - 4):
            for j in range (4, self.width_in - 4):
                p = np.zeros((9,9))
                s = 0
                for m in range (i-4,i+5):
                    for n in range (j-4,j+5):
                        p[m,n] = self.Gray[m,n] / N[i,j]
                        p[m,n] = -p[m,n]*math.log(p[m,n])
                        s += p[m,n]
                H[i,j] = s
        
        M = self.energy_map_without_le(self)
        return H + M
    
    def neighbourmat_forward(self,kernel):
        B,G,R = cv2.split(self.img_out)
        res = np.absolute(cv2.filter2D(B,-1,kernel=kernel,anchor=(-1, -1)))+\
              np.absolute(cv2.filter2D(G,-1,kernel=kernel,anchor=(-1, -1)))+\
              np.absolute(cv2.filter2D(R,-1,kernel=kernel,anchor=(-1, -1)))
        return res
    
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
