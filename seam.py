# -*- coding: utf-8 -*-
"""
Created on Sat May 26 13:30:25 2018

@author: mayiping
"""

import sys
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
from energy_class_new import ENERGY
#import Image
import cv2


class seam:
    def __init__(self, filename_in, filename_out, m_out, n_out, type, opt_deep_layer=0):
        self.filename_in=filename_in
        self.filename_out=filename_out
        self.m_out=m_out
        self.n_out=n_out
        self.img_in=cv2.imread(filename_in).astype(np.float64)
        self.img_out=self.img_in
        self.energy_solver=ENERGY(type)
        self.opt_deep_layer = opt_deep_layer

    def simple_carve(self):
        m, n, c=self.img_out.shape
        if n>self.n_out:
            self.collapse(self.n_out)
    
        if m>self.m_out:
            self.img_out=np.transpose(self.img_out, (1, 0, 2))
            self.collapse(self.m_out)
            self.img_out=np.transpose(self.img_out, (1, 0, 2))
        
        m, n, c=self.img_out.shape
        
        if n<self.n_out:
            self.enlarge(self.n_out)
                
        if m<self.m_out:
            self.img_out=np.transpose(self.img_out, (1, 0, 2))
            self.enlarge(self.m_out)
            self.img_out=np.transpose(self.img_out, (1, 0, 2))
                            #plt.imshow(self.img_out)
                #print(np.typeDict['self.img_out'])
        cv2.imwrite(self.filename_out, self.img_out)


    '''
    def opt_carve(self):
        m, n, _=self.img_out.shape
        r=abs(m-self.m_out)
        c=abs(n-self.n_out)
        e=np.zeros((r+1, c+1), dtype=np.float64)
        eseamr=np.zeros((r+1, c+1, n), dtype=np.int16)
        eseamc=np.zeros((r+1, c+1, m), dtype=np.int16)
        direction=np.zeros((r+1, c+1), dtype=np.int16)
        for i in range(0, r+1, 1):
            for j in range(0, c+1, 1):
                if j>0:
                    img_rc=construct_img(i, j-1, eseamr, eseamc, direction)
                    M=self.energy_solver.compute_energy(img_rc)
                    eseamc=Find_seam(M)
    
    def construct_img(r, c, eseamr, eseamc, direction):
        img_o=self.img_in
        while r>0 or c>0:
            if c>0 and direction[r, c]==0:
                Remove_seam(eseamc[r, c], img_o)
            elif r>0 and direction[r, c]==1:
                img_o=np.transpose(img_o, (1, 0, 2))
                Remove_seam(eseamr[r, c], img_o)
                img_o=np.transpose(img_o, (1, 0, 2))
            else:
                print("optimal seam err!")
                sys.exit(2)
            r-=1
            c-=1
        return img_o

    def get_seam_energy(self, M, seam_point_list):
        for i in range()
    '''
    
    # 先默认竖向seam
    def Find_seam(self, M): # M is a matrix of energy
        m, n = M.shape
        seam_point_list = np.zeros((m, ), dtype=np.int16)
        cumulative_energy = np.zeros((m, n), dtype = np.float64)
        last_pos = np.zeros((m, n), dtype = np.int16)
        cumulative_energy[0, :] = M[0, :]
        for i in range(1, m):
            for j in range(n):
                if j == 0:
                    cumulative_energy[i, j] = cumulative_energy[i-1, j:j+2].min()+M[i, j]
                    last_pos[i, j] = np.argmin(M[i-1, j:j+2])
                elif j == n-1:
                    cumulative_energy[i, j] = cumulative_energy[i-1, j-1: n].min()+M[i, j]
                    last_pos[i, j] = np.argmin(M[i-1, j-1:n])+j-1
                else:
                    cumulative_energy[i, j] = cumulative_energy[i-1, j-1: j+2].min()+M[i, j]
                    last_pos[i, j] = np.argmin(M[i-1, j-1: j+2])+j-1
        seam_point_list[-1]=np.argmin(cumulative_energy[-1, :])
        for i in range(m-2, -1, -1):
            seam_point_list[i] = last_pos[i+1, seam_point_list[i+1]]
        return seam_point_list


        '''
        seam_point_list = np.zeros((m, ), dtype=np.int16)
        seam_point_list[-1] = np.argmin(M[-1])
        for i in range(m-2, -1, -1):
            tmp=seam_point_list[i+1]
            if tmp==0:
                seam_point_list[i]=np.argmin(M[i, :2])
            elif tmp==n-1:
                seam_point_list[i]=np.argmin(M[i, n-2:])+n-2
            else:
                seam_point_list[i]=np.argmin(M[i, tmp-1 :tmp+2])+tmp-1
        #print(seam_point_list[i])
        return seam_point_list #seam_point_list 用来储存seam路线上的所有点的坐标
        '''
    ##compute_energy(img)
    '''
    def enlarge(self, n_o):
        m, n, c=self.img_out.shape
        M=self.energy_solver.compute_energy(self.img_out)
        while n<n_o:
            seam_point_list=self.Find_seam(M)
            self.img_out, M=self.Duplicate_seam(seam_point_list, self.img_out, M)
            m, n, c=self.img_out.shape
    '''
    
    def enlarge(self, n_o):
        m, n, c=self.img_out.shape
        mask=self.img_out
        mask_cd=np.zeros((m, n), dtype=np.int16)
        mask_cd[:]=np.arange(n)
        seam_point_list=np.zeros((m, ),dtype=np.int16)
        while n<n_o:
            M=self.energy_solver.compute_energy(mask, self.opt_deep_layer)
            seam_point_list_org=self.Find_seam(M)
            for i in range(m):
                seam_point_list[i]=mask_cd[i][seam_point_list_org[i]]
            self.img_out=self.Duplicate_seam(seam_point_list, self.img_out)
            mask=self.Remove_seam(seam_point_list_org, mask)
            mask_cd=self.remove_mask(seam_point_list_org, mask_cd)
            m, n, c=self.img_out.shape


    def collapse(self, n_o):
        m, n, c =self.img_out.shape
        while n>n_o:
            M=self.energy_solver.compute_energy(self.img_out, self.opt_deep_layer)
            seam_point_list=self.Find_seam(M)
            self.img_out=self.Remove_seam(seam_point_list, self.img_out)
            m, n, c=self.img_out.shape
    #print(M, '\n')
#print(seam_point_list, '\n')

    def Remove_seam(self, seam_point_list, img_i): # X is the orginal matrix, M is X's energy matrix
        m, n, c = img_i.shape
        img_o = np.zeros((m, n-1, 3), dtype=np.float32)
        for i in range(m):
            img_o[i, :, :]=np.delete(img_i[i], [seam_point_list[i]], axis=0)
        return img_o #返回裁剪之后的图片(400,600,3)
            
    def remove_mask(self, seam_point_list, mask):
        m, n=mask.shape
        mask_o=np.zeros((m, n-1), dtype=np.int16)
        for i in range(m):
            mask[i, seam_point_list[i]:]=mask[i, seam_point_list[i]:]+1
            mask_o[i, :]=np.delete(mask[i], [seam_point_list[i]])
        return mask_o

    def Duplicate_seam(self, seam_point_list, img_i):  # X is the orginal matrix, M is X's energy matrix
        m, n, c = img_i.shape
        img_o = np.zeros((m, n+1, 3), dtype=np.float32)
        for i in range(m):
            tmp=seam_point_list[i]
            img_o[i, :tmp+1, :]=img_i[i, :tmp+1, :]
            img_o[i, tmp+1, :]=img_i[i, tmp, :]
            img_o[i, tmp+2:, :]=img_i[i, tmp+1:, :]
        return img_o#返回拉长之后的图片(400,600,3)

'''
    def Duplicate_seam(self, seam_point_list, img_i, M):  # X is the orginal matrix, M is X's energy matrix
        max_energy=268435456
        m, n, c = img_i.shape
        img_o = np.zeros((m, n+1, 3), dtype=np.float32)
        M_out=np.zeros((m, n+1), dtype=np.float64)
        for i in range(m):
            tmp=seam_point_list[i]
            img_o[i, :tmp+1, :]=img_i[i, :tmp+1, :]
            img_o[i, tmp+1, :]=img_i[i, tmp, :]
            img_o[i, tmp+2:, :]=img_i[i, tmp+1:, :]
            M_out[i, :tmp]=M[i, :tmp]
            M_out[i][tmp]=max_energy
            M_out[i][tmp+1]=max_energy
            M_out[i, tmp+2:]=M[i, tmp+1:]
        return img_o, M_out #返回拉长之后的图片(400,600,3)
'''

def main():
    usage = "usage: %prog <input image> <width> <height> <energy type> <output image> \n"
    usage += "for energy type:\n"
    usage += "0=regular energy without entropy term\n"
    usage += "1=regular energy with entropy term\n"
    usage += "2=forward energy\n"
    usage += "666=deep-based energy"
    
    
    if len(sys.argv)!=7 and len(sys.argv)!=6 and len(sys.argv)!=2:
        print ("Incorrect Usage; please see python seam.py --help")
        sys.exit(2)
    
    
    if len(sys.argv)==2 and sys.argv[1]=="--help":
        print(usage)
        sys.exit(2)
    
    else:
        print(sys.argv)
        filename_in=sys.argv[1]
        n_out=int(sys.argv[2])
        m_out=int(sys.argv[3])
        type=int(sys.argv[4])
        if type != 666 and (type>2 or type<0):
            sys.exit(2)
        filename_out=sys.argv[5]
        if len(sys.argv)==7:
            opt_deep_layer=int(sys.argv[6])
            seam_carve=seam(filename_in, filename_out, m_out, n_out, type, opt_deep_layer)
        else:
            seam_carve=seam(filename_in, filename_out, m_out, n_out, type)
        seam_carve.simple_carve()
    


if __name__ == "__main__":
    main()


