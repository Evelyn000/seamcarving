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
#import cv2

seam('coast.jpg', 'coast_modified.jpg', 600, 399, "without_le")


class seam:
    def __init__(self, filename_in, filename_out, m_out, n_out, type):
        self.filename_in=filename_in
        self.filename_out=filename_out
        self.m_out=m_out
        self.n_out=n_out
        self.img_in=mpimg.imread(filename).astype(np.float64)
        self.img_out=img_in
        self.energy_solver=ENERGY(type)

    def simple_carve(self):
        m, n, c=self.img_out.shape
        if n>self.n_out:
            collapse(n_out)
        elif m>self.m_out:
            img_out=np.transpose(img_out, (1, 0, 2))
            collapse(m_out)
            img_out=np.transpose(img_out, (1, 0, 2))
        elif n<self.n_out:
            enlarge(n_out)
        
    # 先默认竖向seam
    def Find_seam(self, M): # M is a matrix of energy
        m, n = M.shape
        seam_point_list = np.zeros((m, 1), dtype=np.int16)
        seam_point_list[-1] = np.argmin(M[-1])
        for i in range(m-2, -1, 1):
            if seam_point_list[i+1]==0:
                seam_point_list[i]=np.argmin(M[i, :2])
            elif seam_point_list[i+1]==n-1:
                seam_point_list[i]=np.argmin(M[i, n-2:])+n-2
            else:
                seam_point_list[i]=np.argmin(M[i, seam_point_list[i+1]-1 :2])+seam_point_list[i+1]-1
        return seam_point_list #seam_point_list 用来储存seam路线上的所有点的坐标

    ##compute_energy(img)
    def enlarge(self, n_o):
        img_mask=M


    def collapse(self, n_o):
        m, n, c =self.img_out.shape
        while n>self.n_o:
            M=energy_solver.compute_energy(self.img_out)
            seam_point_list=Find_seam(M)
            img_out=Remove_seam(seam_point_list, self.img_out)
            m, n, c=img_out.shape
            print(seam_point_list, '\n')

    def Remove_seam(self, seam_point_list, img_i): # X is the orginal matrix, M is X's energy matrix
        m, n, c = img_i.shape
        img_o = np.zeros((m, n-1, 3), dtype=np.float32)
        for i in range(m):
            img_o[i, :, :]=np.delete(img_i[i], [seam_point_list[i]], axis=0)
        return img_o #返回裁剪之后的图片(400,600,3)
     
    def Duplicate_seam(self, seam_point_list, img_i):  # X is the orginal matrix, M is X's energy matrix
        m, n, c = img_i.shape
        img_o = np.zeros((m, n+1, 3), dtype=np.float32)
        for i in range(m):
            img_o[i, :seam_point_list[i]+1, :]=img_i[i, :seam_point_list[i]+1, :]
            img_o[i, seam_point_list[i]+1, :]=img_i[i, seam_point_list[i]+1, :]
            img_o[i, seam_point_list[i]+2:, :]=img_i[i, seam_point_list[i]+2:, :]
        return img_o #返回拉长之后的图片(400,600,3)

def main():
    from optparse import OptionParser
    import os 
    usage = "usage: %prog -i [input image] -r [width] [height] -o [output name] \n" 
    usage+= "where [width] and [height] are the resolution of the new image"
    parser = OptionParser(usage=usage)
    (options, args) = parser.parse_args()
    parser.add_option("-i", "--image", dest="input_image", help="Input Image File")
    parser.add_option("-r", "--resolution", dest="resolution", help="Output Image size [width], [height]", nargs=2)
    parser.add_option("-o", "--output", dest="output", help="Output Image File Name")
    parser.add_option("-v", "--verbose", dest="verbose", help="Trigger Verbose Printing", action="store_true")
    parser.add_option("-m", "--mark", dest="mark", help="Mark Seams Targeted. Only works for deleting", action="store_true")
    # discuss options here
    if not options.input_image or not options.resolution:
        print ("Incorrect Usage; please see python seam.py --help")
        sys.exit(2)
    if options.verbose:
        global verbose
        verbose = True
    if not options.output:
        output_image = os.path.splitext(options.input_image)[0] + ".coast.jpg"
    else:
        output_image = options.output
	
    try: 
        input_image = options.input_image
        resolution = ( int(options.resolution[0]), int(options.resolution[1]) )
    except:
        print ("Incorrect Usage; please see python CAIS.py --help")
        sys.exit(2)
        

#if __name__ == "__main__":
#	main()

