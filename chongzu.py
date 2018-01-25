import os
import sys
import cv2
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt 
from functools import reduce
dirfile = '/home/xm/桌面/img/data/wn' # 挑选出来的图片所在的文件夹，这个文件夹不用动，应该这里面的数据是挑出来的
second_folder = '/home/xm/桌面/img/data_1/wn' #第二个切割出来的数据集所在的文件夹的位置
info_path = second_folder + '/info.txt'
with open(info_path,'rb') as f:
    info = pickle.load(f)
info = info['info_arr']
for i in range(info.shape[0]):
    file_name_1 = info[:,1][i]   #已经排好序的图片
    for file_name_2 in os.listdir(dirfile): #file_name_2 表示的是未排序好的图像
        if file_name_1 == file_name_2:
            file_path = dirfile + '/' + file_name_2
            image_copy = Image.open(file_path)
##            image_copy_array = np.array(image_copy)
            print('begin to write ........')
            save_file = second_folder + '//' + 'img{}'.format(i+1) + '.bmp'
            image_copy.save(save_file)
            print('finish writing ........')
