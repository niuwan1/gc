#首先将info的信息读取出来
import pandas as pd
import numpy as np
import pickle
path = '/home/xm/桌面/img/data/wn/info.txt'
info = pd.read_csv(path,header = None,delim_whitespace = True)
info_arr = np.array(info)
index = []
for i in range(info_arr.shape[0]):
    if info_arr[i][0] not in index:
        index.append(info_arr[i][0])
index_1 = [np.where(info_arr[:,0] == index[i]) for i in range(len(index))]
#index_1中存取的是29种图片的位置信息，同一种图片已经按照顺序排好了，一种图片是一个arr
#index_1 得到的是29中图片的名字，得到的数据的格式是这样的[(arr(),)],list里面有tuple，tuple里面有arr
print(index)
for i in range(len(index_1)):
    if i == 0:
        index_arr = index_1[i][0]
    else:
        index_arr = np.concatenate([index_arr,index_1[i][0]],axis = 0)

info_arr = info_arr[index_arr]
print('index_arr:\n',index_arr)
print(info_arr)

