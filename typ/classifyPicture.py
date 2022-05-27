
import os
import time
import cv2
import numpy as np
from sklearn.decomposition import TruncatedSVD
from skimage.feature import local_binary_pattern

from .Tree import DecisionTree
from  . import debug

class classifyPicture:
    def __init__(self) -> None:
        self.svd = TruncatedSVD(n_components=1)
        self.tree = None

    def get_photos_str(self, dir):
        """
        从文件夹中读取图片, 文件夹中 只能有子文件夹，子文件夹中只能有图片
        子文件夹的名字，将作为图片类型记录
        返回第一个为 图片路径字符串数组
        第二个为图片分类字符串数组
        """
        X = []
        Y = []
        for subdir in os.listdir(dir):
            for photo in os.listdir(dir+'/'+subdir):
                X.append(dir+'/'+subdir+'/'+photo)
                Y.append(subdir)
        # X = np.array(X)
        # Y = np.array(Y).reshape(-1,1)
        return X,Y

    def __read_photos(self, photos, types, size=50):
        """
        :phots ,一个字符串数组， 每个字符串是图片地址\n
        :types, 一个字符串数组，每个字符串是图片类型
        :size , 默认100, 对于图片大小进行缩放
        """
        photos_arr = []
        photos_id = []
        tags = []

        for (file, tag) in zip(photos, types):
            if not tag in tags:
                tags.append(tag)
            # 读取图片，并调整大小
            image = cv2.imread(file)
            image = cv2.resize(image, (size,size),interpolation=cv2.INTER_CUBIC)

            # 计算颜色直方图
            hist_size = int(size)
            hist0 = cv2.calcHist([image], [0], None, [hist_size], [0,255])
            hist1 = cv2.calcHist([image], [1], None, [hist_size], [0,255])
            hist2 = cv2.calcHist([image], [2], None, [hist_size], [0,255])
            # 颜色直方图的几种整合方式
            hist = hist0 + hist1 + hist2    # 简单加和

            if not self.only_color:
            # 提取纹理特征
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    #获取灰度图
                lbp = local_binary_pattern(gray_image, 3,32)            # 用skimage库 提取纹理特征
                lbp = self.svd.fit_transform(lbp)                       # 由 size x size 降维为 size x 1

            if self.only_color:
                photo_arr = hist.ravel()
            else:
                photo_arr = np.append(hist.ravel(),lbp.ravel())

            photos_arr.append(photo_arr)
            photos_id.append(tags.index(tag))
            
        photos_arr = np.array(photos_arr)
        photos_id = np.array(photos_id).reshape(-1,1)
        return photos_arr,photos_id,tags

    def list_split(self, lst, n=2):
        """
        :lst ,一个 字符串 数组
        :n ,整数， 默认为2 表示大约 1/n 分割到第一个返回值， 剩余分割到第二个返回值
        """
        flag = 0
        train = []
        test = []
        n =4
        for i in lst:
            if flag%n != 0:
                train.append(i)
                flag += 1
            else:
                test.append(i)
                flag += 1

        return train,test


    def train(self, photos_str, types_str, only_color = True):
        """
        训练决策树，先从参数中读取图片，再进行决策树训练

        :photos_str, 图片地址字符串数组
        :types_str , 图片类型字符串数组
        :only_color, 默认True, 仅仅使用图片的颜色特征，为False则还会使用图片的纹理特征
        """
        self.only_color = only_color

        debug.debug_time_start()
        photos,types,self.tags = self.__read_photos(photos_str, types_str)
        debug.debug_print_time("read time")

        train_set = np.append(photos, types, axis = 1)
        self.tree = DecisionTree()
        debug.debug_time_start()
        self.tree.train(train_set)
        debug.debug_print_time("二叉树训练用时")
        


    def predict(self, photos_str,types_str, display = False, only_rate=False):
        """
        :photos_str, 图片地址字符串数组
        :types_str , 图片类型字符串数组
        :display, 默认为False, 是否显示预测结果。
        :only_rate,默认False, 是否仅仅显示预测正确率；仅当display为 True有效
        """
        if self.tree == None:
            print("没有进行训练")
            return None
        read_time = time.perf_counter()
        photos, types, _ = self.__read_photos(photos_str, types_str)
        print(f'read time:{time.perf_counter() - read_time:.8f}s')
        test_set  = np.append(photos, types, axis=1)
        re = self.tree.predict(test_set)

        if display:
            correct = 0
            for i in range(len(re)):
                if not only_rate:
                    print("%s -> %s %s" % (photos_str[i], self.tags[re[i].astype(int)], re[i] == types[i]))
                if re[i] == types[i]:
                    correct += 1
            print('正确率：', correct/len(photos))

        return re


    def simple_start(self, dir, display=True, only_rate=True):
        """
        简单测试用的函数， 接受目录，读取目录中的图片文件，并用 50%样本进行训练， 用50% 样本进行测试

        :dir 是一个目录字符串， 仅仅有airplane和forest两个文件夹
        :display, 默认为False, 是否显示预测结果。
        :only_rate,默认False, 是否仅仅显示预测正确率; 仅当display为 True有效
        """
        photos, types = self.get_photos_str(dir)
        photos_train_str, photos_test_str = self.list_split(photos)
        types_train_str, types_test_str = self.list_split(types)

        self.train(photos_train_str, types_train_str, only_color=False )
        print("测试样本的：")
        self.predict(photos_test_str, types_test_str, display, only_rate)
        print("训练样本的：")
        self.predict(photos_train_str,types_train_str, display, only_rate)






T = classifyPicture()
T.simple_start('/home/tt/machine_test')