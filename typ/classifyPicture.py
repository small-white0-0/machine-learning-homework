
import os
import time
import cv2
import numpy as np
from sklearn.decomposition import TruncatedSVD
from skimage.feature import local_binary_pattern

from .Tree import DecisionTree
from . import debug


class classify_picture:
    def __init__(self) -> None:
        pass

    def get_tag(self, index):
        """
        :index, 图片类型的序号，整数

        返回一个字符串
        """
        return self.tags[index]

    def train(self, photos_str, types_str, size=50, only_color=True):
        """
        训练决策树，先从参数中读取图片，再进行决策树训练

        :photos_str, 图片地址字符串数组
        :types_str , 图片类型字符串数组
        :size , 默认50, 对于图片大小进行缩放
        :only_color, 默认True, 仅仅使用图片的颜色特征，为False则还会使用图片的纹理特征

        无返回值
        """
        self.photos_processing = photos_processing(
            size=size, only_color=only_color)

        debug.debug_time_start()
        photos, types, self.tags = self.photos_processing.read_photos(
            photos_str, types_str)
        debug.debug_print_time("read time")

        train_set = np.append(photos, types, axis=1)
        self.tree = DecisionTree()
        debug.debug_time_start()
        self.tree.train(train_set)
        debug.debug_print_time("二叉树训练用时")

    def predict(self, photos_str, types_str, display=False, only_rate=False):
        """
        :photos_str, 图片地址字符串数组
        :types_str , 图片类型字符串数组
        :display, 默认为False, 是否显示预测结果。
        :only_rate,默认False, 是否仅仅显示预测正确率；仅当display为 True有效

        返回一个 行向量， 里面的数字代表 图片类型
        """
        if self.tree == None:
            print("没有进行训练!!!")
            return None

        debug.debug_time_start()
        photos, types, _ = self.photos_processing.read_photos(
            photos_str, types_str)
        debug.debug_print_time("图片读取和处理时间")

        test_set = np.append(photos, types, axis=1)
        re = self.tree.predict(test_set)

        if display:
            correct = 0
            for i in range(len(re)):
                if not only_rate:
                    print(
                        "%s -> %s %s" % (photos_str[i], self.get_tag(re[i].astype(int)), re[i] == types[i]))
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
        photos, types = photos_processing.get_photos_str(dir)
        photos_train_str, photos_test_str = photos_processing.list_split(
            photos)
        types_train_str, types_test_str = photos_processing.list_split(types)

        self.train(photos_train_str, types_train_str, only_color=False)

        print("测试样本的：")
        self.predict(photos_test_str, types_test_str,
                     display=display, only_rate=only_rate)

        print("训练样本的：")
        self.predict(photos_train_str, types_train_str,
                     display=display, only_rate=only_rate)


class photos_processing:
    def __init__(self, size=50, only_color=True) -> None:
        """
        :size , 默认50, 对于图片大小进行缩放
        :only_color, 默认True, 仅仅使用图片的颜色特征，为False则还会使用图片的纹理特征
        """

        self.svd = TruncatedSVD(n_components=1)
        self.size = size
        self.only_color = only_color

    @staticmethod
    def list_split(lst, n=4):
        """
        :lst ,一个 字符串 数组
        :n ,整数， 默认为4 表示大约 (n-1)/n 的 lst 分割到第一个返回值， 剩余分割到第二个返回值

        返回两个数组
        """
        flag = 0
        train = []
        test = []
        for i in lst:
            if flag % n != 0:
                train.append(i)
                flag += 1
            else:
                test.append(i)
                flag += 1

        return train, test

    @staticmethod
    def get_photos_str(dir):
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
        return X, Y

    def read_photos(self, photos, types):
        """
        :phots ,一个字符串数组， 每个字符串是图片地址
        :types, 一个字符串数组，每个字符串是图片类型

        参数可以通过 get_photos_str 获取

        返回两个矩阵和一个数组。
        第一个矩阵 图片特征矩阵
        第二个矩阵 图片类型序号矩阵
        第三个数组 图片类型序号和图片类型名（字符串）对应关系的数组
        """

        photos_arr = []
        photos_id = []
        tags = []

        size = self.size
        only_color = self.only_color

        for (file, tag) in zip(photos, types):
            if not tag in tags:
                tags.append(tag)
            # 读取图片，并调整大小
            image = cv2.imread(file)
            image = cv2.resize(image, (size, size),
                               interpolation=cv2.INTER_CUBIC)

            # 计算颜色直方图
            hist_size = int(size)
            hist0 = cv2.calcHist([image], [0], None, [hist_size], [0, 255])
            hist1 = cv2.calcHist([image], [1], None, [hist_size], [0, 255])
            hist2 = cv2.calcHist([image], [2], None, [hist_size], [0, 255])
            # 颜色直方图的几种整合方式
            hist = hist0 + hist1 + hist2    # 简单加和

            if not only_color:
                # 提取纹理特征
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 获取灰度图
                lbp = local_binary_pattern(
                    gray_image, 3, 32)            # 用skimage库 提取纹理特征
                # 由 size x size 降维为 size x 1
                lbp = self.svd.fit_transform(lbp)

            if only_color:
                photo_arr = hist.ravel()
            else:
                photo_arr = np.append(hist.ravel(), lbp.ravel())

            photos_arr.append(photo_arr)
            photos_id.append(tags.index(tag))

        photos_arr = np.array(photos_arr)
        photos_id = np.array(photos_id).reshape(-1, 1)
        return photos_arr, photos_id, tags

    def read_photos_by_dir(self, dir):
        """
        就是简单的把get_photos_str 和 read_photos结合了一下，参数要求和作用和它们一样
        PS: 没有list_split 的分割作用

        返回两个矩阵和一个数组。
        第一个矩阵 图片特征矩阵
        第二个矩阵 图片类型序号矩阵
        第三个数组 图片类型序号和图片类型名（字符串）对应关系的数组
        """
        photos, types = self.get_photos_str(dir)
        return self.read_photos(photos, types)
