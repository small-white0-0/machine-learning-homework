import os
import cv2
import numpy as np
from sklearn.decomposition import TruncatedSVD
from skimage.feature import local_binary_pattern


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
            for photo in os.listdir(dir + '/' + subdir):
                X.append(dir + '/' + subdir + '/' + photo)
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
            try:
                image = cv2.resize(image, (size, size),
                               interpolation=cv2.INTER_CUBIC)
            except:
                continue
            # 计算颜色直方图
            hist_size = int(size)
            hist0 = cv2.calcHist([image], [0], None, [hist_size], [0, 255])
            hist1 = cv2.calcHist([image], [1], None, [hist_size], [0, 255])
            hist2 = cv2.calcHist([image], [2], None, [hist_size], [0, 255])
            # 颜色直方图的几种整合方式
            hist = hist0 + hist1 + hist2  # 简单加和

            if not only_color:
                # 提取纹理特征
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 获取灰度图
                lbp = local_binary_pattern(
                    gray_image, 3, 32)  # 用skimage库 提取纹理特征
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


