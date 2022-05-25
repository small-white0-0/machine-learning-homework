
import os
import time
import cv2
import numpy as np
from sklearn.decomposition import PCA
# from sklearn.cross_validation import train_test_split

from Tree import DEBUG, DecisionTree


class classifyPicture:
    def __init__(self) -> None:
        pass

    def read_photos_by_dir(self, dir):
        """
        从文件夹中读取图片
        """
        X = []
        Y = []
        A = [0,0]
        i = 0
        for subdir in os.listdir(dir):
            for photo in os.listdir(dir+'/'+subdir):
                A[i] = subdir
                photo_file = dir+'/'+subdir+'/'+photo
                image = cv2.imread(photo_file)
                hist = cv2.calcHist([image], [0,1], None, [256,256], [0,256,0,256])
                X.append(hist.flatten())
                Y.append(i)
            i += 1
        X = np.array(X)
        Y = np.array(Y).reshape(-1,1)
        return X,Y,A

    def read_photos_by_array(self, photos, types):
        """
        :phots ,一个字符串数组， 每个字符串是图片地址\n
        :types, 一个字符串数组，每个字符串是图片类型
        """
        photos_arr = []
        photos_id = []
        tags = []
        for (file, tag) in zip(photos, types):
            if not tag in tags:
                tags.append(tag)

            image = cv2.imread(file)
            hist = cv2.calcHist([image], [0,1], None, [256,256], [0,256,0,256])
            photos_arr.append(hist.flatten())
            photos_id.append(tags.index(tag))

        photos_arr = np.array(photos_arr)
        photos_id = np.array(photos_id).reshape(-1,1)
        return photos_arr,photos_id,tags

    def array_split(arr):
        flag = False
        train = []
        test = []
        for i in arr:
            if flag:
                train.append(i)
                flag = False
            else:
                test.append(i)
                flag = True

        train = np.array(train)
        test = np.array(test)
        return train,test

    





# 读取图片


# 数组分类


# 选择50%作为训练集
t =time.perf_counter()
X,Y, tags = read_photos("/home/tt/machine_test")
print(len(X))
print(f'imread time:{time.perf_counter() - t:.8f}s')


t =time.perf_counter()
pca = PCA(n_components=0.9,whiten=True)
X = pca.fit_transform(X)
print(f'pca time:{time.perf_counter() - t:.8f}s')


Z = np.append(X, Y, axis=1)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=1)
# X_train,X_test = array_split(X)
# Y_train,Y_test = array_split(Y)
t =time.perf_counter()
Z_train,Z_test = array_split(Z)
# print(len(X), len(X_test), len(Y_train), len(Y_test))


TB = DecisionTree()
t =time.perf_counter()
TB.train(Z_train)
print(f'train time:{time.perf_counter() - t:.8f}s')
print(len(Z_test))

t =time.perf_counter()
re = TB.predict(Z_test)
print(f'predict time:{time.perf_counter() - t:.8f}s')

print("re:",re)
correct = 0
for (i,j) in zip(Z_test[:,-1],re):
    if j == i:
        correct +=1

print("正确率：",correct/len(Z_test))