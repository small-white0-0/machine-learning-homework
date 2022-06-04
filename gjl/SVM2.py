import numpy as np


class LinearSVM(object):

    def __init__(self, dataset_size, vector_size, c):
        self.a = np.zeros(dataset_size, np.float_)
        self.w = np.zeros(vector_size, np.float_)
        self.b = 0
        self.C = c
        self.IJ = set()

    def train(self, dataset):
        i = 0
        while i != -1:
            j = self.__find_j(dataset, i)
            self.IJ.add(i)
            self.IJ.add(j)
            self.__smo(dataset, i, j)
            self.__update_w(dataset)
            self.__update_b(dataset)
            i = self.__find_i(dataset)

    def __find_i(self, dataset):
        i = -1
        for k in range(len(dataset)):
            K = {k}
            if self.IJ - K == self.IJ:
                xk = dataset[k][0]
                yk = dataset[k][-1]
                if self.a[k] == 0:
                    if yk*(np.dot(self.w, xk) + self.b) < 1:
                        i = k
                        break
                elif self.a[k] == self.C:
                    if yk*(np.dot(self.w, xk) + self.b) > 1:
                        i = k
                        break
                elif 0 < self.a[k] < self.C:
                    if yk * (np.dot(self.w, xk) + self.b) != 1:
                        i = k
                        break
        return i

    def __find_j(self, dataset, i):
        yi = dataset[i][-1]
        xi = dataset[i][0]
        Ei = np.dot(self.w, xi) + self.b - yi
        max_Ei_Ej = 0
        for k in range(len(dataset)):
            K = {k}
            if self.IJ - K == self.IJ:
                yk = dataset[k][-1]
                xk = dataset[k][0]
                Ek = np.dot(self.w, xk) + self.b - yk
                if abs(Ei - Ek) >= max_Ei_Ej:
                    j = k
        return j

    def __smo(self, dataset, i, j):
        yi = dataset[i][-1]
        xi = dataset[i][0]
        yj = dataset[j][-1]
        xj = dataset[j][0]

        Ei = np.dot(self.w, xi) + self.b - yi
        Ej = np.dot(self.w, xj) + self.b - yj
        eta = np.dot(xi - xj, xi - xj)
        a_new_unclipped = self.a[j] + yj * (Ei - Ej) / eta
        # 截断记录i的`拉格朗⽇乘⼦`并计算记录j的`拉格朗⽇乘⼦`
        ksai = -self.__calculate_ksai(dataset, i, j)
        a_new = self.__quadratic_programming(a_new_unclipped, yi, yj, ksai)
        if a_new >= 0:
            self.a[j] = a_new
            self.a[i] = (ksai - a_new * yj) * yi

    def __update_b(self, dataset):
        sum_b = 0
        count = 0
        for k in range(self.a.__len__()):
            if self.a[k] != 0:
                label = dataset[k][-1]
                vector = np.array(dataset[k][0])
                sum_b += label - np.dot(self.w, vector)
                count += 1
        if count == 0:
            self.b = 0
        else:
            self.b = sum_b / count

    def __update_w(self, dataset):
        w = np.zeros(len(dataset[0][0]))
        for k in range(len(dataset)):
            y = dataset[k][-1]
            x = dataset[k][0]
            w += self.a[k] * y * x
        self.w = w

    def __calculate_ksai(self, dataset, i, j):
        yi = dataset[i][-1]
        yj = dataset[j][-1]
        #通过置零来遍历所有数据，除去指定的i和j
        dataset[i][-1] = 0
        dataset[j][-1] = 0
        sum_ksai = 0
        for k in range(len(dataset)):
            y = dataset[k][-1]
            sum_ksai += self.a[k] * y
        dataset[i][-1] = yi
        dataset[j][-1] = yj
        return sum_ksai

    def __quadratic_programming(self, a_new_unclipped, yi, yj, ksai):
        if yi * yj == 1:
            L = max(0, ksai - self.C)
            H = min(self.C, ksai)
        else:
            L = max(0, -ksai)
            H = min(self.C, self.C - ksai)
        if a_new_unclipped > H:
            a_new = H
        elif a_new_unclipped < L:
            a_new = L
        else:
            a_new = a_new_unclipped
        return a_new

    def predict(self, vector):
        result = np.dot(self.w, vector) + self.b
        if result >= 0:
            return 1
        else:
            return -1
