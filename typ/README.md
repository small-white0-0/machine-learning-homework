# 基本使用

## 依赖安装

依赖写在了requirements.txt, 可以使用python工具安装

## 提供的类

主要有：

DecisionTree: 实现决策树的类
photos_processing: 对图片进行处理的类
classify_picture: 基于上面的类，进行图片分类的类

方法中没有默认值的参数是必选参数， 有默认值的参数是可选参数

## 快速开始

创建一个 classify_picture 的对象。传递给对象的 simple_start 方法 1 个字符串，目录字符串——该目录下仅仅能够有airplane,forest两个文件夹，
该方法会自动读取目录下的图片，并自动均分样本分别用来进行训练和测试，并输出结果。

simple_start 可选参数：

1. display, 默认为False, 是否显示预测结果。
2. only_rate,默认False, 是否仅仅显示预测正确率; 仅当display为 True有效

例子：

```python
T = classify_picture()
T.simple_start('/home/tt/machine_test')
# 如果在Windows下,注意用反斜杠
T.simple_start('D:/machine_test')
```

## 正常使用

创建一个 `classify_picture` 的对象。 先用 `photos_processing` 的 `get_photos_str` 方法获取图片地址数组和图片类型数组， 所给参数要求和 `simple_start` 的 `dir` 参数一致 。 获取到数组后可以用`classify_picture` 的对象的 `train` 方法进行训练， 再用 `predict` 方法进行预测。 predict返回一个 行向量，其中数字代表了图片类型，可以使用 `classify_picture` 的对象的 `get_tag` 获取类型名。 可根据需要使用。

例子：

```python
# 获取图片地址数组
photos, types = photos_processing.get_photos_str('/home/tt/machine_test')

# 将数组对半分
# 也可以将图片放于不同文件夹，使用两次 photos_processing.get_photos_str
photos_train_str, photos_test_str = photos_processing.list_split(
    photos)
types_train_str, types_test_str = photos_processing.list_split(types)


A = classify_picture()
# 进行训练
A.train(photos_train_str, types_train_str, only_color=False)

print("测试样本的：")
A.predict(photos_test_str, types_test_str, display=True, only_rate=False)
# 进行测试
print("训练样本的：")
A.predict(photos_train_str, types_train_str, display=True, only_rate=False)
```

## debug信息展示

在 typ/debug中 有一个全局变量 DEBUG ，如果为True, 则会输出额外的关于时间和树深度的信息。 修改为 False ，则不会显示。