# 基本使用
## 快速开始
创建一个 classifyPicture 的对象。传递给给对象的 simple_start 方法 一个目录字符串——该目录下仅仅能够有airplane,forest两个文件夹，
该方法会自动读取目录下的图片，并自动均分样本分别用来进行训练和测试，并输出结果。

simple_start 可选参数：

1. display, 默认为False, 是否显示预测结果。
2. only_rate,默认False, 是否仅仅显示预测正确率; 仅当display为 True有效

例子：

```python
T = classifyPicture()
T.simple_start('/home/tt/machine_test')

```
## 正常使用

创建一个 classifyPicture 的对象。 先用 get_photos_str 方法获取图片地址数组， 然后用 list_split 对获取到的数组分裂， 一部分用于训练，一部分用于测试。用该对象的 train 方法进行训练，然后用 predict 方法进行测试。

例子：

```python
# 获取图片地址数组
photos, types = self.get_photos_str(dir)
# 将图片对半分
photos_train_str, photos_test_str = self.list_split(photos)
types_train_str, types_test_str = self.list_split(types)

# 进行训练
self.train(photos_train_str, types_train_str, only_color=False )
# 进行测试
print("测试样本的：")
self.predict(photos_test_str, types_test_str, display, only_rate)
print("训练样本的：")
self.predict(photos_train_str,types_train_str, display, only_rate)
```
