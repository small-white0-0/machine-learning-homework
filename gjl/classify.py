import picture
import SVM2


# 获取图片地址数组
photos, types = picture.photos_processing.get_photos_str('PICTURE')
# 将数组对半分
# 也可以将图片放于不同文件夹，使用两次 photos_processing.get_photos_str
photos_train_str, photos_test_str = picture.photos_processing.list_split(photos)
types_train_str, types_test_str = picture.photos_processing.list_split(types)

a = picture.photos_processing(16, True)
photos_arr, photos_id, tags = a.read_photos(photos_train_str, types_train_str)

x = {}
for i in range(len(photos_id)):
    if photos_id[i] == 0:
        x[i] = [photos_arr[i], -1]
    else:
        x[i] = [photos_arr[i], 1]

linearsvm = SVM2.LinearSVM(len(x), len(x[0][0]), 5)
linearsvm.train(x)

b = picture.photos_processing(16, True)
photos_arr_b, photos_id_b, tags_b = b.read_photos(photos_test_str, types_test_str)

y = {}
for i in range(len(photos_id_b)):
    if photos_id[i] == 0:
        y[i] = [photos_arr_b[i], -1]
    else:
        y[i] = [photos_arr_b[i], 1]

for k in range(len(photos_id_b)):
    y_predict = linearsvm.predict(y[k][0])
    if y_predict == -1:
        print('飞机')
    elif y_predict == 1:
        print('森林')
