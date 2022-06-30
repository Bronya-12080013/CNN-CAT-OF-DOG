import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np


def load_train(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []

    print('Going to read training images')  # 去读训练用文件
    for fields in classes:
        index = classes.index(fields)  # 取下标
        print('Now going to read {} files (Index: {} )'.format(fields, index))  # 注意是{}  #填数输出
        path = os.path.join(train_path, fields, '*g')  # 字符串的下标式拼接  把所有参数用'/'连接  #'*g'应该是以g结尾的文件吧 用于所有的jpg文件
        # print(path)
        files = glob.glob(path)  # 获取路径的所有文件
        for fl in files:
            image = cv2.imread(fl)
            # cv2.imshow('img',image)
            # cv2.waitKey(1)

            # http://t.csdn.cn/YGbiV
            image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)  # 改图片大小
            image = image.astype(np.float32)  # 转数据格式
            image = np.multiply(image, 1.0 / 255.0)  # 归一化
            images.append(image)  # 把这张图片放到上面定义的images里面
            label = np.zeros(len(classes))  # 1*2矩阵
            label[index] = 1.0  # 辨别是猫是狗
            labels.append(label)  # 放进去
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)  # 放路径名 判断是猫是狗
    # 转数据类型
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls

class DataSet(object):
    # 注意这里是 __init__      #我之前搞错成 __int__  导致下面调用时没用到初始化方法，报错：DataSet() takes no arguments
    def __init__(self, images, labels, img_names, cls):
        self._num_examples = images.shape[0]

        self._images = images
        self._labels = labels
        self._img_names = img_names
        self._cls = cls
        self._epochs_done = 0  # 已经做了几个了?
        self._index_in_epoch = 0

    # @property装饰器来创建只读属性，@property装饰器会将方法转换为相同名称的只读属性
    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def img_name(self):
        return self._img_names

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done


    def next_batch(self, batch_size):
            """Return  the next 'batch_size' example from this data set."""
            start = self._index_in_epoch
            self._index_in_epoch += batch_size

            if self._index_in_epoch > self._num_examples:
                self._index_in_epoch += 1
                start = 0
                self._index_in_epoch = batch_size
                assert batch_size <= self._num_examples
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[
                                                                                                 start:end]




def read_train_sets(train_path, image_size, classes, validation_size):
    class DataSets(object):  # 区别上面的DataSet()方法
        pass

    # 新建 data_sets
    data_sets = DataSets()

    images, labels, img_names, cls = load_train(train_path, image_size, classes)
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls)  # 洗牌 打乱顺序 所有的一起打乱，对应关系不变

    if isinstance(validation_size, float):  # 如果对象的类型与参数二的类型（classinfo）相同则返回 True，否则返回 False
        validation_size = int(
            validation_size * images.shape[0])  # 根据上述数据 images.shape 为[1000,64,64,3] #images.shape[0]表示有1000张图片 我们取20%
    # 开切
    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_img_names = img_names[:validation_size]
    validation_cls = cls[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_img_names = img_names[validation_size:]
    train_cls = cls[validation_size:]

    # data_sets是上面建的
    data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
    data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

    return data_sets




