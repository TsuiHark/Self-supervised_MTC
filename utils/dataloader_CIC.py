import torch.utils.data as Data
import struct
import numpy as np
import torch
import os
from collections import Counter
import random
from sklearn.model_selection import train_test_split


def load_mnist(path="data", kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    with open(labels_path, 'rb') as label_path:
        _, n = struct.unpack('>II', label_path.read(8))
        labels = np.fromfile(label_path, dtype=np.uint8)

    with open(images_path, 'rb') as img_path:
        _, num, rows, cols = struct.unpack('>IIII', img_path.read(16))
        images = np.fromfile(img_path, dtype=np.uint8).reshape(len(labels), 1, 28, 28)

    return images, labels


"""训练集"""
# training set
x1, y1 = load_mnist(path=r"D:\Projects\MTMAE-main\datasets\CICMalAnal2017_5", kind='train')                             # 读取训练数据
X_train, y_train = x1, y1                                                                                               # 取270,000个样本组成训练集
X_subtrain, _, y_subtrain, _ = train_test_split(X_train, y_train, train_size=0.1, random_state=1412)                   # 1%, 2%, 5%, 10%

X_subtrain = torch.Tensor(X_subtrain)
y_subtrain = torch.LongTensor(y_subtrain)
train_subset = Data.TensorDataset(X_subtrain, y_subtrain)


"""测试集"""
# testing set
x2, y2 = load_mnist(path=r"D:\Projects\MTMAE-main\datasets\CICMalAnal2017_5", kind='test')                              # 读取测试数据
X_test, y_test = x2, y2                                                                                                 # 取30,000个样本组成测试集
X_test = torch.Tensor(X_test)
y_test = torch.LongTensor(y_test)
test_set = Data.TensorDataset(X_test, y_test)







