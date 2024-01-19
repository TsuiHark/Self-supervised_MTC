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


# """训练集"""
# # training set
# x1, y1 = load_mnist(path=r"D:\Projects\MTMAE-main\datasets\ISCX-12", kind='train')                                    # 读取训练数据
# X_train, _, y_train, _ = train_test_split(x1, y1, train_size=35100, random_state=1412)                                # 取35,100个样本组成训练集
# X_subtrain, _, y_subtrain, _ = train_test_split(X_train, y_train, train_size=0.1, random_state=1412)                  # 1%, 2%, 5%, 10%   1412
#
# X_subtrain = torch.Tensor(X_subtrain)
# y_subtrain = torch.LongTensor(y_subtrain)
# train_subset = Data.TensorDataset(X_subtrain, y_subtrain)
#
# """测试集"""
# # testing set
# x2, y2 = load_mnist(path=r"D:\Projects\MTMAE-main\datasets\ISCX-12", kind='test')                                       # 读取测试数据
# X_test, _, y_test, _ = train_test_split(x2, y2, train_size=3900, random_state=1412)                                     # 取3,900个样本组成测试集
# X_test = torch.Tensor(X_test)
# y_test = torch.LongTensor(y_test)
# test_set = Data.TensorDataset(X_test, y_test)


"""训练集"""
# training set
x1, y1 = load_mnist(path=r"D:\Projects\MTMAE-main\datasets\ISCX-6NonVPN", kind='train')                                      # 读取训练数据
X_train, _, y_train, _ = train_test_split(x1, y1, train_size=24000, random_state=1412)                                  # 取35,100个样本组成训练集
X_subtrain, _, y_subtrain, _ = train_test_split(X_train, y_train, train_size=23999, random_state=1412)                     # 1%, 2%, 5%, 10%   1412
# from collections import Counter
# # 使用 Counter 统计每个标签出现的次数
# label_counts = Counter(y_subtrain)
#
# # 打印每个标签和对应的出现次数
# for label, count in label_counts.items():
#     print(f"标签 {label}: 出现次数 {count}")

X_subtrain = torch.Tensor(X_subtrain)
y_subtrain = torch.LongTensor(y_subtrain)
train_subset = Data.TensorDataset(X_subtrain, y_subtrain)

"""测试集"""
# testing set
x2, y2 = load_mnist(path=r"D:\Projects\MTMAE-main\datasets\ISCX-6NonVPN", kind='t10k')                                       # 读取测试数据
X_test, _, y_test, _ = train_test_split(x2, y2, train_size=2690, random_state=1412)                                     # 取3,900个样本组成测试集
X_test = torch.Tensor(X_test)
y_test = torch.LongTensor(y_test)
test_set = Data.TensorDataset(X_test, y_test)



# """训练集 11000"""
# # training set
# x1, y1 = load_mnist(path=r"D:\Projects\MTMAE-main\datasets\ISCX-6VPN", kind='train')                                    # 读取训练数据
# X_train, _, y_train, _ = train_test_split(x1, y1, train_size=11000, random_state=1412)                                  # 取35,100个样本组成训练集
# X_subtrain, _, y_subtrain, _ = train_test_split(X_train, y_train, train_size=0.5, random_state=1412)                    # 1%, 2%, 5%, 10%   1412
# # from collections import Counter
# # # 使用 Counter 统计每个标签出现的次数
# # label_counts = Counter(y_subtrain)
# #
# # # 打印每个标签和对应的出现次数
# # for label, count in label_counts.items():
# #     print(f"标签 {label}: 出现次数 {count}")
#
# X_subtrain = torch.Tensor(X_subtrain)
# y_subtrain = torch.LongTensor(y_subtrain)
# train_subset = Data.TensorDataset(X_subtrain, y_subtrain)
#
# """测试集 1253"""
# # testing set
# x2, y2 = load_mnist(path=r"D:\Projects\MTMAE-main\datasets\ISCX-6VPN", kind='t10k')                                       # 读取测试数据
# X_test, _, y_test, _ = train_test_split(x2, y2, train_size=1250, random_state=1412)                                     # 取3,900个样本组成测试集
#
# X_test = torch.Tensor(X_test)
# y_test = torch.LongTensor(y_test)
# test_set = Data.TensorDataset(X_test, y_test)






