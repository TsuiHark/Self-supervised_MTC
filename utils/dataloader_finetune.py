import torch.utils.data as Data
import struct
import numpy as np
import torch
import os
from collections import Counter
import random




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
x_data_train, y_data_train = load_mnist(path=r"D:\Projects\MTMAE-main\datasets\USTC2016-20", kind='train')
# x_data_train, y_data_train = load_mnist(path=r"D:\Projects\MTMAE-main\datasets\CIC-AAGM2017_3", kind='train')
# x_data_train, y_data_train = load_mnist(path=r"D:\Projects\MTMAE-main\datasets\CICMalAnal2017_5", kind='train')
# x_data_train, y_data_train = load_mnist(path=r"D:\Projects\MTMAE-main\datasets\ISCX-12", kind='train')
x_data_train_tensor = torch.Tensor(x_data_train)
y_data_train_tensor = torch.LongTensor(y_data_train)
train_set = Data.TensorDataset(x_data_train_tensor, y_data_train_tensor)


"""测试集"""
# testing set
x_data_test, y_data_test = load_mnist(path=r"D:\Projects\MTMAE-main\datasets\USTC2016-20", kind='t10k')
# x_data_test, y_data_test = load_mnist(path=r"D:\Projects\MTMAE-main\datasets\CIC-AAGM2017_3", kind='test')
# x_data_test, y_data_test = load_mnist(path=r"D:\Projects\MTMAE-main\datasets\CICMalAnal2017_5", kind='test')
# x_data_test, y_data_test = load_mnist(path=r"D:\Projects\MTMAE-main\datasets\ISCX-12", kind='test')
x_data_test_tensor = torch.Tensor(x_data_test)
y_data_test_tensor = torch.LongTensor(y_data_test)
test_set = Data.TensorDataset(x_data_test_tensor, y_data_test_tensor)


def create_random_subdataset(trainset, K, n, random_state):
    # 设置随机数种子以保持可复现性
    torch.manual_seed(random_state)
    random.seed(random_state)

    # 获取原始数据集的长度
    dataset_size = len(trainset)
    print("原始训练集中样本量：", dataset_size)

    # 计算每个类别的样本数量
    class_counts = [0] * K
    for _, label in trainset:
        class_counts[label] += 1

    # 初始化子数据集
    subdataset_data = []
    subdataset_labels = []
    remaining_data = []
    remaining_labels = []

    # 从每个类别中准确随机抽取n比例的样本
    for i in range(dataset_size):
        data, label = trainset[i]
        if class_counts[label] > 0 and random.random() < n:
            subdataset_data.append(data)
            subdataset_labels.append(label)
            class_counts[label] -= 1
        else:
            remaining_data.append(data)
            remaining_labels.append(label)

    # 创建子数据集的 TensorDataset
    subdataset = Data.TensorDataset(torch.stack(subdataset_data), torch.LongTensor(subdataset_labels))
    remaining_dataset = Data.TensorDataset(torch.stack(remaining_data), torch.LongTensor(remaining_labels))
    dataset_size = len(subdataset)
    dataset_size_ = len(remaining_dataset)
    print("抽样训练集中样本量：", dataset_size)
    print("剩余训练集中样本量：", dataset_size_)

    return subdataset, remaining_dataset


train_subset, remaining_subset = create_random_subdataset(train_set, 20, 0.01, random_state=2023)


if __name__ == '__main__':
    print(x_data_train.shape)
    print(x_data_test.shape)

    element_counts = Counter(y_data_train)
    for element, count in element_counts.items():
        print(f"元素 {element} 出现 {count} 次")
    #
    # element_counts = Counter(y_data_test)
    # for element, count in element_counts.items():
    #     print(f"元素 {element} 出现 {count} 次")





