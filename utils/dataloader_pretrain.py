import torch.utils.data as Data
import struct
import numpy as np
import torch
import os


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
x_data_train, y_data_train = load_mnist(path=r"D:\Projects\MTMAE-main\datasets\BigDataset", kind='train')
x_data_train_tensor = torch.Tensor(x_data_train)
y_data_train_tensor = torch.LongTensor(y_data_train)
train_dataset = Data.TensorDataset(x_data_train_tensor, y_data_train_tensor)

"""测试集"""
# testing set
x_data_test, y_data_test = load_mnist(path=r"D:\Projects\MTMAE-main\datasets\BigDataset", kind='t10k')
x_data_test = torch.Tensor(x_data_test)
y_data_test = torch.LongTensor(y_data_test)
test_dataset = Data.TensorDataset(x_data_test, y_data_test)


if __name__ == '__main__':
    # 训练集
    print("训练集")
    print(x_data_train_tensor.shape)
    print(y_data_train_tensor.shape)
    # print(y_data_train_tensor[:100])
    # 测试集
    print("测试集")
    print(x_data_test.shape)
    print(x_data_test.shape)
