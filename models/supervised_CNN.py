import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from torch.utils.data import DataLoader
import os
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F

from utils.dataloader_USTC import train_subset, test_set
# from utils.dataloader_CIC import train_subset, test_set
# from utils.dataloader_ISCX import train_subset, test_set


# 设置随机种子以确保结果可复现
torch.manual_seed(0)

batch_size = 64  # 你可以根据需要设置批次大小
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


# 定义CNN模型
class CNNModel(nn.Module):
    def __init__(self, num_class=20):
        super(CNNModel, self).__init__()
        # 第一个卷积层：32个5x5的卷积核，ReLU激活函数
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding="same")
        # 第一个最大池化层：2x2池化窗口
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # 第二个卷积层：64个5x5的卷积核，ReLU激活函数
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding="same")
        # 第二个最大池化层：2x2池化窗口
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # 全连接层1：1024个神经元，ReLU激活函数
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        # Dropout 层：用于减轻过拟合
        self.dropout = nn.Dropout(0.5)
        # 输出层：10个神经元，使用 softmax 激活函数输出类别概率
        self.fc2 = nn.Linear(1024, num_class)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # 将多维张量展平为一维
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 初始化模型、损失函数和优化器
num_classes = 20
warmup_epochs = 20

model = CNNModel(num_class=num_classes).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.05)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=warmup_epochs, T_mult=1, eta_min=0)

# 训练模型
num_epochs = 50
accuracy_max = 0
save_dir_model = "../weights/classifier_"


def train():
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # 测试模型
        model.eval()
        predicted_labels = []
        true_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

            accuracy = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels, average='macro')
            recall = recall_score(true_labels, predicted_labels, average='macro')
            f1 = f1_score(true_labels, predicted_labels, average='macro')
            print(
                f"Epoch {epoch + 1}/{num_epochs}:Accuracy: {accuracy:.4f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

            if (epoch + 1) == 100:
                torch.save(model.state_dict(), os.path.join(save_dir_model, f"model_CNN_ISCX_{epoch + 1}.pth"))

            if accuracy >= accuracy_max:
                torch.save(model.state_dict(), os.path.join(save_dir_model, "Amodel_CNN_ISCX_max.pth"))
                print(f"Epoch: {epoch + 1}")


"""Test"""


def test():
    print("Testing")
    predicted_labels = []
    true_labels = []
    model_test = CNNModel(num_class=num_classes).cuda()
    model_test.load_state_dict(torch.load(r"D:\Projects\MTMAE-main\weights\classifier_\Amodel_CNN_USTC_max.pth"))

    from thop import profile
    dummy_input = torch.randn(1, 1, 28, 28).cuda()
    flops, params = profile(model_test, (dummy_input,))
    print('flops: ', flops, 'params: ', params)

    total_params = sum(p.numel() for p in model_test.parameters())
    print(total_params)
    exit()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model_test(images)
            _, predicted = torch.max(outputs, 1)
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    # 评估模型
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('Confusion Matrix:')
    print(np.array(conf_matrix))


if __name__ == '__main__':
    # train()
    test()
