import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from utils.dataloader_pretrain import train_dataset
from torch.utils.data import DataLoader
import os
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
from modules.model_finetune import Classifier, CAE_finetuneModel
from tqdm import tqdm
import sys

# from utils.dataloader_USTC import train_subset, test_set
# from utils.dataloader_CIC import train_subset, test_set
from utils.dataloader_ISCX import train_subset, test_set


# 设置随机种子以确保结果可复现
torch.manual_seed(0)


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, input):
        z = self.encoder(input)
        output = self.decoder(z)
        return output


save_dir = "../weights/encoder"
save_dir_run = "../runs"


def pretrain():
    pretrain_num_epochs = 300
    model = ConvAutoencoder().cuda()
    criterion_pretrain = nn.MSELoss()
    optimizer_pretrain = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.05)
    pretrain_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    for epoch in range(pretrain_num_epochs):
        total_loss = 0
        pretrain_loader = tqdm(pretrain_loader, file=sys.stdout)
        for images, labels in pretrain_loader:
            images, labels = images.cuda(), labels.cuda()
            optimizer_pretrain.zero_grad()
            outputs = model(images)
            loss = criterion_pretrain(outputs, images)
            loss.backward()
            optimizer_pretrain.step()
            total_loss += loss.detach().item()

        print("epoch loss:", total_loss / len(train_dataset) * 128)

        with open(os.path.join(save_dir_run, "loss/pretrain_loss_CAE.txt"), "a") as f:
            f.write(str(total_loss / len(train_dataset) * 128) + "\n")

        # 保存模型
        if (epoch + 1) == 100:
            torch.save(model.encoder, os.path.join(save_dir, f'./CAE_encoder.pth'))


save_dir_model = "../weights/classifier_"
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)


def finetune():
    num_classes = 6
    batch_size = 64  # 你可以根据需要设置批次大小
    finetune_num_epochs = 50  # 100
    warmup_epochs = 20  # 20

    finetune_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    Encoder = torch.load(r'D:\Projects\MTMAE-main\weights\encoder\新建文件夹\CAE_encoder.pth')
    # Encoder = torch.load(r'D:\Projects\MTMAE-main\weights\encoder\CAE_encoder.pth')
    classifierHead = Classifier(latent_dim=128, cls_dim=num_classes)  # 修改cls-dim
    model = CAE_finetuneModel(Encoder, classifierHead).cuda()

    criterion_finetune = nn.CrossEntropyLoss()
    optimizer_finetune = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.05)
    scheduler = CosineAnnealingWarmRestarts(optimizer_finetune, T_0=warmup_epochs, T_mult=1, eta_min=0)
    accuracy_max = 0
    # finetune
    for epoch in range(finetune_num_epochs):
        model.train()

        for images, labels in finetune_loader:
            images, labels = images.cuda(), labels.cuda()
            optimizer_finetune.zero_grad()
            outputs = model(images)
            loss = criterion_finetune(outputs, labels)
            loss.backward()
            optimizer_finetune.step()
        scheduler.step()

        # val
        model.eval()
        predicted_labels = []
        true_labels = []

        with torch.no_grad():
            for batch, data in enumerate(test_loader):
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=1)
        recall = recall_score(true_labels, predicted_labels, average='macro')
        f1 = f1_score(true_labels, predicted_labels, average='macro')
        if accuracy > accuracy_max:
            accuracy_max = accuracy

        print(f"Epoch {epoch + 1}/{finetune_num_epochs}:Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

        # if (epoch + 1) % 100 == 0:
        #     torch.save(model.state_dict(), os.path.join(save_dir_model, f"model_CAE_XXX_{epoch + 1}.pth"))

        if accuracy >= accuracy_max:
            # torch.save(model, os.path.join(save_dir_model, "Amodel_CAE_ISCX_max.pth"))
            torch.save(model, os.path.join(save_dir_model, "Amodel_temp_max.pth"))



def test():
    print("Testing")
    predicted_labels = []
    true_labels = []
    # model_test = torch.load(os.path.join(save_dir_model, "Amodel_CAE_USTC_max.pth")).cuda()
    model_test = torch.load(os.path.join(save_dir_model, "Amodel_temp_max.pth")).cuda()

    # from torchsummary import summary
    # summary(model_test, input_size=(1, 28, 28))
    #
    # from thop import profile
    # dummy_input = torch.randn(1, 1, 28, 28).cuda()
    # flops, params = profile(model_test, (dummy_input,))
    # print('flops: ', flops, 'params: ', params)
    #
    # total_params = sum(p.numel() for p in model_test.parameters())
    # print(total_params)
    # exit()

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
    # print(np.array(conf_matrix))


if __name__ == '__main__':
    # pretrain()
    finetune()
    test()
