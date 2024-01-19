import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import torch.utils.data as Data
from modules.model_finetune import Classifier, MTMAE_finetuneModel
from models.model_MTMAE import encoder
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# from utils.dataloader_USTC import train_subset, test_set
# from utils.dataloader_CIC import train_subset, test_set
from utils.dataloader_ISCX import train_subset, test_set

from torch.cuda import Event


num_classes = 6
batch_size = 64
epochs = 50 # ###################
lr = 0.0005
weight_decay = 0.05
warmup_epochs = 10  # ###################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_dir_run = "./runs"
save_dir_model = "./weights/classifier_"
data_loader = Data.DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
test_loader = Data.DataLoader(dataset=test_set, batch_size=256, shuffle=False)
# ---------------------------------------------------------------------------------------------------------------------#


classifierHead = Classifier(latent_dim=128, cls_dim=num_classes)  # 修改cls-dim


def finetune():
    Encoder = encoder(img_size=28, patch_size=4, in_chans=1, embed_dim=128,
                      encoder_depth=8, num_heads=8, mlp_ratio=3.)
    # load the pretrained weights
    weights_dict = Path(r"D:\Projects\MTMAE-main\weights\encoder\encoder6_300.pth")
    Encoder.load_state_dict(torch.load(weights_dict, map_location=torch.device('cpu')))
    print(f"Load weights from {weights_dict} successfully.")
    model = MTMAE_finetuneModel(Encoder, classifierHead)

    model.to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=warmup_epochs, T_mult=1, eta_min=0)
    # ---------------------------------------------------------------------------------------------------------------------#

    accuracy_max = 0
    accuracy = 0
    # finetune
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        train_Subset_data_loader = tqdm(data_loader, file=sys.stdout)
        for batch, data in enumerate(train_Subset_data_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item()
            # print("epoch", epoch, "batch", batch, "loss:", loss.detach().item())
        scheduler.step()
        # print("epoch loss:", total_loss / len(train_Subset) * batch_size)
        # with open(os.path.join(save_dir_run, "loss/finetune.txt"), "a") as f:
        #     f.write(str(total_loss / len(train_Subset_data_loader) * batch_size) + "\n")

        # val
        model.eval()
        correct = 0
        total = 0
        predicted_labels = []
        true_labels = []

        with torch.no_grad():
            for batch, data in enumerate(test_loader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        accuracy = accuracy_score(true_labels, predicted_labels)
        if accuracy > accuracy_max:
            accuracy_max = accuracy
        precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=1)
        recall = recall_score(true_labels, predicted_labels, average='macro')
        f1 = f1_score(true_labels, predicted_labels, average='macro')

        print(f"Epoch {epoch + 1}/{epochs}:Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
        # with open(os.path.join(save_dir_run, "acc/temp_finetune_acc.txt"), "a") as f:
        #     f.write(str(accuracy) + "\n")
        # with open(os.path.join(save_dir_run, "acc/temp_finetune_pre.txt"), "a") as f:
        #     f.write(str(precision) + "\n")
        # with open(os.path.join(save_dir_run, "acc/temp_finetune_rc.txt"), "a") as f:
        #     f.write(str(recall) + "\n")
        # with open(os.path.join(save_dir_run, "acc/temp_finetune_f1.txt"), "a") as f:
        #     f.write(str(f1) + "\n")

        # if (epoch + 1) % 100 == 0:
        #     torch.save(model.state_dict(), os.path.join(save_dir_model, f"model_temp_{epoch+1}.pth"))

        if accuracy >= accuracy_max:
            # torch.save(model.state_dict(), os.path.join(save_dir_model, "Amodel_MAE_CIC_max.pth"))
            torch.save(model, os.path.join(save_dir_model, "Amodel_temp_max.pth"))


def test():
    print("Testing")
    predicted_labels = []
    true_labels = []
    total_batches = len(test_loader)
    total_samples = total_batches * test_loader.batch_size

    # Encoder = encoder(img_size=28, patch_size=4, in_chans=1, embed_dim=128,
    #                   encoder_depth=8, num_heads=8, mlp_ratio=3.)
    # model_test = MTMAE_finetuneModel(Encoder, classifierHead).cuda()
    # path = torch.load(os.path.join(save_dir_model, "Amodel_MAE_USTC_max.pth"), map_location=torch.device('cpu'))
    # model_test.load_state_dict(path)

    model_test = torch.load(os.path.join(save_dir_model, "Amodel_temp_max.pth")).cuda()

    # 创建CUDA事件
    start_event = Event(enable_timing=True)
    end_event = Event(enable_timing=True)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            # 记录开始时间
            start_event.record()

            outputs = model_test(images)
            # 记录结束时间
            end_event.record()
            torch.cuda.synchronize()  # 等待GPU完成所有操作
            # 计算时间
            elapsed_time_ms = start_event.elapsed_time(end_event)

            _, predicted = torch.max(outputs, 1)
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    # 评估模型
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=1)
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    # 计算推理速度
    # total_inference_time_sec = elapsed_time_ms / 1000.0  # 转换为秒
    # inference_speed = total_samples / total_inference_time_sec

    # print(f"Total inference time: {total_inference_time_sec:.2f} seconds")
    # print(f"Inference speed: {inference_speed:.2f} samples per second")

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    # print('Confusion Matrix:')
    # print(conf_matrix)

    # from thop import profile
    # dummy_input = torch.randn(1, 1, 28, 28).cuda()
    # flops, params = profile(model_test, (dummy_input,))
    # print('flops: ', flops, 'params: ', params)
    #
    # total_params = sum(p.numel() for p in model_test.parameters())
    # print(total_params)
    # exit()

    # model_test = torch.load(os.path.join(save_dir_model, "Amodel_MAE_USTC_max.pth")).cuda()


if __name__ == '__main__':
    finetune()
    test()
