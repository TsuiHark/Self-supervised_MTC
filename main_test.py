import torch
import numpy as np
import torch.utils.data as Data
from modules.model_finetune import Classifier, MTMAE_finetuneModel
from utils.dataloader_finetune import train_Subset, test_Subset
from models.model_MTMAE import encoder
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

batch_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_loader = Data.DataLoader(dataset=train_Subset, batch_size=batch_size, shuffle=True)
test_Subset_data_loader = Data.DataLoader(dataset=test_Subset, batch_size=batch_size, shuffle=False)

Encoder = encoder(img_size=28, patch_size=4, in_chans=1, embed_dim=128,
                  encoder_depth=8, num_heads=8, mlp_ratio=3.)

classifierHead = Classifier(latent_dim=128, cls_dim=3)  # 修改cls_dim
model = MTMAE_finetuneModel(Encoder, classifierHead)

# load the pretrained weights
# weights_dict = Path(r"D:\Projects\MTMAE-main\weights\model\Amodel_USTC_max.pth")
# weights_dict = Path(r"D:\Projects\MTMAE-main\weights\model\Amodel_ISCX_max.pth")
# weights_dict = Path(r"D:\Projects\MTMAE-main\weights\model\Amodel_CICMalAnal_max.pth")
weights_dict = Path(r"D:\Projects\MTMAE-main\weights\model\Amodel_CICAAGM_max.pth")

model.load_state_dict(torch.load(weights_dict, map_location=torch.device('cpu')))
print(f"Load weights from {weights_dict} successfully.")
model.to(device)

model.eval()
correct = 0
total = 0
predicted_labels = []
true_labels = []
with torch.no_grad():
    for batch, data in enumerate(test_Subset_data_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # Store predicted and true labels for later calculation
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
accuracy = 100.0 * correct / total
precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')
# cm = confusion_matrix(true_labels, predicted_labels)
print(f"Accuracy: {accuracy:.5f}%, Precision: {precision:.5f}, Recall: {recall:.5f}, F1-score: {f1:.5f}")
# print(cm)


def calculate_false_alarm_rate(y_true, y_pred, positive_class):
    # 计算混淆矩阵
    confusion = confusion_matrix(y_true, y_pred)

    # 获取正类别的行和列索引
    positive_idx = positive_class
    FP = np.sum(confusion[:, positive_idx]) - confusion[positive_idx, positive_idx]
    TN = np.sum(confusion) - np.sum(confusion[positive_idx, :]) - np.sum(confusion[:, positive_idx]) + confusion[
        positive_idx, positive_idx]

    false_alarm_rate = FP / (FP + TN)
    return false_alarm_rate

