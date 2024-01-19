from models.model_MTMAE import encoder, decoder
from utils.dataloader_pretrain import train_dataset
from utils.dataloader_finetune import train_subset
import torch
from torchvision.utils import save_image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./vis/input_images", exist_ok=True)
os.makedirs("./vis/output_images", exist_ok=True)

encoder = encoder()  # .to(device)
model_path_en = r"D:\Projects\MTMAE-main\weights\encoder\encoder6_300.pth"
encoder.load_state_dict(torch.load(model_path_en, map_location=torch.device('cpu')))
encoder.eval()

decoder = decoder()  # .to(device)
model_path_de = r"D:\Projects\MTMAE-main\weights\decoder\decoder6_300.pth"
decoder.load_state_dict(torch.load(model_path_de, map_location=torch.device('cpu')))
decoder.eval()

data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=128,
                                          shuffle=False)
# for epoch in range(1):
#     with torch.no_grad():
#         for step, data in enumerate(data_loader):
#             images, labels = data
#             latent, ids_restore = encoder(images, mask_ratio=0.6)
#             predict = decoder(latent, ids_restore)
#
#             batches_done = epoch * len(data_loader) + step
#             if batches_done % 40 == 0:
#                 save_image(images.data[:9], "./vis/input_images/%d.png" % batches_done, nrow=3, normalize=True)
#                 save_image(predict.data[:9], "./vis/output_images/%d.png" % batches_done, nrow=3, normalize=True)

data_loader_ = torch.utils.data.DataLoader(dataset=train_subset,
                                          batch_size=500,
                                          shuffle=False)

# Initialize empty lists to store images
label_0_images = []
label_1_images = []
label_2_images = []
label_3_images = []

for epoch in range(1):
    with torch.no_grad():
        for step, data in enumerate(data_loader_):
            images, labels = data

            for i in range(len(labels)):
                # Check the label and add the image to the corresponding list
                if labels[i] == 10 and len(label_0_images) < 9:
                    label_0_images.append(images[i])
                elif labels[i] == 11 and len(label_1_images) < 9:
                    label_1_images.append(images[i])
                elif labels[i] == 17 and len(label_2_images) < 9:
                    label_2_images.append(images[i])
                elif labels[i] == 18 and len(label_3_images) < 9:
                    label_3_images.append(images[i])

                # If we have collected 9 images for each label, break the loop
                if len(label_0_images) == 9 and len(label_1_images) == 9 and len(label_2_images) == 9 and len(label_3_images) == 9:
                    break

# Convert the lists of images to tensors
label_0_images = torch.stack(label_0_images)
label_1_images = torch.stack(label_1_images)
label_2_images = torch.stack(label_2_images)
label_3_images = torch.stack(label_3_images)

# Save the images
# save_image(label_0_images, "./vis/label_0_images.png", nrow=3, normalize=True)
save_image(label_1_images, "./vis/label_1_images.png", nrow=3, normalize=True)
save_image(label_2_images, "./vis/label_2_images.png", nrow=3, normalize=True)
# save_image(label_3_images, "./vis/label_3_images.png", nrow=3, normalize=True)

