import torch.nn as nn
import torch

class Classifier(nn.Module):
    def __init__(self, latent_dim, cls_dim):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, cls_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class MTMAE_finetuneModel(nn.Module):
    def __init__(self, encoder, classifier):
        super(MTMAE_finetuneModel, self).__init__()
        self.encoder = encoder
        self.classifierHead = classifier

    def forward(self, x):
        x, _ = self.encoder(x)
        x_cls = x[:, :1, :]
        x_cls = torch.squeeze(x_cls, dim=1)
        x = self.classifierHead(x_cls)
        return x


class CAE_finetuneModel(nn.Module):
    def __init__(self, encoder, classifier):
        super(CAE_finetuneModel, self).__init__()
        self.encoder = encoder
        self.classifierHead = classifier

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        x = self.classifierHead(x)
        return x


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False
    print("Freezing done!")
    return model


if __name__ == '__main__':
    import sys
    import torch

    sys.path.append("..")
    from models.model_MTMAE import encoder

    BackboneNetwork = encoder()
    BackboneNetwork = freeze_model(BackboneNetwork)

    classifier = Classifier(latent_dim=128, cls_dim=20)

    # instantiate a complete classification network model
    model_test = MTMAE_finetuneModel(BackboneNetwork, classifier)
    x = torch.randn(64, 1, 28, 28)
    y = model_test(x)
    print(y[0])

    # for name, param in model_test.named_parameters():
    #     print(name, param.requires_grad)
