import torch.nn as nn
from torch.linalg import vecdot
from torchvision.models import resnet34, ResNet34_Weights


class EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.pre_trained = nn.Sequential(
            *list(model.children())[:-1],
        )

        self.conv = nn.Conv2d(512, 64, 1)
        self.linear = nn.Linear(1, 1)

    def change_freezing(self, mode):
        for param in self.pre_trained.parameters():
            param.requires_grad = mode

    def forward_embedding(self, x):
        emb = self.pre_trained(x)
        emb = self.conv(emb)
        emb = emb[:, :, 0, 0]
        return emb

    def forward(self, current_img, context_img):
        e_curr = self.forward_embedding(current_img)
        e_context = self.forward_embedding(context_img)
        dot_prod = vecdot(e_curr, e_context)

        emb = dot_prod.unsqueeze(-1)
        o = self.linear(emb).squeeze(-1)
        return o


def accuracy(predicted_logits, actual, pred_threshold=0.5):
    predicted_sigmoid = nn.functional.sigmoid(predicted_logits)
    predicted = (predicted_sigmoid > pred_threshold).float()
    return (actual == predicted).float().mean()
