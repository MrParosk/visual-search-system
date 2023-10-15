import numpy as np
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
from torch.linalg import vecdot

from dataset import CalTechDataset, image_dataset_path


train_size = 0.9
ds = CalTechDataset(image_dataset_path)
batch_size = 64


train_ds, val_ds = random_split(
    ds,
    [
        int(train_size * len(ds)),
        len(ds) - int(train_size * len(ds))
    ]
)

train_loader = DataLoader(
    train_ds, 
    batch_size=batch_size,
    num_workers=2,
    collate_fn=CalTechDataset.collate_fn,
    shuffle=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    collate_fn=CalTechDataset.collate_fn
)


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


lr = 1e-3
device = "cuda"
num_epochs = 5

pred_threshold = 0.3

model = EmbeddingModel()
model.change_freezing(False)
model = model.to(device)

model = torch.compile(model)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):

    model.train()
    epoch_train_loss, epoch_train_acc = [], []
    for current_img, context_img, same_class in train_loader:
        model.zero_grad()

        current_img = current_img.to(device)
        context_img = context_img.to(device)
        same_class = same_class.to(device)
        output = model(current_img, context_img)

        batch_loss = nn.functional.binary_cross_entropy_with_logits(output, same_class)
        batch_loss.backward()
        optimizer.step()

        batch_acc = accuracy(output, same_class)
        epoch_train_acc.append(batch_acc.cpu().detach().numpy())
        epoch_train_loss.append(batch_loss.cpu().detach().numpy())

    epoch_train_loss = np.mean(epoch_train_loss)
    epoch_train_acc = np.mean(epoch_train_acc)

    print(f"Train loss: {epoch_train_loss:.4f}")
    print(f"Train accuracy: {epoch_train_acc:.4f}")

    model.eval()
    epoch_val_loss, epoch_val_acc = [], []
    with torch.no_grad():
        for current_img, context_img, same_class in val_loader:
            current_img = current_img.to(device)
            context_img = context_img.to(device)
            same_class = same_class.to(device)

            output = model(current_img, context_img)

            batch_acc = accuracy(output, same_class)
            epoch_val_acc.append(batch_acc.cpu().detach().numpy())
            epoch_val_loss.append(batch_loss.cpu().detach().numpy())

    epoch_val_loss = np.mean(epoch_val_loss)
    epoch_val_acc = np.mean(epoch_val_acc)

    print(f"Val loss: {epoch_val_loss:.4f}")
    print(f"Val accuracy: {epoch_val_acc:.4f}")
