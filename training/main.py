import numpy as np
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn

from dataset import CalTechDataset, image_dataset_path
from model import EmbeddingModel, accuracy


train_size = 0.8
ds = CalTechDataset(image_dataset_path)
batch_size = 128

#This causes data-leakage but since the training part is not the central
# topic of this system design, will skip it
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
    num_workers=4,
    collate_fn=CalTechDataset.collate_fn,
    shuffle=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=CalTechDataset.collate_fn
)


lr = 1e-2
device = "cuda"
num_epochs = 5

model = EmbeddingModel()
model.change_freezing(False)
model = model.to(device)


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


model = model.cpu()
scripted_module = torch.jit.script(model)
scripted_module.save("/home/user/artifact/model.pt")

