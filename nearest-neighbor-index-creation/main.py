import glob

import faiss
import torch
from torchvision import io
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, transforms_path, file_path_wildcard):
        self.transforms = torch.jit.load(transforms_path)
        self.files = glob.glob(file_path_wildcard)
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        img_array = io.read_image(file)
        img_array = self.transforms(img_array)
        return img_array[0, ...]


ds = ImageDataset(
    transforms_path="/home/user/artifact/transforms.pt",
    file_path_wildcard="/home/user/data/caltech-101/**/*.jpg"
)

data_loader = DataLoader(
    ds,
    batch_size=64,
    shuffle=False,
    num_workers=4,
)


device = "cuda"
model = torch.jit.load("/home/user/artifact/model.pt")
model = model.to(device)
index = faiss.IndexFlatIP(64)

with torch.no_grad():
    for imgs_array in data_loader:
        imgs_array = imgs_array.to(device)

        embedding = model.forward_embedding(imgs_array)
        embedding = embedding.cpu().numpy()

        index.add(embedding)

faiss.write_index(index, "/home/user/artifact/index.fs")
