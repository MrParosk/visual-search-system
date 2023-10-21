from pathlib import Path
import random
from collections import defaultdict

import torch
from torchvision import io
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from image_transforms import get_transforms


image_dataset_path = "/home/user/data/caltech-101"


def get_classes(image_dataset_path):
    classes = []
    for path in Path(image_dataset_path).glob("*"):
        classes.append(path.name)
    return classes


def plot_example_images(image_dataset_path, nrows=2, ncols=2):
    classes = get_classes(image_dataset_path)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)

    for x_idx, row in enumerate(ax):
        for y_idx, col in enumerate(row):
            c = classes[x_idx + y_idx * nrows]

            image_path = Path(image_dataset_path) / c / "image_0001.jpg"
            image = io.read_image(str(image_path))
            image = image.permute((1, 2, 0)).numpy()
            col.imshow(image)
            col.set_title(c)

    plt.show()


class CalTechDataset(Dataset):
    def __init__(self, image_dataset_path, transforms, frac_positive = 0.3):
        self.transforms = transforms
        self.frac_positive = frac_positive

        self.classes = get_classes(image_dataset_path)

        self.images_paths = []
        self.class_to_image = defaultdict(list)

        for c in self.classes:
            class_images_path = (Path(image_dataset_path) / c).glob("*.jpg")

            for p in class_images_path:
                self.images_paths.append({"class": c, "path": str(p)})
                self.class_to_image[c].append(str(p))

    def __len__(self):
        return len(self.images_paths)

    def load_transform_image(self, image_path):
        image = io.read_image(image_path)
        image = self.transforms(image)
        return image

    def get_idx_and_context_image(self, idx):
        idx_image_path = self.images_paths[idx]

        if random.random() > self.frac_positive:
            other_class = random.sample(self.classes, 1)[0]
        else:
            other_class = idx_image_path["class"]
        
        other_image_class_images = self.class_to_image[other_class]
        other_image_path = random.sample(other_image_class_images, 1)[0]
        
        same_class = idx_image_path["class"] == other_class

        return idx_image_path["path"], other_image_path, same_class

    def __getitem__(self, idx):
        idx_image_path, other_image_path, same_class = self.get_idx_and_context_image(idx)
        idx_image = self.load_transform_image(idx_image_path)
        other_image = self.load_transform_image(other_image_path)
        same_class = torch.tensor(same_class, dtype=torch.float32).view(1)
        return (idx_image, other_image, same_class)

    @staticmethod
    def collate_fn(batch):
        current_img = torch.concat([b[0] for b in batch])
        context_img = torch.concat([b[1] for b in batch])
        same_class = torch.concat([b[2] for b in batch])
        return (current_img, context_img, same_class)


if __name__ == "__main__":
    transforms = get_transforms()
    ds = CalTechDataset(image_dataset_path, transforms)

    nrows = 5
    fig, ax = plt.subplots(nrows=nrows, ncols=2)

    for row in ax:
        idx = random.randint(0, len(ds) - 1)
        idx_image_path, other_image_path, same_class = ds.get_idx_and_context_image(idx)

        idx_image = io.read_image(idx_image_path)
        idx_image = idx_image.permute((1, 2, 0)).numpy()

        context_image = io.read_image(other_image_path)
        context_image = context_image.permute((1, 2, 0)).numpy()

        row[0].imshow(idx_image)
        row[1].imshow(context_image)

        row[0].set_title(same_class)

    plt.show()
