import torch
from torchvision import io
from torchvision.transforms import RandomCrop


crop = RandomCrop(size=(250, 250), pad_if_needed=True)

mean = torch.Tensor([0.485, 0.456, 0.406]).view((3, 1, 1))
std = torch.Tensor([0.229, 0.224, 0.225]).view((3, 1, 1))


def gray_to_rgb(image_array):
    if image_array.shape[0] == 1:
        image_array = image_array.repeat(3, 1, 1)
    return image_array


def load_transform_image(image_path):
    image = io.read_image(image_path)
    image = gray_to_rgb(image)
    image = crop(image)
    image = image / 255.0
    image = (image - mean) / std
    image = image.unsqueeze(0)
    return image
