import torch
from torchvision.transforms import RandomCrop, Normalize, Resize

import warnings
warnings.filterwarnings("ignore")


class ToGrayToThreeChannels(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        return x        


class Uint8ToFloat(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() / 255.0


class Unsqueeze(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0)


def get_transforms():
    transforms = torch.nn.Sequential(
        Resize((256, 256)),
        RandomCrop(size=(224, 224)),
        ToGrayToThreeChannels(),
        Uint8ToFloat(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        Unsqueeze()
    )

    return transforms
