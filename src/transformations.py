from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from typing import List


def get_train_transform(img_size: int, mean: List[float], std: List[float], interpolation = InterpolationMode.BICUBIC):
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (img_size, img_size), interpolation=interpolation,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            # transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            transforms.CenterCrop(size=img_size),
            transforms.ToTensor(),
            normalize_transform(mean, std),
        ]
    )
    return train_transform


def get_valid_transform(img_size: int, mean: List[float], std: List[float]):
    valid_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize_transform(mean, std),
        ]
    )
    return valid_transform


def normalize_transform(mean: List[float], std: List[float]):
    normalize = transforms.Normalize(
        mean=mean,
        std=std,
    )
    return normalize
