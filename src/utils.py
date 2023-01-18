import os
import random
from typing import Union

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def read_img(
    image_path: str, img_size: Union[int, None] = None, return_array: bool = False
) -> Union[Image.Image, np.ndarray]:
    """_summary_

    Parameters
    ----------
    image_path : str
        _description_
    img_size : Union[int, None], optional
        _description_, by default None
    return_array : bool, optional
        _description_, by default False

    Returns
    -------
    Union[Image.Image, np.ndarray]
        _description_
    """
    img = Image.open(image_path).convert("L")
    if img_size is not None:
        img = img.resize((img_size, img_size))
    if return_array:
        img = np.array(img)
    return img


def seed_everything(seed: int):
    """Funstion sets seeds for reproducibility of training process

    Parameters
    ----------
    seed : int
        _description_
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_transform():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    )
    return transform
