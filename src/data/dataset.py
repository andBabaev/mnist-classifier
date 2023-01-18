from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils import read_img


class MNISTDataset(Dataset):
    def __init__(
        self,
        img_paths: List[str],
        labels: Optional[List[int]] = None,
        n_classes: int = 10,
        preprocess_fn=None,
    ):
        """_summary_

        Parameters
        ----------
        img_paths : List[str]
            List of paths to images of mnist dataset
        labels : List[int], np.ndarray or None
            List of labels of mnist images for training mode. Or `None` for evaluate mode
        n_classes : int, optional
            Number of classes, by default 10
        preprocess_fn : _type_, optional
            Function for preprocessing of inputs, by default None
        """
        self.images = []
        for img_path in tqdm(img_paths, unit="images"):
            self.images.append(read_img(img_path))
        self.labels = labels
        self.n_classes = n_classes
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.images)

    def get_processed_label(self, idx: int) -> torch.Tensor:
        """Return OHE verions of a label

        Parameters
        ----------
        idx : int
            label id

        Returns
        -------
        torch.Tensor
            OHE label
        """
        label = self.labels[idx]
        label_ohe = torch.zeros(self.n_classes)
        label_ohe[label] = 1
        return torch.FloatTensor(label_ohe)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:

        image = self.images[idx]
        if self.preprocess_fn is not None:
            preproc_image = self.preprocess_fn(image)

        if self.labels is not None:
            label = self.get_processed_label(idx)
        else:
            label = torch.FloatTensor()

        return preproc_image, label
