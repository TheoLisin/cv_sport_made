import torch
import pandas as pd
from torch.utils.data import Dataset
from torch import Tensor
from PIL import Image
from pathlib import Path
from typing import Optional, Union


class SportacusDataset(Dataset):
    """Sport dataset."""

    def __init__(
        self,
        info_pd: pd.DataFrame,
        img_dir: Union[Path, str],
        min_shape_size: Optional[int] = None,
        transform=None,
        test: bool = False,
    ):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.pd_df = info_pd
        self.test = test

        if min_shape_size is not None and test is False:
            self._rm_small_images(min_shape_size)

        if test is False:
            self._construct_target()
            label_counts = (
                self.pd_df.loc[:, ["label", "target"]]
                .groupby("target")
                .count()["label"]
            )
            self.class_w = torch.tensor((label_counts / len(self.pd_df)).to_numpy())
            self.class_w = self.class_w.float()
        else:
            self.target = []

    def _construct_target(self):
        labels = sorted(self.pd_df.label.unique())
        self.label_enc = {label: i for i, label in enumerate(labels)}
        self.label_dec = {num: label for label, num in self.label_enc.items()}
        self.pd_df["target"] = self.pd_df.label.apply(
            func=lambda x: self.label_enc.get(x)
        )

    def _rm_small_images(self, min_s: int):
        idx_to_delete = []
        for idx, img_name in self.pd_df.image_id.items():
            path = self.img_dir / img_name
            img = Image.open(path)
            img.size
            h, w = img.size
            ch = len(img.getbands())

            if h < min_s or w < min_s or ch == 1:
                idx_to_delete.append(idx)

        self.pd_df = self.pd_df.drop(idx_to_delete, axis=0).reset_index(drop=True)

    def __len__(self):
        return len(self.pd_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.pd_df.iloc[idx, :]
        name = data.image_id
        if self.test:
            target = -1
        else:
            target = data.target
        path = self.img_dir / name
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        if isinstance(img, Tensor):
            img = img.float()
            return img, target

        return img, target
