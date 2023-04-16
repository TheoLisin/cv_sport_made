# import lightning.pytorch as pl
import pytorch_lightning as pl
import os
# from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
from pathlib import Path
from torch import set_float32_matmul_precision
from torch.cuda import empty_cache
from torchvision.models.efficientnet import (
    efficientnet_v2_l,
    EfficientNet_V2_L_Weights,
    efficientnet_v2_m,
    EfficientNet_V2_M_Weights,
)
from torchvision.models.vision_transformer import (
    vit_b_16,
    ViT_B_16_Weights,
)
from torchvision.transforms.functional import InterpolationMode
from sklearn.model_selection import train_test_split

from dataset import SportacusDataset
from plmodel import make_optimizer, make_lr_scheduler, SportacusModule
from classification import ClassificationHead
from transformations import get_train_transform, get_valid_transform

IMG_SIZE_VIT = 224
IMG_SIZE_EFN = 480
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

EFN_START_SIZE = 1280
VIT_START_SIZE = 768


def init_datasets():
    data_path = Path("/workspaces/cv/src/data")
    train_folder = data_path / "train"
    test_folder = data_path / "test"
    train_csv = data_path / "train.csv"
    test_csv = data_path / "test.csv"
    # transforms = EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms()
    train_transform = get_train_transform(IMG_SIZE_EFN, MEAN, STD)
    val_transform = get_valid_transform(IMG_SIZE_EFN, MEAN, STD)

    df = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    train_ind, val_ind = train_test_split(df.index, test_size=0.1, shuffle=False)

    train_dataset = SportacusDataset(
        df.iloc[train_ind, :],
        train_folder,
        min_shape_size=40,
        transform=train_transform,
    )
    val_dataset = SportacusDataset(
        df.iloc[val_ind, :],
        train_folder,
        min_shape_size=40,
        transform=val_transform,
    )
    test_dataset = SportacusDataset(
        df_test,
        test_folder,
        None,
        val_transform,
        test=True,
    )
    return train_dataset, val_dataset, test_dataset


def init_model(freeze_body: bool = True):
    # sportacus = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
    sportacus = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
    # sportacus = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    if freeze_body:
        for param in sportacus.parameters():
            param.requires_grad = False

    classification_head = ClassificationHead(30, start_size=EFN_START_SIZE, depth=4)

    # depends on model type (ViT: heads)
    sportacus.classifier = classification_head
    # sportacus.heads = classification_head
    return sportacus


def main():
    sportacus = init_model(freeze_body=True)
    train_dataset, val_dataset, test_dataset = init_datasets()
    set_float32_matmul_precision("high")
    # model = SportacusModule(
    #     sportacus,
    #     make_optimizer,
    #     train_dataset,
    #     val_dataset,
    #     test_dataset,
    #     make_lr_scheduler,
    #     64,
    # )

    model = SportacusModule.load_from_checkpoint(
        '/workspaces/cv/src/models_checkpoint/ep4_fet58_head_efn.ckpt',
        model=sportacus,
        optimizer_fn=make_optimizer,
        trainset=train_dataset,
        valset=val_dataset,
        testset=test_dataset,
        lr_scheduler_fn=make_lr_scheduler,
        batch_size=16,
    )

    # train_layers = ['classifier', 'features.8', 'features.7', 'features.6', 'features.5']

    # for name, par in model.named_parameters():
    #     par.requires_grad = False
    #     for name_l in train_layers:
    #         if name_l in name:
    #             par.requires_grad = True
    #             break
            

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        "./src/models_checkpoint",
        monitor="val_loss",
        # save_top_k=2,
        # every_n_train_steps=245,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=100,
        callbacks=[
            pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", patience=5),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            checkpoint_callback,
        ],
    )
    trainer.fit(model)
    trainer.test(ckpt_path='best')


if __name__ == "__main__":
    empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
    main()
    empty_cache()
    
