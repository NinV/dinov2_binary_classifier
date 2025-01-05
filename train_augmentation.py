from dataclasses import dataclass, field
import torch
import torch.nn as nn
import tyro
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import Settings


@dataclass
class Config:
    save_dir: str = 'saved_model'
    model_size: str = 'base'
    batch_size: int = 16
    max_epochs: int = 10
    imsize: int = 224       # Must be multiplier of 14, because ViT backbone use 14x14 grid
    backbone_num_patches: int = 14

    lr: float = 1e-5
    random_seed: int = 42
    test_size: float = 0.2


backbone_archs = {
    "small": ("vits14", 384),   # (backbone_name, out_dim)
    "base": ("vitb14", 768),
    "large": ("vitl14", 1024),
}


def build_model(backbone_size):
    backbone_arch, out_dim = backbone_archs[backbone_size]
    backbone_name = f"dinov2_{backbone_arch}"
    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    return backbone_model, out_dim


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, out_dim=1024):
        super().__init__()
        self.out_dim = out_dim
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(out_dim, 1)

    def forward(self, x):
        x = self.dropout(x)
        return self.fc1(x)


class Dinov2Classifier(nn.Module):
    def __init__(self, backbone_size):
        super().__init__()
        self.backbone, out_dim = build_model(backbone_size)
        self.classifier = LinearClassifier(out_dim)

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)


def get_transform(cfg, mode='train'):
    if mode == 'train':
        crop_max_size = int(cfg.imsize * 1.2)
        crop_min_size = int(cfg.imsize * 0.8)
        transform = A.Compose([
            A.Resize(crop_max_size, crop_max_size),
            A.RandomSizedCrop(min_max_height=(crop_min_size, crop_max_size), size=(cfg.imsize, cfg.imsize), p=1),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
            ToTensorV2(),
        ])

    elif mode == 'test_simple':
        transform = A.Compose([
            A.Resize(cfg.imsize, cfg.imsize),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
            ToTensorV2(),
        ])

    elif mode == 'test_complex':
        transform = A.Compose([
            A.Resize(cfg.imsize, cfg.imsize),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
            ToTensorV2(),
        ])
    else:
        raise ValueError(f'Unrecognized mode {mode}')
    return transform

class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.label_map = {'real': 0, 'fake': 1}  # Map 'real' -> 0, 'fake' -> 1

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row['path']
        label = self.label_map[row['label']]
        # image = Image.open(img_path).convert('RGB')  # Ensure images are in RGB format
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']

        return image, label


def prepare_data(cfg: Config):
    settings = Settings()
    df = pd.DataFrame(
        {
            "label": label,
            "kind": kind,
            "path": path,
        }
        for label, kind, path in settings.iterate_all_train_images()
    )
    df_train, df_val = train_test_split(df, test_size=cfg.test_size,
                                        random_state=cfg.random_seed, stratify=df['label'])
    train_dataset = ImageDataset(df_train, transform=get_transform(cfg, 'train'))
    val_dataset = ImageDataset(df_val, transform=get_transform(cfg, 'test_simple'))
    return train_dataset, val_dataset


class BinaryClassifier(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.net = Dinov2Classifier(cfg.model_size)
        self.train_ds, self.val_ds = prepare_data(self.cfg)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.cfg.batch_size,
                          shuffle=True, persistent_workers=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, shuffle=False, persistent_workers=True, num_workers=4)

    def training_step(self, batch_data, batch_idx):
        images, labels = batch_data  # Unpack the batch
        logits = self.net(images)  # Forward pass
        labels = labels.float().unsqueeze(1)  # Ensure labels match output shape for binary classification
        loss = self.criterion(logits, labels)  # Compute binary cross-entropy loss

        # Log the loss for visualization in TensorBoard or other loggers
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch_data, batch_idx):
        images, labels = batch_data
        logits = self.net(images)
        labels = labels.float().unsqueeze(1)
        loss = self.criterion(logits, labels)

        # Calculate metrics like accuracy if needed
        preds = torch.sigmoid(logits) > 0.5  # Apply sigmoid for binary classification thresholding
        acc = (preds == labels).float().mean()

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.lr)
        return optimizer


def main(cfg: Config):
    checkpoint_callback = ModelCheckpoint(
        dirpath="save_models",
        save_top_k=1,
        monitor="val_acc",
        mode='max',
        save_last=True
    )
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="gpu",  # Use "cpu" if no GPU is available
        devices=1,  # Number of GPUs or CPUs
        callbacks=checkpoint_callback,
        num_sanity_val_steps=0
    )
    model = BinaryClassifier(cfg)
    trainer.fit(model)


if __name__ == '__main__':
    main(tyro.cli((Config)))
