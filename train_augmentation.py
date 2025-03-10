import pathlib
import json
from dataclasses import asdict
import tyro
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from models import Config, BinaryClassifier


def main(cfg: Config):
    saved_dir = pathlib.Path(cfg.save_dir)
    saved_dir.mkdir(exist_ok=True, parents=True)

    with open(saved_dir / "config.json", 'w') as json_file:
        json.dump(asdict(cfg), json_file, indent=2)

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.save_dir,
        save_top_k=1,
        monitor="val_acc",
        mode='max',
        save_last=True
    )
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="gpu",  # Use "cpu" if no GPU is available
        devices=1,          # Number of GPUs or CPUs
        callbacks=checkpoint_callback,
        num_sanity_val_steps=0
    )
    model = BinaryClassifier(cfg)
    trainer.fit(model)


if __name__ == '__main__':
    main(tyro.cli(Config))
