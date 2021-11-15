import argparse
import logging
import os
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.utilities import rank_zero_info

from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import GPUAccelerator
from pytorch_lightning.plugins import NativeMixedPrecisionPlugin, DDPPlugin

from fisheye_dataset import FishEyeDataset
from fisheye_model import fisheye_model

log = logging.getLogger(__name__)


# 模型微调
class ModelFinetuning(BaseFinetuning):

    def __init__(self, milestones: tuple = (5, 10), train_bn: bool = False):
        self.milestones = milestones
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: pl.LightningModule):
        self.freeze(modules=pl_module.feature_extractor, train_bn=self.train_bn)

    def finetune_function(self, pl_module: pl.LightningModule, epoch: int, optimizer: Optimizer, opt_idx: int):
        if epoch == self.milestones[0]:
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor[-5:], optimizer=optimizer, train_bn=self.train_bn
            )

        elif epoch == self.milestones[1]:
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor[:-5], optimizer=optimizer, train_bn=self.train_bn
            )


class FisheyeDataModule(LightningDataModule):

    def __init__(
            self,
            data_root="data",
            num_workers=0,
            batch_size=8,
            test_mode=False
    ):
        super().__init__()

        self.data_root = data_root
        self._num_workers = num_workers
        self._batch_size = batch_size
        self.test_mode = test_mode

    def __dataloader(self, train: bool):
        """Train/validation loaders."""
        if train:
            dataset = FishEyeDataset(self.data_root, "train")
        else:
            if self.test_mode:
                dataset = FishEyeDataset(self.data_root, "test")
            else:
                dataset = FishEyeDataset(self.data_root, "val")
        return DataLoader(dataset=dataset, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=train)

    def train_dataloader(self):
        log.info("Training data loaded.")
        return self.__dataloader(train=True)

    def val_dataloader(self):
        log.info("Validation data loaded.")
        return self.__dataloader(train=False)


#  --- Pytorch-lightning module ---


class FisheyeTrainingModel(pl.LightningModule):

    def __init__(
            self,
            backbone: str = "fisheye",
            train_bn: bool = True,
            milestones: tuple = (5, 10),
            batch_size: int = 32,
            lr: float = 1e-2,
            lr_scheduler_gamma: float = 1e-1,
            num_workers: int = 6,
            **kwargs,
    ) -> None:
        """
        Args:
            dl_path: Path where the data will be downloaded
        """
        super().__init__()
        self.backbone = backbone
        self.train_bn = train_bn
        self.milestones = milestones
        self.batch_size = batch_size
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_workers = num_workers

        self.__build_model()

        # self.train_acc = Accuracy()
        # self.valid_acc = Accuracy()
        self.save_hyperparameters()

    def __build_model(self):
        """Define model layers & loss."""

        self.model = fisheye_model()

        # 3. Loss:
        self.loss_func = F.smooth_l1_loss

    def forward(self, x):
        """Forward"""
        x = self.model(x)
        return x

    def loss(self, pred, gt):
        return self.loss_func(input=pred, target=gt)

    def training_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y_pred = self.forward(x)
        y_gt = y.view(-1, 3).type_as(x)
        print(y_pred[0], y_gt[0])
        # 2. Compute loss
        train_loss = self.loss(y_pred, y_gt)

        # 3. Compute accuracy:
        self.log("train_mse_loss", train_loss, prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y_pred = self.forward(x)
        y_gt = y.view(-1, 3).type_as(x)

        # 2. Compute loss
        val_loss = self.loss(y_pred, y_gt)

        # 3. Compute accuracy:
        self.log("val_mse_loss", val_loss, prog_bar=True)

    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        # rank_zero_info(
        #     f"The model will start training with only {len(trainable_parameters)} "
        #     f"trainable parameters out of {len(parameters)}."
        # )
        optimizer = optim.Adam(trainable_parameters, lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)
        scheduler = MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_scheduler_gamma)
        return [optimizer], [scheduler]


def main():
    """Train the model.
    Args:
        args: Model hyper-parameters
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        default="data",
        type=str,
    )

    parser.add_argument(
        "--num-workers", default=0, type=int, metavar="W", help="number of CPU workers", dest="num_workers"
    )

    parser.add_argument(
        "--epochs", default=30, type=int, metavar="N", help="total number of epochs", dest="nb_epochs"
    )
    parser.add_argument(
        "--batch-size", default=8, type=int, metavar="B", help="batch size", dest="batch_size"
    )
    parser.add_argument(
        "--gpus", type=int, default=1, help="number of gpus to use"
    )
    parser.add_argument(
        "--lr", "--learning-rate", default=1e-3, type=float, metavar="LR", help="initial learning rate", dest="lr"
    )
    parser.add_argument(
        "--lr-scheduler-gamma",
        default=1e-1,
        type=float,
        metavar="LRG",
        help="Factor by which the learning rate is reduced at each milestone",
        dest="lr_scheduler_gamma",
    )
    parser.add_argument(
        "--train-bn",
        default=True,
        type=bool,
        metavar="TB",
        help="Whether the BatchNorm layers should be trainable",
        dest="train_bn",
    )
    parser.add_argument(
        "--milestones", default=[20, 25], type=list, metavar="M", help="List of two epochs milestones"
    )

    args = parser.parse_args()

    datamodule = FisheyeDataModule(
        data_root="data", batch_size=args.batch_size, num_workers=args.num_workers
    )
    model = FisheyeTrainingModel(**vars(args))
    # finetuning_callback = ModelFinetuning(milestones=args.milestones)

    # accelerator = GPUAccelerator(
    #     precision_plugin=NativeMixedPrecisionPlugin(),
    #     training_type_plugin=DDPPlugin(),
    # )   

    trainer = pl.Trainer(
        weights_summary=None,
        progress_bar_refresh_rate=1,
        num_sanity_val_steps=0,
        gpus=args.gpus,
        max_epochs=args.nb_epochs,
        # accelerator = accelerator
        # callbacks=[finetuning_callback]
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
