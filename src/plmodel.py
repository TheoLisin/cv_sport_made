# import lightning.pytorch as pl
import pytorch_lightning as pl
from numpy import ndarray
from torch import optim, Tensor
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Callable
from sklearn.metrics import f1_score

from dataset import SportacusDataset


def lr(step):
    return 0.1 + 0.9 * (0.998) ** step


def make_lr_scheduler(optimizer):
    # return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

def make_optimizer(model: Module):
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, nesterov=True)
    return optimizer


class SportacusModule(pl.LightningModule):
    def __init__(
        self,
        model: Module,
        optimizer_fn: Callable[[Module], optim.Optimizer],
        trainset: SportacusDataset,
        valset: SportacusDataset,
        testset: SportacusDataset,
        lr_scheduler_fn: Optional[optim.lr_scheduler.LRScheduler] = None,
        batch_size=32,
        test_save_path: str = "./src/data/answer.csv",
    ):
        super().__init__()
        self._model = model
        self._optimizer_fn = optimizer_fn
        self._lr_scheduler_fn = lr_scheduler_fn
        self._criterion = CrossEntropyLoss(weight=trainset.class_w)
        self._batch_size = batch_size
        self._trainset = trainset
        self._valset = valset
        self._testset = testset
        self.test_save_path: str = test_save_path

    def forward(self, input):
        return self._model(input)
    
    def step(self, batch: Tuple[Tensor, Tensor], step_name: str, prog_bar: bool = False):
        x, y = batch
        logits: Tensor = self._model(x)
        loss: Tensor = self._criterion(logits, y)
        loss_name = f"{step_name}_loss"
        f1_name = f"{step_name}_f1"
        self.log(loss_name, loss, prog_bar=prog_bar)
        self.log(f1_name, self._f1(logits, y))
        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx):
        # x, y = batch
        # logits: Tensor = self._model(x)
        # loss: Tensor = self._criterion(logits, y)
        # self.log("train_loss", loss)
        return self.step(batch, "train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx):
        # x, y = batch
        # logits: Tensor = self._model(x)
        # loss = self._criterion(logits, y)
        # self.log("val_loss", loss)
        # self.log("val_f1", self._f1(logits, y))
        # self.log("loss_check", loss)
        # self.logger.experiment.add_scalars(
        #     "loss",
        #     {"val": loss},
        #     global_step=self.global_step,
        # )
        # self.logger.experiment.add_scalars(
        #     "f1",
        #     {"val": self._f1(logits, y)},
        #     global_step=self.global_step,
        # )
        return self.step(batch, "val", True)

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx):
        x, _ = batch
        logits: Tensor = self._model(x)
        preds: ndarray = logits.argmax(dim=1).cpu().detach().numpy()
        preds_list = preds.squeeze().tolist()
        self._testset.target.extend(preds_list)

        if len(self._testset) == len(self._testset.target):
            dec = self._trainset.label_dec
            dec_target = [dec[lb] for lb in self._testset.target]
            test_df = self._testset.pd_df.copy()
            test_df["label"] = dec_target
            test_df.to_csv(self.test_save_path, sep=",", index=False)

        return

    def train_dataloader(self):
        return DataLoader(
            self._trainset,
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=16,
        )

    def val_dataloader(self):
        return DataLoader(
            self._valset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=16,
        )

    def test_dataloader(self):
        return DataLoader(
            self._testset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=16,
        )

    def configure_optimizers(self):
        optimizer = self._optimizer_fn(self._model)
        lr_scheduler = self._lr_scheduler_fn(optimizer)
        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]

    def _f1(self, logits: Tensor, labels: Tensor):
        np_target = labels.cpu().detach().numpy()
        np_logits = logits.argmax(dim=1).cpu().detach().numpy()
        return f1_score(np_target, np_logits, average="weighted")
