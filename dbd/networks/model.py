import lightning as pl
import torch
import torchmetrics
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models


def focal_loss(logits, targets, alpha=1.0, gamma=2.0, reduction="mean"):
    ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    loss = alpha * (1 - pt) ** gamma * ce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def l1_loss(weights):
    return sum(p.abs().sum() for p in weights)


class Model(pl.LightningModule):
    def __init__(self, lr: float = 1e-4, total_epochs: int = 100):
        super().__init__()
        self.example_input_array = torch.zeros((32, 3, 224, 224), dtype=torch.float32)
        self.nb_classes = 10

        self.model = self.build_model()
        self.lr = lr
        self.total_epochs = total_epochs

        self.metrics_train = torchmetrics.F1Score(
            task="multiclass",
            num_classes=self.nb_classes,
            average="macro",
            validate_args=False,
        )
        # self.metrics_val = torchmetrics.F1Score(task='multiclass', num_classes=self.nb_classes, average="macro", validate_args=False)

        self.metrics_val = torchmetrics.MetricCollection(
            [
                torchmetrics.F1Score(
                    task="multiclass",
                    num_classes=self.nb_classes,
                    average="macro",
                    validate_args=False,
                ),
                torchmetrics.Precision(
                    task="multiclass",
                    num_classes=self.nb_classes,
                    average="none",
                    validate_args=False,
                ),
                torchmetrics.Recall(
                    task="multiclass",
                    num_classes=self.nb_classes,
                    average="none",
                    validate_args=False,
                ),
            ],
        )

    def build_model(self):
        weights = models.ShuffleNet_V2_X0_5_Weights.DEFAULT
        model = models.shufflenet_v2_x0_5(weights=weights)
        model.fc = torch.nn.Linear(1024, self.nb_classes)

        # weights = models.MobileNet_V3_Small_Weights.DEFAULT
        # model = models.mobilenet_v3_small(weights=weights)
        # model.classifier[-1] = torch.nn.Linear(1024, self.nb_classes)

        return model

    def compute_loss(self, pred, y):
        # loss = torch.nn.functional.cross_entropy(pred, y)
        focal_error = focal_loss(pred, y)
        l1_error = l1_loss(self.model.parameters())

        return focal_error + 1e-6 * l1_error

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        loss = self.compute_loss(pred, y)
        self.log("loss/train", loss)

        # Accumulate metrics
        self.metrics_train.update(pred, y)

        return loss

    def on_train_epoch_end(self):
        metrics_train = self.metrics_train.compute()
        self.log_dict({"F1/train": torch.mean(metrics_train)})

        self.metrics_train.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        loss = self.compute_loss(pred, y)
        self.log("loss/val", loss)

        # Accumulate metrics
        self.metrics_val.update(pred, y)

        return loss

    def on_validation_epoch_end(self):
        metrics_val = self.metrics_val.compute()

        self.log_dict({f"Recall/val_{i}": score for i, score in enumerate(metrics_val["MulticlassRecall"])})
        self.log_dict(
            {f"Precision/val_{i}": score for i, score in enumerate(metrics_val["MulticlassPrecision"])},
        )
        self.log_dict({"F1/val": torch.mean(metrics_val["MulticlassF1Score"])})

        self.metrics_val.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        pred = self(x)
        pred = torch.argmax(pred, dim=-1)
        return pred

    def forward(self, x):
        pred = self.model(x)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0)
        # optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-5)

        scheduler = CosineAnnealingLR(optimizer, T_max=self.total_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
