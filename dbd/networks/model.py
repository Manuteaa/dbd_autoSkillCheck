import pytorch_lightning as pl
import torchmetrics
import torch
import torchvision.models as models


class Model(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.example_input_array = torch.zeros((32, 3, 224, 224), dtype=torch.float32)
        self.nb_classes = 11

        self.model = self.build_model()
        self.lr = lr

        self.acc_score_train = torchmetrics.Accuracy(task='multiclass', num_classes=self.nb_classes, average="none", validate_args=False)
        self.acc_score_val = torchmetrics.Accuracy(task='multiclass', num_classes=self.nb_classes, average="none", validate_args=False)

        # self.metrics_val = torchmetrics.MetricCollection([
        #     torchmetrics.F1Score(task='multiclass', num_classes=self.nb_classes, average="none", validate_args=False),
        #     torchmetrics.Accuracy(task='multiclass', num_classes=self.nb_classes, average="none", validate_args=False)
        # ])

    def build_model(self):
        # weights = models.MobileNet_V3_Large_Weights.DEFAULT
        # model = models.mobilenet_v3_large(weights=weights)
        # model.classifier[-1] = torch.nn.Linear(1280, self.nb_classes)

        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = torch.nn.Linear(1024, self.nb_classes)

        # weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        # model = models.convnext_tiny(weights=weights, num_classes=self.nb_classes)

        return model

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        loss = torch.nn.functional.cross_entropy(pred, y)
        self.log("loss/train", loss)

        # Accumulate metrics
        self.acc_score_train.update(pred, y)

        return loss

    def on_train_epoch_end(self):
        acc_score_train = self.acc_score_train.compute()
        self.log_dict({"Acc/train_{}".format(i): score for i, score in enumerate(acc_score_train)})
        self.log_dict({"Acc/train_mean": torch.mean(acc_score_train)})

        self.acc_score_train.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        loss = torch.nn.functional.cross_entropy(pred, y)
        self.log("loss/val", loss)

        # Accumulate metrics
        self.acc_score_val.update(pred, y)

        return loss

    def on_validation_epoch_end(self):
        acc_score_val = self.acc_score_val.compute()
        self.log_dict({"Acc/val_{}".format(i): score for i, score in enumerate(acc_score_val)})
        self.log_dict({"Acc/val_mean": torch.mean(acc_score_val)})

        self.acc_score_val.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        pred = self(x)
        pred = torch.argmax(pred, dim=-1)
        return pred

    def forward(self, x):
        pred = self.model(x)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        # optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        # scheduler = ExponentialLR(optimizer, gamma=0.9)

        return optimizer
