import pytorch_lightning as pl
import torchmetrics
import torch
import torchvision.models as models

class Model(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.example_input_array = torch.zeros((32, 3, 224, 224), dtype=torch.float32)
        self.nb_classes = 8

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.lr = lr

        self.metrics_train = torchmetrics.MetricCollection([
            torchmetrics.Precision(task='multiclass', num_classes=self.nb_classes, average=None),
            torchmetrics.Recall(task='multiclass', num_classes=self.nb_classes, average=None)
        ])

        self.metrics_val = torchmetrics.MetricCollection([
            torchmetrics.Precision(task='multiclass', num_classes=self.nb_classes, average=None),
            torchmetrics.Recall(task='multiclass', num_classes=self.nb_classes, average=None)
        ])

    def build_encoder(self):
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        encoder = models.mobilenet_v3_large(weights=weights)

        # weights = models.MNASNet0_5_Weights.DEFAULT
        # encoder = models.mnasnet0_5(weights=weights)

        # Freeze encoder
        # for param in encoder.parameters():
        #     param.requires_grad = False

        return encoder

    def build_decoder(self):
        return torch.nn.Sequential(
            torch.nn.Linear(1000, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, self.nb_classes),
            # torch.nn.Softmax()  # Use logits instead
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        loss = torch.nn.functional.cross_entropy(pred, y)

        self.log("loss/train", loss)
        self.metrics_train.update(pred, y)
        return loss

    def on_train_epoch_end(self):
        results = self.metrics_train.compute()

        metrics_pres = {"pres/train_{}".format(i): score for i, score in enumerate(results['MulticlassPrecision'])}
        metrics_rec = {"rec/train_{}".format(i): score for i, score in enumerate(results['MulticlassRecall'])}
        metrics_pres.update(metrics_rec)

        self.log_dict(metrics_pres)
        self.metrics_train.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        loss = torch.nn.functional.cross_entropy(pred, y)

        self.log("loss/val", loss)
        self.metrics_val.update(pred, y)
        return loss

    def on_validation_epoch_end(self):
        results = self.metrics_val.compute()

        metrics_pres = {"pres/val_{}".format(i): score for i, score in enumerate(results['MulticlassPrecision'])}
        metrics_rec = {"rec/val_{}".format(i): score for i, score in enumerate(results['MulticlassRecall'])}
        metrics_pres.update(metrics_rec)

        self.log_dict(metrics_pres)
        self.metrics_val.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        pred = self(x)
        pred = torch.argmax(pred, dim=-1)
        return pred

    def forward(self, x):
        z = self.encoder(x)
        pred = self.decoder(z)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
