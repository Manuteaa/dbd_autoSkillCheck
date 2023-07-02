import pytorch_lightning as pl
import torchmetrics
import torch
import torchvision.models as models

class Model(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.lr = lr

        self.precision_train = torchmetrics.Precision(task='multiclass', num_classes=3, average=None)
        self.recall_train = torchmetrics.Recall(task='multiclass', num_classes=3, average=None)

        self.precision_val = torchmetrics.Precision(task='multiclass', num_classes=3, average=None)
        self.recall_val = torchmetrics.Recall(task='multiclass', num_classes=3, average=None)

    def build_encoder(self):
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        encoder = models.mobilenet_v3_large(weights=weights)

        # weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        # encoder = models.convnext_tiny(weights=weights)

        # Freeze encoder
        for param in encoder.parameters():
            param.requires_grad = False

        return encoder

    def build_decoder(self):
        return torch.nn.Sequential(
            torch.nn.Linear(1000, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 3),
            # torch.nn.Softmax()  # Use logits instead
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        loss = torch.nn.functional.cross_entropy(pred, y)
        pres_0, pres_1, pres_2 = self.precision_train(pred, y)
        rec_0, rec_1, rec_2 = self.recall_train(pred, y)

        self.log("loss/train", loss)
        self.log_dict({
                  'pres/train_0': pres_0, 'pres/train_1': pres_1, 'pres/train_2': pres_2,
                  'rec/train_0': rec_0, 'rec/train_1': rec_1, 'rec/train_2': rec_2
        })

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        loss = torch.nn.functional.cross_entropy(pred, y)
        pres_0, pres_1, pres_2 = self.precision_val(pred, y)
        rec_0, rec_1, rec_2 = self.recall_val(pred, y)

        self.log("loss/val", loss)
        self.log_dict({
                    'pres/val_0': pres_0, 'pres/val_1': pres_1, 'pres/val_2': pres_2,
                    'rec/val_0': rec_0, 'rec/val_1': rec_1, 'rec/val_2': rec_2
        })

        return loss

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        return optimizer
