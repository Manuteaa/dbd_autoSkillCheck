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

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=3, average=None)

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
        acc_0, acc_1, acc_2 = self.accuracy(pred, y)

        self.log("loss/train", loss)
        self.log('acc/train_0', acc_0, prog_bar=True, on_step=False, on_epoch=True)
        self.log('acc/train_1', acc_1, prog_bar=True, on_step=False, on_epoch=True)
        self.log('acc/train_2', acc_2, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        pred = self(x)

        loss = torch.nn.functional.cross_entropy(pred, y)
        acc_0, acc_1, acc_2 = self.accuracy(pred, y)

        self.log("loss/val", loss)
        self.log('acc/val_0', acc_0, prog_bar=True, on_step=False, on_epoch=True)
        self.log('acc/val_1', acc_1, prog_bar=True, on_step=False, on_epoch=True)
        self.log('acc/val_2', acc_2, prog_bar=True, on_step=False, on_epoch=True)

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
