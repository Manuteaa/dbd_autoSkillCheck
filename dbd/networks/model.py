import pytorch_lightning as pl
import torchmetrics
import torch
import torchvision.models as models

class Model(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=3)

    def build_encoder(self):
        # weights = models.MobileNet_V2_Weights.DEFAULT
        # encoder = models.mobilenet_v2(weights=weights)

        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        encoder = models.convnext_tiny(weights=weights)

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
        self.accuracy(pred, y)

        self.log("train_loss", loss)
        self.log('train_accuracy', self.accuracy, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        pred = self(x)

        loss = torch.nn.functional.cross_entropy(pred, y)
        self.accuracy(pred, y)

        self.log("val_loss", loss)
        self.log('val_accuracy', self.accuracy, prog_bar=True, on_step=False, on_epoch=True)

    def forward(self, x):
        z = self.encoder(x)
        pred = self.decoder(z)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
