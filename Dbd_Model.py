import pytorch_lightning as pl
import torchmetrics
import torch
import torchvision.models as models

class My_Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.encoder.eval()

        # freeze params
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=3)

    def build_encoder(self):
        # weights = models.MobileNet_V2_Weights.DEFAULT
        # return models.mobilenet_v2(weights=weights)
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        return models.mobilenet_v3_small(weights=weights)

    def build_decoder(self):
        return torch.nn.Sequential(
            torch.nn.Linear(1000, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 3),
            torch.nn.Softmax()
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        pred = self.decoder(z)

        loss = torch.nn.functional.cross_entropy(pred, y)
        self.accuracy(pred, y)

        self.log("train_loss", loss)
        self.log('train_accuracy', self.accuracy, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        z = self.encoder(x)
        pred = self.decoder(z)

        loss = torch.nn.functional.cross_entropy(pred, y)
        self.accuracy(pred, y)

        self.log("val_loss", loss)
        self.log('val_accuracy', self.accuracy, prog_bar=True, on_step=False, on_epoch=True)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-4)
        return optimizer
