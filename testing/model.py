import pytorch_lightning as pl
import torch

class TestLightningModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, x):
        output = self.model(x)
        return output


