import pytorch_lightning as pl
import logging as log

class TestLightningModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        return output


