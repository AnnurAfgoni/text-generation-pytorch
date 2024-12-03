import torch
import torch.nn as nn
import lightning as L

from omegaconf import DictConfig
from hydra.utils import instantiate


class CharRNN(nn.Module):
    def __init__(self, tokens, n_hidden=256, n_layers=2, drop_prob=0.5):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.chars = tokens

        self.lstm = nn.LSTM(
            len(self.chars), n_hidden, n_layers, dropout=drop_prob, batch_first=True
        )
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, len(self.chars))

    def forward(self, x, hidden):
        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output)
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
            weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
        )
        return hidden


class CharRNNModule(L.LightningModule):
    def __init__(
        self, 
        model: DictConfig, 
        loss_fn: DictConfig,
        optimizer: DictConfig,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = instantiate(model)
        self.loss_fn = instantiate(loss_fn)
        self.optimizer = optimizer

    def forward(self, x, hidden):
        return self.model(x, hidden)

    def training_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.size(0)
        hidden = self.model.init_hidden(batch_size)
        hidden = tuple([h.to(self.device) for h in hidden])

        # Forward pass
        y_hat, hidden = self(x, hidden)
        y = y.view(-1, self.model.fc.out_features)

        # Compute loss
        loss = self.loss_fn(y_hat, y)

        # Log training loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.size(0)
        hidden = self.model.init_hidden(batch_size)
        hidden = tuple([h.to(self.device) for h in hidden])

        # Forward pass
        y_hat, hidden = self(x, hidden)
        y = y.view(-1, self.model.fc.out_features)

        # Compute loss
        val_loss = self.loss_fn(y_hat, y)

        # Log validation loss
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = instantiate(
            self.optimizer, 
            params=self.parameters(), 
            _convert_="partial"
        )
        return optimizer
