import torch
import torch.nn as nn
import pytorch_lightning as pl


class LightningLSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size):
        super(LightningLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.model = LSTM(input_size, hidden_size, output_size)
        self.criterion = nn.CrossEntropyLoss()

    def on_train_start(self):
        self.model.train()

    def training_step(self, batch, batch_index):
        x, x_length, y = batch
        out = self.model(x, x_length)

        loss = self.criterion(out, torch.argmax(y, 1))
        self.log('training_loss', loss.item(), prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.model.eval()

    def validation_step(self, batch, batch_index):
        x, x_length, y = batch
        out = self.model(x, x_length)

        loss = self.criterion(out, torch.argmax(y, 1))
        self.log("validation_loss", loss.item(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.output_layer = nn.Sequential(nn.Dropout(0.5),
                                          nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.5),
                                          nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.5),
                                          nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.5),
                                          nn.Linear(hidden_size, output_size))

    def forward(self, inputs, inputs_length):
        X = torch.nn.utils.rnn.pack_padded_sequence(inputs, inputs_length, batch_first=True, enforce_sorted=False)
        out, hidden = self.lstm(X)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # get last output
        last_seq_idxs = torch.LongTensor([len_idx - 1 for len_idx in inputs_length])
        last_output = X[range(X.shape[0]), last_seq_idxs, :]
        return self.output_layer(last_output)
