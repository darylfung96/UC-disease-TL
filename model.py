from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class LightningLSTM(pl.LightningModule):
    def __init__(self, model, input_size, hidden_size, output_size):
        super(LightningLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.model = model_dict[model](input_size, hidden_size, output_size)
        self.criterion = nn.CrossEntropyLoss()

        self.log_dict = {'validation f1': [], 'validation precision': [], 'validation recall': [], 'validation loss': []}

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

        predictions = out.argmax(1).cpu().detach().numpy()
        y_labels = y.argmax(1).cpu().detach().numpy()
        f1 = f1_score(y_labels, predictions, average='micro')
        precision = precision_score(y_labels, predictions, average='micro')
        recall = recall_score(y_labels, predictions, average='micro')

        self.log_dict['validation loss'].append(loss.item())
        self.log_dict['validation f1'].append(f1)
        self.log_dict['validation precision'].append(precision)
        self.log_dict['validation recall'].append(recall)

        for key, value in self.log_dict.items():
            self.log(key, value[-1], prog_bar=True)
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


class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CNNLSTM, self).__init__()

        self.conv1 = nn.Conv1d(input_size, 500, 2)
        self.bn1 = nn.BatchNorm1d(500)
        self.conv2 = nn.Conv1d(500, 500, 2)

        self.lstm = nn.LSTM(input_size=500, hidden_size=hidden_size, batch_first=True)
        self.output_layer = nn.Sequential(nn.Dropout(0.5),
                                          nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.5),
                                          nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.5),
                                          nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.5),
                                          nn.Linear(hidden_size, output_size))

    def forward(self, inputs, inputs_length):
        conv1_output = F.relu(self.bn1(self.conv1(inputs.transpose(1, 2))))
        conv2_output = F.relu(self.conv2(conv1_output))

        out, hidden = self.lstm(conv2_output.transpose(1, 2))
        last_output = out[:, -1, :]
        return self.output_layer(last_output)


model_dict = {
    'LSTM': LSTM,
    'CNNLSTM': CNNLSTM
}