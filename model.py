from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve, auc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class LightningLSTM(pl.LightningModule):
    def __init__(self, model, input_size, hidden_size, output_size, load_model_filename=None):
        super(LightningLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.model = model_dict[model](input_size, hidden_size, output_size)
        self.criterion = nn.BCELoss()

        self.log_dict = {'validation f1': [], 'validation precision': [], 'validation recall': [], 'validation loss': [], 'validation auc': []}
        self.step_log_dict = {'validation f1': [], 'validation precision': [], 'validation recall': [], 'validation loss': [], 'validation auc': []}
        self.best_state_dict = None
        self.best_f1 = 0

        if load_model_filename is not None:
            pretrained_dict = torch.load(load_model_filename)['state_dict']
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.state_dict() and self.state_dict()[k].shape == v.shape}
            self.load_state_dict(pretrained_dict)

    def on_train_start(self):
        self.model.train()

    def training_step(self, batch, batch_index):
        x, x_length, y = batch
        out = self.model(x, x_length)

        loss = self.criterion(out, y)
        self.log('training_loss', loss.item(), prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.model.eval()
        self.step_log_dict = {'validation f1': [], 'validation precision': [], 'validation recall': [],
                              'validation loss': [], 'validation auc': []}

    def validation_step(self, batch, batch_index):
        x, x_length, y = batch
        out = self.model(x, x_length)

        loss = self.criterion(out, y)
        self.log("validation_loss", loss.item(), prog_bar=True)

        fpr, tpr, thresholds = roc_curve(y.cpu().detach().reshape(-1), out.cpu().detach().reshape(-1))
        auc_value = auc(fpr, tpr)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        predictions = out.cpu().detach()
        predictions[ predictions > optimal_threshold] = 1
        predictions[ predictions <= optimal_threshold] = 0

        f1 = f1_score(y, predictions, average='micro')
        precision = precision_score(y, predictions, average='micro')
        recall = recall_score(y, predictions, average='micro')

        self.step_log_dict['validation loss'].append(loss.item())
        self.step_log_dict['validation f1'].append(f1)
        self.step_log_dict['validation precision'].append(precision)
        self.step_log_dict['validation recall'].append(recall)
        self.step_log_dict['validation auc'].append(auc_value)

        return loss

    def on_validation_epoch_end(self):
        # log the mean of metrics in this epoch
        for key, value in self.step_log_dict.items():
            current_mean = np.mean(np.array(value))
            self.log_dict[key].append(current_mean)

        #TODO error when logging in validation epoch end
        # log to tensorboard
        # for key, value in self.log_dict.items():
        #     self.log(key, value[-1], prog_bar=True)

        if self.best_state_dict is None:
            self.best_f1 = self.log_dict['validation f1'][-1]
            self.best_state_dict = self.model.state_dict()

        # keep the best weight based on the best f1 epoch
        if self.log_dict['validation f1'][-1] > self.best_f1:
            self.best_f1 = self.log_dict['validation f1'][-1]
            self.best_state_dict = self.model.state_dict()

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
        return F.sigmoid(self.output_layer(last_output))


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
        return F.sigmoid(self.output_layer(last_output))


model_dict = {
    'LSTM': LSTM,
    'CNNLSTM': CNNLSTM
}