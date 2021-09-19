from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve, auc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List, Any

from self_distillation import LightningDistillation, FirstLightningDistillation


class LightningLSTM(pl.LightningModule):
    def __init__(self, model, input_size, hidden_size, output_size, max_inputs_length, load_model_filename=None, gradual_unfreezing=False,
                 discr_fine_tune=False, concat_pooling=False,
                 self_distillation: LightningDistillation = FirstLightningDistillation, attention=False, total_epoch=100):
        super(LightningLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.total_epoch = total_epoch

        self.model = model_dict[model](input_size, hidden_size, output_size, max_inputs_length, concat_pooling, self_distillation, attention)
        self.criterion = nn.BCELoss()

        self.log_dict = {'validation f1': [], 'validation precision': [], 'validation recall': [], 'validation loss': [], 'validation auc': []}
        self.step_log_dict = {'validation f1': [], 'validation precision': [], 'validation recall': [], 'validation loss': [], 'validation auc': []}
        self.best_state_dict = None
        self.best_f1 = 0
        if load_model_filename is not None:
            pretrained_dict = torch.load(load_model_filename)['state_dict']
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.state_dict() and self.state_dict()[k].shape == v.shape}
            self.load_state_dict(pretrained_dict, strict=False)

        # gradual unfreezing parameters
        self.gradual_unfreezing = gradual_unfreezing
        if gradual_unfreezing:
            self.freeze_all_layers()
            self.current_unfrozen_layer = 0
        self.discr_fine_tune = discr_fine_tune
        self.concat_pooling = concat_pooling

        # for self distillation
        self.self_distillation = self_distillation
        if self.self_distillation is not None:
            distill_args = {'total_epoch': total_epoch}
            self.self_distillation.init(distill_args, self.model)

    def _unfreeze_next_layer(self):
        list(reversed(list(self.parameters())))[self.current_unfrozen_layer].requires_grad = True
        self.current_unfrozen_layer += 1

    def freeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = False

    def on_train_start(self):
        self.model.train()
        if self.gradual_unfreezing:
            self._unfreeze_next_layer()

    def on_train_epoch_start(self):
        args = {'current_epoch': self.current_epoch, 'model': self.model}
        if self.self_distillation is not None:
            self.self_distillation.on_train_epoch_start(args)

    def training_step(self, batch, batch_index):
        x, x_length, y = batch
        out = self.model(x, x_length)

        loss = self.criterion(out, y)
        if self.self_distillation is not None:
            args = {'loss': loss, 'x': x, 'x_length': x_length, 'y': y,
                    'main_pre_output_layer_features': self.model.main_pre_output_layer_features,
                    'post_main_output_layer': self.model.post_main_output_layer, 'model': self.model}
            loss = self.self_distillation.step(args)

        self.log('training_loss', loss.item(), prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        if self.self_distillation is not None:
            self.self_distillation.on_train_epoch_end(self.model)

    def on_validation_epoch_start(self):
        self.model.eval()
        self.step_log_dict = {'validation f1': [], 'validation precision': [], 'validation recall': [],
                              'validation loss': [], 'validation auc': []}

    def validation_step(self, batch, batch_index):
        x, x_length, y = batch
        out = self.model(x, x_length)

        loss = self.criterion(out, y)
        if self.self_distillation is not None:
            args = {'loss': loss, 'x': x, 'x_length': x_length,
                    'y': y, 'main_pre_output_layer_features': self.model.main_pre_output_layer_features,
                    'post_main_output_layer': self.model.post_main_output_layer, 'model': self.model}
            loss = self.self_distillation.step(args)
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
        if not self.discr_fine_tune:
            return torch.optim.Adam(self.model.parameters(), lr=1e-3)

        lr = 1e-3
        params = []
        for current_param in list(reversed(list(self.model.parameters()))):
            params.append({"params": current_param, "lr": lr})
            lr /= 2.6
        return torch.optim.Adam(params)


class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()

    def forward(self, inputs):
        inputs = inputs.reshape(inputs.shape[0], -1)
        return inputs


class BackBone(nn.Module):
    def __init__(self, max_inputs_length):
        super(BackBone, self).__init__()
        self.feature_loss = nn.L1Loss()
        self.max_inputs_length = max_inputs_length

    def _create_layer_block(self, hidden_size, output_size, attention=False):

        modules = [nn.Dropout(0.5)]
        if attention:
            modules.append(nn.TransformerEncoderLayer(hidden_size, 8, hidden_size, dropout=0.5))
            modules.append(Reshape())

        if attention is False:
            first_hidden_size = hidden_size
        else:
            first_hidden_size = self.max_inputs_length * hidden_size

        modules += [
            nn.Linear(first_hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.5)
        ]
        pre_layer = nn.Sequential(*modules)
        output_layer = nn.Sequential(nn.Linear(hidden_size, output_size))
        return pre_layer, output_layer

    def kd_loss_function(self, output, target_output, temperature=3):
        """Compute kd loss"""
        """
        para: output: middle ouptput logits.
        para: target_output: final output has divided by temperature and softmax.
        """

        output = output / temperature
        output_log_softmax = torch.log_softmax(output, dim=1)
        loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
        return loss_kd

    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss).sum()


class LSTM(BackBone):
    def __init__(self, input_size, hidden_size, output_size, max_inputs_length, concat_pooling=False,
                 self_distillation: LightningDistillation = FirstLightningDistillation, attention=False):
        super(LSTM, self).__init__(max_inputs_length)
        self.concat_pooling = concat_pooling
        self.self_distillation = self_distillation
        self.attention = attention

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        hidden_size = hidden_size
        if self.concat_pooling:
            hidden_size = 3 * hidden_size
        self.output_layer = nn.Sequential(nn.Dropout(0.5),
                                          nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.5),
                                          nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.5),
                                          nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.5),
                                          nn.Linear(hidden_size, output_size))

        if self.self_distillation is not None:
            distillation_args = {'hidden_size': hidden_size, 'output_size': output_size, 'attention': attention,
                                 'create_layer_block': self._create_layer_block}
            self.self_distillation.init(distillation_args, None)
        self.main_pre_output_layer, self.main_output_layer = self._create_layer_block(hidden_size, output_size, attention)

        # outputs
        self.pre_output_layer1_features = None
        self.pre_output_layer2_features = None
        self.pre_output_layer3_features = None
        self.pre_output_layer4_features = None

    def forward(self, inputs, inputs_length):
        X = torch.nn.utils.rnn.pack_padded_sequence(inputs, inputs_length, batch_first=True, enforce_sorted=False)
        out, hidden = self.lstm(X)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        if self.attention:
            last_output = X
        else:
            # get last output
            last_seq_idxs = torch.LongTensor([len_idx - 1 for len_idx in inputs_length])
            last_output = X[range(X.shape[0]), last_seq_idxs, :]

            # concat pooling
            if self.concat_pooling:
                concat_pooling_output = None
                for i in range(X.shape[0]):
                    current_output = X[i:i+1, last_seq_idxs[i]:last_seq_idxs[i]+1, :]
                    before_mean_output = X[i:i+1, :last_seq_idxs[i], :]
                    mean_output = torch.mean(before_mean_output, 1, keepdim=True)
                    max_output = torch.max(before_mean_output, 1, keepdim=True)[0]
                    output = torch.cat([current_output, mean_output, max_output], 1)

                    if concat_pooling_output is None:
                        concat_pooling_output = output
                    else:
                        concat_pooling_output = torch.cat([concat_pooling_output, output], 0)
                last_output = concat_pooling_output
                last_output = last_output.reshape(last_output.shape[0], -1)

        # self distillation
        if self.self_distillation is not None:
            self.self_distillation.forward(last_output)

        self.main_pre_output_layer_features = self.main_pre_output_layer(last_output)
        main_output_layer = self.main_output_layer(self.main_pre_output_layer_features)
        self.post_main_output_layer = F.sigmoid(main_output_layer)

        return self.post_main_output_layer


class CNNLSTM(BackBone):
    def __init__(self, input_size, hidden_size, output_size, max_inputs_length, concat_pooling=False,
                 self_distillation: LightningDistillation=FirstLightningDistillation, attention=False):
        super(CNNLSTM, self).__init__(max_inputs_length)
        self.max_inputs_length -= 2  # TODO change this to be variable instead of fix because of attention
        self.concat_pooling = concat_pooling
        self.self_distillation = self_distillation
        self.attention = attention

        self.conv1 = nn.Conv1d(input_size, 500, 2)
        self.bn1 = nn.BatchNorm1d(500)
        self.conv2 = nn.Conv1d(500, 500, 2)

        self.lstm = nn.LSTM(input_size=500, hidden_size=hidden_size, batch_first=True)

        hidden_size = hidden_size
        if concat_pooling:
            hidden_size = 3 * hidden_size

        if self.self_distillation is not None:
            distillation_args = {'hidden_size': hidden_size, 'output_size': output_size, 'attention': attention,
                                 'create_layer_block': self._create_layer_block}
            self.self_distillation.init(distillation_args, None)
        self.main_pre_output_layer, self.main_output_layer = self._create_layer_block(hidden_size, output_size, attention)

        # outputs
        self.pre_output_layer1_features = None
        self.pre_output_layer2_features = None
        self.pre_output_layer3_features = None
        self.pre_output_layer4_features = None
        self.main_pre_output_layer_features = None

    def forward(self, inputs, inputs_length):
        conv1_output = F.relu(self.bn1(self.conv1(inputs.transpose(1, 2))))
        conv2_output = F.relu(self.conv2(conv1_output))

        out, hidden = self.lstm(conv2_output.transpose(1, 2))

        if self.attention:
            last_output = out
        else:
            last_output = out[:, -1, :]
            # concat pooling
            if self.concat_pooling:
                last_output = torch.cat([out[:, -1:, :], torch.mean(out, 1, keepdim=True),
                                         torch.max(out, 1, keepdim=True)[0]], 1)
                last_output = last_output.reshape(last_output.shape[0], -1)

        # self distillation
        if self.self_distillation is not None:
            self.self_distillation.forward(last_output)

        self.main_pre_output_layer_features = self.main_pre_output_layer(last_output)
        main_output_layer = self.main_output_layer(self.main_pre_output_layer_features)
        self.post_main_output_layer = F.sigmoid(main_output_layer)

        return self.post_main_output_layer


model_dict = {
    'LSTM': LSTM,
    'CNNLSTM': CNNLSTM
}