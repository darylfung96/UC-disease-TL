import matplotlib.pyplot as plt
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import random
import numpy as np
from copy import deepcopy
import torch
import os

from data_preprocessing import process_data
from dataset import MMCDataset
from model import LightningLSTM

os.makedirs('lightning_logs', exist_ok=True)
log_index = len(os.listdir('lightning_logs'))

# can be LSTM or CNNLSTM
model_type = "CNNLSTM"
is_pca = False
pad_in_sequence = True


def start_training(model_type, is_pca, pad_in_sequence):
    # pad the patient samples to 6 time stamp in sequence, if we are using LSTM we do not have to use this
    # this is only for CNN, we could experiment LSTM with this variable set as True too
    pca_text = 'PCA' if is_pca else ''
    pad_text = 'padded' if pad_in_sequence else ''

    if model_type == "CNNLSTM":
        pad_in_sequence = True
    sorted_data, sorted_length, target_data = process_data(pad_in_sequence=pad_in_sequence)

    # do PCA
    if is_pca:
        original_shape = sorted_data.shape
        sorted_data = np.reshape(sorted_data, [-1, 1015])
        pca = PCA(n_components=400)
        sorted_data = pca.fit_transform(sorted_data)
        sorted_data = np.reshape(sorted_data, [original_shape[0], original_shape[1], 400])

    random.seed(100)
    np.random.seed(100)
    torch.manual_seed(100)
    kf = KFold(n_splits=5)

    log_dict = {'validation f1': [], 'validation precision': [], 'validation recall': [], 'validation loss': [],
                'validation confusion matrix': []}
    for index, (train_index, test_index) in enumerate(kf.split(sorted_data)):
        random.seed(100)
        np.random.seed(100)
        torch.manual_seed(100)

        X_train = sorted_data[train_index]
        X_train_length = sorted_length[train_index]
        y_train = target_data[train_index]

        X_test = sorted_data[test_index]
        X_test_length = sorted_length[test_index]
        y_test = target_data[test_index]

        train_dataset = MMCDataset(X_train, X_train_length, y_train)
        test_dataset = MMCDataset(X_test, X_test_length, y_test)
        train_data_loader = DataLoader(train_dataset, batch_size=64)
        test_data_loader = DataLoader(test_dataset, batch_size=64)

        lightning_lstm = LightningLSTM(model=model_type, input_size=train_dataset[0][0].shape[1], hidden_size=32,
                                       output_size=train_dataset.y.shape[1])

        tb_logger = pl_loggers.TensorBoardLogger(f'lightning_logs/{log_index}_{pca_text}_{model_type}_{pad_text}')
        pl_trainer = Trainer(max_epochs=100, callbacks=[EarlyStopping(monitor='validation_loss', patience=6)],
                             checkpoint_callback=ModelCheckpoint('saved_model/model', monitor='validation_loss',
                                                                 save_top_k=1, prefix=f'kfold_{index}'),
                             logger=tb_logger)
        pl_trainer.fit(lightning_lstm, train_data_loader, test_data_loader)

        log_dict['validation f1'].append(lightning_lstm.log_dict['validation f1'])
        log_dict['validation precision'].append(lightning_lstm.log_dict['validation precision'])
        log_dict['validation recall'].append(lightning_lstm.log_dict['validation recall'])
        log_dict['validation loss'].append(lightning_lstm.log_dict['validation loss'])

        # get confusion matrix
        lightning_lstm.model.load_state_dict(lightning_lstm.best_state_dict)
        all_y = []
        all_predictions = []
        for batch in test_data_loader:
            x, x_length, y = batch
            out = lightning_lstm.model(x, x_length)

            predictions = out.argmax(1).cpu().detach().numpy()
            y_labels = y.argmax(1).cpu().detach().numpy()
            all_y += y_labels.tolist()
            all_predictions += predictions.tolist()

        conf_matrix = confusion_matrix(all_y, all_predictions, labels=[0, 1, 2, 3, 4])
        log_dict['validation confusion matrix'].append(conf_matrix)

    # save the mean value for the validation results
    mean_fold_values = {}
    for key, log_value in log_dict.items():
        # get validation confusion matrix
        if key == 'validation confusion matrix':
            mean_value = np.sum(np.array(log_value), 0)
            mean_fold_values[key] = mean_value
            continue

        # since we have early stopping, we want to keep track of this to create an average for plotting
        max_len = 0
        longest_value = None
        for log in log_value:
            if max_len < len(log):
                max_len = len(log)
                longest_value = log
        longest_value = np.array(longest_value)

        # replace the empty values (at the end) because not all folds have the same epoch to the longest epoch
        # since the longest epoch usually have the best performance because of early stopping
        for i in range(len(log_value)):
            temp = deepcopy(longest_value)
            temp[:len(log_value[i])] = log_value[i]
            log_value[i] = temp

        mean_value = np.mean(np.array(log_value), 0)
        mean_fold_values[key] = mean_value

    torch.save(mean_fold_values, f'plots for {pca_text}_{model_type}_{pad_text}.pth')

all_model_types = ["LSTM", "CNNLSTM"]
all_pcas = [True, False]
all_pads = [True, False]
for model_type in all_model_types:
    for is_pca in all_pcas:
        for pad_in_sequence in all_pads:
            start_training(model_type, is_pca, pad_in_sequence)


