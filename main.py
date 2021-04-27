from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import random
import numpy as np
import torch
import os

from data_preprocessing import process_data
from dataset import MMCDataset
from model import LightningLSTM

log_index = len(os.listdir('lightning_logs'))

sorted_data, sorted_length, target_data = process_data()

kf = KFold(n_splits=5)
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

    lightning_lstm = LightningLSTM(input_size=train_dataset[0][0].shape[1], hidden_size=32, output_size=train_dataset.y.shape[1])

    tb_logger = pl_loggers.TensorBoardLogger(f'lightning_logs/{log_index}')
    pl_trainer = Trainer(max_epochs=100, callbacks=[EarlyStopping(monitor='validation_loss', patience=6)],
                                                    checkpoint_callback=ModelCheckpoint('saved_model/model', monitor='validation_loss',
                                                                                        save_top_k=1, prefix=f'kfold_{index}'),
                         logger=tb_logger)
    pl_trainer.fit(lightning_lstm, train_data_loader, test_data_loader)
