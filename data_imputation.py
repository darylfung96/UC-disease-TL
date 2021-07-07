from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl

from dataset import MMCDataset
from data_preprocessing import process_data
from GAIN.gain import GAIN

output_dict = process_data(pad_in_sequence=True)
sorted_data, sorted_length, target_data, missing_data = output_dict['sorted_data'], output_dict['sorted_length'], \
                                                        output_dict['target_data'], output_dict['missing_data']
sorted_data = sorted_data.reshape(-1, sorted_data.shape[-1])
missing_data = missing_data.reshape(-1, missing_data.shape[-1])

dataset = MMCDataset(sorted_data, sorted_length, target_data)
train_data_loader = DataLoader(dataset)

gpus = None
if torch.cuda.is_available():
    gpus = 1
# model_checkpoint = ModelCheckpoint('gain_model_ckpt/best.ckpt', save_top_k=-1, period=5)
trainer = pl.Trainer(max_epochs=100, gpus=gpus)

gain = GAIN()
trainer.fit(gain, train_data_loader)
