from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import MMCDataset
from data_preprocessing import process_data
from GAIN.gain import GAIN

sorted_data, sorted_length, target_data = process_data(pad_in_sequence=True)
sorted_data = sorted_data.reshape(sorted_data.shape[0], -1)

dataset = MMCDataset(sorted_data, sorted_length, target_data)
train_data_loader = DataLoader(dataset)

model_checkpoint = ModelCheckpoint('gain_model_ckpt/best.ckpt', save_top_k=-1, period=5)
trainer = pl.Trainer(max_epochs=100, checkpoint_callback=model_checkpoint)

gain = GAIN()
trainer.fit(gain, train_data_loader)

