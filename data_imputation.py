from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
import argparse
import numpy as np

from dataset import MMCDataset
from data_preprocessing import dataset_list
from GAIN.gain import GAIN

dataset = 'david'
current_dataset = dataset_list[dataset]()

output_dict = current_dataset.process_data(pad_in_sequence=True)
sorted_data, sorted_length, target_data, missing_data = output_dict['sorted_data'], output_dict['sorted_length'], \
                                                        output_dict['target_data'], output_dict['missing_data']
original_shape = sorted_data.shape
sorted_data = sorted_data.reshape(-1, original_shape[-1])
missing_data = missing_data.reshape(-1, original_shape[-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_eval', action="store_true")
    parser.add_argument('--save_data_path', type=str, default='data/imputed_data.npy')
    args = parser.parse_args()

    if args.is_eval:
        print('run evaluation')
        gain = GAIN(dataset)
        gain.load()

        masking = 1 - missing_data
        imputed_data = gain.generate(sorted_data, masking)
        imputed_data = imputed_data.cpu().detach().numpy()
        imputed_data = sorted_data * masking + (1 - masking) * imputed_data
        imputed_data = imputed_data.reshape(*original_shape)
        np.save(args.save_data_path, imputed_data)
        print(f'saved imputed data to {args.save_data_path}')

    else:
        dataset = MMCDataset(sorted_data, sorted_length, target_data)
        train_data_loader = DataLoader(dataset)

        gpus = None
        if torch.cuda.is_available():
            gpus = 1
        # model_checkpoint = ModelCheckpoint('gain_model_ckpt/best.ckpt', save_top_k=-1, period=5)
        trainer = pl.Trainer(max_epochs=100, gpus=gpus)

        gain = GAIN()
        trainer.fit(gain, train_data_loader)