import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, KFold, ShuffleSplit
from sklearn.metrics import confusion_matrix
import random
import numpy as np
import torch
import os

from data_preprocessing import dataset_list
from dataset import MMCDataset
from model import LightningLSTM
from self_distillation import LightningDistillation, FirstLightningDistillation, SecondLightningDistillation

os.makedirs('lightning_logs', exist_ok=True)
log_index = len(os.listdir('lightning_logs'))

# define dataset
# change parameters here
dataset_name = 'mmc7'
# imputed_type = None  # options: [None, 'GAIN', 'mean', 'mice']
imputed_npy_filename = 'data/imputed_data_mmc7.npy'
# taxonomy_order = 'phylum'  # [None, 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

current_dataset = dataset_list[dataset_name](dataset_name)
if dataset_name == 'mmc7':
    onehot_encode = True
else:
    onehot_encode = False

# can be LSTM or CNNLSTM
pca_components = 200
# model_type = "CNNLSTM"
# is_pca = False
pad_in_sequence = True


def start_training(output_dict, model_type, is_pca, pad_in_sequence, taxonomy_order=None, imputed_type=None,
                   prefix='', number_splits=10, load_model_filename=None, gradual_unfreezing=False, discr_fine_tune=False,
                   concat_pooling=False, self_distillation: LightningDistillation=None, attention=False, gpus=None):
    """

    :param output_dict:     Dictionary of the dataset information in the format of:
                            {'sorted_data': X, 'sorted_length: length of X, 'target_data': label,
                            'missing_data': missingNAN values in the dataset }

                            can get this information from ProcessDataset

    :param model_type:      LSTM, CNNLSTM (type of the model)
    :param is_pca:
    :param pad_in_sequence:
    :param prefix:          Add this name into the start of the model saved name or the result saved name
    :return:
    """
    if imputed_type is not None:  # ensure pad is True when imputed type is GAIN because GAIN pads the sequence
        pad_in_sequence = True
    gradual_unfreezing_text = 'g' if gradual_unfreezing else ''
    discr_fine_tune_text = 'df' if discr_fine_tune else ''
    concat_pooling_text = 'conpool' if concat_pooling else ''
    self_distillation_text = self_distillation.__class__.__name__ if self_distillation else ''
    attention_text = 'attention' if attention else ''

    # pad the patient samples to 6 time stamp in sequence, if we are using LSTM we do not have to use this
    # this is only for CNN, we could experiment LSTM with this variable set as True too
    pca_text = 'PCA' if is_pca else ''
    pad_text = 'padded' if pad_in_sequence else ''

    if model_type == "CNNLSTM":
        pad_in_sequence = True

    sorted_data, sorted_length, target_data = output_dict['sorted_data'], output_dict['sorted_length'], \
                                              output_dict['target_data']

    # do PCA
    if is_pca:
        original_shape = sorted_data.shape

        # don't do pca if the components is less than the number of pca components specified
        if original_shape[-1] < pca_components:
            return
        sorted_data = np.reshape(sorted_data, [-1, sorted_data.shape[2]])
        pca = PCA(n_components=pca_components, svd_solver='full')
        sorted_data = pca.fit_transform(sorted_data)
        sorted_data = np.reshape(sorted_data, [original_shape[0], original_shape[1], pca_components])

    random.seed(100)
    np.random.seed(100)
    torch.manual_seed(100)

    log_dict = {'validation f1': [], 'validation precision': [], 'validation recall': [], 'validation loss': [],
                'validation auc': [], 'validation confusion matrix': []}

    # kf = StratifiedKFold(n_splits=10, random_state=100)
    if number_splits == 1:
        kf = ShuffleSplit(n_splits=number_splits, test_size=0.1)
        kf.get_n_splits(sorted_data, target_data)
    else:
        kf = KFold(n_splits=number_splits, random_state=100)

    for index, (train_index, test_index) in enumerate(kf.split(sorted_data, target_data)):
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

        total_epoch = 100

        lightning_lstm = LightningLSTM(model=model_type, input_size=train_dataset[0][0].shape[1], max_inputs_length=train_dataset[0][0].shape[0], hidden_size=32,
                                       output_size=target_data.shape[1], load_model_filename=load_model_filename,
                                       gradual_unfreezing=gradual_unfreezing, discr_fine_tune=discr_fine_tune,
                                       concat_pooling=concat_pooling, self_distillation=self_distillation,
                                       attention=attention, total_epoch=total_epoch)

        tb_logger = pl_loggers.TensorBoardLogger(f'lightning_logs/{prefix}_{attention_text}_{self_distillation_text}_'
                                                 f'{gradual_unfreezing_text}_'
                                f'{discr_fine_tune_text}_{concat_pooling_text}_{prefix}_'
                                f'{dataset_name}_{taxonomy_order}_'
                                f'{pca_text}_{model_type}_{pad_text}_{imputed_type}')
        pl_trainer = Trainer(max_epochs=total_epoch,
                             checkpoint_callback=ModelCheckpoint('saved_model/model', monitor='validation_loss',
                                                                 save_top_k=1, prefix=f'kfold_{index}'),
                             logger=tb_logger, gpus=gpus)
        pl_trainer.fit(lightning_lstm, train_data_loader, test_data_loader)

        log_dict['validation f1'].append(lightning_lstm.log_dict['validation f1'])
        log_dict['validation precision'].append(lightning_lstm.log_dict['validation precision'])
        log_dict['validation recall'].append(lightning_lstm.log_dict['validation recall'])
        log_dict['validation loss'].append(lightning_lstm.log_dict['validation loss'])
        log_dict['validation auc'].append(lightning_lstm.log_dict['validation auc'])

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
    std_fold_values = {}
    for key, log_value in log_dict.items():
        # get validation confusion matrix
        if key == 'validation confusion matrix':
            mean_value = np.sum(np.array(log_value), 0)
            mean_fold_values[key] = mean_value
            continue

        # since we have early stopping, we want to keep track of this to create an average for plotting
        # max_len = 0
        # longest_value = None
        # for log in log_value:
        #     if max_len < len(log):
        #         max_len = len(log)
        #         longest_value = log
        # longest_value = np.array(longest_value)

        # replace the empty values (at the end) because not all folds have the same epoch to the longest epoch
        # since the longest epoch usually have the best performance because of early stopping

        # get the best value for each fold
        for i in range(len(log_value)):
            # temp = deepcopy(longest_value)
            # temp[:len(log_value[i])] = log_value[i]
            # log_value[i] = temp
            #
            best_fold_value = np.max(log_value[i])
            log_value[i] = best_fold_value

        mean_value = np.mean(np.array(log_value), 0)
        std_value = np.std(np.array(log_value))
        mean_fold_values[key] = mean_value
        std_fold_values[key] = std_value

    all_fold_values = {'mean': mean_fold_values, 'std': std_fold_values}
    torch.save(all_fold_values, f'plots/average F1 plots/plots for {attention_text}_'
                                f'{self_distillation_text}_{gradual_unfreezing_text}_'
                                f'{discr_fine_tune_text}_{concat_pooling_text}_{prefix}_'
                                f'{dataset_name}_{taxonomy_order}_'
                                f'{pca_text}_{model_type}_{pad_text}_{imputed_type}.pth')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--pad_in_sequence', action='store_true')
    parser.add_argument('--model_type', default='LSTM')
    parser.add_argument('--gpus', default=None)
    parser.add_argument('--taxonomy_order', default=None, type=str, choices=[None, 'kingdom', 'phylum',
                                                                             'class', 'order', 'family',
                                                                             'genus', 'species'])
    parser.add_argument('--imputed_type', default=None, type=str, choices=[None, 'GAIN', 'mean', 'mice'])
    args = parser.parse_args()

    if args.all:
        all_model_types = ["CNNLSTM", "LSTM"]
        all_pcas = [True, False]
        all_pads = [True, False]
        discr_fine_tunes = [False]
        gradual_unfreezings = [False]
        concat_poolings = [False]
        self_distillations = [None, FirstLightningDistillation(), SecondLightningDistillation()]
        imputed_types = [None, 'GAIN', 'mean', 'mice']
        attentions = [False]  # TODO change this to [True, False]
        os.makedirs('plots/average F1 plots', exist_ok=True)

        for pad_in_sequence in all_pads:
            for imputed_type in imputed_types:
                output_dict = current_dataset.process_data(pad_in_sequence=pad_in_sequence, imputer=imputed_type,
                                                           taxonomy_order=args.taxonomy_order)
                for attention in attentions:
                    for self_distillation in self_distillations:
                        for discr_fine_tune in discr_fine_tunes:
                            for gradual_unfreezing in gradual_unfreezings:
                                for concat_pooling in concat_poolings:
                                    for model_type in all_model_types:
                                        for is_pca in all_pcas:
                                                start_training(output_dict, model_type, is_pca, pad_in_sequence,
                                                               taxonomy_order=args.taxonomy_order, imputed_type=imputed_type,
                                                               prefix="",
                                                               discr_fine_tune=discr_fine_tune, gradual_unfreezing=gradual_unfreezing,
                                                               concat_pooling=concat_pooling, self_distillation=self_distillation, attention=attention, gpus=args.gpus)
    else:
        output_dict = current_dataset.process_data(pad_in_sequence=pad_in_sequence, imputer=args.imputed_type,
                                                   taxonomy_order=args.taxonomy_order)
        start_training(output_dict, args.model_type, args.pca, args.pad_in_sequence,
                       taxonomy_order=args.taxonomy_order, imputed_type=args.imputed_type, prefix="")
