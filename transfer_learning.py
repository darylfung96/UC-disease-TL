import os

from data_preprocessing import dataset_list, find_intersecting_bacteria
from main import start_training


imputer = None  # [None, 'GAIN', mean', mice]
taxonomy_order = 'phylum'
model_type = 'CNNLSTM'
pad_in_sequence = True

allergy_dataset = dataset_list['allergy']('allergy')
mmc7_dataset = dataset_list['mmc7']('mmc7')
allergy_intersecting_dict, mmc7_intersecting_dict = find_intersecting_bacteria(allergy_dataset, mmc7_dataset)
allergy_dataset.process_data(pad_in_sequence=True, imputer=imputer, taxonomy_order='phylum')
output_dict = allergy_dataset.get_selected_features(allergy_intersecting_dict)


# all_model_types = ["LSTM", "CNNLSTM"]
# all_pcas = [True, False]
# all_pads = [True, False]
# os.makedirs('plots/average F1 plots', exist_ok=True)
# for model_type in all_model_types:
#     for is_pca in all_pcas:
#         for pad_in_sequence in all_pads:
#             start_training(output_dict, model_type, is_pca, pad_in_sequence, taxonomy_order=taxonomy_order,
#                            imputed_type=imputer, prefix="mmc7_transfer")

if model_type == 'CNNLSTM':
    pad_in_sequence=  True
start_training(output_dict, model_type,
               is_pca=False, pad_in_sequence=pad_in_sequence, taxonomy_order='phylum',
               imputed_type=None, prefix='best_phylum_mmc7', number_splits=1,
               load_model_filename='saved_model/mmc7_pretrained.ckpt')
