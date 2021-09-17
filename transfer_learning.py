import os

from data_preprocessing import dataset_list, find_intersecting_bacteria
from main import start_training


imputer = 'mice'  # [None, 'GAIN', 'mean', 'mice']
taxonomy_order = 'phylum'
model_type = 'LSTM'
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
    pad_in_sequence = True

pcas = [True, False]
all_imputers = [None, 'GAIN', 'mean', 'mice']
discr_fine_tunes = [True, False]
gradual_unfreezings = [True, False]
concat_poolings = [True, False]
model_types = ['CNNLSTM']

for model_type in model_types:
    for concat_pooling in concat_poolings:
        for discr_fine_tune in discr_fine_tunes:
            for gradual_unfreezing in gradual_unfreezings:
                for imputer in all_imputers:
                    for pca in pcas:
                        start_training(output_dict, model_type,
                                       is_pca=pca, pad_in_sequence=pad_in_sequence, taxonomy_order='phylum',
                                       imputed_type=imputer, prefix='transfer_mmc7_to_allergy', number_splits=10,
                                       load_model_filename='saved_model/mmc7_pretrained.ckpt',
                                       gradual_unfreezing=gradual_unfreezing, discr_fine_tune=discr_fine_tune,
                                       concat_pooling=concat_pooling)
