import pandas as pd

taxonomy_order_dict = {'kingdom': [], 'phylum': []. ''}

data = pd.read_csv('data/DIABIMMUNE_data_16s.csv')
samples_columns = [item.split('|') for item in list(data.columns)[1:]]
for samples_columns_index in range(len(samples_columns)):
    samples_columns[samples_columns_index] = \
        [item.split('__')[1] for item in samples_columns[samples_columns_index]]


...