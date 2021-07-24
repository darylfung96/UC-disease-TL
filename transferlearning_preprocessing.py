import pandas as pd

taxonomy_order_dict = {'kingdom': {}, 'phylum': {}, 'class': {}, 'order': {}, 'family': {}, 'genus': {}}
index_to_order = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']

data = pd.read_csv('data/DIABIMMUNE_data_16s.csv')
samples_columns = [item.split('|') for item in list(data.columns)[1:]]
for samples_columns_index in range(len(samples_columns)):
    samples_columns[samples_columns_index] = \
        [item.split('__')[1] for item in samples_columns[samples_columns_index]]


for sample_columns in samples_columns:
    for idx, order in enumerate(sample_columns):
        if taxonomy_order_dict.get(index_to_order[idx], None) is not None:
            taxonomy_order_dict[index_to_order[idx]] += 1
        else:
            taxonomy_order_dict[index_to_order[idx]] = 1
...