from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

diabimmune_taxonomy_order_dict = {'kingdom': {}, 'phylum': {}, 'class': {}, 'order': {}, 'family': {}, 'genus': {}}
mmc_taxonomy_order_dict = {'kingdom': {}, 'phylum': {}, 'class': {}, 'order': {}, 'family': {}, 'genus': {}, 'species': {}}
index_to_order = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']


def check_intersecting_diab_mmc():
    data = pd.read_csv('data/DIABIMMUNE_data_16s.csv')
    # split | and __ to get the order values
    samples_columns = [item.split('|') for item in list(data.columns)[1:]]
    for samples_columns_index in range(len(samples_columns)):
        samples_columns[samples_columns_index] = \
            [item.split('__')[1] for item in samples_columns[samples_columns_index]]

    # get the count of each order
    for sample_columns in samples_columns:
        for idx, order in enumerate(sample_columns):
            if diabimmune_taxonomy_order_dict[index_to_order[idx]].get(order, None) is not None:
                diabimmune_taxonomy_order_dict[index_to_order[idx]][order] += 1
            else:
                diabimmune_taxonomy_order_dict[index_to_order[idx]][order] = 1

    mmc_data = pd.read_csv('data/mmc7.csv')
    mmc_columns = [item.split('..') for item in list(mmc_data.columns)[24:]]
    for mmc_columns_index in range(len(mmc_columns)):
        mmc_columns[mmc_columns_index] = \
            [item.split('__')[1] for item in mmc_columns[mmc_columns_index][:-1]]

    for sample_columns in mmc_columns:
        for idx, order in enumerate(sample_columns):
            # if empty categorized, then we skip
            if order == '':
                continue

            if mmc_taxonomy_order_dict[index_to_order[idx]].get(order, None) is not None:
                mmc_taxonomy_order_dict[index_to_order[idx]][order] += 1
            else:
                mmc_taxonomy_order_dict[index_to_order[idx]][order] = 1

    for order in index_to_order:
        intersecting = set(diabimmune_taxonomy_order_dict[order].keys()) & set(mmc_taxonomy_order_dict[order].keys())
        non_intersecting = set(diabimmune_taxonomy_order_dict[order].keys()) ^ set(
            mmc_taxonomy_order_dict[order].keys())
        all_order = set.union(set(diabimmune_taxonomy_order_dict[order].keys()),
                              set(mmc_taxonomy_order_dict[order].keys()))
        print('intersecting:')
        print(intersecting)
        print('non intersecting:')
        print(non_intersecting)
        print(f'{order}: {len(intersecting)}/{len(all_order)}')

