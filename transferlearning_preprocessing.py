from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

diabimmune_taxonomy_order_dict = {'kingdom': {}, 'phylum': {}, 'class': {}, 'order': {}, 'family': {}, 'genus': {}}
mmc_taxonomy_order_dict = {'kingdom': {}, 'phylum': {}, 'class': {}, 'order': {}, 'family': {}, 'genus': {}, 'species': {}}
index_to_order = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']


def process_columns(columns, order_separate='|', feature_separate='__'):
    """

    An example of a column would be k__bacteria|p__etc|c__etc|o__etc|f__etc|g__etc|s__etc
    The order separate in this example would be |
    The feature separate in this example is __

    :param samples_columns:
    :param order_separate:
    :param feature_separate:
    :return:
    """
    taxonomy_dictionary = {}

    samples_columns = [item.split(order_separate) for item in columns]

    for samples_columns_index in range(len(samples_columns)):
        samples_columns[samples_columns_index] = \
            [item.split(feature_separate)[1] for item in samples_columns[samples_columns_index]]

    # get the count of each order
    for sample_columns in samples_columns:
        for idx, order in enumerate(sample_columns):
            if taxonomy_dictionary[index_to_order[idx]].get(order, None) is not None:
                taxonomy_dictionary[index_to_order[idx]][order] += 1
            else:
                taxonomy_dictionary[index_to_order[idx]][order] = 1

    return taxonomy_dictionary


def check_intersecting_diab_mmc():
    data = pd.read_csv('data/DIABIMMUNE_data_16s.csv')
    # split | and __ to get the order values
    diabimmune_taxonomy_order_dict = process_columns(list(data.columns)[1:], '|', '__')

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


check_intersecting_diab_mmc()