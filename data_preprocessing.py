from abc import ABC, abstractmethod, abstractproperty
import collections
import numpy as np
from copy import deepcopy
import pandas as pd
import pickle
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

# samples in sequence (this is used to check to make sure that the subjects are in sequence)
SAMPLES_SEQUENCE = [['biopsy', 0], ['stool', 0], ['stool', 4], ['stool', 12], ['biopsy', 52], ['stool', 52]]
missing_num_samples = {'biopsy_0': 0, 'stool_0': 0, 'stool_4': 0, 'stool_12': 0, 'biopsy_52': 0, 'stool_52': 0}
total_num_samples = {'biopsy_0': 0, 'stool_0': 0, 'stool_4': 0, 'stool_12': 0, 'biopsy_52': 0, 'stool_52': 0}
imputer_dict = {'mean': SimpleImputer(missing_values=np.nan, strategy='mean'),
                'mice': IterativeImputer(random_state=0, missing_values=np.nan)}
index_to_order = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']


class ProcessDataset(ABC):

    @property
    def dataset_otu_columns(self):
        raise NotImplementedError

    def process_data(self, pad_in_sequence=True, imputer=None, normalize=True):
        # since GAIN will use pad_in_sequence
        if imputer == 'GAIN':
            assert pad_in_sequence is True

    def categorize(self, pad_in_sequence=True, imputer=None, order='phylum'):
        output_dict = self.process_data(pad_in_sequence, imputer, normalize=False)
        sorted_data = output_dict['sorted_data']

        # get all the names of that particular order
        samples_columns = [item.split('|') for item in self.dataset_otu_columns]
        for samples_columns_index in range(len(samples_columns)):
            samples_columns[samples_columns_index] = \
                [item.split('__')[1] for item in samples_columns[samples_columns_index]]

        idx = index_to_order.index(order)
        # this total_length is to ensure that we are only getting the correct category
        total_length = idx + 1

        order_dict = {}  # the order index of the column of the dataset

        # get all the orders indexes into a dictionary,
        # so later we can select the specific column index and aggregate those orders
        for sample_index, sample_columns in enumerate(samples_columns):
            # if not the correct category we go to next one
            if len(sample_columns) != total_length:
                continue

            if order_dict.get(sample_columns[idx], None) is None:
                order_dict[sample_columns[idx]] = [sample_index]
            else:
                order_dict[sample_columns[idx]].append(sample_index)

        #  ##### aggregate the orders of the columns value and put into a single column #####  #
        # category is the name of the category of the specific order, (phylum) e.g.
        # current_indexes are the indexes to get the values for that category (Firmicutes) e.g.
        new_columns = []
        new_values = []
        for current_category, current_indexes in order_dict.items():
            aggregated_value = np.sum(sorted_data[:, :, current_indexes], 2)
            new_columns.append(current_category)
            new_values.append(aggregated_value)
        new_values = np.array(new_values).transpose(1, 2, 0)

        output_dict['sorted_data'] = new_values
        return output_dict


class ProcessDatasetMMC7(ProcessDataset):
    def __init__(self, dataset_filename):
        self.dataset_filename = dataset_filename
        self.dataset = pd.read_csv(f"data/{self.dataset_filename}.csv")
        self.gain_data_filename = 'data/imputed_data_mmc7.npy'

    @property
    def dataset_otu_columns(self):
        return list(self.dataset.columns)[24:]

    def process_data(self, pad_in_sequence=True, imputer=None, normalize=True):
        super(ProcessDatasetMMC7, self).process_data()

        self.dataset = self.dataset.sort_values(by=['SubjectID', 'collectionWeek', 'sampleType'])

        MAX_TIME_POINTS = 6
        missing_values = None  # this is only used when pad_in_sequence is True

        # remove biopsy for now because they only occur from week 0 and week52
        # df.drop(df[df['sampleType'] == 'biopsy'].index, inplace=True)
        # MAX_TIME_POINTS = 4 # if we remove biopsy

        values = self.dataset.values
        # remove nan values
        nan_indexes = np.argwhere(values[:, 16] != values[:, 16])
        original_values = np.delete(values, [811, 812], axis=0) # remove the whole subject hardcoded first #TODO remove hardcoded value here mmc7.csv

        ### pad the samples with 0
        if pad_in_sequence:
            missing_values = []  # binary values to tell you if this is a missing value or not
            processed_values = []
            index_sample_sequence = 0
            index = 0
            current_subject_id = original_values[0, 1]

            while index < original_values.shape[0]:
                value = original_values[index]

                if value[1] != current_subject_id:
                    while index_sample_sequence != len(SAMPLES_SEQUENCE):
                        # adding missing samples and keep track of the number of them for different weeks
                        missing_key = '_'.join([str(item) for item in SAMPLES_SEQUENCE[index_sample_sequence]])
                        missing_num_samples[missing_key] += 1

                        empty_value = np.zeros(original_values.shape[1])
                        empty_value[:] = np.nan
                        empty_value[1] = current_subject_id
                        processed_values.append(empty_value)
                        missing_values.append(
                            np.ones(
                                original_values.shape[1]))  # add a binary value to show that this is a missing sample
                        index_sample_sequence += 1

                    current_subject_id = value[1]
                    index_sample_sequence = 0

                if value[2] != SAMPLES_SEQUENCE[index_sample_sequence][0] or value[3] != \
                        SAMPLES_SEQUENCE[index_sample_sequence][1]:
                    # adding missing samples and keep track of the number of them for different weeks
                    missing_key = '_'.join([str(item) for item in SAMPLES_SEQUENCE[index_sample_sequence]])
                    missing_num_samples[missing_key] += 1

                    empty_value = np.zeros(original_values.shape[1])
                    empty_value[:] = np.nan
                    empty_value[1] = original_values[index, 1]
                    processed_values.append(empty_value)
                    missing_values.append(
                        np.ones(original_values.shape[1]))  # add a binary value to show that this is a missing sample
                else:
                    # add total key to see how many samples in each week
                    current_key = '_'.join([str(item) for item in SAMPLES_SEQUENCE[index_sample_sequence]])
                    total_num_samples[current_key] += 1

                    processed_values.append(value)
                    missing_values.append(np.zeros(
                        original_values.shape[1]))  # add a binary value to show that this is not a missing sample
                    index += 1

                index_sample_sequence += 1

            original_values = np.array(processed_values)
            missing_values = np.array(missing_values)

        # get the relevant information
        values = original_values[:, 24:].astype(np.float32)
        if missing_values is not None:
            missing_values = missing_values[:, 24:]

        # impute the data if not using GAIN
        if imputer != 'GAIN' and imputer is not None:
            current_imputer = imputer_dict[imputer]
            values = current_imputer.fit_transform(values)
        # if imputation is none just set them to 0
        elif imputer is None:
            values[np.isnan(values)] = 0

        # ### do normalization ### #
        # if GAIN we don't have to normalize because GAIN already normalizes
        if imputer == 'GAIN':
            values = np.load(self.gain_data_filename).astype(np.float32)
            normalized_features = values
        else:
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(values)

        lastSubjectId = original_values[0, 1]
        sorted_data = []  # store all the sorted data
        target_data = []
        sorted_length = []
        current_data = [normalized_features[0]]  # store the temp list for the current subject Id
        zero_padding = np.zeros_like(current_data[0])
        allsubjectid = []

        ### make into shape [samples, timepoints_length, features]
        for current_index in range(1, original_values.shape[0]):
            subjectId = original_values[current_index, 1]
            allsubjectid.append(subjectId)

            # if it's a new subjectId then we append a new row
            if subjectId != lastSubjectId:
                lastSubjectId = subjectId

                # get the actual length
                sorted_length.append(len(current_data))

                # pad data to appropriate time steps
                while len(current_data) < MAX_TIME_POINTS:
                    current_data.append(zero_padding)

                # add the current sample to the whole data and the target value
                sorted_data.append(current_data)
                target_data.append(str(
                    original_values[current_index - 1, 16]))  # previous id because this current index is the next id

                current_data = [normalized_features[current_index]]
            else:
                current_data.append(normalized_features[current_index])

        ### add the last item to the whole data
        sorted_length.append(len(current_data))
        while len(current_data) < MAX_TIME_POINTS:
            current_data.append(zero_padding)
        sorted_data.append(current_data)
        target_data.append(str(original_values[-1, 16]))  # previous id because this current index is the next id

        target_data = np.array(target_data)
        if missing_values is not None:
            missing_data = missing_values.reshape(target_data.shape[0], 6, -1)
        sorted_length = np.array(sorted_length)
        sorted_data = np.stack(sorted_data, 0).astype(np.float32)

        return_dict = {'sorted_data': sorted_data, 'sorted_length': sorted_length,
                       'target_data': np.expand_dims(target_data, 1)}
        if missing_values is not None:
            return_dict['missing_data'] = missing_data
        return return_dict


class ProcessDatasetAllergy(ProcessDataset):
    def __init__(self, dataset_filename):
        self.dataset_filename = dataset_filename
        self.dataset_otu = pd.read_csv(f'data/DIABIMMUNE_data_16s.csv')
        self.dataset_metadata = pd.read_excel('data/DIABIMMUNE_metadata.xlsx', engine="openpyxl")
        self.gain_data_filename = 'data/imputed_data_allergy.npy'
        self.dataset = pd.merge(self.dataset_metadata, self.dataset_otu, on=['SampleID'])

    @property
    def dataset_otu_columns(self):
        return list(self.dataset.columns)[18:]

    def process_data(self, pad_in_sequence=True, imputer=None, normalize=True):
        super(ProcessDatasetAllergy, self).process_data()

        self.dataset = self.dataset.sort_values(by=['subjectID', 'collection_month'])

        all_unique_timepoints = np.unique(self.dataset['collection_month'].values)
        max_timepoints = len(all_unique_timepoints)
        all_samples = np.array([pd.DataFrame(y).values for x, y in self.dataset.groupby('subjectID', as_index=False)])
        labels = np.array([sample[:, 8:11][-1] for sample in all_samples]).astype(np.float32)

        nan_samples = np.where(np.isnan(labels)[:, 0] == True)
        all_samples = np.delete(all_samples, nan_samples, axis=0).tolist()
        labels = np.delete(labels, nan_samples, axis=0)
        lengths = []

        # remove duplicated timepoints
        for sample_index in range(len(all_samples)):
            current_timepoints = all_samples[sample_index][:, 3]
            duplicated_timepoints = [item for item, count in collections.Counter(current_timepoints).items() if count > 1]
            for duplicated_timepoint in duplicated_timepoints:
                indexes = np.where(all_samples[sample_index][:, 3] == duplicated_timepoint)[0][1:]
                all_samples[sample_index] = np.delete(all_samples[sample_index], indexes, axis=0)

        if pad_in_sequence is True:
            for sample_index, current_sample in enumerate(all_samples):
                list_to_add = []
                for idx, current_time in enumerate(all_unique_timepoints):
                    if current_time not in current_sample[:, 3]:
                        list_to_add.append(idx)
                for item in list_to_add:
                    all_samples[sample_index] = np.insert(all_samples[sample_index], item, np.nan, axis=0)
            lengths = np.repeat(max_timepoints, len(all_samples))
        else:
            for sample_index in range(len(all_samples)):
                length = all_samples[sample_index].shape[0]
                all_samples[sample_index] = np.pad(all_samples[sample_index],
                                               [(0, max_timepoints - all_samples[sample_index].shape[0]), (0, 0)],
                                                    constant_values=np.nan)
                lengths.append(length)
            lengths = np.array(lengths)

        all_samples = np.array(all_samples)[:, :, 18:].astype(np.float32)  # TODO make envionment variables too
        missing_data = np.isnan(all_samples).astype(np.float32)

        # impute the data if not using GAIN or there is no imputation setup
        if imputer != 'GAIN' and imputer is not None:
            current_imputer = imputer_dict[imputer]
            original_shape = all_samples.shape
            all_samples = all_samples.reshape(-1, original_shape[-1])
            all_samples = current_imputer.fit_transform(all_samples)
            all_samples = all_samples.reshape(*original_shape)
        # if imputation is none just set them to 0
        elif imputer is None:
            all_samples[np.isnan(all_samples)] = 0

        # ### do normalization ### #
        # if GAIN we don't have to normalize because GAIN already normalizes
        if imputer == 'GAIN':
            all_samples = np.load(self.gain_data_filename).astype(np.float32)
        else:
            # normalize
            if normalize:
                scaler = StandardScaler()
                original_shape = all_samples.shape
                all_samples = all_samples.reshape(-1, original_shape[-1])
                all_samples = scaler.fit_transform(all_samples)
                all_samples.reshape(*original_shape)

        return_dict = {'sorted_data': all_samples,
                       'sorted_length': lengths, 'target_data': labels, 'missing_data': missing_data}
        return return_dict


class ProcessDatasetMeta(ProcessDataset):

    def __init__(self, dataset_filename):
        with open(f'data/{dataset_filename}.data', 'rb') as f:
            dataset = pickle.load(f)
        self.timepoints = dataset['T']
        self.all_unique_timepoints = np.unique([v for timepoint in self.timepoints for v in timepoint])
        self.max_timepoints = len(self.all_unique_timepoints)
        self.original_x = dataset['X']
        self.labels = dataset['y']

    @property
    def dataset_otu_columns(self):
        return None

    def process_data(self, pad_in_sequence=True, imputer=None, normalize=True):
        lengths = []
        self.samples = deepcopy(self.original_x)

        if pad_in_sequence is True:
            for sample_index, timepoint in enumerate(self.timepoints):
                list_to_add = []
                for idx, current_time in enumerate(self.all_unique_timepoints):
                    if current_time not in timepoint:
                        list_to_add.append(idx)
                for item in list_to_add:
                    self.samples[sample_index] = np.insert(self.samples[sample_index], item, np.nan, axis=1)
            lengths = np.repeat(self.max_timepoints, len(self.samples))
        else:
            for sample_index in range(len(self.samples)):
                length = self.samples[sample_index].shape[1]
                self.samples[sample_index] = np.pad(self.samples[sample_index],
                                               [(0, 0), (0,self.max_timepoints - self.samples[sample_index].shape[1])],
                                                    constant_values=np.nan)
                lengths.append(length)
            lengths = np.array(lengths)
        samples = np.array(self.samples).transpose(0, 2, 1).astype(np.float32)
        labels = np.array(self.labels).astype(np.float32)
        samples, labels, lengths = shuffle(samples, labels, lengths, random_state=0)

        # ### normalize ### #
        scaler = StandardScaler()
        original_shape = samples.shape
        samples = samples.reshape(-1, original_shape[-1])
        samples = scaler.fit_transform(samples)
        samples.reshape(*original_shape)

        missing_data = np.isnan(samples).astype(np.float32)

        # impute the data if not using GAIN
        if imputer != 'GAIN' and imputer is not None:
            current_imputer = imputer_dict[imputer]
            original_shape = samples.shape
            samples = samples.reshape(-1, original_shape[-1])
            samples = current_imputer.fit_transform(samples)
            samples = samples.reshape(*original_shape)
        # if imputation is none just set them to 0
        if imputer is None:
            samples[np.isnan(samples)] = 0

        return_dict = {'sorted_data': samples,
                       'sorted_length': lengths, 'target_data': np.expand_dims(labels, 1), 'missing_data': missing_data}
        return return_dict


dataset_list = {'mmc7': ProcessDatasetMMC7, 'david': ProcessDatasetMeta, 'digiulio': ProcessDatasetMeta,
                't1d': ProcessDatasetMeta, 'allergy': ProcessDatasetAllergy}
