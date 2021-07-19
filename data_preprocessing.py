from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

# samples in sequence (this is used to check to make sure that the subjects are in sequence)
SAMPLES_SEQUENCE = [['biopsy', 0], ['stool', 0], ['stool', 4], ['stool', 12], ['biopsy', 52], ['stool', 52]]
missing_num_samples = {'biopsy_0': 0, 'stool_0': 0, 'stool_4': 0, 'stool_12': 0, 'biopsy_52': 0, 'stool_52': 0}
total_num_samples = {'biopsy_0': 0, 'stool_0': 0, 'stool_4': 0, 'stool_12': 0, 'biopsy_52': 0, 'stool_52': 0}
imputer_dict = {'mean': SimpleImputer(missing_values=np.nan, strategy='mean'),
                'mice': IterativeImputer(random_state=0, missing_values=np.nan)}


class ProcessDataset(ABC):
    @abstractmethod
    def process_data(self, pad_in_sequence=True, imputer=None):
        pass


class ProcessDatasetMMC7(ProcessDataset):
    def __init__(self):
        pass

    def process_data(self, pad_in_sequence=True, imputer=None):
        df = pd.read_csv("data/mmc7.csv")
        df = df.sort_values(by=['SubjectID', 'collectionWeek', 'sampleType'])

        MAX_TIME_POINTS = 6
        missing_values = None  # this is only used when pad_in_sequence is True

        # remove biopsy for now because they only occur from week 0 and week52
        # df.drop(df[df['sampleType'] == 'biopsy'].index, inplace=True)
        # MAX_TIME_POINTS = 4 # if we remove biopsy

        values = df.values
        # remove nan values
        nan_indexes = np.argwhere(values[:, 16] != values[:, 16])
        original_values = np.delete(values, nan_indexes, 0)

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
        if imputer is None:
            values[np.isnan(values)] = 0

        # normalize the data
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


class ProcessDatasetDavid(ProcessDataset):
    def __init__(self):
        with open('data/david.data', 'rb') as f:
            dataset = pickle.load(f)
        self.timepoints = dataset['T']
        self.max_timepoints = len(range(-5, 11))
        self.original_x = dataset['X']
        self.labels = dataset['y']

    def process_data(self, pad_in_sequence=True, imputer=None):
        lengths = []
        self.samples = deepcopy(self.original_x)

        if pad_in_sequence is True:
            for sample_index, timepoint in enumerate(self.timepoints):
                list_to_add = []
                for i in range(-5, 11):
                    if i not in timepoint:
                        list_to_add.append(i + 5)
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
        labels = np.array(self.labels)

        missing_data = np.isnan(samples).astype(np.int32)

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


dataset_list = {'mmc7': ProcessDatasetMMC7, 'david': ProcessDatasetDavid}

