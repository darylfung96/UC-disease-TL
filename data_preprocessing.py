import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# samples in sequence (this is used to check to make sure that the subjects are in sequence)
SAMPLES_SEQUENCE = [['biopsy', 0], ['stool', 0], ['stool', 4], ['stool', 12], ['biopsy', 52], ['stool', 52]]
missing_num_samples = {'biopsy_0': 0, 'stool_0': 0, 'stool_4': 0, 'stool_12': 0, 'biopsy_52': 0, 'stool_52': 0}
total_num_samples = {'biopsy_0': 0, 'stool_0': 0, 'stool_4': 0, 'stool_12': 0, 'biopsy_52': 0, 'stool_52': 0}


def process_data(pad_in_sequence=True):
    df = pd.read_csv("data/mmc7.csv")
    df = df.sort_values(by=['SubjectID', 'collectionWeek', 'sampleType'])

    MAX_TIME_POINTS = 6
    missing_values = None  # this is only used when pad_in_sequence is True

    # remove biopsy for now because they only occur from week 0 and week52
    # df.drop(df[df['sampleType'] == 'biopsy'].index, inplace=True)
    # MAX_TIME_POINTS = 4 # if we remove biopsy

    values = df.values

    ### pad the samples with 0
    if pad_in_sequence:
        missing_values = []  # binary values to tell you if this is a missing value or not
        processed_values = []
        index_sample_sequence = 0
        index = 0
        current_subject_id = values[0, 1]

        while index < values.shape[0]:
            value = values[index]

            if value[1] != current_subject_id:
                while index_sample_sequence != len(SAMPLES_SEQUENCE):
                    # adding missing samples and keep track of the number of them for different weeks
                    missing_key = '_'.join([str(item) for item in SAMPLES_SEQUENCE[index_sample_sequence]])
                    missing_num_samples[missing_key] += 1

                    empty_value = np.zeros(values.shape[1])
                    empty_value[1] = current_subject_id
                    processed_values.append(empty_value)
                    missing_values.append(
                        np.ones(values.shape[1]))  # add a binary value to show that this is a missing sample
                    index_sample_sequence += 1

                current_subject_id = value[1]
                index_sample_sequence = 0

            if value[2] != SAMPLES_SEQUENCE[index_sample_sequence][0] or value[3] != SAMPLES_SEQUENCE[index_sample_sequence][1]:
                # adding missing samples and keep track of the number of them for different weeks
                missing_key = '_'.join([str(item) for item in SAMPLES_SEQUENCE[index_sample_sequence]])
                missing_num_samples[missing_key] += 1

                empty_value = np.zeros(values.shape[1])
                empty_value[1] = values[index, 1]
                processed_values.append(empty_value)
                missing_values.append(np.ones(values.shape[1])) # add a binary value to show that this is a missing sample
            else:
                # add total key to see how many samples in each week
                current_key = '_'.join([str(item) for item in SAMPLES_SEQUENCE[index_sample_sequence]])
                total_num_samples[current_key] += 1

                processed_values.append(value)
                missing_values.append(np.zeros(values.shape[1])) # add a binary value to show that this is not a missing sample
                index += 1

            index_sample_sequence += 1

        values = np.array(processed_values)
        missing_values = np.array(missing_values)

    # normalize the data
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(values[:, 24:])
    if missing_values is not None:
        missing_values = missing_values[:, 24:]

    lastSubjectId = values[0, 1]
    sorted_data = []  # store all the sorted data
    target_data = []

    sorted_length = []
    current_data = [normalized_features[0]]  # store the temp list for the current subject Id
    zero_padding = np.zeros_like(current_data[0])

    allsubjectid = []

    ### make into shape [samples, timepoints_length, features]
    for current_index in range(1, values.shape[0]):
        subjectId = values[current_index, 1]
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
            target_data.append(str(values[current_index-1, 16]))  # previous id because this current index is the next id

            current_data = [normalized_features[current_index]]
        else:
            current_data.append(normalized_features[current_index])

    ### add the last item to the whole data
    sorted_length.append(len(current_data))
    while len(current_data) < MAX_TIME_POINTS:
        current_data.append(zero_padding)
    sorted_data.append(current_data)
    target_data.append(str(values[-1, 16]))  # previous id because this current index is the next id

    # remove all nan values
    target_data = np.array(target_data)
    missing_data = missing_values.reshape(target_data.shape[0], 6, -1)
    sorted_length = np.array(sorted_length)
    for nan_values_index in np.where(target_data == 'nan')[0]:
        sorted_data = np.delete(sorted_data, nan_values_index, 0)
        target_data = np.delete(target_data, nan_values_index, 0)
        missing_data = np.delete(missing_data, nan_values_index, 0)
        sorted_length = np.delete(sorted_length, nan_values_index, 0)

    sorted_data = np.stack(sorted_data, 0).astype(np.float32)

    return_dict = {'sorted_data': sorted_data, 'sorted_length': sorted_length, 'target_data': np.expand_dims(target_data, 1), 'missing_data': missing_data}
    return return_dict
