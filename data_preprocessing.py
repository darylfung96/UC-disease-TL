import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


SAMPLES_SEQUENCE = [['biopsy', 0], ['stool', 0], ['stool', 4], ['stool', 12], ['biopsy', 52], ['stool', 52]]
missing_num_samples = {'biopsy_0': 0, 'stool_0': 0, 'stool_4': 0, 'stool_12': 0, 'biopsy_52': 0, 'stool_52': 0}
total_num_samples = {'biopsy_0': 0, 'stool_0': 0, 'stool_4': 0, 'stool_12': 0, 'biopsy_52': 0, 'stool_52': 0}

def process_data(pad_in_sequence=True):
    df = pd.read_csv("data/mmc7.csv")
    df = df.sort_values(by=['SubjectID', 'collectionWeek', 'sampleType'])

    MAX_TIME_POINTS = 6

    # remove biopsy for now because they only occur from week 0 and week52
    # df.drop(df[df['sampleType'] == 'biopsy'].index, inplace=True)
    # MAX_TIME_POINTS = 4 # if we remove biopsy

    values = df.values

    if pad_in_sequence:
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
            else:
                # add total key to see how many samples in each week
                current_key = '_'.join([str(item) for item in SAMPLES_SEQUENCE[index_sample_sequence]])
                total_num_samples[current_key] += 1

                processed_values.append(value)
                index += 1

            index_sample_sequence += 1

        values = np.array(processed_values)

    # normalize the data
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(values[:, 24:])

    lastSubjectId = values[0, 1]
    sorted_data = []  # store all the
    target_data = []
    sorted_length = []
    current_data = [normalized_features[0]]  # store the temp list for the current subject Id
    zero_padding = np.zeros_like(current_data[0])

    allsubjectid = []
    # make into shape [samples, timepoints_length, features]
    for current_index in range(1, values.shape[0]):
        subjectId = values[current_index, 1]
        allsubjectid.append(subjectId)

        # if it's a new subjectId then we append a new row
        if subjectId != lastSubjectId:
            lastSubjectId = subjectId

            # get the actual length
            sorted_length.append(len(current_data))

            # pad data to 4 time steps
            while len(current_data) < MAX_TIME_POINTS:
                current_data.append(zero_padding)

            # add the current sample to the whole data and the target value
            sorted_data.append(current_data)
            target_data.append(str(values[current_index-1, 16]))   # previous id because this current index is the next id

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
    sorted_length = np.array(sorted_length)
    for nan_values_index in np.where(target_data == 'nan')[0]:
        sorted_data = np.delete(sorted_data, nan_values_index, 0)
        target_data = np.delete(target_data, nan_values_index, 0)
        sorted_length = np.delete(sorted_length, nan_values_index, 0)

    sorted_data = np.stack(sorted_data, 0).astype(np.float32)
    return sorted_data, sorted_length, np.expand_dims(target_data, 1)




