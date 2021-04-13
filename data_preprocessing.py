import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def process_data():
    df = pd.read_csv("data/mmc7.csv")
    df = df.sort_values(by=['SubjectID', 'collectionWeek'])

    values = df.values

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
            while len(current_data) < 6:
                current_data.append(zero_padding)

            # add the current sample to the whole data and the target value
            sorted_data.append(current_data)
            target_data.append(str(values[current_index-1, 16]))   # previous id because this current index is the next id

            current_data = [normalized_features[current_index]]
        else:
            current_data.append(normalized_features[current_index])

    ### add the last item to the whole data
    sorted_length.append(len(current_data))
    # pad data to 4 time steps
    while len(current_data) < 6:
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




