import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import useful packages
import keras
import pandas as pd
import numpy as np
from keras import backend as K
import tensorflow as tf
#import dill
from sklearn import preprocessing
#from sklearn.metrics import confusion_matrix,recall_score, precision_score, roc_auc_score, roc_curve, balanced_accuracy_score, f1_score,jaccard_similarity_score
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, LSTM, Activation, RepeatVector,Conv1D,GlobalAveragePooling1D,Flatten,UpSampling1D
import matplotlib.pyplot as plt
from keras.utils import to_categorical

df=pd.read_excel('IBD_data_processed.xlsx')

# # encoder decoder LSTM
# #  不对基因做最大最小化，其本身已经是标准数据
min_max_scaler = preprocessing.MinMaxScaler()
df.loc[:,'SampleType':'antibiotics']=min_max_scaler.fit_transform(df.loc[:,'SampleType':'antibiotics'].values)

start=2
end=4
seq_len=end-start
X=[]
Y1=[]
Y2=[]
#生成序列
for index,data in df.groupby('SubjectID'):
    if(data.iloc[3]['label2']==3):
        print('a')
    if(data.iloc[3]['label2']!=3):
        X.append(data.iloc[start:end,1:-3].values.reshape((1,seq_len,-1)))
        Y1.append(data.iloc[end-1]['label1'])
        Y2.append(data.iloc[end-1]['label2'])
X=np.vstack(X)

# normal LSTM
df.head()
# Split training and testing set
id = df.SubjectID.unique()
index = np.random.rand(len(id)) < 0.7
train_id = id[index]
test_id = id[~index]
train_df = df.loc[df['SubjectID'].isin(train_id)]
test_df = df.loc[df['SubjectID'].isin(test_id)]

# MinMax normalization for training set
train_df['TimePoint_norm'] = train_df['TimePoint']
cols_normalize = train_df.columns.difference(['SubjectID','TimePoint','label1','label2'])
min_max_scaler = preprocessing.MinMaxScaler()
norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]),
                             columns=cols_normalize,
                             index=train_df.index)
join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
train_df = join_df.reindex(columns = train_df.columns)

# MinMax normalization for testing set
test_df['TimePoint_norm'] = test_df['TimePoint']
norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]),
                            columns=cols_normalize,
                            index=test_df.index)
test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
test_df = test_join_df.reindex(columns = test_df.columns)
test_df = test_df.reset_index(drop=True)

# Set a window size (how many time steps we look back)
sequence_length = 2

# function to reshape features into (samples, time steps, features) for each subject in training set
def gen_sequence(id_df, seq_length, seq_cols):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]

# function to generate labels for each patient in training set
def gen_labels(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]

# Define the feature columns
sequence_cols = df.columns.tolist()[2:14]
sequence_cols.extend(['m.' + str(i) for i in range(1, 1016)])
sequence_cols.extend(['TimePoint_norm'])

# Generate feature sequences for each patient in the training set
seq_gen = (list(gen_sequence(train_df[train_df['SubjectID']==id], sequence_length, sequence_cols))
           for id in train_df['SubjectID'].unique())
# Convert the sequences to a numpy array
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)

# generate labels of the whole training set
label_gen = [gen_labels(train_df[train_df['SubjectID']==id], sequence_length, ['label1'])
             for id in train_df['SubjectID'].unique()]
label_array = np.concatenate(label_gen).astype(np.float32)