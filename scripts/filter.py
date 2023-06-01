import tensorflow as tf
import numpy as np
import pandas as pd

# Taken from the EDA notebook
from os import listdir
from os.path import isfile, join
def get_filenames(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]

trainClips = pd.read_csv("audioset_train_strong.tsv", sep="\t")
print(trainClips.head())
trainClips.iloc[:,0] = trainClips.iloc[:,0].str.rstrip("0123456789")
filter = set(trainClips.iloc[:,0].str.rstrip("_"))
del trainClips
print('Filter made.')

#filenames = ["./unbal_train/" + i for i in get_filenames("./unbal_train/")]
#train_dataset = tf.data.TFRecordDataset(filenames)

filenames = ["../data/audioset_v1_embeddings/unbal_train/" + i for i in get_filenames("../data/audioset_v1_embeddings/unbal_train/")]
train_dataset = tf.data.TFRecordDataset(filenames)

print('Dataset loaded.')
#print(train_dataset.cardinality)

# Make Pandas DataFrame out of the training set records.
col = ['video_id', 'time_stamp'] + [str(k) for k in range(0,128)]

placeholder_array = np.array(col)

counter = 0
for raw_record in train_dataset:
    example = tf.train.SequenceExample()
    example.ParseFromString(raw_record.numpy())
    vID = example.context.feature['video_id'].bytes_list.value[0]
    if vID.decode() in filter:
        for i in range(0,len(example.feature_lists.feature_list['audio_embedding'].feature)):
            time = example.context.feature['start_time_seconds'].float_list.value[0] + 0.96*i
            placeholder_array = np.vstack((placeholder_array, np.array([vID, time] + list(example.feature_lists.feature_list['audio_embedding'].feature[0].bytes_list.value[0]))))
        counter += 1
        print(str(counter) + ' matching clips.')

print('Done making the array. Converting to Pandas DF and saving.')

trainFeatures = pd.DataFrame(np.delete(placeholder_array, 0, 0), columns = col)

trainFeatures.to_csv('trainFeatures.csv')
trainFeatures.head()