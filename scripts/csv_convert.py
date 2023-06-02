import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

# Taken from the EDA notebook
from os import listdir
from os.path import isfile, join
def get_filenames(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]


filenames = ["../data/audioset_v1_embeddings/bal_train/" + i for i in get_filenames("../data/audioset_v1_embeddings/bal_train/")]
train_dataset = tf.data.TFRecordDataset(filenames)

# Make Pandas DataFrame out of the training set records.
col = ['video_id', 'time_stamp'] + [str(k) for k in range(0,128)]

placeholder_array = np.array(col)
counter = 0

for raw_record in train_dataset:
    example = tf.train.SequenceExample()
    example.ParseFromString(raw_record.numpy())
    for i in range(0,len(example.feature_lists.feature_list['audio_embedding'].feature)):
        vID = example.context.feature['video_id'].bytes_list.value[0]
        time = example.context.feature['start_time_seconds'].float_list.value[0] + 0.96*i
        placeholder_array = np.vstack((placeholder_array, np.array([vID, time] + list(example.feature_lists.feature_list['audio_embedding'].feature[0].bytes_list.value[0]))))
    counter += 1
    print('Record count: ' + str(counter))

print('Done making the array. Converting to Pandas DF and saving.')

trainFeatures = pd.DataFrame(np.delete(placeholder_array, 0, 0), columns = col)

trainFeatures.to_csv('trainFeatures.csv')
print(trainFeatures.head())

# Make a Pandas DF for the target: whether or not speech is present in each 0.96 second chunk.
trainTargets = trainFeatures[:,:'time_stamp']
del trainFeatures # For RAM's sake. We've already saved this to a CSV.

# Respectively, speech, male speech, female speech, child speech, conversation, and narration.
speech_events = set(['/m/09x0r', '/m/05zppz','/m/02zsn','/m/0ytgt','/m/01h8n0','/m/02qldy'])

# Make a Pandas DataFrame of all instances of speech present.
trainEvents = pd.read_csv("../data/audioset_train_strong.tsv", sep="\t")
for i in range(0,len(trainEvents)):
    ''' Have to make the labels match the feature set,
        and these labels have a "_" followed by trailing digits.'''
    trainEvents.iloc[i,0] = trainEvents.iloc[i,0].rstrip("0123456789")
    trainEvents.iloc[i,0] = trainEvents.iloc[i,0].rstrip("_")
    if (trainEvents.iloc[i,3] in speech_events) == False:
        trainEvents.iloc[i,3] = None
trainEvents = trainEvents.dropna() # Deletes all rows with a None in it, i.e. entries that have no speech.

trainTargets['speech_present'] = False # By default.

# Now check to see if each 0.96 second segment contains speech according to the trainEvents DF.
'''This seems complicated at first glance, but the idea behind it is simple.
Since each clip's events are grouped together in the trainEvents TSV, we just
need to run a search ONCE for each clip label. Once we have it, we don't need
to search again for the next entry's events unless its label is different.
Sadly, the labels are NOT in alphabetical order, making an approach like this
necessary.'''

first_label_match = 0
for i in range(0,len(trainTargets)):
    if first_label_match == 0:
        while (trainTargets.iloc[i,0] != trainEvents.iloc[first_label_match,0]):
            first_label_match += 1
    offset = 0
    while (trainTargets.iloc[i,0] == trainEvents.iloc[first_label_match + offset,0]):
        if trainTargets.iloc[i,1] <= trainEvents.iloc[first_label_match + offset,1]:
            if trainTargets.iloc[i,1] + 0.96 >= trainEvents.iloc[first_label_match + offset,1]:
                trainTargets.iloc[i,2] = True
        if trainTargets.iloc[i,1] >= trainEvents.iloc[first_label_match + offset,1]:
            if trainTargets.iloc[i,1] <= trainEvents.iloc[first_label_match + offset,2]:
                trainTargets.iloc[i,2] = True
        offset += 1
    if i != len(trainTargets) - 1:
        if trainTargets.iloc[i,0] != trainTargets.iloc[i+1,0]:
            first_label_match = 0

trainTargets.to_csv('trainTargets.csv')

# When done, we can drop the labels and time indices, since the orders are the same.