import numpy as np
import pandas as pd
import math

fraction_of_set_to_train = 5 / 6

# Training set currently has 25,687 960 ms pieces of audio.
trainFeatures = pd.read_csv("../data/trainFeaturesFiltered.csv", header = 0, index_col = 0)
trainTargets = pd.read_csv("../data/trainTargetsFiltered.csv", header = 0, index_col = 0)

# Evaluation set currently has 51,044 960 ms pieces of audio.
evalFeatures = pd.read_csv("../data/evalFeaturesFiltered.csv", header = 0, index_col = 0)
evalTargets = pd.read_csv("../data/evalTargetsFiltered.csv", header = 0, index_col = 0)

# Combined, there are 76,731 960 ms pieces of audio.
totalFeatures = pd.concat([trainFeatures, evalFeatures], axis = 0, ignore_index = True)
totalTargets = pd.concat([trainTargets, evalTargets], axis = 0, ignore_index = True)

del trainFeatures, trainTargets, evalFeatures, evalTargets

train_size = math.floor(len(totalFeatures) * fraction_of_set_to_train)

trainFeaturesNew = totalFeatures.iloc[0:train_size,:]
trainTargetsNew = totalTargets.iloc[0:train_size,:]

evalFeaturesNew = totalFeatures.iloc[train_size:len(totalFeatures),:]
evalTargetsNew = totalTargets.iloc[train_size:len(totalTargets),:]

trainFeaturesNew.to_csv('../data/trainFeaturesNewSplit.csv')
trainTargetsNew.to_csv('../data/trainTargetsNewSplit.csv')
evalFeaturesNew.to_csv('../data/evalFeaturesNewSplit.csv')
evalTargetsNew.to_csv('../data/evalTargetsNewSplit.csv')
