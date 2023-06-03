# Introduction/Objective:

In recent years, speech recognition technology has been integrated into many consumer electronics, such as cell phones, vehicles, and smart homes, and will continue to be in demand for the foreseeable future. Speech-in-Noise (SIN) perception is the ability to recognise relevant voice within a noisy background, and is crucial for devices operating in environments where such a background is unavoidable, such as the inside of a moving vehicle. The first step in speech recognition for such devices, and the objective of our neural network model, is to determine whether a given raw audio sample includes someone speaking or not.

# Proposed Solution:

Our solution is to train an artificial neural network (ANN) on Google's AudioSet, which consists of thousands of 10-second audio excerpts from randomly selected YouTube videos. The raw audio set is not available to the public, but rather, 128-dimensional feature vectors, each representing 960 millisecond pieces of the excerpts, are free to download. These features were extracted using a convolutional neural network (CNN), quantized, and subjected to principal component analysis (PCA). Moreover, for a small subset of this dataset, Google provided a temporally detailed label set, describing for each 960 ms piece which sound events (out of 527 possibilites) are present.

We train our ANN on this subset; our objective is binary classification in the sense that the model should determine whether or not speech is present in a 960 ms audio sample just from the 128-dimensional feature vector representing it. Google provided 527 sound event labels, and several of these easily can be considered "speech", and so every feature vector including one or more such events is to be assigned a "true" label. Because the subset is imbalanced, with roughly 18% of excerpts containing speech by our definition, we minimize a weighted binary cross entropy loss function.

# Our model and results:

Our key results are 82.2% accuracy, 66.0% recall, and 64.4% precision, all of which were measured over the evaluation set and averaged across 30 trainings. The latter two are metrics that try to compensate for datasets with imbalanced classes such as ours. Overall, this is still a fairly good result which shows promise for our model's use in speech recognition applications. However, in some testing and examination of the ontology, we note that some human noises, such as laughter, singing, or babbling, tended to be a source of confusion for our model, given their similarities to speech. We classified them as non-speech because we aimed to distinguish deliberate speech from other human-made sounds. More work will need to be done in the future to strengthen the distinction.

# Conclusion

These results show promise for the use of artificial neural networks to recognize speech, though the restricted subset proved to be a hinderance. 

Potential Applications:
- For designing voice assistants for noisy environments.
