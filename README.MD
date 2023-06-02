# Outline

Objective:

- To train a binary classification model to recognize when speech is present in an audio file.

Google dataset description:

- Each 10 second audio clip was split into 0.96 second pieces, each subjected to feature (spectrogram) extraction, a proprietary embedding, PCA, among other things.

- Google provides a label set which lists the sound events contained in 10 second clip, but this doesn't tell us when in each clip they occur.

- Fortunately, Google also provided a ``strong'' label set, which described exactly for each 960 ms piece which sound events were present.

What we did with it:

- Because the ``strong'' labels were only made for 5% of the dataset, we took the intersection of that and the balanced subset. We had to take a similar approach for the evaluation set, because their ``strong'' label set also only covers a subset.

- Google's label set includes X different sound events. 8 of these we classify as ``speech'' for our purposes.

- We processed the event sets to simply say, for each 960 ms piece of each 10 second audio clip, whether or not speech (as defined above) was present.

- This allowed us to use this dataset for our binary classification problem.

Our model:

- Because Google had done most preprocessing already, we only needed to apply normalization (performed in the second layer of our neural network model, immediately below the input).