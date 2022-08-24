# EEG-Net-Matlab
In this repository, I used the model trained in EEG-Net 
Python repository to classify the data received from participants in the experiment. The experiment was based on SSVEP protocol, and the subject could see the stimulus they wanted to choose, and the model could classify the EEG data it received and identify the stimulus subject was looking at.
I could easily transfer the model from Python to Matlab using the importKerasNetwork function implemented in Matlab, but the model only accepts preprocessed data in the frequency domain that is transformed using FFT. So, I had to implement all preprocessing modules in Matlab too. In the src folder, you can run the main.m file to see the outputs of the model. This model received an accuracy of 91% on the test set after being evaluated on our dataset.

Disclaimer: The S02.mat file in the Data folder is only part of our dataset, and the whole data can not be shared as its rights belong to the Telecommunication Company of Iran.
