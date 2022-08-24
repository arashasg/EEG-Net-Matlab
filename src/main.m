dat = load("..\Data\S02.mat"); % loading data
eeg = dat.eeg;
% trial parameters
sample_rate = 512;
resolution = 0.2930;
start_frequency = 3.0;
end_frequency = 35.0;
window_len = 2;
shift_len = 1;

% applying Butterworth filter
filtered_data = get_filtered_eeg(eeg, 4, 45, 4, sample_rate);
% data Segmentation
user_data = get_segmented_epochs(filtered_data, window_len,shift_len, sample_rate);
magnitude_spectrum_feature = magnitude_spectrum_features(user_data, sample_rate, resolution, start_frequency, end_frequency);
[train_data, labels] = get_training_data(magnitude_spectrum_feature);
% getting data prepared to give it as input to model
train_data = reshape(train_data, size(train_data, 1), size(train_data, 2), size(train_data, 3), 1);
% reading the model
net = importKerasNetwork("../models/model.h5");
% getting the outputs
num = 259;
for num = 60:120
    disp(classify(net, reshape(train_data(num,:,:, :),10, 110, 1)))
end

disp(labels(num))