function segmented_data = get_segmented_epochs(data, window_len, shift_len, sample_rate)
    %{
    Returns epoched eeg data based on the window duration and step size.

    Args:
        data (numpy.ndarray): array of samples. 
        window_len (int): window length (seconds).
        shift_len (int): step size (seconds).
        sample_rate (float): sampling rate (Hz).

    Returns:
        (numpy.ndarray): epoched eeg data of shape. 
        (num_classes, num_channels, num_trials, number_of_segments, duration).
    %}
    
    num_classes = size(data, 1);
    num_chan = size(data, 2);
    num_trials = size(data, 4);
    
    duration = int32(window_len*sample_rate);
    data_overlap = (window_len - shift_len)*sample_rate;
    
    number_of_segments = int32(ceil((size(data, 3) - data_overlap)/(duration - data_overlap)));
    segmented_data = zeros(num_classes, num_chan,num_trials, number_of_segments, duration);

    for target = 1:num_classes
        for channel = 1: num_chan
            for trial = 1 : num_trials
                %disp(size(buffer(data(target, channel, :, trial), duration, data_overlap)));
                segmented_data(target, channel, trial, :, :) = buffer(data(target, channel, :, trial), duration, data_overlap);
            end
        end
    end
end
