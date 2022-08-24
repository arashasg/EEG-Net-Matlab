function filtered_data = get_filtered_eeg(eeg, lowcut, highcut, order, sample_rate)
    %{
    Returns bandpass filt
    ered eeg for all channels and trials.

    Args:
        eeg (numpy.ndarray): raw eeg data of shape (num_classes, num_channels, num_samples, num_trials).
        lowcut (float): lower cutoff frequency (Hz).
        highcut (float): lower cutoff frequency (Hz).
        order (int): order of the bandpass filter.
        sample_rate (float): sampling rate (Hz).

    Returns:
        (numpy.ndarray): bandpass filtered eeg of shape (num_classes, num_channels, num_samples, num_trials).
    %}
    num_classes = size(eeg, 1);
    num_chan = size(eeg, 2);
    % total_trial_len = size(eeg, 3)
    num_trials = size(eeg, 4);
    
    trial_len = int32(38+0.135*sample_rate+4*sample_rate-1) - int32(38+0.135*sample_rate);
    filtered_data = zeros(num_classes, num_chan, trial_len, num_trials);

    for target = 1:num_classes
        for channel = 1:num_chan
            for trial = 1:num_trials
                signal_to_filter = squeeze(eeg(target, channel, int32(38+0.135*sample_rate) : int32(38+0.135*sample_rate+4*sample_rate-1), trial));
                filtered_data(target, channel, :, trial) = transpose(butter_bandpass_filter(signal_to_filter, lowcut, highcut, sample_rate, order));%changed

            end
        end
    end
end