function features_data = complex_spectrum_features(segmented_data, fft_sample_rate, fft_resolution, fft_start_frequency, fft_end_frequency)
    %{
    Returns complex spectrum features. Fast Fourier Transform computed based on
    the FFT parameters provided as input. The real and imaginary parts of the input
    signal are concatenated into a single feature vector.

    Args:
        segmented_data (numpy.ndarray): epoched eeg data of shape 
        (num_classes, num_channels, num_trials, number_of_segments, num_samples).
        FFT_PARAMS (dict): dictionary of parameters used for feature extraction.
        FFT_PARAMS['resolution'] (float): frequency resolution per bin (Hz).
        FFT_PARAMS['start_frequency'] (float): start frequency component to pick from (Hz). 
        FFT_PARAMS['end_frequency'] (float): end frequency component to pick upto (Hz). 
        FFT_PARAMS['sampling_rate'] (float): sampling rate (Hz).

    Returns:
        (numpy.ndarray): complex spectrum features of the input EEG.
        (2*n_fc, num_channels, num_classes, num_trials, number_of_segments)
    %}
    
    num_classes = size(segmented_data, 1);
    num_chan = size(segmented_data, 2);
    num_trials = size(segmented_data, 3);
    number_of_segments = size(segmented_data, 4);
    fft_len = size(segmented_data(1, 1, 1, 1, :), 5);
    NFFT = round(fft_sample_rate/fft_resolution);
    fft_index_start = int32(round(fft_start_frequency/fft_resolution));
    fft_index_end = int32(round(fft_end_frequency/fft_resolution))+1;
    features_data = zeros(2*(fft_index_end - fft_index_start),num_chan, num_classes, num_trials, number_of_segments);
    for target = 1 : num_classes
        for channel = 1 : num_chan
            for trial = 1 : num_trials
                for segment = 1 : number_of_segments
                    temp_FFT = squeeze(fft(segmented_data(target, channel, trial, segment, :), NFFT)/fft_len);
                    real_part = real(temp_FFT);
                    imag_part = imag(temp_FFT);
                    features_data(:, channel, target, trial, segment) = cat(1, real_part(fft_index_start:fft_index_end-1,:), imag_part(fft_index_start:fft_index_end-1,:));
                end
            end
        end
    end
end
    
