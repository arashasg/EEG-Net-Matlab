function y = butter_bandpass_filter(data, lowcut, highcut, sample_rate, order)
    %{
    Returns bandpass filtered data between the frequency ranges specified in the input.

    Args:
        data (numpy.ndarray): array of samples. 
        lowcut (float): lower cutoff frequency (Hz).
        highcut (float): lower cutoff frequency (Hz).
        sample_rate (float): sampling rate (Hz).
        order (int): order of the bandpass filter.

    Returns:
        (numpy.ndarray): bandpass filtered data.
    %}
    nyq = 0.5 * sample_rate;
    low = lowcut / nyq;
    high = highcut / nyq;
    [b, a] = butter(order, [low, high]);
    y = filtfilt(b, a, data);
    y = y(2:end,:);
end

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
                filtered_data(target, channel, :, trial) = butter_bandpass_filter(signal_to_filter, lowcut, highcut, sample_rate, order);
            end
        end
    end
end

function segmented_data = buffer(data, duration, data_overlap)
    %{
    Returns segmented data based on the provided input window duration and overlap.

    Args:
        data (numpy.ndarray): array of samples. 
        duration (int): window length (number of samples).
        data_overlap (int): number of samples of overlap.

    Returns:
        (numpy.ndarray): segmented data of shape (number_of_segments, duration).
    %}
    number_segments = int32(ceil((size(data, 0) - data_overlap)/(duration - data_overlap)));
    temp_buf = zeros(number_segments,duration);
    for i = 1:number_segments
        startOfSegment = (i-1)*(duration - data_overlap) + 1;
        endOfSegment = startOfSegment + duration;
        temp_buf(i) = data(startOfSegment:endOfSegment);
    end
    segmented_data = temp_buf;
end

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
    
    num_classes = size(A, 1)
    num_chan = size(A, 2)
    num_trials = size(A, 4)
    
    duration = int32(window_len*sample_rate)
    data_overlap = (window_len - shift_len)*sample_rate
    
    number_of_segments = int32(ceil((num_chan - data_overlap)/(duration - data_overlap)))
    
    segmented_data = zeros(num_classes, num_chan,num_trials, number_of_segments, duration)

    for target = 1:num_classes
        for channel = 1: num_chan
            for trial = 1 : num_trials
                segmented_data(target, channel, trial, :, :) = buffer(data(target, channel, :, trial), duration, data_overlap)
            end
        end
    end
end

function features_data = magnitude_spectrum_features(segmented_data, fft_sample_rate, fft_resolution, fft_start_frequency, fft_end_frequency)
    %{
    Returns magnitude spectrum features. Fast Fourier Transform computed based on
    the FFT parameters provided as input.

    Args:
        segmented_data (numpy.ndarray): epoched eeg data of shape 
        (num_classes, num_channels, num_trials, number_of_segments, num_samples).
        FFT_PARAMS (dict): dictionary of parameters used for feature extraction.
        FFT_PARAMS['resolution'] (float): frequency resolution per bin (Hz).
        FFT_PARAMS['start_frequency'] (float): start frequency component to pick from (Hz). 
        FFT_PARAMS['end_frequency'] (float): end frequency component to pick upto (Hz). 
        FFT_PARAMS['sampling_rate'] (float): sampling rate (Hz).

    Returns:
        (numpy.ndarray): magnitude spectrum features of the input EEG.
        (n_fc, num_channels, num_classes, num_trials, number_of_segments).
    %}
    
    num_classes = size(segmented_data, 1);
    num_chan = size(segmented_data, 2);
    num_trials = size(segmented_data, 3);
    number_of_segments = size(segmented_data, 4);
    fft_len = size(segmented_data(1, 1, 1, 1, :), 1);

    NFFT = round(fft_sample_rate/fft_resolution);
    fft_index_start = int32(round(fft_start_frequency/fft_resolution));
    fft_index_end = int32(round(fft_end_frequency/fft_resolution))+1;

    features_data = zeros((fft_index_end - fft_index_start),num_chan, num_classes,num_trials, number_of_segments);
    
    for target = 1: num_classes
        for channel = 1 : num_chan
            for trial = 1 : num_trials
                for segment = 1: number_of_segments
                    temp_FFT = fft(segmented_data(target, channel, trial, segment, :), NFFT)/fft_len;
                    magnitude_spectrum = 2*abs(temp_FFT);
                    features_data(:, channel, target, trial, segment) = magnitude_spectrum(fft_index_start:fft_index_end,:); %mahale eshteb line 169 in py
                end
            end
        end
    end
end

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
    fft_len = size(segmented_data(1, 1, 1, 1, :), 1);

    NFFT = round(fft_sample_rate/fft_resolution);
    fft_index_start = int32(round(fft_start_frequency/fft_resolution));
    fft_index_end = int32(round(fft_end_frequency/fft_resolution))+1;

    features_data = zeros(2*(fft_index_end - fft_index_start),num_chan, num_classes, num_trials, number_of_segments);
    
    for target = 1 : num_classes
        for channel = 1 : num_chan
            for trial = 1 : num_trials
                for segment = 1 : number_of_segments
                    temp_FFT = fft(segmented_data(target, channel, trial, segment, :), NFFT)/fft_len;
                    real_part = np.real(temp_FFT);
                    imag_part = np.imag(temp_FFT);
                    features_data(:, channel, target, trial, segment) = cat(1, real_part(fft_index_start:fft_index_end,:), imag_part(fft_index_start:fft_index_end,:));
                end
            end
        end
    end
end
    
