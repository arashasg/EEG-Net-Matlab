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

    number_segments = int32(ceil((size(data, 3) - data_overlap)/(duration - data_overlap)));
    temp_buf = zeros(number_segments, duration);
    for i = 1:number_segments
        startOfSegment = (i-1)*(duration - data_overlap) + 1;
        endOfSegment = startOfSegment + duration - 1;
        if endOfSegment > size(data, 3)
            segment = cat(3,data(startOfSegment:size(data, 3)), zeros(1, 1, endOfSegment - size(data, 3)));
        else
            segment = data(startOfSegment:endOfSegment);
        end
        temp_buf(i, :) = segment;
    end
    segmented_data = temp_buf;
end
