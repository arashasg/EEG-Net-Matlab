function [train_data, labels] = get_training_data(features_data)
    features_data = reshape(features_data, size(features_data, 1) , size(features_data, 2), size(features_data, 3), size(features_data, 4) *size(features_data, 5));
    train_data = permute(squeeze(features_data(:, :, 1, :)), [3, 2, 1]);
    for target = 2 : size(features_data, 3)
        train_data = vertcat(train_data, permute(squeeze(features_data(:, :, target, :)), [3, 2, 1]));
    end

    train_data = reshape(train_data, size(train_data, 1), size(train_data, 2), size(train_data, 3), 1);
    total_epochs_per_class = size(features_data, 4);
    features_data = [];
    class_labels = 1:5;
    labels = reshape(transpose(repmat(class_labels, total_epochs_per_class, 2)),1,[]);    
end