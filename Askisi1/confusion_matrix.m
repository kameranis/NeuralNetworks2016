close all;
clear;
clc;

% Load Train and Test Data
fprintf('Loading dataset\n');
load dataSet.mat;

% Preprocessing data
fprintf('Unprocessed data, %d Train Examples with %d features\n', fliplr(size(TrainData)));
[TrainData, TrainDataTargets, TestData, TestDataTargets] = preprocess(TrainData, TrainDataTargets, TestData, TestDataTargets);
fprintf('Preprocessed data, %d Train Examples with %d features\n', fliplr(size(TrainData)));

% Part 4: Investigate optimal neuron number based on F1 score and accuracy
% One layer with 5 to 30 neurons with step 5
[~, ghat] = max(TestDataTargets);
neurons = 30;
func = {'trainlm'};
for j=1:100
    if(mod(j, 10) == 0)
        fprintf('%d ', j);
    end
    predictions = predict(TrainData, TrainDataTargets, TestData, TestDataTargets, neurons, func{1});
    confusion(j, :, :) = confusionmat(ghat, predictions);
end
fprintf('\nNeurons: %2d, Training function: %8s\n', neurons, func{1});
fprintf('%2.2f %2.2f %2.2f %2.2f %2.2f\n', squeeze(mean(confusion)));