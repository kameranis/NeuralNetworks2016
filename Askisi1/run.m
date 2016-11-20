close all;
clear;
clc;

% Load Train and Test Data
fprintf('Loading dataset\n');
load dataSet.mat;

% Preprocessing data
fprintf('Unprocessed data, %d Train Examples with %d features\n', fliplr(size(TrainData)));
[TrainData, TrainDataTargets, TestData, neurons] = preprocess(TrainData, TrainDataTargets, TestData, TestDataTargets);
fprintf('Preprocessed data, %d Train Examples with %d features\n', fliplr(size(TrainData)));
fprintf('Program paused. Press enter to continue.\n\n');
pause;

% Instantiate Neural Network with 2 hidden layers
% The first will have 10 neurons and the second 15
initial_accuracy = testNN(TrainData, TrainDataTargets, TestData, TestDataTargets, [10 15]);
fprintf('Initial Training: Hidden Layer: 2, Neurons: [10 15], Accuracy: %f\n', initial_accuracy);
fprintf('Program paused. Press enter to continue.\n\n');
pause;

% Part 4: Investigate optimal neuron number based on F1 score
% One layer with 5 to 30 neurons with step 5
accuracy_one_hidden = zeros(1, 6);
i = 1;
fprintf('Training for one hidden layer and different neuron number\n');
for neurons = 5:5:30
    accuracy_one_hidden(i) = testNN(TrainData, TrainDataTargets, TestData, TestDataTargets, neurons);
    fprintf('Hidden Layer: 1, Neurons: %d, Accuracy: %f\n', neurons, accuracy_one_hidden(i));
    i = i + 1;
end;
fprintf('Program paused. Press enter to continue.\n\n');
pause;

% Train for 2 hidden layers
fprintf('Training for 2 hidden layers and different neuron numbers\n');
accuracy_two_hidden = zeros(6, 6);
neurons_2d = combvec(5:5:30, 5:5:30);
i = 1;
for neurons = neurons_2d
    accuracy_two_hidden(i) = testNN(TrainData, TrainDataTargets, TestData, TestDataTargets, neurons');
    fprintf('Hidden Layer: 2, Neurons: [%2.0f %2.0f], Accuracy: %f\n', neurons(:), accuracy_two_hidden(i));
    i = i + 1;
end;
fprintf('Program paused. Press enter to continue.\n');
pause;
