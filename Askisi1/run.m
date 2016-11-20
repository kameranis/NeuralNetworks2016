close all;
clear;
clc;

% Load Train and Test Data
fprintf('Loading dataset\n');
load dataSet.mat;

% Preprocessing data
[TrainData, TrainDataTargets, TestData, neurons] = preprocess(TrainData, TrainDataTargets, TestData, TestDataTargets);

% Instantiate Neural Network with 2 hidden layers
% The first will have 10 neurons and the second 15
initial_accuracy = testNN(TrainData, TrainDataTargets, TestData, TestDataTargets, [10 15]);

% Part 4: Investigate optimal neuron number based on F1 score
% One layer with 5 to 30 neurons with step 5
accuracy_one_hidden = zeros(1, 6);
i = 1;
for neurons = 5:5:30
    accuracy_one_hidden(i) = testNN(TrainData, TrainDataTargets, TestData, TestDataTargets, neurons);
    i = i + 1;
end;

% Train for 2 hidden layers
accuracy_two_hidden = zeros(6, 6);
i = 1;
for neurons = combvec(5:5:30, 5:5:30)
    accuracy_two_hidden(i) = testNN(TrainData, TrainDataTargets, TestData, TestDataTargets, neurons');
    i = i + 1;
end;
