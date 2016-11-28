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
initial_accuracy = testNN(TrainData, TrainDataTargets, TestData, TestDataTargets, [10 15], 'trainlm');
fprintf('Initial Training: Hidden Layer: 2, Neurons: [10 15], Accuracy: %f\n', initial_accuracy.accuracy);
fprintf('Program paused. Press enter to continue.\n\n');
pause;

% Part 4: Investigate optimal neuron number based on F1 score
% One layer with 5 to 30 neurons with step 5
funcs = {'traingdx', 'trainlm', 'traingd', 'traingda'};
accuracies = [];
fprintf('Training for one hidden layer and different neuron number\n');
for neurons = 5:5:30
    for func=funcs
        results = [];
        for j=1:20
            result = testNN(TrainData, TrainDataTargets, TestData, TestDataTargets, neurons', func{1});
            results = [results, struct('score', metric(result, 'accuracy'), 'neurons', neurons, 'train_func', func)];
        end
        accuracies = [accuracies aggregate(results)];
        fprintf('Hidden Layer: 1, Accuracy: %f, Neurons: %2.0f, train_func=%s\n', accuracies(end).score, accuracies(end).neurons, accuracies(end).train_func);
    end
end;
fprintf('Program paused. Press enter to continue.\n\n');
pause;

% Train for 2 hidden layers
fprintf('Training for 2 hidden layers and different neuron numbers\n');
neurons_2d = combvec(5:5:30, 5:5:30);
for neurons = neurons_2d
    for func=funcs
        results = [];
        for j=1:20
            result = testNN(TrainData, TrainDataTargets, TestData, TestDataTargets, neurons', func{1});
            results = [results, struct('score', metric(result, 'accuracy'), 'neurons', neurons, 'train_func', func)];
        end
        accuracies = [accuracies aggregate(results)];
        fprintf('Hidden Layer: 2, Neurons: [%2.0f %2.0f], Accuracy: %f, train_func=%s\n', accuracies(end).neurons, accuracies(end).score, accuracies(end).train_func);
    end
end;
fprintf('Program paused. Press enter to continue.\n');
pause;

% Part 6: For the best architecture fine tune
[maxScore, maxArch] = max([accuracies.score]);
train_func = accuracies(maxArch).train_func;
neurons = accuracies(maxArch).neurons;

