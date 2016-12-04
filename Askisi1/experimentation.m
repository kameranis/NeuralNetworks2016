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

% Set architecture from last step
neurons = [10 5];
trainFunc = 'trainlm';

outputTransferFuncs = {'hardlim', 'tansig', 'logsig', 'purelin'};
i = 1;
for func = outputTransferFuncs
    for j=1:2
        [accuracies(j), precision, recall] = testNN(TrainData, TrainDataTargets, TestData, TestDataTargets, neurons, trainFunc, func{1});
        Fsc(j,:) = harmmean([precision recall],2);
    end
    F_score(i,:) = mean(Fsc,1);
    accuracy(i) = mean(accuracies);
    fprintf('Output activation function: %s, Accuracy=%.4f\n', func{1}, accuracy(i));
    i = i + 1;
end

[maxAcc, maxInd] = max(accuracy);
transferFunc = outputTransferFuncs{maxInd};
clear F_score accuracy accuracies

learnFuncs = {'learngd', 'learngdm'};
i = 1;
for learnFunc = learnFuncs
    for j=1:2
        [accuracies(j), precision, recall] = testNN(TrainData, TrainDataTargets, TestData, TestDataTargets, neurons, trainFunc, transferFunc, learnFunc{1});
        Fsc(j,:) = harmmean([precision recall],2);
    end
    F_score(i,:) = mean(Fsc,1);
    accuracy(i) = mean(accuracies);
    fprintf('learning function: %s, Accuracy=%.4f\n', learnFunc{1}, accuracy(i));
    i = i + 1;
end

[maxAcc, maxInd] = max(accuracy);
learnFunc = learnFuncs{maxInd};
clear F_score accuracy accuracies

% See how it deals with no validation
net = newff(TrainData, TrainDataTargets, neurons);
net.layers{length(neurons)}.transferFcn = transferFunc;

% Configure the Neural Network
net.divideParam.trainRatio = 1.0;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 0;

net.trainParam.epochs = 20;
net.trainFcn =  trainFunc;
net.derivFcn = learnFunc;

% Train the Neural Network
net = train(net, TrainData, TrainDataTargets);

% Get output
TestDataOutput = sim(net, TestData);
[accuracy,precision,recall] = eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);
fprintf('No validation accuracy: %.4f\n', accuracy);

