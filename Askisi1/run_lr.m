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

% Change lr
for lr = 0.05:0.05:0.4
    for func = {'traingd', 'traingdx'}
        for i=1:2
            net = newff(TrainData, TrainDataTargets, neurons);

            % Configure the Neural Network
            net.divideParam.trainRatio = 0.8;
            net.divideParam.valRatio = 0.2;
            net.divideParam.testRatio = 0;

            net.trainParam.epochs = 1000;
            net.trainFcn =  func{1};
            net.trainParam.lr = lr;

            % Train the Neural Network
            net = train(net, TrainData, TrainDataTargets);

            % Get output
            TestDataOutput = sim(net, TestData);
            [accuracies(i),precision,recall] = eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);
            Fsc(i,:) = harmmean([precision recall],2);
        end
        F_score(cast(lr*20, 'int32'),:) = mean(Fsc,1);
        accuracy(cast(lr*20, 'int32')) = mean(accuracies);
        fprintf('Training function: %s, lr: %.2f, Accuracy=%.4f \n', func{1}, lr, accuracy(cast(lr*20, 'int32')));
    end
end