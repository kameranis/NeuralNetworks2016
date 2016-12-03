function [ accuracy, precision, recall ] = testNN(TrainData, TrainDataTargets, TestData, TestDataTargets, neurons)
%TESTNN Summary of this function goes here
%   Detailed explanation goes here
net = newff(TrainData, TrainDataTargets, neurons);

% Configure the Neural Network
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.2;
net.divideParam.testRatio = 0;

net.trainParam.epochs = 1000;
net.trainFcn = 'trainlm';

% Train the Neural Network
net = train(net, TrainData, TrainDataTargets);

% Get output
TestDataOutput = sim(net, TestData);
[accuracy,precision,recall] = eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);
%result = struct('accuracy', accuracy, 'precision', precision, 'recall', recall);
end

