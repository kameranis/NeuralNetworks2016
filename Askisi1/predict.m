function TestDataOutput = predict(TrainData, TrainDataTargets, TestData, TestDataTargets, neurons, trainFunc, transferFcn, learnFcn)
%TESTNN Summary of this function goes here
%   Detailed explanation goes here
if nargin < 8
    learnFcn = 'learngdm';
end
if nargin < 7 
    transferFcn = 'tansig';    
end

net = newff(TrainData, TrainDataTargets, neurons);
net.layers{length(neurons)}.transferFcn = transferFcn;


% Configure the Neural Network
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.2;
net.divideParam.testRatio = 0;

net.trainParam.epochs = 1000;
net.trainFcn =  trainFunc;
net.derivFcn = learnFcn;
net.trainParam.showWindow = 0;

% Train the Neural Network
net = train(net, TrainData, TrainDataTargets);

% Get output
TestDataOutput = sim(net, TestData);
[~, TestDataOutput] = max(TestDataOutput);
end

