close all;
clear;
clc;

% Load Train and Test Data
fprintf('Loading dataset');
load dataSet.mat;

% Keep the same amount of instances for every label
examples_per_label = min(sum(TrainDataTargets, 2));
fprintf('Keeping %d examples for each label\n', examples_per_label);

index_1 = find(TrainDataTargets(1,:), examples_per_label);
index_2 = find(TrainDataTargets(2,:), examples_per_label);
index_3 = find(TrainDataTargets(3,:), examples_per_label);
index_4 = find(TrainDataTargets(4,:), examples_per_label);
index_5 = find(TrainDataTargets(5,:), examples_per_label);

indexes = [index_1 index_2 index_3 index_4 index_5];
permutation=randperm(size(indexes,2));
indexes=indexes (permutation);
TrainData=TrainData(:,indexes);
TrainDataTargets=TrainDataTargets(:,indexes) ;

% Remove constant rows
[TrainData, PS] = removeconstantrows(TrainData);
TestData = removeconstantrows('apply', TestData, PS);

% Normalize Data
[TrainData, PS] = mapstd(TrainData);
TestData = mapstd('apply', TestData, PS);

% Implement PCA
[TrainData, PS] = processpca(TrainData, 0.002);
TestData = processpca('apply', TestData, PS);

% Instantiate Neural Network with 2 hidden layers
% The first will have 10 neurons and the second 15
net = newff(TrainData, TrainDataTargets, [10 15]);

% Configure the Neural Network
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.2;
net.divideParam.testRatio = 0;

net.trainParam.epochs = 1000;

% Train the Neural Network
net = train(net, TrainData, TrainDataTargets);

% Get output
TestDataOutput = sim(net, TestData);
[accuracy,precision,recall] = eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);
fprintf('Initial Train: Hidden Layer: 2, Neurons: [10, 15], Accuracy: %f\n', accuracy);

% Part 4: Investigate optimal neuron number based on F1 score
% One layer with 5 to 30 neurons with step 5
for neurons = 5:5:30
    net = newff(TrainData, TrainDataTargets, neurons);
    
    net.divideParam.trainRatio = 0.8;
    net.divideParam.valRatio = 0.2;
    net.divideParam.testRatio = 0;
    
    net.trainParam.epochs = 1000;
    
    net = train(net, TrainData, TrainDataTargets);
    
    TestDataOutput = sim(net, TestData);
    [accuracy,precision,recall] = eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);
    fprintf('Hidden Layer: 1, Neurons: %d, Accuracy: %f\n', neurons, accuracy);
end;

% Train for 2 hidden layers
for neurons_1 = 5:5:30
    for neurons_2 = 5:5:30
        net = newff(TrainData, TrainDataTargets, [neurons_1, neurons_2]);
        
        net.divideParam.trainRatio = 0.8;
        net.divideParam.valRatio = 0.2;
        net.divideParam.testRatio = 0;
        
        net.trainParam.epochs = 1000;
        
        net = train(net, TrainData, TrainDataTargets);
        
        TestDataOutput = sim(net, TestData);
        [accuracy,precision,recall] = eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);
        fprintf('Hidden Layers: 2, Neurons: [%d %d], Accuracy: %f\n', neurons_1, neurons_2, accuracy);
    end;
end;
