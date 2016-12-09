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


% Pruning and Weight Decay

%We choose 1 hidden layer with 30 neurons, as the task commands.

net = newff(TrainData, TrainDataTargets, 30, {'tansig' 'tansig' 'purelin'},'traingd', 'learngd');

net.divideParam.trainRatio=1;
net.divideParam.valRatio=0;
net.divideParam.testRatio=0;
net.trainParam.epochs=1;
net.trainParam.lr = 0.4;
%net.trainParam.goal=0.1

d = 0.1;
l = 0.01;
performance = zeros(1,1000);
non_zero_weights = zeros(1,1000);

for k=1:1000
    old_w = getwb(net);
    [net,tr] = train(net,TrainData,TrainDataTargets);   
    new_w = getwb(net) - l*old_w;
    performance(k) =  tr.perf(2);
    idxs = find(abs(new_w) < d);
    new_w(idxs) = 0;
    non_zero_weights(k) = (length(new_w)-length(idxs));
    net = setwb(net,new_w);
end


 figure;
 bar([1:1000],performance);
  title(sprintf('Performance-Epoch plot')); 
 xlabel('Number of Epochs');
 ylabel('Performance');
 
 figure;
 bar([1:1000],non_zero_weights);
 title(sprintf('Non-zero weights according to epoch of training')); 
 xlabel('Number of Epochs');
 ylabel('Number of non-zero weights');
