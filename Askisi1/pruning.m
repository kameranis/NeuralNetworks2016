clc; clear all;
prepro_nostd;

% Part 4: Pruning

%We choose 1 hidden layer with 30 neurons, as the task commands.
%We also choose purelin for transfer function of output layer. See quest3.m

net = newff(TrainData, TrainDataTargets, [30], {'tansig' 'tansig' 'purelin'},'traingd', 'learngd');

net.divideParam.trainRatio=1;
net.divideParam.valRatio=0;
net.divideParam.testRatio=0;
net.trainParam.epochs=1;
net.trainParam.lr = 0.4;

d = 0.1;
l = 0.01;
performance = [];
non_zero_weights = [];

for i=1:1000
    old_w = getwb(net);
    [net,tr] = train(net,TrainData,TrainDataTargets);   
    new_w = getwb(net) - l*old_w;
    performance = [performance tr.perf(2)];
    idxs = find(abs(new_w) < d);
    new_w(idxs) = 0;
    non_zero_weights = [non_zero_weights (length(new_w)-length(idxs))];
    net = setwb(net,new_w);
    %net.divideParam.trainRatio=0.8;
    %net.divideParam.valRatio=0.2;
    %net.divideParam.testRatio=0;
    %net.trainParam.epochs=1000;
end