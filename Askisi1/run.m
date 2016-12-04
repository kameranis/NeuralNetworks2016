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


% Instantiate Neural Network with 2 hidden layers
% The first will have 10 neurons and the second 15
[accuracy, precision, recall] = testNN(TrainData, TrainDataTargets, TestData, TestDataTargets, [10 15]);
fprintf('Initial Training: Hidden Layer: 1, Neurons: [10 15], Accuracy: %f\n', accuracy );


% Part 4: Investigate optimal neuron number based on F1 score and accuracy
% One layer with 5 to 30 neurons with step 5
%funcs = {'trainlm', 'traingdx', 'traingd', 'traingda'};
accuracies = zeros(1,20);
fprintf('Training for one hidden layer and different neuron number\n');
for neurons = 5:5:30
    %for func=funcs
        results = [];
        for j=1:20
            [accuracies(j), precision, recall] = testNN(TrainData, TrainDataTargets, TestData, TestDataTargets, neurons)% %funcs{1});
             Fsc(j,:) = harmmean([precision recall],2);
        end
        F_score(neurons/5+1,:)=mean(Fsc,1);
        accuracy(neurons/5)=mean(accuracies)
    %end
end;
bar([5:5:30],accuracy);
title('Accuracy according to neurons for two hidden layers'); 
xlabel('First layer');
ylabel('Accuracy');
legend('0','5','10','15','20','25','30');


% % Train for 2 hidden layers
% % fprintf('Training for 2 hidden layers and different neuron numbers\n');
accuracies = [];
for neurons1 = 5:5:30
    for neurons2 = 5:5:30
    %for func=funcs
        for j=1:5
            [accuracies(j), precision, recall] = testNN(TrainData, TrainDataTargets, TestData, TestDataTargets, [neurons1 neurons2]);
             Fsc(j,:) = harmmean([precision recall],2);
        end

        accuracies = [accuracies aggregate(results)];
        fprintf('Hidden Layer: 2, Neurons: [%2.0f %2.0f], Accuracy: %f, train_func=%s\n', accuracies(end).neurons, accuracies(end).score, accuracies(end).train_func);
    end
end;


bar(5:5:30,accuracy);
title('Accuracy according to neurons for two hidden layers'); 
xlabel('First layer');
ylabel('Accuracy');
legend('0','5','10','15','20','25','30');

