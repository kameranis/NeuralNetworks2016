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


% Instantiate Neural Network with 2 hidden layers
% The first will have 10 neurons and the second 15
[accuracy, precision, recall] = testNN(TrainData, TrainDataTargets, TestData, TestDataTargets, [10 15], 'trainlm');
fprintf('Initial Training: Hidden Layer: 1, Neurons: [10 15], Accuracy: %f\n', accuracy );


% Part 4: Investigate optimal neuron number based on F1 score and accuracy
% One layer with 5 to 30 neurons with step 5
funcs = {'trainlm', 'traingdx', 'traingd', 'traingda'};
accuracies = zeros(1,20);
fprintf('Training for one hidden layer and different neuron number\n');
for func = funcs
    for neurons = 5:5:30
        for j=1:20
            [accuracies(j), precision, recall] = testNN(TrainData, TrainDataTargets, TestData, TestDataTargets, neurons,func{1});
             Fsc(j,:) = harmmean([precision recall],2);
        end
        F_score(neurons/5,:) = mean(Fsc,1);
        accuracy(neurons/5) = mean(accuracies);
        fprintf('Neurons=%d Accuracy=%.4f \n', neurons, accuracy(neurons/5));
    end
    
    figure;
    bar([5:5:30],accuracy);
    title(sprintf('Accuracy according to neurons for one hidden layer with %s',func{1})); 
    xlabel('Hidden layer');
    ylabel('Accuracy');
end;

clear accuracies Fsc precision recall

% Train for 2 hidden layers
fprintf('Training for 2 hidden layers and different neuron numbers\n');
for func=funcs
    for neurons1 = 5:5:30
        for neurons2 = 5:5:30
            for j=1:2
                [accuracies(j), precision, recall] = testNN(TrainData, TrainDataTargets, TestData, TestDataTargets, [neurons1 neurons2], func{1});
                Fsc(j,:) = harmmean([precision recall],2);
            end
            F_score2(neurons1/5, neurons2/5, :) = mean(Fsc, 1);
            accuracy2(neurons1/5, neurons2/5) = mean(accuracies);
            fprintf('Neurons: [%2d %2d], Accuracy: %.4f\n', neurons1, neurons2, accuracy2(neurons1/5, neurons2/5));
        end
    end
    bar(5:5:30,accuracy2);
    title('Accuracy according to neurons for two hidden layers'); 
    xlabel('First layer');
    ylabel('Accuracy');
    legend('5','10','15','20','25','30', 'Location', 'eastoutside', 'Orientation', 'vertical');
end;


