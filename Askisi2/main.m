%% Self Organising Maps - SOMs
close all; clear; clc;

% Load simple data
EightData;
QuestionData;
Cities;
GroupData;

datasets = {EightPatterns, QuestionPatterns, CityCoordinates, GroupPatterns};
dataset_names = {'Eight', 'Question', 'Cities', 'Group'};

global IW distances;

% Set SOM parameters
gridSize = [5 5];
orderLR = 0.9;
orderEpochs = 250;
tuneLR = 0.1;

% Create SOM for every dataset
for i=1:4
    mM = minmax(datasets{i});
    somCreate(mM, gridSize);
    somTrainParameters(orderLR, orderEpochs, tuneLR);
    somTrain(datasets{i});
    figure;
    plot2DSomData(IW, distances, datasets{i})
end