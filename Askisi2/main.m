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
gridSizes = [[1 5]; [1 10]; [1 15]; [1 20]; [1 30]; [5 5]; [5 10]; [10 5]; [10 10]]';
orderLR = 0.9;
orderEpochs = 250;
tuneLR = 0.1;

% Create SOM for every dataset
for i=1:2
    for gridSize=gridSizes
        mM = minmax(datasets{i});
        somTrainParameters(orderLR, orderEpochs, tuneLR, @gridtop, @dist);
        somCreate(mM, gridSize');
        somTrain(datasets{i});
        figure;
        plot2DSomData(IW, distances, datasets{i})
    end
end