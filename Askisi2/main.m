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

% Create SOM for different topologies and distance functions
for topology={@gridtop, @hexagonalTopology}
    for distFunction={@dist, @mandist};
        somTrainParameters(orderLR, orderEpochs, tuneLR, topology{1}, distFunction{1});
        
        % Eight Patterns
        gridSize = [10 5];
        mM = minmax(datasets{1});
        somCreate(mM, gridSize);
        somTrain(datasets{1});
        figure;
        plot2DSomData(IW, distances, datasets{1})

        % Question Patterns
        gridSize = [1 20];
        mM = minmax(datasets{2});
        somCreate(mM, gridSize);
        somTrain(datasets{2});
        figure;
        plot2DSomData(IW, distances, datasets{2})
    end
end

% Create SOM for different grid sizes for gridtop and dist
somTrainParameters(orderLR, orderEpochs, tuneLR, @gridtop, @dist);

for i=1:2
    for gridSize=gridSizes
        mM = minmax(datasets{i});
        somCreate(mM, gridSize');
        somTrain(datasets{i});
        figure;
        plot2DSomData(IW, distances, datasets{i});
    end
end

% Solve TSP using SOM
somTrainParameters(orderLR, orderEpochs, tuneLR, @gridtop, @ring_dist);

mM = minmax(CityCoordinates);
somCreate(mM, [1 60]);
somTrain(CityCoordinates);
figure;
plot2DSomData(IW, distances, CityCoordinates);

% Classification using SOM
somTrainParameters(orderLR, orderEpochs, tuneLR, @hexagonalTopology, @dist);

mM = minmax(GroupPatterns);
somCreate(mM, [5 5]);
somTrain(GroupPatterns);
figure;
somShow(IW, [5 5]);