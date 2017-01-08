%% Self Organising Maps - SOMs
close all; clear; clc;

[datasets, dataset_names] = loadDatasets();

% Set SOM parameters
orderLR = 0.9;
orderEpochs = 250;
tuneLR = 0.1;

% 2A2
runTopologies(datasets, orderLR, orderEpochs, tuneLR);

%%
% 2A3
runGridSizes(datasets, orderLR, orderEpochs, tuneLR);

% 2B
runTSP(datasets{3}, orderLR, orderEpochs, tuneLR);

% 2C

runClassification(datasets{4}, orderLR, orderEpochs, tuneLR);

% Document Classification


