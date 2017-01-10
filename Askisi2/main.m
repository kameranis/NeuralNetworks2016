%% Self Organising Maps - SOMs
close all; clear; clc;

[datasets, dataset_names] = loadDatasets();

%% Set SOM parameters
orderLR = 0.9;
orderEpochs = 250;
tuneLR = 0.1;

%% 2A2 Create SOMs with different topologies
runTopologies(datasets, dataset_names, orderLR, orderEpochs, tuneLR);

%% 2A3 Create SOMs with different grid sizes
runGridSizes(datasets, dataset_names, orderLR, orderEpochs, tuneLR);

%% 2B Solve the TSP problem using SOM
runTSP(datasets{3}, dataset_names{3}, orderLR, orderEpochs, tuneLR);

%% 2C Perform Classification using SOM
runClassification(datasets{4}, dataset_names{4}, orderLR, orderEpochs, tuneLR);

%% Document Classification
[neuron_title, neuron_term, selected_neurons, final] = documents(orderLR, orderEpochs, tuneLR);


