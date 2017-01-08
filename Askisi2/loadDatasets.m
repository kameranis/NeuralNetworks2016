function [ datasets, dataset_names ] = loadDatasets()
% Load simple data
EightData;
QuestionData;
Cities;
GroupData;

datasets = {EightPatterns, QuestionPatterns, CityCoordinates, GroupPatterns};
dataset_names = {'Eight', 'Question', 'Cities', 'Group'};
end

