function [TrainData, TrainDataTargets, TestData, TestDataTargets] = preprocess(TrainData, TrainDataTargets, TestData, TestDataTargets)
%PREPROCESS Summary of this function goes here
%   Detailed explanation goes here
% Keep the same amount of instances for every label
examples_per_label = min(sum(TrainDataTargets, 2));
fprintf('Keeping %d examples for each label\n', examples_per_label);

index_1 = find(TrainDataTargets(1,:), examples_per_label);
index_2 = find(TrainDataTargets(2,:), examples_per_label);
index_3 = find(TrainDataTargets(3,:), examples_per_label);
index_4 = find(TrainDataTargets(4,:), examples_per_label);
index_5 = find(TrainDataTargets(5,:), examples_per_label);

indexes = [index_1 index_2 index_3 index_4 index_5];
permutation=randperm(size(indexes,2));
indexes=indexes (permutation);
TrainData=TrainData(:,indexes);
TrainDataTargets=TrainDataTargets(:,indexes) ;

% Remove constant rows
[TrainData, PS] = removeconstantrows(TrainData);
TestData = removeconstantrows('apply', TestData, PS);

% Normalize Data
[TrainData, PS] = mapstd(TrainData);
TestData = mapstd('apply', TestData, PS);

% Implement PCA
[TrainData, PS] = processpca(TrainData, 0.01);
TestData = processpca('apply', TestData, PS);

end

