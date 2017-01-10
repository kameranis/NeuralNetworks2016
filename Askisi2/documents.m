function [neuron_title, neuron_term, selected_neurons, final] = documents(orderLR, orderEpochs, tuneLR) 

%% 4A
load NIPS500.mat
global IW;

dim1 = 10;
dim2 = 10;

%% Compute term frequency–inverse document frequency
tic;
new_patterns = tfidf1(Patterns)';
fprintf('Term frequency–inverse document frequency was computed in %.4f\n', toc);

%% 4B Train Document SOM 
tic;
somTrainParameters(orderLR, orderEpochs, tuneLR, @hexagonalTopology, @dist);
MinMax = minmax(new_patterns);
somCreate(MinMax, [dim1 dim2]);
somTrain(new_patterns);
figure;
somShow(IW, [dim1 dim2]);
fprintf('Documents SOM was trained in %.2f\n', toc);

%% 4Ci Compute how many documents belong to each neuron
tic;
neuron_docs = zeros(N, 1);
for i = 1:size(new_patterns, 2)
    neuron_docs = neuron_docs + somOutput(new_patterns(:,i));
end
fprintf('Computing number of patterns closest to each neuron took %.4f\n', toc);

%% 4Cii Compute the title of the closest document to each neuron
tic;
title_dist = negdist(IW, new_patterns);
[~, idx] = max(title_dist, [], 2);
neuron_title = titles(idx);
fprintf('Computing title of closest document to each neuron took %.4f\n', toc);

%% 4Ciii Compute the 3 largest terms for each neuron
tic;
[~, ord] = sort(IW, 2, 'descend');
neuron_term = terms(ord(:, 1:3));
fprintf('Computing the three largest terms of each neuron took %.4f\n', toc);

%% 4Civ Neurons where "network" and "function" are both above 30% of the maximum value
tic;
max_vals = max(IW, [], 2);
max_term = 0.3 * max_vals;
selected_neurons = find((IW(:, ismember(terms, 'network')) > max_term) + (IW(:, ismember(terms, 'function')) > max_term) > 1);
fprintf('Computing the neurons with strong "network" and "function" presence took %.4f\n', toc);

%%  4Cv Percent of maximum value of terms from 4Ciii
tic;
final = 100 * reshape(mean(IW(:, ord(:,1:3))), [100 3]) ./ repmat(max_vals, [1 3]);
fprintf('Computing 4Cv took %.4f\n', toc);
