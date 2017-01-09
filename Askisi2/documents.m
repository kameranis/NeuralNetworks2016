function documents(orderLR, orderEpochs, tuneLR) 

%% 4A
load NIPS500.mat
global IW;

dim1 = 10;
dim2 = 10;

tic;
new_patterns = tfidf1(Patterns)';
toc
somTrainParameters(orderLR, 10, tuneLR, @hexagonalTopology, @dist);
tic
%% 4B
MinMax = minmax(new_patterns);
somCreate(MinMax, [dim1 dim2]);
somTrain(new_patterns);
figure;
somShow(IW, [dim1 dim2]);
toc
%% 4Ci
tic;
neuron_docs = zeros(dim1*dim2, 1);
for i = 1:size(new_patterns, 2)
    neuron_docs = neuron_docs + somOutput(new_patterns(:,i));
%     winner = find(somOutput(new_patterns(:,i)));
%     neuron_docs(winner) = neuron_docs(winner) + 1;
end
toc
%% 4Cii
tic;
title_dist = negdist(IW, new_patterns);
[~, idx] = max(title_dist, [], 2);
neuron_title = titles(idx);
toc

%% 4Ciii
tic;
[~, ord] = sort(IW, 2, 'descend');
neuron_term = terms(ord(:, 1:3));

toc;

%% 4Civ
max_vals = max(IW, [], 2);
max_term = 0.3*max_vals;
selected_neurons = find((IW(:, ismember(terms, 'network')) > max_term) + (IW(:, ismember(terms, 'function')) > max_term) > 1);

%%  4Cv
final = 100 * reshape(mean(IW(:, ord(:,1:3))), [100 3]) ./ repmat(max_vals, [1 3]);
