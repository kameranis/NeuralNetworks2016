function documents(orderLR, orderEpochs, tuneLR) 

%% 4A
load NIPS500.mat
global  IW;

dim1 = 10;
dim2 = 10;

new_patterns = tfidf1(Patterns);

%% 4B
MinMax = minmax(new_patterns');
somCreate(MinMax, [dim1 dim2]);
somTrainParameters(0.9, 50, 0.01, @hexagonalTopology, @dist);
somTrain(new_patterns);
figure;
somShow(IW, [dim1 dim2]);

%% 4Ci

neuron_docs = zeros(dim1*dim2, 1);
for i = 1 : size(new_patterns, 2)
    winner = find(somOutput(new_patterns(:,i)));
    neuron_docs(winner) = neuron_docs(winner) + 1;
end

%% 4Cii
inputs = zeros(8296, 500);
neuron_title = zeros(dim1*dim2, 1);
for i = 1 : dim1*dim2
    for j = 1 : size(new_patterns,2)
        temp(j) = negdist(IW(i,:), new_patterns(:,j));
    end
    [~, idx] = max(title_dist);
    neuron_title{i} = titles{idx};
end

%% 4Ciii
neuron_term = cell(dim1*dim2, 3);
for i = 1 : dim1*dim2
    [~, ord] = sort(IW(i,:));
    neuron_term{i, 1} = terms{ord(end)};
    neuron_term{i, 2} = terms{ord(end-1)};
    neuron_term{i, 3} = terms{ord(end-2)};
end

%% 4Civ
max_vals = max(IW. [], 2)
max_term = 0.3*max_vals;
term1 = find(ismember(terms, 'network'));
term2 = find(ismember(terms, 'function'));

selected_neurons = find((IW(:, term1) > max_term) && (IW(:, term2) > max_term));

%%  4Cv
for i = 1 : dim1*dim2
    [tmp,indices] = sort(IW(i,:));
    last = indices(end-2:end);
    mean_value=mean(IW(:,last));
    final(i,1:3)=mean_value ./ max_values(i);
end
final=100*final;
