function runClassification( GroupPatterns, dataset_name, orderLR, orderEpochs, tuneLR )

global IW

% Classification using SOM
tic;

somTrainParameters(orderLR, orderEpochs, tuneLR, @hexagonalTopology, @dist);

mM = minmax(GroupPatterns);
somCreate(mM, [5 5]);
somTrain(GroupPatterns);
figure;
somShow(IW, [5 5]);
filename = sprintf('pictures/%s', dataset_name);
print(gcf, filename, '-dpng', '-r0');

time = toc;
fprintf('Classification ran for a total of %.2f seconds\n', time);

end

