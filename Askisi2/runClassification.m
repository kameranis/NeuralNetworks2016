function runClassification( GroupPatterns, orderLR, orderEpochs, tuneLR )

global IW

% Classification using SOM
somTrainParameters(orderLR, orderEpochs, tuneLR, @hexagonalTopology, @dist);

mM = minmax(GroupPatterns);
somCreate(mM, [5 5]);
somTrain(GroupPatterns);
figure;
somShow(IW, [5 5]);
end

