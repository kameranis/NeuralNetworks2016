function runTSP( CityCoordinates, orderLR, orderEpochs, tuneLR )

global IW distances

% Solve TSP using SOM
somTrainParameters(orderLR, orderEpochs, tuneLR, @gridtop, @ring_dist);

mM = minmax(CityCoordinates);
somCreate(mM, [1 60]);
somTrain(CityCoordinates);
figure;
plot2DSomData(IW, distances, CityCoordinates);
end

