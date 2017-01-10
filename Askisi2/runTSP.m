function runTSP( CityCoordinates, dataset_name, orderLR, orderEpochs, tuneLR )

global IW distances

% Solve TSP using SOM
tic;

somTrainParameters(orderLR, orderEpochs, tuneLR, @gridtop, @ring_dist);

mM = minmax(CityCoordinates);
somCreate(mM, [1 60]);
somTrain(CityCoordinates);
figure;
plot2DSomData(IW, distances, CityCoordinates);
filename = sprintf('pictures/%s', dataset_name);
print(gcf, filename, '-dpng', '-r0');

time = toc;
fprintf('TSP ran for a total of %.2f seconds\n', time);

end

