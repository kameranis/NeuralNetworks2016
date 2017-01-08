function runGridSizes( datasets, orderLR, orderEpochs, tuneLR )

global IW distances;

gridSizes = [[1 5]; [1 10]; [1 15]; [1 20]; [1 30]; [5 5]; [5 10]; [10 5]; [10 10]]';

% Create SOM for different grid sizes for gridtop and dist
somTrainParameters(orderLR, orderEpochs, tuneLR, @gridtop, @dist);

for i=1:2
    for gridSize=gridSizes
        mM = minmax(datasets{i});
        somCreate(mM, gridSize');
        somTrain(datasets{i});
        figure;
        plot2DSomData(IW, distances, datasets{i});
    end
end

end

