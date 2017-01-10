function runGridSizes( datasets, dataset_names, orderLR, orderEpochs, tuneLR )

global IW distances;

gridSizes = [[1 5]; [1 10]; [1 15]; [1 20]; [1 30]; [5 5]; [5 10]; [10 5]; [10 10]]';

% Create SOM for different grid sizes for gridtop and dist
tic;

somTrainParameters(orderLR, orderEpochs, tuneLR, @gridtop, @dist);

for i=1:2
    for gridSize=gridSizes
        mM = minmax(datasets{i});
        somCreate(mM, gridSize');
        somTrain(datasets{i});
        figure;
        plot2DSomData(IW, distances, datasets{i});
        filename = sprintf('pictures/%s %s', dataset_names{i}, mat2str(gridSize'));
        print(gcf, filename, '-dpng', '-r0');
    end
end
time = toc;
fprintf('Grid sizes ran for a total of %.2f seconds\n', time);
end

