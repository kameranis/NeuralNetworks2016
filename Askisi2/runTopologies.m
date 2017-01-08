function runTopologies( datasets, orderLR, orderEpochs, tuneLR )

global IW distances;

% Create SOM for different topologies and distance functions
for topology={@gridtop, @hexagonalTopology}
    for distFunction={@dist, @mandist};
        somTrainParameters(orderLR, orderEpochs, tuneLR, topology{1}, distFunction{1});
        
        % Eight Patterns
        gridSize = [5 5];
        mM = minmax(datasets{1});
        somCreate(mM, gridSize);
        somTrain(datasets{1});
        figure;
        plot2DSomData(IW, distances, datasets{1})

        % Question Patterns
        gridSize = [1 15];
        mM = minmax(datasets{2});
        somCreate(mM, gridSize);
        somTrain(datasets{2});
        figure;
        plot2DSomData(IW, distances, datasets{2})
    end
end

end

