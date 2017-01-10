function runTopologies( datasets, dataset_names, orderLR, orderEpochs, tuneLR )

global IW distances;

% Create SOM for different topologies and distance functions
tic;

for topology={@gridtop, @hexagonalTopology}
    for distFunction={@dist, @mandist};
        somTrainParameters(orderLR, orderEpochs, tuneLR, topology{1}, distFunction{1});
        
        top = functions(topology{1});
        d = functions(distFunction{1});
        
        % Eight Patterns
        gridSize = [5 5];
        mM = minmax(datasets{1});
        somCreate(mM, gridSize);
        somTrain(datasets{1});
        figure;
        plot2DSomData(IW, distances, datasets{1})
        filename = sprintf('pictures/%s %s %s', dataset_names{1}, top.function, d.function);
%         filename
%         class(filename)
        print(gcf, filename, '-dpng', '-r0');

        % Question Patterns
        gridSize = [1 15];
        mM = minmax(datasets{2});
        somCreate(mM, gridSize);
        somTrain(datasets{2});
        figure;
        plot2DSomData(IW, distances, datasets{2})
        filename = sprintf('pictures/%s %s %s', dataset_names{2}, top.function, d.function);
        print(gcf, filename, '-dpng', '-r0');
    end
end
time = toc;
fprintf('Topologies ran for a total of %.2f seconds\n', time);

end

