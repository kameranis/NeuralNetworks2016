function [ output ] = aggregate( results )
%AGGREGATE Out of N runs, aggregates scores

if length(results) > 5
    [maxValue, maxRow] = max([results.score]);
    [minValue, minRow] = min([results.score]);
    results([maxRow, minRow]) = [];
end

output = struct('score', mean([results.score]), 'neurons', results(1).neurons, 'train_func', results(1).train_func);
end

