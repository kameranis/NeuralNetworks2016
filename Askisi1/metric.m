function [ score ] = metric(values , options)
%Placeholder for more complex metric functions

    switch options
        case 'fscore'
            precision = mean(values.precision);
            recall = mean(values.recall);
            score = 2*precision*recall/(precision+recall);
        case 'accuracy'
            score = values.accuracy;
            
    end
end