function somTrainParameters(setOrderLR,setOrderSteps,setTuneLR,tuneTopology,tuneDistanceFunction)

global distances maxNeighborDist tuneND orderLR orderSteps tuneLR topology distanceFunction;

maxNeighborDist = ceil(max(max(distances)));
tuneND = 1;

orderLR = setOrderLR; 
orderSteps = setOrderSteps;
tuneLR = setTuneLR;
topology = tuneTopology;
distanceFunction = tuneDistanceFunction;