# heuristic-binning-entropy

pip3 install heuristic-binning-entropy==0.2.1

import heuristic_binning_entropy

from heuristic_binning_entropy import entropyHeuristicBinning

import numpy as np

inp = np.random.random(200)*100

a , b , c , d = entropyHeuristicBinning.heuristicbinning(inp)

### suggested number of bins
a

### steps of number of bins for plotting
b

### entropy measure by number of bins for plotting
c

### logistic function fitted , measure of entropy for plotting
d


a , b , c , d = entropyHeuristicBinning.heuristicbinningkl(inp)

### suggested number of bins
a

### steps of number of bins for plotting
b

### entropy measure by number of bins for plotting
c

### logistic function fitted , measure of entropy for plotting
d