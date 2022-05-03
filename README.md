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

## 0.2.1

as stated previously; "Given a set of numbers (independent variable) that has no target measure (dependent variable), what would be a good estimation of the number of 'Bins'. The challenge to balance viewing the data 'too close' or 'too far' is well known. Previously, entropy methods that were used are when there is a target measure, in contrast this library estimates the number of bins without needing a target measure."
Added bin estimation via increment of KL distance; by measuring bin vs (bin + 1) of its KL distance, the distance will diminish as the number of bin increases. In other words, there is not much change in binned distribution (vs its bin +1 distribution) as bin increases, again finding diminishing change at the knee using the  'Kneedle' method ('Kneedle' method finds the maximum rotation location of a discrete data points, Villie & Colleagues 2011 )..

## 0.0.2

Given a set of numbers (independent variable) that has no target measure (dependent variable), what would be a good estimation of the number of 'Bins'. The challenge to balance viewing the data 'too close' or 'too far' is well known. Previously, entropy methods that were used are when there is a target measure, in contrast this library estimates the number of bins without needing a target measure.

Estimation is done circa the location where; as the number of 'Bins' increases, but there is no 'significant' increase of 'surprise'. To improve stability of estimation, said 'Bins' vs surprise is function-fitted using a logistic function which then consequently the knee is estimated using the 'Kneedle' method ('Kneedle' method finds the maximum rotation location of a discrete data points, Villie & Colleagues 2011 )
