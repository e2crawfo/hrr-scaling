
Purpose
-------

**cleanup-scaling** demonstrates that neural cleanup memories designed using the 
Neural Engineering Framework (NEF) are capable of encoding human-scale structured 
knowledge representations with relatively modest resource requirements.

Method
------

Currently, we use WordNet as our large-scale structured representation.
The code reads WordNet data files and constructs ID-vectors for each synset
(i.e. WordNet concept) randomly. A Hierarchical Reduced Representation (HRR)
is then used to create semantic pointers for each synset. These semantic pointers
effectively store the relations between synsets, and can easily be stored and
decoded in neurons using NEF techniques. Neural populations are constructed to implement the HRR operation of unbinding. Finally,
a small neural population is created for each synset, which is designed to 
look for the corresponding ID vector on its input, and output its
corresponding semantic pointer when its activation is high enough. These 
cleanup nodes get input from the unbinding operation. 

The completed neural network permits us to *neurally* traverse the WordNet graph,
while using an amount of neural resources that is linear in the number of 
concepts encoded. A battery of tests is run on the network to check that
the graph can be reliably traversed.

How to use
---------
Coming soon...


