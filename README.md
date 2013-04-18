
Purpose
-------

**cleanup-scaling** demonstrates that neural cleanup memories designed using the 
Neural Engineering Framework (NEF) are capable of encoding human-scale structured 
knowledge representations with relatively modest resource requirements.

Methods
------

Currently, we use WordNet as our large-scale structured representation. WordNet is essentially a Graph wherein the nodes 
correspond to synsets (WordNet terminology for concepts), and the edges correspond to relations between synsets (each 
edge also has a type, which corresponds to the type of relation). This package reads WordNet data files to construct the
WordNet graph in memory. It then employs a Holographic Reduced Representation (HRR), a type of Vector Symbolic 
Architecture, to encode the WordNet graph in vectors. Each synset is randomly assigned an *ID-vector* in a high-
dimensional vector space. HRR techniques are then used to create a *semantic pointers* for each synset. The semantic 
pointers are vectors of the same dimension as the ID-vectors, andeffectively store a given synset's relations with other 
synsets. Conveniently, we can use NEF techniques to construct neural ensembles that store semantic pointers and even 
extract their relational information. For example, if a synset A is related to synset B via a relation-type R, we can 
construct a network of spiking neurons, which, given the semantic pointer for A and a vector representing R, can 
determine the semantic pointer for B. This same network can be used for all synsets in the vocabulary. Effectively, this 
means we can neurally traverse the edges of the WordNet Graph.

One operation required for this is *unbinding*, which takes the semantic pointer for A and the vector representing the 
relation R and returns B', a noisy version of the ID-vector for B. Populations for performing this operation can be 
implemented in neurons through a straightforward application of NEF techniques. To remove the noise from B' and map from 
the ID-vector for B to the semantic pointer for B, we feed the result of the unbinding operation into a neural 
associative cleanup memory. This consists of a collection of small neural populations, each assigned to a unique WordNet 
synset. When one of these populations receives an input that is sufficiently similar to the ID-vector for its assigned 
synset, it outputs the corresponding semantic pointer. Otherwise, the population is silent, effectively outputting 0. 
Finally, the outputs of these ensembles are summed. Returning to our example, B' will be passed into the neural 
associative cleanup memory. If the cleanup memory is correctly constructed, the population assigned to the synset B will 
become active while all other populations remain silent. The output will thus be a clean version of the semantic pointer 
for B. This can then be fed back into the network as the input, permitting recursive graph traversal, or used in other 
cognitive tasks. See [Stewart et al. 2009](http://ctnsrv.uwaterloo.ca/cnrglab/node/65) for more details on the 
associative cleanup memory.

The final spiking neural network permits traveral of the WordNet graph. Moreover, the neural resource requirements are 
linear in the number of concepts encoded. This represents a significant step forward for connectionist representations 
of structured information.

This code constructs the neural network, and provides a battery of tests that can be run on the network to demonstrate 
that the WordNet graph can be reliably traversed.

How to use
---------
Coming soon...


