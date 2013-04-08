
Purpose
-------

**cleanup-scaling** demonstrates that neural cleanup memories designed using the 
Neural Engineering Framework (NEF) are capable of encoding human-scale structured 
knowledge representations with relatively modest resource requirements.

Methods
------

Currently, we use WordNet as our large-scale structured representation.
The code reads WordNet data files and for each synset, randomly constructs a unique *ID-vector* in a high-dimensional vector space
(i.e. WordNet concept) randomly. A Holographic Reduced Representation (HRR)
is then used to create *semantic pointers* for each synset. These semantic pointers
effectively store a given synset's relations with other synsets. In particular, we can use NEF techniques to neurally extract this relation information.
If a synset A is related to synset B via a relation-type R, then given the semantic pointer for A and a vector representing R, we can determine the semantic 
pointer for B, all in spiking neurons. Effectively, this means we can neurally traverse the edges of the Wordnet Graph.

The one operation required for this is *unbinding*, which takes the semantic pointer for A and the vector representing R and returns a noisy version of the ID-vector for B. Populations for performing
this operation can be created through a straightforward application of NEF techniques.
To remove the noise and map from the ID-vector to the semantic pointer, we feed the result of the unbinding operation into a neural associative cleanup memory. Structurally, this consists
of a collection of small neural populations, each assigned to a Wordnet synset. Each population is constantly "looking for" the ID vector for its assigned synset. When a population receives an input that
is sufficiently similar to its ID vector, it outputs the semantic pointer for its synset. The output of the associative cleanup memory is typically a clean version of the semantic pointer for B. This can then be fed back into the network as the input, permitting recursive graph traversal, or used in other cognitive tasks. See this paper [Stewart et al. 2009](http://ctnsrv.uwaterloo.ca/cnrglab/node/65) for more details on the associative cleanup memory.

The final spiking neural network permits traveral of the WordNet graph.
Moreover, the neural resource requirements are linear in the number of 
concepts encoded. This is represents a significant step forward for connectionist representations of structured information.

This code constructs the network, and provides a battery of tests that can be run on the network to demonstrate that the graph can be reliably traversed.

How to use
---------
Coming soon...


