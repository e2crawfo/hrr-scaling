
Purpose
-------

**cleanup-scaling** is intended to demonstrate the scaling capabilities of the Semantic Pointer Architecture (SPA) 
and its approach to connectionist knowledge representation [(Eliasmith 2013)](http://amazon.com/How-Build-Brain-Architecture-Architectures/dp/0199794545). We accomplish this by creating a spiking neural network capable of encoding  WordNet, a lexical database consisting of ~117,000 items, and traversing the primary relations therein. We show that our technique can encode this human-scale structured knowledge base using much fewer neural resources than any previous approach would require.

Methods
------
WordNet is essentially a graph (i.e. nodes and edges) wherein the nodes 
correspond to synsets (WordNet terminology for concepts), and the edges correspond to relations between synsets and are labelled with a relation-type. 
The two primary relation-types are holonymy and metonymy, which can roughly be thought of as **isA** and **partOf**, respectively and these are the two relation types that we encode in our model.

This package reads WordNet data files to construct the WordNet graph in memory. It then employs a Holographic Reduced Representation (HRR) [(Plate 2003)](http://www.amazon.com/Holographic-Reduced-Representation-Distributed-Information/dp/1575864304/ref=sr_1_2?s=books&ie=UTF8&qid=1373560701&sr=1-2&keywords=tony+plate), 
a type of Vector Symbolic Architecture, to encode the WordNet graph in high-dimensional vectors (we typically use 512 dimensions). Roughly speaking, each synset in the WordNet graph is 
assigned a vector which encodes that synset's relations to other synsets. A different HRR operations can later be used to extract this information from these vectors. 
As an example, consider the synset **dog**. Dogs are canines, and dogs (in the wild) are members of packs, so 
(a subset of) a dog's relational structure can be represented as:

**dog** = **isA(canine)** and **partOf(pack)**

Using HRR operations, a vector is created for **dog** which encodes all the information on the right hand side of this equation. Later, an extraction operation can be performed. Performing
the extraction operation with the vector for **dog** and the vector for **isA** results in a noisy version of the vector for **canine**. So performing the extraction operation effectively
amounts to traversing an edge in the WordNet graph.

Our goal is to encode the WordNet graph in *spiking neurons*, and so we require some way to relate the above scheme to a spiking neural network. 
Our tool for this purpose is the Neural Engineering Framework (NEF), a principled approach to constructing populations of
spiking neurons that represent and manipulate high-level vectors [(Eliasmith & Anderson 2003)](http://www.amazon.com/Neural-Engineering-Representation-Neurobiological-Computational/dp/0262550601). The NEF makes it straightforward to implement the mathematically defined HRR extraction operation in spiking neurons.
Moreover, the operation can be implemented using a number of neurons linear in the dimensionality of the vectors used in the HRR scheme, which is fixed at 512 for our model.

The other required ingredient is a *neural cleanup memory*. This is required because the HRR operation that extracts the information stored in the vectors is inherently noisy.
Cleanup memories are capable of recognizing noisy vectors that are in our vocabulary (i.e. correspond to items in WordNet), and map them to clean versions of themselves. 
A nice feature of our cleanup memory is that it is purely feedforward, meaning that no "settling time" is required. The cleanup memory is the primary neural resource consumer in our model,
as it has to store all ~117,000 synsets in WordNet. However, we show that it scales linearly with the number of items in the vocabulary, requiring roughly 20 neurons per item. See [Stewart et al. 2010](http://compneuro.uwaterloo.ca/publications/stewart2009.html) for more details on the associative cleanup memory.

With these two peices, the neural implementation of the HRR extraction operation and the neural cleanup memory, we are able to traverse the WordNet graph. 
Our network takes as input a vector corresponding to a WordNet synset and another vector corresponding to
a WordNet relation-type, and returns the vector corresponding to the target of that relation. This can then be fed back into the network as input, allowing further edge traversals, and ultimately,
arbitrary traversals of the WordNet graph using the subset of relations that we have encoded.

The code in this package constructs the described neural network, and provides a battery of tests that can be run on the network to demonstrate 
that the WordNet graph can be reliably traversed. Results of these tests, and further details of the model, can be found in an upcoming paper.

How to use
---------
Coming soon...

References
----------

Eliasmith, C. (2013). *How to build a brain: A neural architecture for biological cognition*. New York, NY: Oxford University Press.

Elasimith, C., & Anderson, C. H. (2003). *Neural engineering: Computation, representation and dynamics in neurobiological systems*. Cambridge, MA: MIT Press.

Plate, T. A. (2003). *Holographic reduced representations*. Stanford, CA: CSLI Publication.

Stewart, T., Tang, Y., & Eliasmith, C. (2010). A biologically realistic cleanup memory: Autoassociation in spiking neurons. *Cognitive Systems Research*.
