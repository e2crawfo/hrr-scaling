## Purpose

**hrr-scaling** is intended to demonstrate the scaling capabilities of the Semantic Pointer Architecture (SPA) and its approach to connectionist knowledge representation [(Eliasmith 2013)](http://compneuro.uwaterloo.ca/research/spa.html). We accomplish this by creating a spiking neural network capable of encoding WordNet, a lexical database consisting of ~117,000 items, and traversing the primary relations therein. We show that our technique can encode this human-scale structured knowledge base using much fewer neural resources than any previous approach would require. Our results were outlined in a [paper presented at CogSci 2013](http://mindmodeling.org/cogsci2013/papers/0099/paper0099.pdf) (Crawford 2013). A longer, more detailed version has been published in *Cognitive Science* (Crawford 2015).

## Methods
See either of the above papers for a detailed overview of our methods. Briefly, we use a particular vector symbolic architecture, called the Semantic Pointer Architecture (which can be viewed as a neural variant of Holographic Reduced Representations (Plate, 2003)), to encode the WordNet graph in vectorial form. We then employ the [Neural Engineering Framework](http://compneuro.uwaterloo.ca/research/nef.html), a principled approach to creating populations of spiking neurons that represent and transform vectors (Eliasmith and Anderson, 2003), to create a spiking neural network capable of traversing this vectorial representation of the WordNet graph in a biologically plausible number of neurons.

## Setup
Running this package requires [python 2.7](http://www.python.org/getit/) and [numpy/scipy](http://www.scipy.org/install.html). For displaying graphical views of neural data, [matplotlib](http://matplotlib.org/users/installing.html) is required, though it can be omitted if you just want to view the performance.

The model supports GPU acceleration through nVidia's CUDA API, and indeed this is all but required for running the model with all ~117,000 concepts in WordNet. However, if you don't have access to a CUDA-capable GPU, you can still run the model with a reduced number of concepts using the -p command line argument (see below).

To obtain the package, simply clone this repository onto your machine using the ``git clone`` command.

#### GPU Setup
If you don't intend to use a GPU to run simulations, this section can be safely skipped. Otherwise read on, and in just a few simple steps you can be running simulations at GPU-accelerated speeds
on your CUDA-capable GPU(s). Note that this has only been tested on Ubuntu. 

1. Install the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads).

2. Compile the *neuralGPUCleanup* shared library. The source code for this library can be found in the *neuralGPUCleanup* subdirectory of this repository. If step 1 was completed properly, it should be as simple as typing ``make``, though one does sometimes run into pitfalls. More info on this coming soon, in particular, a FAQ addressing common errors encountered in the process of compiling CUDA libraries.

## Running Experiments
Users interact with the package through the ``run.py`` script. ``run.py`` handles all the heavy lifting of loading the WordNet graph into memory, converting it into a vectorial representation, and creating a spiking neural network capable of traversing the edges in the WordNet graph encoded by those vectors. A number of command line options are provided which provide control over which experiments are run and under what conditions.

#### Minimal Example
A minimal example that will execute quickly but still illustrate the capabilities of the network, can be run with the following command:

```
python run.py --jump 1 -p 0.001 -d 128 --probe-all
```

This command picks out a subgraph of WordNet containing only 0.1% of the ~117,000 concepts in WordNet (``--p 0.001``), creates a vectorized representation of that subgraph using vectors with 128 dimensions (``-d 128``), and then uses that vectorized representation to create a spiking neural network capable of traversing the edges in that sub-graph. ``--probe-all`` instructs the code to attach a probe to each of the ensembles inside the associative memory so that the activations of those populations can be recorded and plotted. ``--jump 1`` instructs the code to perform one instance of the jump test on the network, randomly picking an edge in the subgraph and testing the ability of the network to traverse (or jump along) it.

Each time the ``run.py`` script is executed, a subdirectory marked with the time of execution is created in the ``results`` directory. A convenience symbolic link called ``latest`` is also created to point at the most recently created subdirectory. Inside the ``results`` directory are a number of different files. One file will be created for each type of test that was executed, containing details from the execution of that type of test. If the neural model was used (i.e. the --abstract keyword was not supplied), and plotting was not turned off, then plots of neural activity will be stored here as well. The file called ``results`` contains most data gathered through the simulation. In there, fields containing the word "score" give the overall results on each test. E.g. the field called 'jump_score_correct' contains the overall scores for the jump test.


#### Experiments from our papers
The experiments in our CogSci 2013 paper can be run with the following commands:

```
python run.py --jump 100 --hier 20 --sent 30 --runs 20
```

(jump is the nickname used throughout the code for the simple extraction test).  This command can be parsed as: run 100 instances of the simple extraction test, 20 instances of the hierarchical extraction test, and 30 instances of the sentence extraction test, and repeat all of that 20 times (for stat-gathering).

In our 2015 journal paper, we change the sentence test ("sent" in the command above) to a "deep" sentence test, where we construct and extract from sentences that contain embedded clauses. We also require the role vectors used to create the deep sentences to be *unitary* vectors (see the paper for mrore details). Consequently, to run the experiments from that paper, use the command:

```
python run.py --jump 100 --hier 20 --deep 30 --runs 20 --unitary-roles
```

In the 2015 paper, we compare the neural model to an "abstract" version of the model, which operates under the same principles but skips the neural implementation, computing all the HRR operations exactly. To run the same tests on the abstract model, simply supply the ``--abstract`` command line option:

```
python run.py --abstract --jump 100 --hier 20 --deep 30 --runs 20 --unitary-roles
```

#### GPU Notes
To tell the package to use GPU acceleration when running experiments, supply the --gpus command line argument, followed by the number of GPU's you want to use to run the model (e.g. --gpus 1). For example, to run the Deep Sentence Test experiment from the paper using 1 GPU (assuming the the GPU library has been installed properly), one would type:

```
python run.py --deep 30 --runs 20 --gpus 1
```

## Additional Command Line Options
This section provides more complete descriptions of some of the most useful command line options. An exhaustive list can be obtained by running ``python run.py -h``.

``-p P`` : P is a float between 0 and 1. Permits specification of the fraction of the total number of WordNet synsets to use. Defaults to 1.0. Useful for running smaller versions of the model, particularly if you don't have access to a GPU.

``--gpus G`` : G is a non-negative integer. Defaults to 0. Specifies the number of CUDA-capable GPU's to use in executing the neural simulation. At least one is required to run the full model in a reasonable amount of time.

``-d D`` : D is a positive integer. Permits specification of the dimensions of the vectors used to encode WordNet. Defaults to 512. With smaller values, the average similarity of any two randomly chosen vectors increases, making the associative memory less accurate. Extraction, using the combination of approximate inverse and circular convolution, also becomes less noisy as the dimensionality increases.

``--no-ids`` : Specifies that id vectors should not be used. Instead, both the index vectors and the stored vectors in the associative memory are semantic pointers. This tends to reduce the performance of the associative memory (because the average similarity of two semantic pointers is much higher than that of two randomly chosen vectors). However, it frees us from having to keep track of two vectors per WordNet concept, simplifying the model somewhat.

``--unitary-rels`` : Specifies that the relation-type vectors that are used to create semantic pointers should be unitary vectors. For initary vectors, circular convolution is the exact inverse instead of the approximate inverse, so performance typically improves.


## References

Crawford, E., Gingerich, M., and Eliasmith, C. (2013). Biologically plausible, human-scale knowledge representation. In *35th Annual Conference of the Cognitive Science Society*, 412-417.

Crawford, E., Gingerich, M., and Eliasmith, C. (2015). Biologically plausible, human-scale knowledge representation. *Cognitive Science*. doi: 10.1111/cogs.12261

Eliasmith, C. (2013). *How to build a brain: A neural architecture for biological cognition*. New York, NY: Oxford University Press.

Eliasmith, C., & Anderson, C. H. (2003). *Neural engineering: Computation, representation and dynamics in neurobiological systems*. Cambridge, MA: MIT Press.

Plate, T. A. (2003). *Holographic reduced representations*. Stanford, CA: CSLI Publication.
