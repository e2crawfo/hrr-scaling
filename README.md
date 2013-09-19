Purpose
=====

**cleanup-scaling** is intended to demonstrate the scaling capabilities of the Semantic Pointer Architecture (SPA) and its approach to connectionist knowledge representation [(Eliasmith 2013)](http://compneuro.uwaterloo.ca/research/spa.html). We accomplish this by creating a spiking neural network capable of encoding WordNet, a lexical database consisting of ~117,000 items, and traversing the primary relations therein. We show that our technique can encode this human-scale structured knowledge base using much fewer neural resources than any previous approach would require. Our results were outlined in a [paper presented at CogSci 2013](http://mindmodeling.org/cogsci2013/papers/0099/paper0099.pdf). A longer, more detailed version featuring additional empirical data is currently in review.

Methods
=======
See our CogSci 2013 paper for a detailed over view of our methods. Briefly, we use a particular vector symbolic architecture, called the Semantic Pointer Architecture (which can be viewed as a neural varient of Holographic Reduced Representations (Plate, 2003)), to encode the WordNet graph in vectorial form. We then employ the [Neural Engineering Framework](http://compneuro.uwaterloo.ca/research/nef.html), a principled approach to creating populations of spiking neurons that represent and transform vectors (Eliasmith and Anderson, 2003), to create a spiking neural network capable of traversing this vectorial representation of the WordNet graph in a biologically plausible number of neurons.

Setup
==========
Running this package requires [python 2.7](http://www.python.org/getit/) and [numpy/scipy](http://www.scipy.org/install.html). For displaying graphical views of results, [matplotlib](http://matplotlib.org/users/installing.html) is required, though it can be omitted if you're fine with viewing the data textually.

The model supports GPU acceleration through nVidia's CUDA API, and indeed this is all but required for running the model with all ~117,000 concepts in WordNet. 
However, if you don't have access to a CUDA-capable GPU, you can still run the model with a reduced number of concepts. 

To obtain the package, simply clone this repository onto your machine using the ``git clone`` command.

####GPU Setup
If you don't intend to use a GPU to run simulations, this section can be safely skipped. Otherwise read on, and in just a few simple steps you can be running simulations at GPU-accelerated speeds
on your CUDA-capable GPU(s). Note that this has only been tested on Ubuntu. 

1. Install the CUDA runtime libraries. Detailed instructions for this process can be found [here]().

2. Compile the *neuralGPUCleanup* shared library. The source code for this library can be found in the *neuralGPUCleanup* subdirectory of this repository. If step 1 was completed properly, it should be as simple as typing ``make``, though one does sometimes run into pitfalls. More info on this coming soon, in particular, a FAQ addressing common errors encountered in the process of compiling CUDA libraries.

Running Experiments
==========
Users interact with the package through a python script called ``run_script.py`` which is located in the ``python`` subdirectory. ``run_script.py`` handles all the heavy lifting of loading the WordNet graph into memory, converting it into 
a vectorial representation, and, finally, creating a spiking neural network capable of traversing the edges in the WordNet graph encoded by those vectors. A number of command line options are provided which provide control over which experiments are run and under what conditions.

In our CogSci 2013, paper we outline 3 experiments that we perform on our model to ensure that it encodes the WordNet graph. These are the Single Edge Traversal, Hierarchy Traversal and
Sentence Encoding. In the code, these are nicknamed ``Jump Test``, ``Hierarchy Test`` and ``Sentence Test``, respectively. To construct the model and run a test on it, call ``run_script.py`` as follows:

```
python run_script.py <test-type> <batches> <trials>
```

where **test-type** is one of j, h or s indicating the type of test to run, **batches** is a positive integer indicating the number of batches to run, and **trials** is a positive integer indicating the number of trials of the chosen test per batch. For example to run 10 batches of the Jump test with 100 trials each, one would type:

```
python run_script.py j 10 100
```

####Experiments from paper
The experiments in our CogSci 2013 paper can be run with the following commands:

```
python run_script.py j 20 100 
python run_script.py h 20 20 
python run_script.py s 20 30
```

####Non-GPU Notes
If not using a GPU, you'll probably want to run your experiments with the -p command line argument, which builds a model containing only a subset of the concepts in WordNet. For example, to run the Hierarchy Test experiment from the paper, but using only 10% of the concepts in Wordnet, one would type

```
python run_script.py h 20 20 -p 0.1
```

####GPU Notes
To tell the package to use GPU acceleration when running experiments, supply the --gpus command line argument, followed by the number of GPU's you want to use to run the model (e.g. --gpus 1). For example, to run the Sentence Test experiment from the paper using 1 GPU (assuming the the GPU library has been installed properly), one would type:

```
python run_script.py s 20 30 --gpus 1
```

Viewing Results
==========
Running simulations wouldn't be much use if we couldn't gather cold hard stats summarizing the results. Such data is stored in the aptly named ``results`` directory. Each time ``run_script.py`` is invoked, a file is generated in this directory. The filename is a concatenation of the test type (i.e. jump, hierarchical or sentence), the time the invocation occurred (with coarser units occuring closing to the front of the string), and a string indicating some of the parameters of the test. For example, a Jump Test that was started on September 19, 2013 at 12:58:42 AM has the name:

```
jump_results_2013_09_19_12_58_42_n
```

The *n* on the end simply signifies that we performed the task using a neural network. Best not to worry about this too much.

The results file contains a number of sections. At the beginning, there is a list of all the parameter values of the network that was used in that experiment. This is followed by data printed to the file detailing the results of each file. Most importantly, after the completion of each batch, a *Bootsrap Summary* is printed out. This gives summary gives various stats that the system is recording, including the mean performance, as well as more specific information such as the average match between the output vector and the desired vector for starting vectors with various out-degrees, etc. The final section gives a histogram of the distribution of the out-degree of the WordNet concepts.

####Viewing Results Graphically
Coming soon...

Command Line Options
==========
This section provides more complete descriptions of the most important command line options. An exhaustive list can be obtained by running ``python run_script.py -h``.

-p P : P is a float between 0 and 1. Permits specification of the percentage of the total number of WordNet synsets to use. Defaults to 1.0. Useful for running smaller versions of the full model, particularly if you don't have access to a GPU.

--gpus G : G is a non-negative integer. Defaults to 0. Specifies the number of CUDA-capable GPU's to use in executing the neural simulation. At least one is required to run the full model in a reasonable amount of time.

-d D : D is a positive integer. Permits specification of the dimensions of the vectors used to encode WordNet. Defaults to 512. With smaller values, the average similarity of any two randomly chosen vectors increases, making the cleanup operation more difficult.

-i : Specifies that the id vectors of the cleanup be identical to the semantic pointers, rather than randomly chosen vectors as is the default. Typically, this makes cleanup noisier since the average similarity of two semantic pointers is much higher than that of two randomly chosen vectors.

-u : Requires that all semantic pointers be defined in terms of unitary vectors, as defined in (Plate 2003). This directly addresses the problem with requiring the id vectors be used as semantic pointers, as it reduces the average similarity of the semantic pointers. So specifying ``-i -u`` will typically build a model that is more successful than one built with ``-u`` alone. These phenomena are investigated in our forthcoming paper.

References
=========

Eliasmith, C. (2013). *How to build a brain: A neural architecture for biological cognition*. New York, NY: Oxford University Press.

Eliasmith, C., & Anderson, C. H. (2003). *Neural engineering: Computation, representation and dynamics in neurobiological systems*. Cambridge, MA: MIT Press.

Plate, T. A. (2003). *Holographic reduced representations*. Stanford, CA: CSLI Publication.
