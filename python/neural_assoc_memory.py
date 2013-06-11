#NeuralAssociativeMemory!
from assoc_memory import AssociativeMemory
from gpu_cleanup import GPUCleanup

import ccm
import string

from ccm.lib import hrr
from ccm.lib import nef

import numpy
import random
import datetime
import sys

class NeuralAssociativeMemory(AssociativeMemory):

  _type = "Neural"

  def __init__(self, indices, items, identity, unitary, bidirectional=False, threshold=0.3, neurons_per_item=20, neurons_per_dim=50, thresh_min=-0.9,
      thresh_max=0.9, use_func=False, timesteps=100, dt=0.001, seed=None, threads=1, useGPU = True, output_dir=".", probes = [], print_output=True, pstc=0.02, quick=False):

    self.useGPU = useGPU
    self.threshold = threshold
    self.transfer_func = lambda x: 1 if x > self.threshold else 0
    self.print_output = print_output
    self.quick_gpu = quick

    self.threads=threads
    if seed is not None:
      random.seed(seed)

    self.output_dir = output_dir

    self.sentence_results_file=None
    self.jump_results_file=None
    self.hierarchical_results_file=None
    self.active_results_file=None

    self.unitary = unitary
    self.identity = identity
    self.bidirectional = bidirectional
    self.return_vec = True

    self.runtimes_file=open(self.output_dir+'/neural_runtimes', 'a')
    if not self.runtimes_file:
      self.runtimes_file=open(self.output_dir+'/neural_runtimes2', 'a')

    #in the "core" case, indices will be the id vectors, items will be the structured vectors.
    self.indices=indices
    self.items=items

    self.dim = len(self.indices[self.indices.keys()[0]])
    self.num_items = len(self.indices)
    self.neurons_per_item = neurons_per_item
    self.neurons_per_dim = neurons_per_dim
    self.use_func=use_func
    self.dt=dt
    self.timesteps=timesteps

    maximum = numpy.sqrt(1.0 / self.dim)
    minimum = -maximum

    print "Creating item_node array"
    self.item_node = nef.make_array_HRR('Item', neurons_per_dim, self.dim, 1, minimum, maximum, maximum=maximum, minimum=minimum, pstc=pstc)

    print "Creating query_node array"
    self.query_node = nef.make_array_HRR('Query', neurons_per_dim, self.dim, 1, minimum, maximum, maximum=maximum, minimum=minimum, pstc=pstc)

    print "Creating unbind_results_node array"
    self.unbind_results_node = nef.make_array_HRR('UnbindResult', neurons_per_dim, self.dim, 1, minimum, maximum, maximum=maximum, minimum=minimum, pstc=pstc)

    self.unbind_measure = nef.ArrayNode(self.dim)
    self.unbind_results_node.connect(self.unbind_measure)

    print "Creating results_node array"
    self.results_node = nef.ArrayNode(self.dim)
    self.results_node_spiking = nef.make_array_HRR('Result', neurons_per_dim, self.dim, 1, minimum, maximum, maximum=maximum, minimum=minimum, pstc=pstc)

    print "Creating unbind array"
    self.unbind_node = nef.make_convolution('Unbind', self.item_node, self.query_node, self.unbind_results_node, neurons_per_dim, quick=True, invert_second=True, pstc_in=pstc, pstc_out=pstc)

    print "Creating associator nodes"

    self.max_thresh = .9
    self.min_thresh = 0

    #create a single associator ensemble to use as a template for the GPUCleanup class. 
    associator_node = nef.ScalarNode(min=self.min_thresh, max=self.max_thresh)
    associator_node.configure(neurons=self.neurons_per_item,threshold_min=self.min_thresh,threshold_max=self.max_thresh,
                    saturation_range=(200,200),apply_noise=False)

    probeFunctions = [lambda x: x, self.transfer_func]
    probeFunctionNames = ["identity", "transfer"]

    scale = 10.0 

    item_keys = self.items.keys()
    scaled_items = [scale * self.items[key] for key in item_keys]
    indices = [self.indices[key] for key in item_keys]

    for probe in probes:
      if probe.itemKey :
        probe.itemIndex = item_keys.index(probe.itemKey)

    self.associator_node = GPUCleanup(4, self.dt, False, indices, scaled_items, self.unbind_results_node.pstc, associator_node, probeFunctions = probeFunctions, probeFunctionNames = probeFunctionNames, probes = probes, probeFromGPU=True, transfer=self.transfer_func, print_output=print_output, quick=quick)

    self.associator_node.connect(self.results_node_spiking)
    self.unbind_results_node.connect(self.associator_node, tau=pstc)
    self.results_node_spiking.connect(self.results_node)
    self.associator_node.connectToProbes(self.unbind_results_node)

    print "Done creating network"

  def write_to_runtime_file(self, delta):
    print >> self.runtimes_file, self.threads,",",self.dim,",",self.num_items,",",self.neurons_per_item,",",self.neurons_per_dim,",",self.timesteps,",",delta
    print self.threads,",",self.dim,",",self.num_items,",",self.neurons_per_item,",",self.neurons_per_dim,",",self.timesteps,",",delta

  def unbind_and_associate(self, item, query, *args, **kwargs):
    then = datetime.datetime.now()

    self.item_node.set(item)
    self.query_node.set(query)

    print_debug_info = False
    print_neuron_data = False

    print "beginning simulation"

    i = 0
    for j in range(self.timesteps):
      if self.threads > 1:
        self.item_node.tick_multithreaded(threads=self.threads, dt=self.dt)
      else:
        self.item_node.tick(dt=self.dt)

      if i % 10 == 0:
        self.print_unbind_results_node_agreement()

      if print_debug_info:
        self.print_debug_info()
      if print_neuron_data and not self.useGPU:
        self.print_neuron_data()

      i += 1

    #vector = self.results_node.array()
    vector = self.results_node.array()

    #reset them all so they can be used again right away next time
    self.reset_nodes()

    now = datetime.datetime.now()
    self.write_to_runtime_file(now - then)

    return [vector]

  def finish(self):
    self.item_node.kill_multithreaded()

  def reset_nodes(self):
    self.item_node.reset()
    self.query_node.reset()
    self.unbind_results_node.reset()
    self.unbind_node.reset()
    self.unbind_node.array()

    self.unbind_measure.reset()
    self.results_node_spiking.reset()
    self.results_node.reset()

    self.associator_node.reset()

  def drawTransferGraph(self, indices=None):
    self.drawGraph(["transfer"], indices)

  def drawIdentityGraph(self, indices=None):
    self.drawGraph(["identity"], indices)

  def drawCombinedGraph(self, indices=None):
    self.drawGraph(["identity", "transfer"], indices)

  def drawGraph(self, functions, indices=None):
    if self.useGPU:

      if indices:
        item_keys = self.items.keys()
        indices = [item_keys.index(i) if type(i) is tuple else i for i in indices]

      self.associator_node.drawGraph(functions, indices=indices)

  def print_unbind_results_node_agreement(self):
    if len(self.tester.current_target_keys) > 0:
      vector = self.indices[self.tester.current_target_keys[0]]
      print "Result node agreements: ", hrr.HRR(data=vector).compare(hrr.HRR(data=self.unbind_measure.array()))
    else:
      print "Result node agreements: no expected match"

  def print_debug_info(self):
    print >> sys.stderr, "printing results_node norm: ", numpy.linalg.norm(self.results_node_spiking.array())

    print >> sys.stderr, "printing agreements" 
    agreements = []
    for i, vec in enumerate(self.items):
      agreements.append((i,hrr.HRR(data=self.items[vec]).compare(hrr.HRR(data=self.results_node_spiking.array()))))
    print >> sys.stderr, agreements

    print >> sys.stderr, "dot product of unbind_measure with each of the id vecs"
    dot_prods = []
    for i, vec in enumerate(self.indices):
      dot_prods.append((i,hrr.HRR(data=self.indices[vec]).compare(hrr.HRR(data=self.unbind_measure.array()))))
    print >> sys.stderr, dot_prods

    print >> self.jump_results_file, "Result vector:"
    print >> self.jump_results_file, self.results_node_spiking.array()



  def print_neuron_data(self):
    print "Printing accumulator values:"
    for i, node in enumerate(self.associator_node):
      node.accumulator.printVal("associator node " + str(i), node.inputs[0].tau)

    print "Decoder:"
    print self.associator_node[0].get_decoder(self.transfer_func)

    print "alpha"
    print self.associator_node[0].alpha
    print "bias"
    print self.associator_node[0].Jbias

    #print self.associator_node
    print "Printing decoded values:"
    for i, node in enumerate(self.associator_node):
      print "associator node "+str(i) + "decoded val:", node.array()

    print "J_threshold:", self.associator_node[0].J_threshold


  def print_config(self, output_file):
    super(NeuralAssociativeMemory, self).print_config(output_file)

    output_file.write("Quick GPU: " + str(self.quick_gpu) + "\n")
    output_file.write("Neurons per item: " + str(self.neurons_per_item) + "\n")
    output_file.write("Neurons per dim: " + str(self.neurons_per_dim) + "\n")
    output_file.write("Min thresh: " + str(self.max_thresh) + "\n")
    output_file.write("Max thresh: " + str(self.min_thresh) + "\n")
