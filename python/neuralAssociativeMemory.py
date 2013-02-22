#NeuralAssociativeMemory!
import associativeMemory
from gpuCleanup import GPUCleanup

import ccm
import string

from ccm.lib import hrr
from ccm.lib import nef

import numpy
import random
import datetime
import sys
thresh=0.3

def transfer(x):
    if x>thresh: return 1
    return 0

class NeuralAssociativeMemory(associativeMemory.AssociativeMemory):
  def __init__(self, indices, items, neurons_per_item=10, neurons_per_dim=100,thresh=0.3, thresh_min=-0.9, thresh_max=0.9, use_func=False, timesteps=100, dt=0.001, seed=None, threads=1, useGPU = True, output_dir=".", probes = []):

    self.useGPU = useGPU

    self.threads=threads
    if seed is not None:
      random.seed(seed)

    self.output_dir = output_dir

    self.sentence_results_file=None
    self.jump_results_file=None
    self.hierarchical_results_file=None

    self.runtimes_file=open(self.output_dir+'/neural_runtimes', 'a')
    if not self.runtimes_file:
      self.runtimes_file=open(self.output_dir+'/neural_runtimes2', 'a')



    #always use vector indexing so that for the hierarchical tests, we don't map back to a key, since thats kind of cheating
    #this only really affects the behaviour of unbind_and_associate
    self.vector_indexing = True

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
    self.item_node = nef.make_array_HRR('Item', neurons_per_dim, self.dim, 1, minimum, maximum, maximum=maximum, minimum=minimum) 

    print "Creating query_node array"
    self.query_node = nef.make_array_HRR('Query', neurons_per_dim, self.dim, 1, minimum, maximum, maximum=maximum, minimum=minimum) 

    print "Creating unbind_results_node array"
    self.unbind_results_node = nef.make_array_HRR('UnbindResult', neurons_per_dim, self.dim, 1, minimum, maximum, maximum=maximum, minimum=minimum) 

    self.unbind_measure = nef.ArrayNode(self.dim)
    self.unbind_results_node.connect(self.unbind_measure)

    print "Creating results_node array"
    self.results_node1 = nef.ArrayNode(self.dim)
    self.results_node = nef.make_array_HRR('Result', neurons_per_dim, self.dim, 1, minimum, maximum, maximum=maximum, minimum=minimum) 
    self.results_node.connect(self.results_node1)

#    self.results_node = nef.SpikingNode(512)
#    self.results_node.configure(neurons=self.neurons_per_dim,threshold_min=thresh_min,threshold_max=thresh_max,
#                    saturation_range=(200,200),apply_noise=False)
#    self.results_node.configure_spikes(pstc=0.02,dt=.001)

    print "Creating unbind array"
    self.unbind_node = nef.make_convolution('Unbind', self.item_node, self.query_node, self.unbind_results_node, neurons_per_dim, quick=True, invert_second=True)

    print "Creating associator nodes"

    max_thresh = .9 
    min_thresh = 0 
    #create a single associator ensemble to use as a template for the GPUCleanup class. 
    associator_node = nef.ScalarNode(min=min_thresh, max=max_thresh)
    associator_node.configure(neurons=self.neurons_per_item,threshold_min=min_thresh,threshold_max=max_thresh,
                    saturation_range=(200,200),apply_noise=False)
                    #saturation_range=(200,200),apply_noise=False)

    probeFunctions = [lambda x: x, transfer]
    probeFunctionNames = ["identity", "transfer"]
   
    scale = 1.0

#    if useGPU:
    item_keys = self.items.keys()
    scaled_items = [scale * self.items[key] for key in item_keys]
    indices = [self.indices[key] for key in item_keys]

    #correct probes which are specified by their key rather than the item index
    for probe in probes:
      if type(probe.itemIndex) is tuple:
        probe.itemIndex = item_keys.index(probe.itemIndex)

    self.associator_node = GPUCleanup(1, self.dt, False, indices, scaled_items, self.unbind_results_node.pstc, associator_node, probeFunctions = probeFunctions, probeFunctionNames = probeFunctionNames, probes = probes, probeFromGPU=True)

    self.associator_node.connect(self.results_node)
    self.unbind_results_node.connect(self.associator_node, tau=0.02)
    self.associator_node.connectToProbes(self.unbind_results_node)

    #else:
    #  self.associator_node = [associator_node]
    #  self.associator_node.extend([associator_node.clone() for i in range(self.num_items - 1) ])  

    #self.cleanup_measure = [nef.ArrayNode(1) for i in range(self.num_items)]

#    i = 0
#    for key in self.items.keys():
#
#      if not useGPU:
#        self.unbind_results_node.connect(self.associator_node[i], weight=numpy.array([self.indices[key]]))        # connect A to B[i]
#
#        if use_func:
#            self.associator_node[i].connect(self.results_node,func=transfer,weight=scale * numpy.array([self.items[key]]).T)
#        else:
#            self.associator_node[i].connect(self.results_node,weight=scale * numpy.array([self.items[key]]).T)
#
#        self.associator_node[i].configure_spikes(pstc=0.02,dt=dt)
#
#      #self.associator_node[i].connect(self.cleanup_measure[i])
#      #self.associator_node[i].connect(self.cleanup_measure[i], weight =numpy.array([scale]), func=transfer)
#          
#     # self.associator_nodes[i].configure(neurons=self.neurons_per_item,threshold_min=thresh_min,threshold_max=thresh_max,
#     #                 saturation_range=(200,200),apply_noise=False)
#
#      i = i + 1

    print "Done creating network"
       
  def write_to_runtime_file(self, delta):
    print >> self.runtimes_file, self.threads,",",self.dim,",",self.num_items,",",self.neurons_per_item,",",self.neurons_per_dim,",",self.timesteps,",",delta

  def unbind_and_associate(self, item, query, *args, **kwargs):
    then = datetime.datetime.now()

    #item = [random.random() for i in range(len(item))]
    #query = [random.random() for i in range(len(item))]
    self.item_node.set(item)
    self.query_node.set(query)

    print_debug_info = False
    print_neuron_data = False

    print >> sys.stderr, "beginning simulation"
    print "beginning simulation"

    for j in range(self.timesteps):  
      if self.threads > 1:
        self.item_node.tick_multithreaded(threads=self.threads, dt=self.dt)
      else:
        self.item_node.tick(dt=self.dt)

      if "urn_agreement" in kwargs:
        self.print_unbind_results_node_agreement(kwargs["urn_agreement"])

      #print "tick ",j
      #print >> sys.stderr, "tick ",j

      if print_debug_info:
        self.print_debug_info()
      if print_neuron_data and not self.useGPU:
        self.print_neuron_data()

    #matches = [hrr.HRR(data=self.results_node.array()).compare(hrr.HRR(data=self.items[key])) for key in self.items.keys()]
    matches = [hrr.HRR(data=self.results_node1.array()).compare(hrr.HRR(data=self.items[key])) for key in self.items.keys()]

    max_match =  numpy.max(matches)
    max_index =  numpy.argmax(matches)

    del matches[max_index]

    print >> self.jump_results_file, "max dot product and index thereof"
    print >> self.jump_results_file, max_match, max_index

    second_max_match =  numpy.max(matches)
    second_max_index =  numpy.argmax(matches)

    print >> self.jump_results_file, "second max dot product and index thereof"
    print >> self.jump_results_file, second_max_match, second_max_index

    #vector = self.results_node.array()
    vector = self.results_node1.array()

    #reset them all so they can be used again right away next time
    self.reset_nodes()
    
    now = datetime.datetime.now()
    self.write_to_runtime_file(now - then)
    
    if self.vector_indexing:
      return [vector]
    else:
      return [self.get_key_from_vector(vector, indices)]
        
  def finish(self):
    self.item_node.kill_multithreaded()
    #pass

  def reset_nodes(self):
    self.item_node.reset()
    self.query_node.reset()
    self.unbind_results_node.reset()
    self.unbind_node.reset()
    self.unbind_node.array()

    self.unbind_measure.reset()
    self.results_node.reset()
    self.results_node1.reset()

    if self.useGPU:
      self.associator_node.reset()
    else:
      for cn in self.associator_node:
        cn.reset()

  def drawTransferGraph(self, indices=None):
    self.drawGraph(["transfer"], indices)

  def drawIdentityGraph(self, indices=None):
    self.drawGraph(["identity"], indices)

  def drawCombinedGraph(self, indices=None):
    self.drawGraph(["identity", "transfer"], indices)

  def drawGraph(self, functions, indices=None):
    if indices:
      item_keys = self.items.keys()
      indices = [item_keys.index(i) if type(i) is tuple else i for i in indices]

    if self.useGPU:
      self.associator_node.drawGraph(functions, indices=indices)


  def print_unbind_results_node_agreement(self, key):
    print "Result node agreements: ", hrr.HRR(data=self.indices[key]).compare(hrr.HRR(data=self.unbind_measure.array()))

  def print_debug_info(self):
    print >> sys.stderr, "printing results_node norm: ", numpy.linalg.norm(self.results_node.array())

#    print >> sys.stderr, "printing associator node activations"
#    enumerated_activation = [(i, self.associator_measure[i].array()) for i in range(len(self.associator_measure))]
#    activation = [r[1] for r in enumerated_activation]
#    print >> sys.stderr, "max associator node activation: ", numpy.max(activation), ", at index: ", numpy.argmax(activation)
#
#    print >> sys.stderr, "printing associator node inputs"
#    if not self.useGPU:
#      enumerated_input_vals = [(i, self.associator_node[i].inputs[0].apply_weight(self.associator_node[i].inputs[0].array)) for i in range(len(self.associator_node))]
#      input_vals = [r[1] for r in enumerated_input_vals]
#
#      print >> sys.stderr, "max associator node input: ", numpy.max(input_vals), ", at index: ", numpy.argmax(input_vals)
#
    print >> sys.stderr, "printing agreements" 
    agreements = []
    for i, vec in enumerate(self.items):
      agreements.append((i,hrr.HRR(data=self.items[vec]).compare(hrr.HRR(data=self.results_node.array()))))
    print >> sys.stderr, agreements

    print >> sys.stderr, "dot product of unbind_measure with each of the id vecs"
    dot_prods = []
    for i, vec in enumerate(self.indices):
      dot_prods.append((i,hrr.HRR(data=self.indices[vec]).compare(hrr.HRR(data=self.unbind_measure.array()))))
    print >> sys.stderr, dot_prods

    print >> self.jump_results_file, "Result vector:"
    print >> self.jump_results_file, self.results_node.array()





  def print_neuron_data(self):
    print "Printing accumulator values:"
    for i, node in enumerate(self.associator_node):
      node.accumulator.printVal("associator node " + str(i), node.inputs[0].tau)

    print "Decoder:"
    print self.associator_node[0].get_decoder(transfer)

    print "alpha"
    print self.associator_node[0].alpha
    print "bias"
    print self.associator_node[0].Jbias

    #print self.associator_node
    print "Printing decoded values:"
    for i, node in enumerate(self.associator_node):
      print "associator node "+str(i) + "decoded val:", node.array()

    print "J_threshold:", self.associator_node[0].J_threshold

