from ctypes import *
import numpy
import datetime
import string

try:
  import matplotlib.pyplot as plt
except:
  pass

import copy

from ccm.lib import nef

#returns a type (specifically a pointer to type "t" with depth "depth")
def recursive_c_pointer_type(t, depth):
  return ( t if depth < 1 else POINTER(recursive_c_pointer_type(t,depth-1)))

#returns a carray with depth levels of indirection
def convert_to_carray(l, t, depth):
  if depth < 1:
    return

  carray = (recursive_c_pointer_type(t, depth-1) * len(l))()

  for i in range(len(l)):
    if depth == 1:
      carray[i] = l[i]
    elif len(l[i]) > 0:
      #if we ever have an empty list, we just don't descend there and leave it as a null pointer
      carray[i] = convert_to_carray(l[i], t, depth - 1)

  return cast(carray, recursive_c_pointer_type(t, depth))

class GPUCleanup(nef.ArrayNode):

  #index_vectors and result_vectors should both be lists of vectors
  def __init__(self, devices, dt, auto, index_vectors, result_vectors, tau, node, probeFunctions=[], 
      probeFunctionNames=[], probes=[], pstc=0.02, probeFromGPU=False, transfer = lambda x: x, print_output=True, quick=False):

      self.libNeuralCleanupGPU = CDLL("libNeuralCleanupGPU.so")

      self.inputs=[]
      self.outputs=[]
      self.dimensions=len(index_vectors[0])

      self._input = numpy.zeros(self.dimensions)
      self._output = numpy.zeros(self.dimensions)


      self._all_nodes=None

      self._c_input = None
      self._c_output = None

      self.pstc = pstc

      self.numVectors = len(index_vectors)
      self.index_vectors = index_vectors

      #tmep
      self._decoded_values = numpy.zeros(self.numVectors)

      decoder = node.get_decoder(func=transfer)
      encoder = node.basis
      alpha = node.alpha
      Jbias = node.Jbias
      t_rc = node.t_rc
      t_ref = node.t_ref

      self.numNeuronsPerItem = len(alpha)

      self.dt = dt

      c_index_vectors = convert_to_carray(index_vectors, c_float, 2)
      c_result_vectors = convert_to_carray(result_vectors, c_float, 2)

      c_encoder = convert_to_carray(encoder, c_float, 1)
      c_decoder = convert_to_carray(decoder, c_float, 1)

      c_alpha = convert_to_carray(alpha, c_float, 1)
      c_Jbias = convert_to_carray(Jbias, c_float, 1)

      self.elapsed_time = 0.0

      self.probeFromGPU = probeFromGPU
      self.probes = []
      if len(probes) > 0:
        self.probeFunctions = dict(zip(probeFunctionNames, probeFunctions))
        #setup probes
        self.probeData = {}
        self.probes = filter(lambda p: p.name in self.probeFunctions, probes)

        self.history = {}
        for p in probes:
          start = nef.ScalarNode(min=node.min, max=node.max)
          start.configure(neurons=self.numNeuronsPerItem,threshold_min=node.min,threshold_max=node.max,
                     saturation_range=(200,200),apply_noise=False)
          start.configure_spikes(pstc=self.pstc,dt=self.dt)

          end = nef.ArrayNode(1)

          start.connect(end, func=self.probeFunctions[p.name])

          history = []
          self.probeData[p] = (start, end, history)
          self.history[p] = []

        #only return spikes for nodes that have a probe on them
        itemIndices = [p.itemIndex for p in probes]
        returnSpikes = [1 if i in itemIndices else 0 for i in range(self.numVectors)]
      else:
        returnSpikes = [0 for i in range(self.numVectors)]

        #for displaying graphs
      self.numItemsReturningSpikes = len(filter(lambda x: x, returnSpikes))

      self._spikeIndices = numpy.where(returnSpikes)[0]
      self._spikes = dict((i, numpy.zeros(self.numNeuronsPerItem)) for i in self._spikeIndices)

      c_returnSpikes = convert_to_carray(returnSpikes, c_int, 1)

      print "Print output:", print_output, " ", int(print_output)

      self.libNeuralCleanupGPU.setup(c_int(devices), c_float(dt), c_int(self.numVectors), 
                                     c_int(self.dimensions), c_int(int(auto)), c_index_vectors, 
                                     c_result_vectors, c_float(tau), c_encoder, c_decoder, 
                                     c_int(self.numNeuronsPerItem), c_alpha, c_Jbias, c_float(t_ref), 
                                     c_float(t_rc), c_returnSpikes, c_int(int(print_output)), c_int(int(quick)) )

      self.mode='gpu_cleanup'

      self.time_points = []

  def tick_accumulator(self, dt):
      for i in self.inputs:
        i.transmit_in(dt)

  def get_output_array_gpu_cleanup(self, conn, dt):
      return self._output

  def add_input_array_gpu_cleanup(self, conn, tau, dt):
      self._input = conn.array

  def _calc_output(self):
      self._c_input = convert_to_carray(self._input, c_float, 1)
      self._c_output = convert_to_carray(numpy.zeros(self.dimensions), c_float, 1)
      self._c_spikes = convert_to_carray(numpy.zeros(self.numItemsReturningSpikes * self.numNeuronsPerItem), c_float, 1)
      self._c_decoded_values = convert_to_carray(self._decoded_values, c_float, 1)

      self.libNeuralCleanupGPU.step(self._c_input, self._c_output, self._c_spikes, 
                                    c_float(self.elapsed_time), c_float(self.elapsed_time + self.dt), self._c_decoded_values)

      #make sure output is NOT scaled by dt_over_tau,
      #we let that happen in the termination of results node
      self.time_points.append( self.elapsed_time )
      self.elapsed_time = self.elapsed_time + self.dt
      for i in range(len(self._output)):
        self._output[i] = self._c_output[i]

      for i in range(self.numVectors):
        self._decoded_values[i] = self._c_decoded_values[i]

#      if int(self.elapsed_time * 1000) % 10 == 0:
#        fig=plt.figure()
#        plt.hist(self._decoded_values, bins = 20)
#        plt.show()
#        date_time_string = str(datetime.datetime.now())
#        date_time_string = reduce(lambda y,z: string.replace(y,z,"_"), [date_time_string,":","."," ","-"])
#        plt.savefig('graphs/decoded_values'+date_time_string+".png")
#
      #get spikes
      spikeIndex = 0
      for i in self._spikeIndices:
        s = self._spikes[i]
        for j in range(self.numNeuronsPerItem):
          s[j] = self._c_spikes[ spikeIndex + j ] * self.dt

        #print self._spikes[i]
        #print "********************"

        spikeIndex += self.numNeuronsPerItem

      #update probes
      if len(self.probes) > 0:
        for probe in self.probes:
          start, end, history = self.probeData[probe]
          
          #this is from when we were actually probing the stuff from the GPU
          #itemIndex = probe.itemIndex
          if self.probeFromGPU:
            start._set_spikes( self._spikes[probe.itemIndex] )
            start.tick()

          history.append( copy.deepcopy(end.value() ))
          #print "Probe :", probe, ", val: ", end.value()

      return 

  def drawGraph(self, functionNames, indices=None):
    fig = plt.figure()

    line_types = ["-", "--", "-.", ":"]
    line_types = line_types[0:min(len(functionNames), len(line_types))]
    ltd = {}
    for i, fn in enumerate(functionNames):
      ltd[fn] = line_types[ i % len(functionNames)]

    if indices:
      indices = filter(lambda x: x < self.numVectors and x >= 0, indices)

    l = []

    for probe in self.probes:
      if indices is None or probe.itemIndex in indices:
        if probe.name in functionNames:
          plt.plot(self.time_points_prev, self.history[probe], ltd[probe.name])
          l.append(str(probe.itemKey) + ", " + probe.name)

    plt.legend(l, loc=2)

    plt.show()

    date_time_string = str(datetime.datetime.now())
    date_time_string = reduce(lambda y,z: string.replace(y,z,"_"), [date_time_string,":","."," ","-"])
    plt.savefig('graphs/neurons_'+date_time_string+".png")
  
  def connectToProbes(self, node):
    if self.probeFromGPU:
      return

    #print "connecting to probes!"
    for p in self.probes:
      weight = self.index_vectors[p.itemIndex]
      node.connect( self.probeData[p][0], weight=copy.deepcopy(weight))

  def kill(self):
      self.libNeuralCleanupGPU.kill()

  def reset(self, probes=None):
      self.time_points_prev = self.time_points
      self.time_points = []

      self.elapsed_time = 0.0
      
      if len(self.probes) > 0:
        for p in self.probeData:

          start = self.probeData[p][0]
          end = self.probeData[p][1]
          history = self.probeData[p][2]

          start.reset()
          end.reset()
          self.history[p] = history

          self.probeData[p] = (start, end, [])

      self.libNeuralCleanupGPU.reset()

  
