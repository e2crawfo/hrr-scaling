from ctypes import *
import numpy
import datetime
import string
import collections
from probe import Probe

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

  #index_vectors and item_vectors should both be lists of vectors
  def __init__(self, devices, dt, auto, index_vectors, item_vectors, node, probe_spec=[],
      pstc=0.02, transfer = lambda x: x, print_output=True, quick=False, scale=1.0):

      self.libNeuralCleanupGPU = CDLL("libNeuralCleanupGPU.so")

      self.inputs=[]
      self.outputs=[]
      self.dimensions=len(index_vectors.values()[0])

      self._input = numpy.zeros(self.dimensions)
      self._output = numpy.zeros(self.dimensions)

      #this is required for nodes
      self._all_nodes=None

      self._c_input = None
      self._c_output = None

      self.pstc = pstc

      self.numVectors = len(index_vectors)
      self.index_vectors = index_vectors

      self._decoded_values = numpy.zeros(self.numVectors)

      decoder = node.get_decoder(func=transfer)
      encoder = node.basis
      alpha = node.alpha
      Jbias = node.Jbias
      t_rc = node.t_rc
      t_ref = node.t_ref

      self.numNeuronsPerItem = len(alpha)

      self.dt = dt

      c_index_vectors = convert_to_carray(index_vectors.values(), c_float, 2)
      c_item_vectors = convert_to_carray(item_vectors.values(), c_float, 2)

      c_encoder = convert_to_carray(encoder, c_float, 1)
      c_decoder = convert_to_carray(decoder, c_float, 1)

      c_alpha = convert_to_carray(alpha, c_float, 1)
      c_Jbias = convert_to_carray(Jbias, c_float, 1)

      self.elapsed_time = 0.0

      key_indices = {}
      i = 0
      for key in item_vectors:
        key_indices[key] = i
        i+=1

      self.probe_data = {}
      return_spikes = [0 for i in range(self.numVectors)]
      self._spikes = []

      for ps in probe_spec:
        item_index, name, func = ps

        index = key_indices[item_index]

        if item_index not in self.probe_data:
          start = nef.ScalarNode(min=node.min, max=node.max)
          start.configure(neurons=self.numNeuronsPerItem, threshold_min=node.min,
              threshold_max=node.max, saturation_range=(200,200), apply_noise=False)
          start.configure_spikes(pstc=self.pstc, dt=self.dt)

          self.probe_data[item_index] = (start, [], numpy.zeros(self.numNeuronsPerItem))
        else:
          start = self.probe_data[item_index][0]

        probe = Probe(str(item_index) + "," + name, self.dt)
        probe.probe_by_connection(start, func)

        self.probe_data[item_index][1].append(probe)

        self._spikes.append((item_index, index))

        return_spikes[index] = 1

      self._spikes = collections.OrderedDict( [ (x[0], self.probe_data[x[0]][2]) for x in sorted(self._spikes, key = lambda x:x[1]) ] )

      c_return_spikes = convert_to_carray(return_spikes, c_int, 1)

      del key_indices


      c_devices = convert_to_carray(devices, c_int, 1)
      num_devices = len(devices)

      self.libNeuralCleanupGPU.setup(c_int(num_devices), c_devices, c_float(dt), c_int(self.numVectors), 
                                     c_int(self.dimensions), c_int(int(auto)), c_index_vectors, 
                                     c_item_vectors, c_float(self.pstc), c_encoder, c_decoder, 
                                     c_int(self.numNeuronsPerItem), c_alpha, c_Jbias, c_float(t_ref), 
                                     c_float(t_rc), c_return_spikes, c_int(int(print_output)), c_int(int(quick)) )

      self.mode='gpu_cleanup'

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
      self._c_spikes = convert_to_carray(numpy.zeros(len(self._spikes) * self.numNeuronsPerItem), c_float, 1)
      self._c_decoded_values = convert_to_carray(self._decoded_values, c_float, 1)

      self.libNeuralCleanupGPU.step(self._c_input, self._c_output, self._c_spikes, 
                                    c_float(self.elapsed_time), c_float(self.elapsed_time + self.dt), self._c_decoded_values)

      #make sure output is NOT scaled by dt_over_tau,
      #we let that happen in the termination of results node
      for i in range(len(self._output)):
        self._output[i] = self._c_output[i]

      for i in range(self.numVectors):
        self._decoded_values[i] = self._c_decoded_values[i]

      #get spikes
      spike_offset = 0
      #print self._spikes
      for i in self._spikes:
        for j in range(self.numNeuronsPerItem):
          self._spikes[i][j] = self._c_spikes[ spike_offset + j ] * self.dt

        spike_offset += self.numNeuronsPerItem

      #update probes
      for key in self.probe_data:
        node, probes, spikes = self.probe_data[key]

        node._set_spikes( spikes )
        node.tick()

        for p in probes:
          p.probe()

      return

  #currently have to graph all probes on a given node at once
  def plot(self, item_indices, run_index):

    indices_to_probe = item_indices
  #currently have to graph all probes on a given node at once
  def plot(self, item_indices=[], run_index=-1):

    indices_to_plot = item_indices
    if not indices_to_plot:
      indices_to_plot = self.probe_data
    else:
      indices_to_plot = filter(lambda x: x in probe_data, indices_to_plot)

    if not indices_to_plot:
      return

    line_types = ["-", "--"]

    first = True

    legend = []

    for key in indices_to_plot:
      s, probes, spikes = self.probe_data[key]

      for i in range(len(probes)):
        p = probes[i]

        p.plot(run_index, line_types[i], init=first, legend=legend)

        first = False
    
    plt.legend(legend, loc=2)

    plt.show()

    date_time_string = str(datetime.datetime.now()).split('.')[0]
    date_time_string = reduce(lambda y,z: string.replace(y,z,"_"), [date_time_string,":","."," ","-"])
    plt.savefig('../graphs/neurons_'+date_time_string+".png")

#
#  def drawGraph(self, functionNames, indices=None):
#    fig = plt.figure()
#
#    line_types = ["-", "--", "-.", ":"]
#    line_types = line_types[0:min(len(functionNames), len(line_types))]
#    ltd = {}
#    for i, fn in enumerate(functionNames):
#      ltd[fn] = line_types[ i % len(functionNames)]
#
#    if indices:
#      indices = filter(lambda x: x < self.numVectors and x >= 0, indices)
#
#    l = []
#
#    for probe in self.probes:
#      if indices is None or probe.itemIndex in indices:
#        if probe.name in functionNames:
#          plt.plot(self.time_points_prev, self.history[probe], ltd[probe.name])
#          l.append(str(probe.itemKey) + ", " + probe.name)
#
#    plt.legend(l, loc=2)
#
#    plt.show()
#
#    date_time_string = str(datetime.datetime.now())
#    date_time_string = reduce(lambda y,z: string.replace(y,z,"_"), [date_time_string,":","."," ","-"])
#    plt.savefig('graphs/neurons_'+date_time_string+".png")
#
  def kill(self):
    self.libNeuralCleanupGPU.kill()

  def reset(self):

    self.elapsed_time = 0.0

    for key in self.probe_data:
      node, probes, spikes = self.probe_data[key]
      node.reset()
      for p in probes:
        p.reset()

    self.libNeuralCleanupGPU.reset()


