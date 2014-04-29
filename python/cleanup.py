#cleanup.py

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

threshold_func = lambda x: (x > 0.3 and x) or 0.0

class Cleanup(nef.ArrayNode):

  def __init__(self, dt, auto, index_vectors, item_vectors, cleanup_node, probe_spec=[],
               pstc=0.02, print_output=True, scale=1.0):

      self.inputs=[]
      self.outputs=[]
      self.dimensions=len(index_vectors.values()[0])

      self._all_nodes=None

      self.pstc = pstc

      self.numVectors = len(index_vectors)
      self.index_vectors = index_vectors
      self.item_vectors = item_vectors

      self.dt = dt

      self.elapsed_time = 0.0

      self.cleanup_nodes = [cleanup_node]
      self.cleanup_nodes.extend( [cleanup_node.clone() for i in range(self.numVectors-1)] )

      for cn in self.cleanup_nodes:
        cleanup_node.configure_spikes(pstc=self.pstc, dt=self.dt)

      key_indices = {}
      i = 0
      for key in self.item_vectors:
        key_indices[key] = i
        i+=1

      self.probe_data = {}

      for ps in probe_spec:
        item_index, name, func = ps

        if not item_index in self.probe_data:
          self.probe_data[item_index] = []

        probe = Probe(str(item_index) + "," + name, self.dt)
        probe.probe_by_connection(self.cleanup_nodes[key_indices[item_index]], func)

        self.probe_data[item_index].append(probe)

      del key_indices

      self.mode='cleanup'
      self.time_points = []

  def tick_accumulator(self, dt):
      for i in self.inputs:
        i.transmit_in(dt)

      for cn in self.cleanup_nodes:
        cn.tick_accumulator(dt)

        for conn in cn.outputs:
          conn.pop2.tick_accumulator(dt)

  def get_output_array_cleanup(self, conn, dt):
      self._array = numpy.zeros(self.dimensions)

      sims = []
      for cn, iv, i in zip(self.cleanup_nodes, self.item_vectors.values(), range(len(self.item_vectors))):
        decoder = cn.get_decoder(threshold_func)

        decoded_value = (cn.activity_to_array(cn._output, decoder=decoder))[0]
        sims.append(decoded_value)

        #if int(self.elapsed_time * 1000) % 10 == 0:
        #  print i, ", ", similarity

        if abs(decoded_value - 0.0) > 0.0000005:
          self._array += iv * decoded_value

      if int(self.elapsed_time * 1000) % 1 == 0:
          argmax = max(range(len(sims)), key = lambda x: sims[x])
          print "highest output index: ", argmax, ": ", self.index_vectors.keys()[argmax], ": ", sims[argmax]

      return self._array


  def add_input_array_cleanup(self, conn, tau, dt):
      """
      Called by the connection object when updating the input to this node. Weights on the 
      connection coming into this node will not be applied. Each node has its own weights
      to apply, namely the node's index vector.

      :param conn: The connection object calling this function. Supplies the input through conn._array
      :type Connection:

      :param tau: The post-synaptic time constant to use in applying this input to the efferent neurons
      :type float:

      :param dt: The simulation time step
      :type float:
      """
      print "in add input array"

      sims = [numpy.dot(iv, conn.array) for iv in self.index_vectors.values()]

      if int(self.elapsed_time * 1000) % 1 == 0:
          argmax = max(range(len(sims)), key = lambda x: sims[x])
          print "highest input index: ", argmax, ": ", self.index_vectors.keys()[argmax], ": ", sims[argmax]

      for cn, s in zip(self.cleanup_nodes, sims):
        cn.accumulator.add(numpy.array([s]), tau, dt)

      self._input = conn.array

  def _calc_output(self):
      #for cn in self.cleanup_nodes:
      #  cn.tick()

      for cn in self.cleanup_nodes:
          for conn in cn.outputs:
              conn.transmit_out(self.dt)

      for cn in self.cleanup_nodes:
        cn._calc_output()
        cn.clear_state()

        for conn in cn.outputs:
            conn.pop2._calc_output()
            conn.pop2.clear_state()

      self.time_points.append( self.elapsed_time )
      self.elapsed_time += self.dt

      for key in self.probe_data:
        probes = self.probe_data[key]

        for p in probes:
          p.probe()

      return

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

    fig = plt.figure()
    for key in indices_to_plot:
      probes = self.probe_data[key]

      for i, p in enumerate(probes):
        h = p.history[run_index]
        plt.plot(h[0], h[1], line_types[i], label=p.name)

    plt.legend(loc=4)
    plt.show()

    date_time_string = str(datetime.datetime.now()).split('.')[0]
    date_time_string = reduce(lambda y,z: string.replace(y,z,"_"), [date_time_string,":","."," ","-"])
    plt.savefig('../graphs/neurons_'+date_time_string+".png")

  def reset(self):

    self.elapsed_time = 0.0

    for cn in self.cleanup_nodes:
      cn.reset()

    for key in self.probe_data:
      probes = self.probe_data[key]
      for p in probes:
        p.reset()

