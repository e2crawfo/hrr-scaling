#cleanup.py

import numpy
import datetime
import string
from probe import Probe

import collections

try:
  import matplotlib.pyplot as plt
except:
  pass

import copy

from ccm.lib import nef

threshold_func = lambda x: (x > 0.3 and x) or 0.0

class Cleanup(nef.ArrayNode):

  def __init__(self, dt, auto, index_vectors, result_vectors, cleanup_node, probe_spec=[], 
               pstc=0.02, print_output=True):

      print "index vectors type: ", type(index_vectors)
      print "result vectors type: ", type(result_vectors)
      self.inputs=[]
      self.outputs=[]
      self.dimensions=len(index_vectors[0])

      self._all_nodes=None

      self.pstc = pstc

      self.numVectors = len(index_vectors)
      self.index_vectors = index_vectors
      self.result_vectors = result_vectors

      self.dt = dt

      self.elapsed_time = 0.0

      self.cleanup_nodes = [cleanup_node]
      self.cleanup_nodes.extend( [cleanup_node.clone() for i in range(self.numVectors-1)] )

      for cn in self.cleanup_nodes:
        cleanup_node.configure_spikes(pstc=self.pstc, dt=self.dt)

      self.probe_data = {}

      for ps in probe_spec:
        item_index, name, func = ps

        if not item_index in self.probe_data:
          self.probe_data[item_index] = []

        probe = Probe(name)
        probe.probe_by_connection(self.cleanup_nodes[item_index], func)

        self.probe_data[item_index].append(probe)

      self.mode='cleanup'
      self.time_points = []

  def tick_accumulator(self, dt):
      for i in self.inputs:
        i.transmit_in(dt)

      for cn in self.cleanup_nodes:
        cn.tick_accumulator(dt)

  def get_output_array_cleanup(self, conn, dt):
      self._array = numpy.zeros(self.dimensions)

      sims = []
      for cn, rv, i in zip(self.cleanup_nodes, self.result_vectors, range(len(self.result_vectors))):
        decoder = cn.get_decoder(threshold_func)

        #not sure if i'm supposed to divide by dt here or not
        similarity = (cn.activity_to_array(cn._output, decoder=decoder))[0]
        sims.append(similarity)

        #if int(self.elapsed_time * 1000) % 10 == 0:
        #  print i, ", ", similarity

        if similarity > 0.0:
          self._array += rv * similarity

      #if int(self.elapsed_time * 1000) % 10 == 0:
      argmax = max(range(len(sims)), key = lambda x: sims[x])
      print "output: ", argmax, ": ", sims[argmax]

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

      sims = [numpy.dot(iv, conn.array) for iv in self.index_vectors]

      #if int(self.elapsed_time * 1000) % 10 == 0:
      argmax = max(range(len(sims)), key = lambda x: sims[x])
      print "input: ", argmax, ": ", sims[argmax]

      for cn, s in zip(self.cleanup_nodes, sims):
        cn.accumulator.add(numpy.array([s]), tau, dt)

      self._input = conn.array

  def _calc_output(self):
      #for cn in self.cleanup_nodes:
      #  cn.tick()
      for cn in self.cleanup_nodes:
        cn._calc_output()

      self.time_points.append( self.elapsed_time )
      self.elapsed_time += self.dt

      for key in self.probe_data:
        probes = self.probe_data[key]

        for p in probes:
          p.probe()

      return

  #currently have to graph all probes on a given node at once
  def plot(self, item_indices, run_index):

    indices_to_probe = item_indices
    if not indices_to_probe:
      indices_to_probe = self.probe_data

    line_types = ["-", "--"]

    first = True

    for key in item_indices:
      probes = self.probe_data[key]

      for i in range(len(probes)):
        p = probes[i]

        p.plot(run_index, line_types[i], draw = True, init=first)

        first = False

    plt.show()

    date_time_string = str(datetime.datetime.now())
    date_time_string = reduce(lambda y,z: string.replace(y,z,"_"), [date_time_string,":","."," ","-"])
    plt.savefig('graphs/neurons_'+date_time_string+".png")

  def reset(self):

    self.elapsed_time = 0.0

    for cn in self.cleanup_nodes:
      cn.reset()

    for key in self.probe_data:
      probes = self.proba_data[key] 
      for p in probes:
        p.reset()

