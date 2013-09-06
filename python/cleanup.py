#cleanup.py

import numpy
import datetime
import string

try:
  import matplotlib.pyplot as plt
except:
  pass

import copy

from ccm.lib import nef

threshold_func = lambda x: (x > 0.3 and x) or 0.0

class Cleanup(nef.ArrayNode):

  def __init__(self, dt, auto, index_vectors, result_vectors, cleanup_node, probeFunctions={}, probes=[], 
               pstc=0.02, print_output=True):

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

      #setup probes
      self.probes = []
      self.probeFunctions = probeFunctions

      if len(probes) > 0:
        self.probeData = {}
        self.probes = filter(lambda p: p.name in self.probeFunctions, probes)

        self.history = {}
        for p in probes:
          recorder = nef.ArrayNode(1)
          self.cleanup_nodes[p.itemIndex].connect(recorder, func=self.probeFunctions[p.name])

          self.probeData[p] = (recorder, [])
          self.history[p] = []

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

      if len(self.probes) > 0:
        for probe in self.probes:
          recorder, history = self.probeData[probe]
          history.append( copy.deepcopy(recorder.value()) )


  def drawGraph(self, functionNames, indices=None):
    """
    Draw a graph of the decoded value reprsented by a node that was being probed.

    :param functionNames: Valid function names for the nodes picked out by indices
    :type (listof string):

    :param indices: List of indices of nodes in the cleanup memory. If None, then data from
    all probes are plotted. Otherwise, only those whose index appears in this list are plotted
    data from the the
    :type (listof int)
    """

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
          plt.plot(self.time_points_prev, self.history[probe][-1], ltd[probe.name])
          l.append(str(probe.itemKey) + ", " + probe.name)

    plt.legend(l, loc=2)

    plt.show()

    date_time_string = str(datetime.datetime.now())
    date_time_string = reduce(lambda y,z: string.replace(y,z,"_"), [date_time_string,":","."," ","-"])
    plt.savefig('graphs/neurons_'+date_time_string+".png")

  def reset(self, probes=None):

      for cn in self.cleanup_nodes:
        if cn.node == "spike":
          cn.reset()

      self.time_points_prev = self.time_points
      self.time_points = []

      self.elapsed_time = 0.0

      if len(self.probes) > 0:
        for p in self.probeData:

          recorder = self.probeData[p][0]
          history = self.probeData[p][1]

          recorder.reset()
          self.history[p].append( history )

          self.probeData[p] = (recorder, [])


