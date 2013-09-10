#probe.py

import copy

from ccm.lib import nef

class Probe:
  def __init__(self, name):
    self.filter_node = nef.ArrayNode(1)
    self.dt = None

    self.time_constant = 0.01

    # a list of pairs(tuples) of arrays, the first of each pair are the timestamps,
    # the second are the values
    self.history = []

    self.current_times = None
    self.current_values = None

    self.reset()

    #self.probe_function = None
    #self.probe_args = None

#for printing on graphs
    self.name = name

  def probe_by_connection(self, node, func):
    node.connect( self.filter_node, func=func)

#  def probe_by_function():
#    pass

  def probe(self):
    #so the first probe we do should be at time t = 0.0
    self.current_times.append( self.elapsed_time )
    self.elapsed_time += self.dt

    self.current_values.append( copy.deepcopy(end.value()))

  def plot(self, index, line_type="-", init=False, shutdown=False):
    if init:
      fig = plt.figure()

    h = self.history[index]
    plt.plot(self.hist, self.history[probe], line_type)

    if shutdown:
      plt.show()

      date_time_string = str(datetime.datetime.now())
      date_time_string = reduce(lambda y,z: string.replace(y,z,"_"), [date_time_string,":","."," ","-"])
      plt.savefig('graphs/neurons_'+date_time_string+".png")

  def kill(self):
    self.libNeuralCleanupGPU.kill()

  def reset(self):
    self.elapsed_time = 0.0

    if self.current_times and self.current_values:
      self.history.append( (self.current_times, self.current_values) )

    self.current_times = []
    self.current_values = []

