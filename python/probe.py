#probe.py

import copy
from ccm.lib import nef
import matplotlib.pyplot as plt
import numpy as np

class Probe:
  def __init__(self, name, dt):
    self.filter_node = nef.ArrayNode(1)
    self.dt = dt

    self.time_constant = 0.01

    # a list of pairs(tuples) of arrays, the first of each pair are the timestamps,
    # the second are the values
    self.history = []

    self.current_times = None
    self.current_values = None

    self.reset()

    self.name = name

  def probe_by_connection(self, node, func):
    node.connect( self.filter_node, func=func)

#  def probe_by_function():
#    pass

  def probe(self):
    #so the first probe we do should be at time t = 0.0
    self.current_times.append( self.elapsed_time )
    self.elapsed_time += self.dt
    value = self.filter_node.array()


    self.current_values.append( copy.deepcopy(value) )

  def reset(self):
    self.elapsed_time = 0.0

    if self.current_times and self.current_values:
      self.history.append( (self.current_times, self.current_values) )

    self.current_times = []
    self.current_values = []

