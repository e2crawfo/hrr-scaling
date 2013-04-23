from .values import HRRNode    
from .core import ArrayNode
from .spikes import SpikingNode
from ..hrr import HRR

import numpy

class NetworkArrayNode(ArrayNode):
    """Collects a set of NEFEnsembles into a single network."""
    serialVersionUID=1
    def __init__(self,name,nodes,noise=0):
        #order of nodes is important, so make sure its right when its passed in
        #dim is the total dimension of the network array
          
        self.nodes=[]
        self.name=name
        self.nodes=nodes
        self.dimensions=0
        self.neurons=0
        self.last_func = None

        for n in nodes:
            self.neurons+=n.neurons
            self.dimensions+=n.dimensions

        self.pstc=nodes[0].pstc
        self.inputs=[]
        self.outputs=[]
        self.array_noise=noise
        
        self._all_nodes=None
        
        self.mode = 'networkarray'
        self._output = []
        self._array = None
    
    def clear_state(self):
      self._array = None
      for n in self.nodes:
        n.clear_state()

#value should just be an array, internally we turn it into an HRR
    def set(self, value, calc_output=True):
      d = 0
      for n in self.nodes:
        n.set(value[d:d+n.dimensions], calc_output)
        d = d + n.dimensions

    def tick_accumulator(self,dt):
      for i in self.inputs:
        i.transmit_in(dt)

      i = 0 
      for n in self.nodes:
        #print "ticking accumulator in networkarray:" + self.name + ", " + str(i) + " of " + str(len(self.nodes))
        n.tick_accumulator(dt)
        i = i + 1

    def _calc_output(self):
      i = 0
      for n in self.nodes:
        #print "calc output in networkarray:" + self.name + ",  " + str(i) + " of " + str(len(self.nodes))
        n._calc_output()
        i = i + 1

    def get_output_array_networkarray(self, conn, dt):
        #print "in get_output_array in networkarray:" + self.name
        decoder = None

        i = 0
        #print self.last_func
        #print conn.func

        if conn.func is None:
          if self._array is not None and self.last_func == "NoFunc":
            return self._array

          self.last_func = "NoFunc"

        elif self.last_func == conn.func:
          if self._array is not None:
            return self._array
        else:
          self.last_func = conn.func

        self._array = []

        for n in self.nodes:
          #print "get_output_array in networkarray:" + self.name + ",  " + str(i) + " of " + str(len(self.nodes))
          #notice how the decoders are taken to be stored in the network array

          if n.mode == 'spike' or n.mode == 'rate':

            if decoder is None:
              decoder = n.get_decoder(conn.func)

            #decoder = n.get_decoder(conn.func)
            self._array.extend(n.activity_to_array(n._output, decoder=decoder)/dt)
            #self._array = numpy.append( _array, n.actvity_to_array(n._output, decoder=decoder))
          else:
            self._array.extend(conn.apply_func(n._output))
            #self._array = numpy.append( _array, conn.apply_func(n._output))

          i = i + 1

        self._array = numpy.array(self._array)

        return self._array


    def add_input_array_networkarray(self, conn, tau, dt):
        i=0
        #print "add_input_array in networkarray:" + self.name 
        array = conn.apply_weight(conn.array)

        d = 0

        for n in self.nodes:
          #print "add_input_array in networkarray:" + self.name + ",  " + str(i) + " of " + str(len(self.nodes))
          n.accumulator.add(array[d:d+n.dimensions], tau, dt)
          d = d + n.dimensions
          i=i+1

    def array(self):
        if self._array is None:
          array = numpy.array([]) 

          for n in self.nodes:
            array = numpy.append(array, n.array())
          return array
        else:
          return self._array 

    def reset(self):
      for n in self.nodes:
        if n.mode == "spike":
          n.reset()

def make_array_HRR(name, neurons, length, dimensions, thresh_min=-0.9, thresh_max=0.9, maximum=1.0, minimum=-1.0, pstc=0.02, dt=0.001, sr=(100,200), apply_noise=False, quick=False, encoders=[], radius=3 ):

    node = SpikingNode(dimensions,noise=0,max=maximum,min=minimum)
    node.configure(neurons=neurons,threshold_min=thresh_min,threshold_max=thresh_max,
                  saturation_range=sr,apply_noise=apply_noise)
    #node.plot_tuning_curves(node.min, .001, node.max)
    #print "max",node.max
    #print "min",node.min

    nodes = [node]

    nodes.extend( [node.clone() for i in range(length-1)] )

    for n in nodes:
      #if reuse_params and alphas is not None and Jbiases is not None and basis is not None:
        #n.configure(neurons=neurons,threshold_min=thresh_min,threshold_max=thresh_max,
        #              saturation_range=sr,apply_noise=apply_noise, alphas=alphas, Jbiases=Jbiases, basis=basis, reuse=True)
      #else:
      #  n.configure(neurons=neurons,threshold_min=thresh_min,threshold_max=thresh_max,
      #                saturation_range=sr,apply_noise=apply_noise)

      n.configure_spikes(pstc=pstc,dt=dt)

#      if reuse_params and not thresholds:
#        thresholds = n.data_thresholds

#      if reuse_params and not saturations:
#        saturations = n.data_saturations
      


    
    return NetworkArrayNode(name, nodes)




