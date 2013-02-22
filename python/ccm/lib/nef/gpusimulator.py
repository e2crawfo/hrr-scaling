#from ca.nengo.model.impl import *
#from ca.nengo.model import *
#from ca.nengo.model.nef.impl import *
#from ca.nengo.model.nef import NEFEnsemble
#from ca.nengo.model.neuron.impl import * #from ca.nengo.model.neuron import *
#from ca.nengo.util import *
#from ca.nengo.util.impl import NodeThreadPool, NEFGPUInterface
#from ca.nengo.sim.impl import LocalSimulator

from ccm.lib.nef.spikes import SpikingNode
from ccm.lib.nef.activity import ActivityNode

#from spikes import SpikingNode

from random import *
from ctypes import *
import numpy

def convert_to_dynamic_carray(vals, val_type):

    carray = (val_type * len(val))
    for i in [0, len(val)-1]:
      carray[i] = val[i]
    
    return cast(carray, POINTER(val))


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



class GPUSimulator:
    def __init__(self, input_node, ran_seed):

        self.libNengoGPU = CDLL("libNengoGPU.so")

        self.gpu_nodes=[]
        self.non_gpu_nodes=[]

        self.gpu_projections=[]
        self.non_gpu_projections=[]

        self.on_to_gpu_projections = []
        self.off_of_gpu_projections = []

        self.projections=[]

        seed(ran_seed)


        self.initialize(input_node)

    def initialize(self,input_node):

        num_devices = 1


        nodes = input_node.all_nodes()

        gpu_node_index = 0
        for node in nodes:
          if isinstance(node, ActivityNode) and node.use_gpu and (node.mode == 'spike' or node.mode == 'activity'):
            self.gpu_nodes.append(node)
            node.using_gpu = True
            node.gpu_node_index = gpu_node_index
            gpu_node_index = gpu_node_index + 1
          else:
            self.non_gpu_nodes.append(node)
            node.using_gpu = False

        projections = input_node.all_connections()

        for proj in projections:
          if proj.pop1.using_gpu and proj.pop2.using_gpu:
            self.gpu_projections.append(proj)
          else:
            if proj.pop1.using_gpu:
              proj.node_index = proj.pop1.gpu_node_index

              for j in range(len(proj.pop1.outputs)): #loop 2
                if proj.pop1.outputs[j] == proj:
                  proj.origin_index = j
                  break #exit loop 2

              self.off_of_gpu_projections.append(proj)
              proj.off_of_gpu = True

            elif proj.pop2.using_gpu:

              proj.node_index = proj.pop2.gpu_node_index

              for j in range(len(proj.pop2.inputs)): #loop 2
                if proj.pop2.inputs[j] == proj:
                  proj.term_index = j
                  break #exit loop 2
                  
              self.on_to_gpu_projections.append(proj)
              proj.on_to_gpu = True

            self.non_gpu_projections.append(proj)
            proj.array = numpy.zeros(len(proj.weight[-1]))


        #process our nodes and projections, put them in the form expected by the c GPU library.

        transforms = []
        termination_dimensions = []
        termination_tau = []
        is_decoded_termination = []

        encoders = []

        decoders = []
        decoder_dimensions = []

        neuron_data = []

        ensemble_data = []
        network_array_data = []

        projection_data = [[0] * 6 for p in self.gpu_projections]

        device_for_ensemble = [randint(0,num_devices-1) for n in self.gpu_nodes]

        for i in range(len(self.gpu_projections)):
          self.gpu_projections[i].proj_id = i

        for p in self.non_gpu_projections:
          p.proj_id = -1

        node_index = 0

        for n in self.gpu_nodes:
          transforms.append([i.weight for i in n.inputs])

          termination_dimensions.append([len(i.weight[0]) for i in n.inputs])
          termination_tau.append([i.tau for i in n.inputs])
          
          is_decoded_termination.append([1] * len(n.inputs))

          encoders.append(n.basis)
          print n.basis

          #neuron data
          nd = []
          nd.append(n.neurons)
          nd.append(n.t_rc) #taurc
          nd.append(n.t_ref) #tauref

          nd.append(0) #these two don't matter
          nd.append(0)

          nd.extend(n.Jbias)
          nd.extend(n.alpha)

          neuron_data.append(nd)

          ##Think I have to make sure the "decoder mode" is NxS
          ##The way this is called, no noise will be added
          decoders.append([n.get_decoder(o.func) for o in n.outputs])
          decoder_dimensions.append([len(d[0]) for d in decoders[-1]])

          #ensemble data
          ensemble_data.append( n.dimensions )
          ensemble_data.append( n.neurons )
          ensemble_data.append( len(n.outputs) )
          ensemble_data.append( sum(termination_dimensions[-1]) )
          ensemble_data.append( sum(decoder_dimensions[-1]) )
          ensemble_data.append( max(termination_dimensions[-1]) )
          ensemble_data.append( max(decoder_dimensions[-1]) )
          ensemble_data.append( len(termination_dimensions[-1]) )
          ensemble_data.append( 0 )
          ensemble_data.append( 0 )
          
          #network array data
          network_array_data.append( node_index );
          network_array_data.append( node_index + 1);
          network_array_data.append( len(termination_dimensions[-1]) )
          network_array_data.append( sum(termination_dimensions[-1]) )
          network_array_data.append( len(n.outputs) )
          network_array_data.append( sum(decoder_dimensions[-1]) )
          network_array_data.append( n.neurons )

          for input_index in range(len(n.inputs)):
            i = n.inputs[input_index]

            if i.proj_id != -1:
              projection_data[i.proj_id][2] = node_index
              projection_data[i.proj_id][3] = input_index
              projection_data[i.proj_id][4] = termination_dimensions[-1][input_index]
              projection_data[i.proj_id][5] = -1
          # end input loop
          
          for output_index in range(len(n.outputs)):
            o = n.outputs[output_index]

            if o.proj_id != -1:
              projection_data[o.proj_id][0] = node_index
              projection_data[o.proj_id][1] = output_index
          # end output loop

          node_index = node_index + 1

        # end node loop
            
        #should change this eventually so we don't bring all the data back every time
        origin_required_by_cpu = [[1] * len(d) for d in decoder_dimensions]

        is_spiking_ensemble = [(1 if isinstance(n, SpikingNode) and n.mode == 'spike' else 0) for n in self.gpu_nodes]
        #is_spiking_ensemble = [0 + is_spiking_ensemble[i] for i in range(len(nodes))]

        collect_spikes = [0] * len(self.gpu_nodes)



        #translation! now put all this data, which we have in the form of python lists, into c arrays so we can
        #pass it into the c gpu function
        transforms_carray = convert_to_carray(transforms, c_float, 4)
        termination_dimensions_carray = convert_to_carray(termination_dimensions, c_int, 2)
        termination_tau_carray = convert_to_carray(termination_tau, c_float, 2)


        encoders_carray = convert_to_carray(encoders, c_float, 3)
        decoders_carray = convert_to_carray(decoders, c_float, 4)

        decoder_dimensions_carray = convert_to_carray(decoder_dimensions, c_int, 2)
        neuron_data_carray = convert_to_carray(neuron_data, c_float, 2)

        ensemble_data_carray = convert_to_carray(ensemble_data, c_int, 1)

        network_array_data_carray = convert_to_carray(network_array_data, c_int, 1)
        
        is_decoded_termination_carray = convert_to_carray(is_decoded_termination, c_int, 2)
        is_spiking_ensemble_carray = convert_to_carray(is_spiking_ensemble, c_int, 1)
        collect_spikes_carray = convert_to_carray(collect_spikes, c_int, 1)
        device_for_ensemble_carray = convert_to_carray(device_for_ensemble, c_int, 1)

        origin_required_by_cpu_carray = convert_to_carray(origin_required_by_cpu, c_int, 2)

        projection_data_carray = convert_to_carray(projection_data, c_int, 2)

        sizes = convert_to_carray([len(self.gpu_nodes), len(self.gpu_projections)], c_int, 1)

        #call the GPU setup function!
        self.libNengoGPU.setup(num_devices, sizes, transforms_carray, termination_dimensions_carray, \
            termination_tau_carray, is_decoded_termination_carray, encoders_carray, decoders_carray, \
            decoder_dimensions_carray, neuron_data_carray, network_array_data_carray, ensemble_data_carray, \
            is_spiking_ensemble_carray, collect_spikes_carray, origin_required_by_cpu_carray, \
            projection_data_carray, device_for_ensemble_carray)


        #make c arrays for sending input to GPU each step
        input_array = []

        for i in range(len(self.gpu_nodes)):

          ensemble_inputs = [] 
          input_connections = self.gpu_nodes[i].inputs

          for j in range(len(input_connections)):
            if input_connections[j] in self.on_to_gpu_projections:
              ensemble_inputs.append([0.0] * termination_dimensions[i][j])
            else:
              ensemble_inputs.append([])

          input_array.append(ensemble_inputs)

        self.input_carray = convert_to_carray(input_array, c_float, 3)

        self.ensemble_num_inputs = convert_to_carray([len(t) for t in termination_dimensions], c_int, 1)
        self.input_sizes = convert_to_carray(termination_dimensions, c_int, 2)


        #make c arrays for getitng output from GPU each step

        output_array = []
        for i in range(len(self.gpu_nodes)):

          ensemble_outputs = [] 

          output_connections = self.gpu_nodes[i].outputs

          for j in range(len(output_connections)):
            if output_connections[j] in self.off_of_gpu_projections:
              ensemble_outputs.append([0.0] * decoder_dimensions[i][j])
            else:
              ensemble_outputs.append([])

          output_array.append(ensemble_outputs)

        self.output_carray = convert_to_carray(output_array, c_float, 3)

        self.ensemble_num_outputs = convert_to_carray([len(d) for d in decoder_dimensions], c_int, 1)
        self.output_sizes = convert_to_carray(decoder_dimensions, c_int, 2)

        self.spikes = convert_to_carray([[0.0] * n.neurons for n in self.gpu_nodes], c_float, 2)
        self.num_neurons = convert_to_carray([n.neurons for n in self.gpu_nodes], c_int, 1)
  
    def step(self,start,dt):
        #here just have to get the input from projections that go from off the gpu to on the gpu
        #and load it into the input_carray

        for n in self.non_gpu_nodes:
            n._clear_inputs()
            n.accumulator.tick(dt)

        for proj in self.non_gpu_projections:
            f=getattr(proj.pop1,'_transmit_%s'%proj.type())
            f(proj,dt)
        
        for n in self.non_gpu_nodes:
            n._calc_output()
            n._value=None
            n._array=None
        
        for proj in self.on_to_gpu_projections:
            node_index = proj.node_index
            term_index = proj.term_index

            for i in range(self.input_sizes[node_index][term_index]):
              self.input_carray[node_index][term_index][i] = proj.array[i]

        self.libNengoGPU.step(self.input_carray, self.ensemble_num_inputs, self.input_sizes, self.output_carray,\
            self.ensemble_num_outputs, self.output_sizes, self.spikes, self.num_neurons, c_float(start), c_float(start + dt))

        #put the output from the gpu in the relevant connections (projections)
        for proj in self.off_of_gpu_projections:
            node_index = proj.node_index
            origin_index = proj.origin_index

            for i in range(self.output_sizes[node_index][origin_index]):
              proj.array[i] = self.output_carray[node_index][origin_index][i] 

        #here just have to get the output from projections that go from on the gpu to off the gpu
        #this data is in output_carray

        #finally, run the nodes which were not run on the gpu

    def kill(self):
      for n in self.gpu_nodes:
        n.gpu_mode = False
      
      for proj in self.non_gpu_projections:
        proj.on_to_gpu = False
        proj.off_of_gpu = False

      self.libNengoGPU.kill()

