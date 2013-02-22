#from ca.nengo.model.impl import *
#from ca.nengo.model import *
#from ca.nengo.model.nef.impl import *
#from ca.nengo.model.nef import NEFEnsemble
#from ca.nengo.model.neuron.impl import * #from ca.nengo.model.neuron import *
#from ca.nengo.util import *
#from ca.nengo.util.impl import NodeThreadPool, NEFGPUInterface
#from ca.nengo.sim.impl import LocalSimulator

from ccm.lib.nef.SpikingNode import *
from random import *
from ctypes import *

def convert_to_dynamic_carray(vals, val_type):

    carray = (val_type * len(val))
    for i in [0, len(val)-1]:
      carray[i] = val[i]
    
    return cast(carray, POINTER(val))


class GPUSimulator:
    def __init__(self, input_node, ran_seed):

        self.libNengoGPU = CDLL("libNengoGPU.so")

        self.gpu_nodes=[]
        self.non_gpu_nodes=[]

        self.gpu_projections=[]
        self.non_gpu_projections=[]

        self.projections=[]

        seed(ran_seed)


        self.initialize(input_node)

    def initialize(self,input_node):

        num_devices = 1


        nodes = input_node.all_nodes()

        for node in nodes:
          if isinstance(node, ActivityNode):
            gpu_nodes.append(node)
          else:
            non_gpu_nodes.append(node)


        projections = input_node.all_connections()

        for proj in projections:
          if proj.pop1 in gpu_nodes and proj.pop2 in gpu_nodes:
            gpu_projections.append(proj)
          else:
            non_gpu_projections.append(proj)


        #process our nodes and projections, put them in the form expected by the c GPU library.

        transforms = []
        termination_dimension = []
        termination_tau = []
        is_decoded_termination = []

        encoders = []

        decoders = []
        decoder_dimensions = []

        neuron_data = []

        ensemble_data = []
        network_array_data = []

        projection_data = [[0] * 6] * len(gpu_projections)

        for i in range(i):
          p.id = i

        node_index = 0

        for n in self.gpu_nodes:
          transforms.append([i.weight for i in n.inputs])
          termination_dimensions.append([len(i.weight[0]) for i in n.inputs])
          termination_tau.append([i.tau for i in n.inputs])
          
          is_decoded_termination.append([1] * len(n.inputs))

          encoders.append(n.basis)

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
          decoders.append([n.get_decoder(o) for o in n.outputs])
          decoder_dimensions.append([len(d[0]) for d in decoders[-1])


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
          network_array_data.append( n.dimensions )
          network_array_data.append( n.neurons )
          network_array_data.append( len(n.outputs) )
          network_array_data.append( sum(termination_dimensions[-1]) )
          network_array_data.append( sum(decoder_dimensions[-1]) )
          network_array_data.append( max(termination_dimensions[-1]) )
          network_array_data.append( max(decoder_dimensions[-1]) )
          network_array_data.append( len(termination_dimensions[-1]) )
          network_array_data.append( 0 )
          network_array_data.append( 0 )

          device_for_ensemble.append(randint(0,num_devices-1))

          for input_index in range(len(n.inputs)):
            i = n.inputs[input_index]

            projection_data[i.id][2] = node_index
            projection_data[i.id][3] = input_index
            projection_data[i.id][4] = termination_dimensions[-1][input_index]
            projection_data[i.id][5] = -1
          # end input loop
          
          for output_index in range(len(n.outputs)):
            o = n.outputs[output_index]

            projection_data[o.id][0] = node_index
            projection_data[o.id][1] = output_index
          # end output loop

          node_index = node_index + 1

        # end node loop
            
        origin_required_by_CPU = [[1] * len(d) for d in decoder_dimensions]

        is_spiking_ensemble = [(1 if isinstance(n, SpikingNode) else 0) for n in nodes]
        #is_spiking_ensemble = [0 + is_spiking_ensemble[i] for i in range(len(nodes))]

        collect_spikes = [0] * len(n)



        #translation! now put all this data, which we have in the form of python lists, into c arrays so we can
        #pass it into the c gpu function

        transforms_carray = (POINTER(POINTER(POINTER(c_float))) * len(transforms))()

        for i in range(len(transforms)):
          temp1 = (POINTER(POINTER(c_float)) * len(transforms[i]))()

          for j in range(len(transforms[i]))
            temp2 = (POINTER(c_float) * len(transforms[i][j]))()

            for k in range(len(transforms[i][j])):
              temp2[k] = convert_to_dynamic_carray(transforms[i][j][k], c_float)

            temp1[j] = cast(temp2, POINTER(POINTER(c_float)))
           
          transforms_carray[i] = cast(temp1, POINTER(POINTER(POINTER(c_float))))
        
        transforms_carray = cast(transforms_carray, POINTER(POINTER(POINTER(POINTER(c_float)))))

        transforms_carray = convert_to



        termination_dimensions_carray = (POINTER(c_int) * len(transforms))()

        for i in range(len(termination_dimensions)):
          termination_dimensions_carray[i] = convert_to_dynamic_carray(terminations_dimensions[i], c_int)

        termination_dimensions_carray = cast(termination_dimensions_carray, POINTER(POINTER(c_int)))


        termination_transform_tau_carray = convert_to_dynamic_carray(terminations_tau, c_float)
        for i in range(len(termination_tau)):
          termination_tau_carray[i] = convert_to_dynamic_carray(terminations_tau[i], c_int)

        termination_tau_carray = cast(termination_tau_carray, POINTER(POINTER(c_int)))



        encoders_carray = (POINTER(POINTER(c_float)) * len(encoders))()

        for i in [0, len(encoders)-1]:
          encoder_carray = (POINTER(c_float) * len(self.encoders[i]))()

          for j in [0,len(encoders[i])-1]:
            encoder_carray[j] = convert_to_dynamic_carray(encoders[i][j], c_float)
          
          encoders_carray[i] = cast(encoder_carray, POINTER(POINTER(c_float)))

        encoders_carray = cast(encoders_carray, POINTER(POINTER(POINTER(c_float))))
          


        decoders_carray = (POINTER(POINTER(POINTER(c_float))) * len(decoders))()

        for i in range(len(decoders)):
          temp1 = (POINTER(POINTER(c_float)) * len(decoders[i]))()

          for j in range(len(decoders[i])):
            temp2 = (POINTER(c_float) * len(decoders[i][j]))()

            for k in range(len(decoders[i][j])):
              temp2[k] = convert_to_dynamic_carray(decoders[i][j][k], c_float)
            
            temp1[j] = cast(temp2, POINTER(POINTER(c_float)))
          
          decoders_carray[i] = cast(temp1, POINTER(POINTER(POINTER(c_float))))

        decoders_carray = cast(decoders_carray, POINTER(POINTER(POINTER(POINTER(c_float)))))



        neuron_data_carray = (POINTER(c_float) * len(neuron_data))

        for i in [0, len(neuron_data)-1]:
          neuron_data_carray[i] = convert_to_dynamic_carray(neuron_data[i], c_float)
        
        neuron_data_carray = cast(neuron_data_carray, POINTER(POINTER(c_float)))



        ensemble_data_carray = convert_to_dynamic_carray(ensemble_data, c_int)
        network_array_data_carray = convert_to_dynamic_carray(network_array_data, c_int)
        
        is_spiking_ensemble_carray = convert_to_dynamic_carray(is_spiking_ensemble, c_int)
        collect_spikes_carray = convert_to_dynamic_carray(collect_spikes, c_int)
        device_for_ensemble_carray = convert_to_dynamic_carray(device_for_ensemble, c_int)


        origin_required_by_cpu_carray = (POINTER(c_int) * len(origin_required_by_cpu))

        for i in [0, len(origin_required_by_cpu)-1]:
          origin_required_by_cpu_carray[i] = convert_to_dynamic_carray(origin_required_by_cpu[i], c_int)
        
        origin_required_by_cpu_carray = cast(origin_required_by_cpu_carray, POINTER(POINTER(c_int)))


        projection_data_carray = (POINTER(c_int) * len(projection_data))

        for i in [0, len(projection_data)-1]:
          projection_data_carray[i] = convert_to_dynamic_carray(projection_data[i], c_int)
        
        projection_data_carray = cast(projection_data_carray, POINTER(POINTER(c_int)))

        sizes = convert_to_dynamic_carray([len(nodes)] * 2, c_int)

        #call the GPU setup function!
        libNengoGPU.setup(num_devices, sizes, transforms_carray, termination_dimensions_carray, \
            termination_tau_carray, is_decoded_termination_carray, encoders_carray, decoders_carray, \
            decoder_dimensions_carray, neuron_data_carray, network_array_data_carray, ensemble_data_carray, \
            is_spiking_ensemble_carray, collect_spikes_carray, origin_required_by_cpu_carray, \
            projection_data_carray, device_for_ensemble_carray)


        input_carray = (POINTER(POINTER(c_float)) * len(nodes))()

        for i in [0, len(nodes)-1]:
          temp = (POINTER(c_float) * len(self.transforms[i]))()

          for j in [0,len([i])-1]:
            temp[j] = convert_to_dynamic_carray([0] * termination_dimensions[i][j], c_float)
          
          input_carray[i] = cast(temp, POINTER(POINTER(c_float)))

        input_carray = cast(input_carray, POINTER(POINTER(POINTER(c_float))))

        self.input_carray = input_carray
          


        decoders_carray = (POINTER(POINTER(POINTER(c_float))) * len(decoders))()
        self.ensemble_num_inputs = [len(t) for t in termination_dimensions]
        self.input_sizes = termination_dimensions

        self.output_array = 
        self.ensemble_num_outputs = [len(d) for d in decoder_dimensions]
        self.output_sizes = decoder_dimensions

        self.spikes = [[0.0] * 6]

    #returns a type
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
        else:
          carray[i] = convert_to_carray(l[i], t, depth - 1)

      return cast(carray, recursive_c_pointer_type(t, depth))





    def step(self,start,end):
        
        libNengoGPU.step()


    def kill(self):

        #call the setup function of the GPU library
        #don't forget to check how many GPU's we have available, and request a good number. 
        # also don't forget to pick a good assignment of nodes to devices (random is probably a good start)



    #def reset(self,randomize=False):
    #    for n in self.nodes: n.reset(randomize)

    def step(self,start,end):

      #call the GPU step function...how do we get the input and where do we put the input?

        if self.thread_pool is not None:
            self.thread_pool.step(start,end)
        else:    
            for p in self.projections:
                p.termination.setValues(p.origin.getValues())
            for n in self.nodes:
                n.run(start,end)

            #for t in self.tasks:
            #    t.run(start,end)

    def kill(self):
      #call the GPU kill function
