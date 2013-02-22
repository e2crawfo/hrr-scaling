from ctypes import *
import numpy

from ccm.lib import nef

thresh=0.3
def transfer(x):
    if x>thresh: return 1
    return 0

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
  def __init__(self, devices, dt, auto, index_vectors, result_vectors, tau, node, pstc=0.02):
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

      numVectors = len(index_vectors)

      decoder = node.get_decoder(func=transfer)
      encoder = node.basis
      alpha = node.alpha
      Jbias = node.Jbias
      t_rc = node.t_rc
      t_ref = node.t_ref

      num_neurons = len(alpha)

      self.dt = dt
      print "Decoder sum: ", sum(decoder)

      c_index_vectors = convert_to_carray(index_vectors, c_float, 2)
      c_result_vectors = convert_to_carray(result_vectors, c_float, 2)

      c_encoder = convert_to_carray(encoder, c_float, 1)
      c_decoder = convert_to_carray(decoder, c_float, 1)

      c_alpha = convert_to_carray(alpha, c_float, 1)
      c_Jbias = convert_to_carray(Jbias, c_float, 1)

      self.elapsed_time = 0.0

      print "starting GPU setup"
      self.libNeuralCleanupGPU.setup(c_int(devices), c_float(dt), c_int(numVectors), c_int(self.dimensions), c_int(int(auto)), c_index_vectors, c_result_vectors, c_float(tau), c_encoder, c_decoder, c_int(num_neurons), c_alpha, c_Jbias, c_float(t_ref), c_float(t_rc)) 
      print "done GPU setup"

      self.mode='gpu_cleanup'

  def tick_accumulator(self, dt):
      for i in self.inputs:
        i.transmit_in(dt)

  def get_output_array_gpu_cleanup(self, conn, dt):
      return self._output

  def add_input_array_gpu_cleanup(self, conn, tau, dt):
      self._input = conn.array

  def _calc_output(self):
      
      self._c_spikes = convert_to_carray(numpy.zeros(10), c_float, 1)
      self._c_input = convert_to_carray(self._input, c_float, 1)
      self._c_output = convert_to_carray(numpy.zeros(self.dimensions), c_float, 1)

      self.libNeuralCleanupGPU.step(self._c_input, self._c_output, self._c_spikes, c_float(self.elapsed_time), c_float(self.elapsed_time + self.dt))

      #make sure output is NOT scaled by dt_over_tau,
      #we let that happen in the termination of results node
      self.elapsed_time = self.elapsed_time + self.dt
      for i in range(len(self._output)):
        self._output[i] = self._c_output[i]

  def kill(self):
      self.libNeuralCleanupGPU.kill()

  def reset(self):
      self.libNeuralCleanupGPU.reset()

  
