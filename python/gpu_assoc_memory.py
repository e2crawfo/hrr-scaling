from ctypes import POINTER, c_float, c_int, cast, CDLL

import numpy as np
import nengo
from nengo.utils.distributions import Uniform

# try:
#     import matplotlib.pyplot as plt
# except:
#     pass


# Returns a type (specifically a pointer to type "t" with depth "depth")
def recursive_c_pointer_type(t, depth):
    return (t if depth < 1 else
            POINTER(recursive_c_pointer_type(t, depth - 1)))


# Returns a carray with depth levels of indirection
def convert_to_carray(l, t, depth):
    if depth < 1:
        return

    carray = (recursive_c_pointer_type(t, depth-1) * len(l))()

    for i in range(len(l)):
        if depth == 1:
            carray[i] = l[i]
        elif len(l[i]) > 0:
            # If we ever have an empty list, we just don't
            # descend there and leave it as a null pointer
            carray[i] = convert_to_carray(l[i], t, depth - 1)

    return cast(carray, recursive_c_pointer_type(t, depth))


class AssociativeMemoryGPU(object):

    # index_vectors and stored_vectors should both be lists of vectors
    # identical is whether all ensembles should be exactly the same
    def __init__(self, devices, index_vectors, stored_vectors, threshold=0.3,
                 neurons_per_item=20, dt=0.001, pstc=0.02, tau_rc=0.02,
                 tau_ref=0.002, intercepts=Uniform(0.0, 0.3),
                 max_rates=Uniform(200, 200), identical=False, do_print=False):

        self.libNeuralCleanupGPU = CDLL("libNeuralAssocGPU.so")

        threshold_func = lambda x: 1 if x > threshold else 0
        assoc_encoders = np.ones((neurons_per_item, 1))

        self.index_vectors = index_vectors
        self.stored_vectors = stored_vectors
        self.num_items = len(index_vectors)
        dimensions = len(index_vectors.values()[0])
        num_devices = len(devices)

        if identical:
            model = nengo.Network("Associative Memory")
            with model:
                assoc = nengo.Ensemble(neurons_per_item, 1,
                                       intercepts=intercepts,
                                       max_rates=max_rates,
                                       encoders=assoc_encoders,
                                       label="assoc",
                                       radius=0.5)

                dummy = nengo.Ensemble(1, 1)

                conn = nengo.Connection(assoc, dummy, function=threshold_func)

            sim = nengo.Simulator(model)

            gain = sim.data[assoc]['gain']
            bias = sim.data[assoc]['bias']
            decoders = sim.data[conn]['decoders']
        else:
            pass

        # Arrays
        index_vector_vals = index_vectors.values()
        stored_vector_vals = stored_vectors.values()

        c_index_vectors = convert_to_carray(index_vector_vals, c_float, 2)
        c_stored_vectors = convert_to_carray(stored_vector_vals, c_float, 2)
        c_decoders = convert_to_carray(decoders, c_float, 1)
        c_gain = convert_to_carray(gain, c_float, 1)
        c_bias = convert_to_carray(bias, c_float, 1)
        c_devices = convert_to_carray(devices, c_int, 1)

        # Scalars
        c_num_devices = c_int(num_devices)
        c_num_items = c_int(self.num_items)
        c_dimensions = c_int(dimensions)
        c_neurons_per_item = c_int(neurons_per_item)
        c_pstc = c_float(pstc)
        c_tau_ref = c_float(tau_ref)
        c_tau_rc = c_float(tau_rc)
        c_dt = c_float(dt)
        c_do_print = c_int(int(do_print))
        c_identical = c_int(int(identical))

        gpu_lib = self.libNeuralAssocGPU
        gpu_lib.setup(c_num_devices, c_devices, c_dt, c_num_items,
                      c_dimensions, c_index_vectors, c_stored_vectors, c_pstc,
                      c_decoders, c_neurons_per_item, c_gain, c_bias,
                      c_tau_ref, c_tau_rc, c_identical)

        # setup arrays needed in step function
        self._input = np.zeros(dimensions)
        self._output = np.zeros(dimensions)

        self._c_input = None
        self._c_output = None

        self.elapsed_time = 0.0
        self.mode = 'gpu_cleanup'

    def step(self, input_vector):
        output_vector = np.zeros()
        return output_vector

    def reset(self):

        self.elapsed_time = 0.0

        for key in self.probe_data:
            node, probes, spikes = self.probe_data[key]
            node.reset()
            for p in probes:
                p.reset()

        self.libNeuralCleanupGPU.reset()
