from ctypes import POINTER, c_float, c_int, cast, CDLL

import numpy as np
import nengo
from nengo.dists import Uniform
from collections import OrderedDict


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

    # index_vectors and stored_vectors should both be OrderedDicts
    # identical is whether all ensembles should be exactly the same
    def __init__(self, devices, index_vectors, stored_vectors, threshold=0.3,
                 neurons_per_item=20, dt=0.001, pstc=0.02, tau_rc=0.02,
                 tau_ref=0.002, radius=1.0, intercepts=Uniform(0.0, 0.3),
                 max_rates=Uniform(200, 350), identical=False,
                 probe_keys=[], do_print=False, seed=None, num_steps=1,
                 eval_points=None, collect_spikes=False):

        if not isinstance(index_vectors, OrderedDict):
            raise ValueError("index_vectors must be an OrderedDict")

        if not isinstance(stored_vectors, OrderedDict):
            raise ValueError("stored_vectors must be an OrderedDict")

        if not index_vectors.keys() == stored_vectors.keys():
            raise ValueError("Key order in index_vectors and stored_vectors"
                             "must be the same.")

        self.libNeuralAssocGPU = CDLL("libNeuralAssocGPU.so")

        threshold_func = lambda x: 1 if x > threshold else 0
        encoders = np.ones((neurons_per_item, 1))

        self.index_vectors = index_vectors
        self.stored_vectors = stored_vectors
        self.num_items = len(index_vectors)
        self.dimensions = len(index_vectors.values()[0])
        self.dt = dt
        self.num_steps = num_steps
        self.neurons_per_item = neurons_per_item
        num_devices = len(devices)

        if identical:
            model = nengo.Network("Associative Memory")
            with model:
                neuron_type = nengo.LIF(
                    tau_rc=tau_rc, tau_ref=tau_ref)

                assoc = nengo.Ensemble(
                    n_neurons=neurons_per_item, dimensions=1,
                    intercepts=intercepts, max_rates=max_rates,
                    encoders=encoders, label="assoc", seed=seed,
                    radius=radius, eval_points=eval_points,
                    neuron_type=neuron_type)

                dummy = nengo.Ensemble(1, 1)

                conn = nengo.Connection(assoc, dummy,
                                        function=threshold_func, seed=seed)

            sim = nengo.Simulator(model)

            gain = sim.data[assoc].gain
            bias = sim.data[assoc].bias
            decoders = sim.data[conn].decoders[0]
        else:
            raise NotImplementedError(
                "GPU can currently only be used if identical is also supplied")

            # Currently does not work
            model = nengo.Network("Associative Memory")
            with model:
                dummy = nengo.Ensemble(1, 1)

                for i in range(self.num_items):

                    neuron_type = nengo.LIF(
                        tau_rc=tau_rc, tau_ref=tau_ref)

                    assoc = nengo.Ensemble(
                        n_neurons=neurons_per_item, dimensions=1,
                        intercepts=intercepts, max_rates=max_rates,
                        encoders=encoders, label="assoc", radius=radius,
                        neuron_type=neuron_type)

                    conn = nengo.Connection(assoc, dummy,
                                            function=threshold_func)

            sim = nengo.Simulator(model)

            gain = []
            bias = []
            decoders = []

            for i in range(self.num_items):
                gain.append(sim.data[assoc].gain)
                bias.append(sim.data[assoc].bias)
                decoders.append(sim.data[conn].decoders[0])

            gain = np.array(gain).T
            bias = np.array(bias).T
            decoders = np.array(decoders).T

        # Arrays
        index_vector_vals = index_vectors.values()
        stored_vector_vals = stored_vectors.values()

        c_index_vectors = convert_to_carray(index_vector_vals, c_float, 2)
        c_stored_vectors = convert_to_carray(stored_vector_vals, c_float, 2)
        c_decoders = convert_to_carray(decoders, c_float, 1)

        neuron_depth = 1 if identical else 2
        c_gain = convert_to_carray(gain, c_float, neuron_depth)
        c_bias = convert_to_carray(bias, c_float, neuron_depth)
        c_devices = convert_to_carray(devices, c_int, neuron_depth)

        keys = index_vectors.keys()

        probe_map = {pk: keys.index(pk) for pk in probe_keys}
        sorted(probe_keys, key=lambda x: probe_map[x])
        self.probe_map = dict(zip(probe_keys, range(len(probe_keys))))
        probe_indices = [probe_map[pk] for pk in probe_keys]

        c_probe_indices = convert_to_carray(probe_indices, c_int, 1)

        # Scalars
        c_num_devices = c_int(num_devices)
        c_num_items = c_int(self.num_items)
        c_dimensions = c_int(self.dimensions)
        c_neurons_per_item = c_int(neurons_per_item)
        c_pstc = c_float(pstc)
        c_tau_ref = c_float(tau_ref)
        c_tau_rc = c_float(tau_rc)
        c_radius = c_float(radius)
        c_dt = c_float(dt)
        c_identical = c_int(int(identical))
        c_do_print = c_int(int(do_print))
        c_num_probes = c_int(len(probe_indices))
        c_do_spikes = c_int(int(collect_spikes))
        c_num_steps = c_int(num_steps)

        gpu_lib = self.libNeuralAssocGPU
        gpu_lib.setup(c_num_devices, c_devices, c_dt, c_num_items,
                      c_dimensions, c_index_vectors, c_stored_vectors, c_pstc,
                      c_decoders, c_neurons_per_item, c_gain, c_bias,
                      c_tau_ref, c_tau_rc, c_radius, c_identical, c_do_print,
                      c_probe_indices, c_num_probes, c_do_spikes, c_num_steps)

        self.elapsed_time = 0.0
        self.n_steps = 0
        self.mode = 'neural_assoc_gpu'

        # setup arrays needed in step function
        if num_steps == 1:
            self._output = np.zeros(self.dimensions)
            self._probes = np.zeros(len(probe_indices))
            self._spikes = np.zeros(len(probe_indices) * neurons_per_item)

            self._c_input = None
            self._c_output = convert_to_carray(self._output, c_float, 1)
            self._c_probes = convert_to_carray(self._probes, c_float, 1)
            self._c_spikes = convert_to_carray(self._spikes, c_float, 1)
        else:
            self._output = np.zeros((num_steps, self.dimensions))
            self._probes = np.zeros((num_steps, len(probe_indices)))
            self._spikes = np.zeros(
                (num_steps, len(probe_indices) * neurons_per_item))

            self._c_input = None

            _output = np.zeros(num_steps * self.dimensions)
            self._c_output = convert_to_carray(_output, c_float, 1)

            _probes = np.zeros(num_steps * len(probe_indices))
            self._c_probes = convert_to_carray(_probes, c_float, 1)

            _spikes = np.zeros(
                num_steps * len(probe_indices) * neurons_per_item)
            self._c_spikes = convert_to_carray(_spikes, c_float, 1)

    def step(self, input_vector):

        if self.num_steps > 1:
            return self.multi_step(input_vector)

        self._c_input = convert_to_carray(input_vector, c_float, 1)

        c_start_time = c_float(self.elapsed_time)
        c_end_time = c_float(self.elapsed_time + self.dt)
        c_n_steps = c_int(self.n_steps)

        gpu_lib = self.libNeuralAssocGPU
        gpu_lib.step(self._c_input, self._c_output,
                     self._c_probes, self._c_spikes,
                     c_start_time, c_end_time,
                     c_n_steps)

        for i in range(self._output.size):
            self._output[i] = self._c_output[i]

        for i in range(self._probes.size):
            self._probes[i] = self._c_probes[i]

        for i in range(self._spikes.size):
            self._spikes[i] = self._c_spikes[i]

        self.n_steps += 1
        self.elapsed_time += 1

        return self._output

    def multi_step(self, input):
        self._c_input = convert_to_carray(
            np.reshape(input, self.num_steps * self.dimensions), c_float, 1)

        c_start_time = c_float(self.elapsed_time)
        c_end_time = c_float(self.elapsed_time + self.dt * self.n_steps)
        c_n_steps = c_int(self.n_steps)

        gpu_lib = self.libNeuralAssocGPU
        gpu_lib.step(self._c_input, self._c_output,
                     self._c_probes, self._c_spikes,
                     c_start_time, c_end_time, c_n_steps)

        shape0 = self._output.shape[0]
        shape1 = self._output.shape[1]
        for i in range(shape0):
            for j in range(shape1):
                self._output[i, j] = self._c_output[i * shape1 + j]

        shape0 = self._probes.shape[0]
        shape1 = self._probes.shape[1]
        for i in range(shape0):
            for j in range(shape1):
                self._probes[i, j] = self._c_probes[i * shape1 + j]

        shape0 = self._spikes.shape[0]
        shape1 = self._spikes.shape[1]
        for i in range(shape0):
            for j in range(shape1):
                self._spikes[i, j] = self._c_spikes[i * shape1 + j]

        self.n_steps += self.n_steps
        self.elapsed_time += self.dt * self.n_steps

        return self._output

    def probe_func(self, probe_key):
        """
        Return a function which gets the value of a probe in the associative
        memory. The idea is to create a node whose output is this function,
        and then to probe that node. That lets us hook the output of the GPU
        simulation into the normal probe system.
        """

        probe_index = self.probe_map[probe_key]

        def f(t):
            return self._probes[probe_index]

        def multi_step_f(t):
            time_index = int(t / self.dt)
            return self._probes[time_index, probe_index]

        if self.num_steps == 1:
            return f
        else:
            return multi_step_f

    def spike_func(self, spike_key):
        """
        Same trick as with probe_func.
        """

        # can use probe map here, spikes are in same order as probes
        spike_index = self.probe_map[spike_key]

        lo_index = spike_index * self.neurons_per_item
        hi_index = (spike_index + 1) * self.neurons_per_item

        def f(t):
            return self._spikes[lo_index:hi_index]

        def multi_step_f(t):
            time_index = int(t / self.dt)
            return self._spikes[time_index, lo_index:hi_index]

        if self.num_steps == 1:
            return f
        else:
            return multi_step_f

    def kill(self):
        self.libNeuralAssocGPU.kill()

    def reset(self):
        self.elapsed_time = 0.0
        self.n_steps = 0
        self.libNeuralAssocGPU.reset()
