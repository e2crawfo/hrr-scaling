# Neural Extraction Algorithm
from gpu_assoc_memory import AssociativeMemoryGPU
from new_neural_assoc_memory import NewNeuralAssociativeMemory

import datetime
from collections import OrderedDict

import numpy as np

from nengo.networks import CircularConvolution, EnsembleArray
from nengo.utils.distributions import Uniform
import nengo.utils.numpy as npext
import nengo


def make_func(cls, attr):
    def f(t):
        return getattr(cls, attr)
    return f


class FastNeuralAssociativeMemory(NewNeuralAssociativeMemory):

    _type = "Neural"

    def __init__(self, index_vectors, stored_vectors, threshold=0.3,
                 neurons_per_item=20, neurons_per_dim=50, timesteps=100,
                 dt=0.001, tau_rc=0.02, tau_ref=0.002, pstc=0.005,
                 output_dir=".", probe_keys=[], plot=False, ocl=[],
                 gpus=[], identical=False):
        """
        A GPU neural associative memory with a unique simulation strategy.
        Instead of making one big model with all the necessary components in
        it, we make multiple models, and simulate each completely before
        simulating the next one. The main benefit of this is that the GPU won't
        have to pause every time step to load data on and off. This is made
        possible by the fact that the network is completely feedforward, and so
        earlier components can be simulated without requiring input from later
        components.

        index_vectors and stored_vectors are both dictionaries mapping from
        tuples of the form (POS, number), indicating a synset, to numpy
        ndarrays containing the assigned vector
        """

        then = datetime.datetime.now()

        self.unitary = False
        self.bidirectional = False
        self.identity = False

        self.ideal_dot = None
        self.second_dot = None

        self.output_dir = output_dir

        self.return_vec = True

        self.index_vectors = index_vectors
        self.stored_vectors = stored_vectors

        self.runtimes_file = open(self.output_dir+'/neural_runtimes', 'a')

        self.dim = len(self.index_vectors.values()[0])
        self.num_items = len(self.index_vectors)
        self.neurons_per_item = neurons_per_item
        self.neurons_per_dim = neurons_per_dim
        self.dt = dt
        self.timesteps = timesteps
        self.plot = plot
        self.gpus = gpus
        self.ocl = ocl

        seed = np.random.randint(npext.maxint)

        self.threshold = threshold
        self.transfer_func = lambda x: 1 if x > self.threshold else 0

        radius = 5.0 / np.sqrt(self.dim)

        self.A_input_vector = np.zeros(self.dim)
        self.B_input_vector = np.zeros(self.dim)

        synapse = pstc

        unbind_model = nengo.Network(label="Unbind", seed=seed)
        with unbind_model:
            self.A_input_func = make_func(self, "A_input_vector")
            self.B_input_func = make_func(self, "B_input_vector")

            A_input = nengo.Node(output=self.A_input_func, size_out=self.dim)
            B_input = nengo.Node(output=self.B_input_func, size_out=self.dim)

            A = EnsembleArray(nengo.LIF(neurons_per_dim), self.dim,
                              label="A", radius=radius)
            B = EnsembleArray(nengo.LIF(neurons_per_dim), self.dim,
                              label="B", radius=radius)
            cconv = CircularConvolution(nengo.LIF(int(2 * neurons_per_dim)),
                                        self.dim, invert_b=True)
            D = EnsembleArray(nengo.LIF(neurons_per_dim), self.dim,
                              label="D", radius=radius)

            nengo.Connection(A_input, A.input)
            nengo.Connection(B_input, B.input)

            nengo.Connection(A.output, cconv.A, synapse=synapse)
            nengo.Connection(B.output, cconv.B, synapse=synapse)
            nengo.Connection(cconv.output, D.input, synapse=synapse)

            unbind_probe = nengo.Probe(D.output, 'output', synapse=0.02)

        intercepts = Uniform(0.0, 0.3)
        max_rates = Uniform(200.0, 200.0)

        # scale = 10.0
        assoc_radius = 0.5

        # Add a nengo.Node which calls out to a GPU library for
        # simulating the associative memory

        self.assoc_memory = \
            AssociativeMemoryGPU(gpus, index_vectors, stored_vectors,
                                 threshold=threshold,
                                 neurons_per_item=neurons_per_item,
                                 intercepts=intercepts,
                                 max_rates=max_rates, tau_ref=tau_ref,
                                 tau_rc=tau_rc, radius=assoc_radius,
                                 do_print=False, identical=identical,
                                 probe_keys=probe_keys, seed=seed,
                                 num_steps=self.timesteps)

        self.assoc_output = np.zeros((1, self.dim))

        assoc_probes = OrderedDict()
        transfer_probes = OrderedDict()

        output_model = nengo.Network(label="Output", seed=seed)
        with output_model:
            input = nengo.Node(output=self.output_func)
            output = EnsembleArray(nengo.LIF(neurons_per_dim),
                                   self.dim, label="output",
                                   radius=radius)
            nengo.Connection(input, output.input, synapse=synapse)
            output_probe = nengo.Probe(output.output, 'output', synapse=0.02)

            for k in probe_keys:
                n = nengo.Node(output=self.assoc_memory.probe_func(k))
                probe = nengo.Probe(n, synapse=0.02)

                transfer_probes[k] = probe
            # end output model

        self.D_probe = unbind_probe
        self.assoc_probes = assoc_probes
        self.transfer_probes = transfer_probes
        self.output_probe = output_probe

        print "Building simulators"
        self.unbind_simulator = self.build_simulator(unbind_model)
        self.output_simulator = self.build_simulator(output_model)
        self.simulator = self.output_simulator

        now = datetime.datetime.now()
        self.write_to_runtime_file(now - then, "setup")

    def unbind_and_associate(self, item, query, *args, **kwargs):
        self.print_instance_difficulty(item, query)

        then = datetime.datetime.now()

        self.A_input_vector = item
        self.B_input_vector = query

        print "Simulating unbind model"
        self.unbind_simulator.run(self.timesteps * self.dt)

        assoc_input = self.unbind_simulator.data[self.D_probe]

        print "Simulating associative memory"
        self.assoc_output = self.assoc_memory.multi_step(assoc_input)

        self.output_index = 0

        print "Simulating output"
        self.output_simulator.run(self.timesteps * self.dt)

        now = datetime.datetime.now()
        self.write_to_runtime_file(now - then, "unbind")

        # for plotting
        self.data = {k: v for (k, v) in self.output_simulator.data.iteritems()}
        self.data[self.D_probe] = self.unbind_simulator.data[self.D_probe]

        if self.plot:
            self.plot_cleanup_activities()

        vector = self.data[self.output_probe][-1, :]
        return [vector]

    def output_func(self, t):
        index = int(t / self.dt)
        return self.assoc_output[index]

    def finish(self):
        pass

    def reset_nodes(self):
        pass

    def write_to_runtime_file(self, delta, label=''):
        to_print = [self.dim, self.num_items,
                    self.neurons_per_item, self.neurons_per_dim,
                    self.timesteps, "OCL: "+str(self.ocl),
                    "GPUS: "+str(self.gpus), "fast", delta]
        print >> self.runtimes_file, label, \
            ": " ",".join([str(tp) for tp in to_print])
