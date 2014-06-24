# Neural Extraction Algorithm
from gpu_assoc_memory import AssociativeMemoryGPU
from neural_extraction import NeuralExtraction

import datetime
from collections import OrderedDict

import numpy as np

from nengo.networks import EnsembleArray
import nengo


def make_func(cls, attr):
    def f(t):
        return getattr(cls, attr)
    return f


class FastNeuralExtraction(NeuralExtraction):

    _type = "Neural"

    def __init__(self, *args, **kwargs):
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
        super(FastNeuralExtraction, self).__init__(*args, **kwargs)

    def setup_simulator(self):
        self.unbind_model = nengo.Network(label="Unbind", seed=self.seed)
        self.build_unbind(self.unbind_model)

        self.build_association()

        self.output_model = nengo.Network(label="Output", seed=self.seed)
        self.build_output(self.output_model)

        print "Building simulators"
        self.unbind_simulator = self.build_simulator(self.unbind_model)
        self.output_simulator = self.build_simulator(self.output_model)

        self.simulator = self.output_simulator

    def build_association(self):

        tau_rc = self.assoc_params.tau_rc
        tau_ref = self.assoc_params.tau_ref
        radius = self.assoc_params.radius
        eval_points = self.assoc_params.eval_points
        intercepts = self.assoc_params.intercepts

        # Add a nengo.Node which calls out to a GPU library for
        # simulating the associative memory
        self.assoc_memory = \
            AssociativeMemoryGPU(self.gpus, self.index_vectors,
                                 self.stored_vectors,
                                 threshold=self.threshold,
                                 neurons_per_item=self.neurons_per_item,
                                 tau_ref=tau_ref, tau_rc=tau_rc,
                                 eval_points=eval_points,
                                 intercepts=intercepts,
                                 radius=radius, do_print=False,
                                 identical=self.identical,
                                 probe_keys=self.probe_keys,
                                 seed=self.seed,
                                 num_steps=self.timesteps)

        self.assoc_output = np.zeros((1, self.dim))

    def build_output(self, model):

        assoc_probes = OrderedDict()
        threshold_probes = OrderedDict()

        synapse = self.assoc_params.synapse

        with model:
            input = nengo.Node(output=self.assoc_output_func)
            output = EnsembleArray(nengo.LIF(self.neurons_per_dim),
                                   self.dim, label="output",
                                   radius=self.radius)
            nengo.Connection(input, output.input, synapse=synapse)
            self.output_probe = nengo.Probe(output.output, 'output',
                                            synapse=0.02)

            for k in self.probe_keys:
                n = nengo.Node(output=self.assoc_memory.probe_func(k))
                probe = nengo.Probe(n, synapse=synapse)

                threshold_probes[k] = probe

        self.assoc_probes = assoc_probes
        self.threshold_probes = threshold_probes

    def extract(self, item, query, *args, **kwargs):
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
            self.plot_simulation()

        vector = self.data[self.output_probe][-1, :]
        return [vector]

    def assoc_output_func(self, t):
        index = int(t / self.dt)
        return self.assoc_output[index]

    def reset_nodes(self):
        pass

    def write_to_runtime_file(self, delta, label=''):
        to_print = [self.dim, self.num_items,
                    self.neurons_per_item, self.neurons_per_dim,
                    self.timesteps, "OCL: "+str(self.ocl),
                    "GPUS: "+str(self.gpus), "fast", delta]
        print >> self.runtimes_file, label, \
            ": " ",".join([str(tp) for tp in to_print])

    def print_config(self, output_file):
        super(FastNeuralExtraction, self).print_config(output_file)
        output_file.write("Fast neural extractor config:")
