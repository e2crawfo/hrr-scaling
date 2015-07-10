# Neural Extractor Algorithm
from hrr_scaling.extractor import Extractor
from hrr_scaling.gpu_assoc_memory import AssociativeMemoryGPU
from hrr_scaling.tools import hrr

import string
import datetime
from collections import OrderedDict, namedtuple
import warnings
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import nengo
from nengo.dists import Gaussian
from nengo.networks import CircularConvolution, EnsembleArray
from nengo.spa import AssociativeMemory
from nengo.dists import Uniform
import nengo.utils.numpy as npext

ocl_imported = True
try:
    import pyopencl
    import nengo_ocl
except:
    ocl_imported = False


def make_func(obj, attr):
    def f(t):
        return getattr(obj, attr)
    return f

AssocParams = namedtuple('AssocParams',
                         ['tau_rc', 'tau_ref', 'synapse',
                          'radius', 'eval_points', 'intercepts'])


class NeuralExtractor(Extractor):

    _type = "Neural"

    def __init__(self, index_vectors, stored_vectors, threshold=0.3,
                 neurons_per_item=20, neurons_per_dim=50, timesteps=100,
                 dt=0.001, tau_rc=0.02, tau_ref=0.002, synapse=0.005,
                 output_dir=".", probe_keys=[], plot=False, show=False,
                 ocl=[], gpus=[], identical=False, collect_spikes=False):
        """
        index_vectors and stored_vectors are both dictionaries mapping from
        tuples of the form (POS, number), indicating a synset, to numpy
        ndarrays containing the assigned vector

        Synapses: synapse arg controls synapses between connections that do
        not involve association populations. The input and output probes use
        synapses of 0.02. Connections involving association populations have
        their own synapse, which is set in the init function.
        """

        then = datetime.datetime.now()

        self.ideal_dot = None
        self.second_dot = None

        self.return_vec = True

        self.output_dir = output_dir

        self.index_vectors = index_vectors
        self.stored_vectors = stored_vectors

        self.runtimes_file = open(self.output_dir+'/neural_runtimes', 'a')

        self.dimension = len(self.index_vectors.values()[0])
        self.num_items = len(self.index_vectors)
        self.neurons_per_item = neurons_per_item
        self.neurons_per_dim = neurons_per_dim
        self.dt = dt
        self.timesteps = timesteps
        self.plot = plot
        self.show = show
        self.gpus = gpus
        self.ocl = ocl
        self.probe_keys = probe_keys
        self.identical = identical
        self.seed = np.random.randint(npext.maxint)
        self.collect_spikes = collect_spikes

        self.threshold = threshold
        self.threshold_func = lambda x: 1 if x > self.threshold else 0

        # association population parameters
        intercepts_low = 0.29
        intercepts_range = 0.00108
        n_eval_points = 750
        eval_point_mean = 0.39
        eval_point_std = 0.32

        intercepts = Uniform(intercepts_low, intercepts_low + intercepts_range)
        eval_points = np.random.normal(eval_point_mean,
                                       eval_point_std,
                                       (n_eval_points, 1))

        self.assoc_params = AssocParams(tau_rc=0.034, tau_ref=0.0026,
                                        synapse=0.005, radius=1.0,
                                        eval_points=eval_points,
                                        intercepts=intercepts)

        # other population parameters
        n_eval_points = 750

        # TODO: potentially use SubvectorLength distribution here.
        self.eval_points = Gaussian(0, 0.06)
        self.radius = 5.0 / np.sqrt(self.dimension)
        self.synapse = synapse

        self.A_input_vector = np.zeros(self.dimension)
        self.B_input_vector = np.zeros(self.dimension)

        self.setup_simulator()

        now = datetime.datetime.now()
        self.write_to_runtime_file(now - then, "setup")

    def setup_simulator(self):
        self.model = nengo.Network(label="Extractor", seed=self.seed)

        print "Specifiying model"
        # The order is important here
        self.build_unbind(self.model)
        self.build_output(self.model)
        self.build_association(self.model)

        print "Building simulator"
        self.simulator = self.build_simulator(self.model)
        print "Done building simulator"

    def build_unbind(self, model):
        A_input_func = make_func(self, "A_input_vector")
        B_input_func = make_func(self, "B_input_vector")

        neurons_per_dim = self.neurons_per_dim
        radius = self.radius
        eval_points = self.eval_points
        synapse = self.synapse
        dimension = self.dimension

        with model:
            A_input = nengo.Node(output=A_input_func, size_out=dimension)
            B_input = nengo.Node(output=B_input_func, size_out=dimension)

            A = EnsembleArray(
                n_neurons=neurons_per_dim, n_ensembles=dimension, label="A",
                radius=radius, eval_points=eval_points)

            B = EnsembleArray(
                n_neurons=neurons_per_dim, n_ensembles=dimension, label="B",
                radius=radius, eval_points=eval_points)

            cconv = CircularConvolution(
                n_neurons=int(2 * neurons_per_dim), dimensions=dimension,
                invert_b=True)

            D = EnsembleArray(
                n_neurons=neurons_per_dim, n_ensembles=dimension, label="D",
                radius=radius, eval_points=eval_points)

            A_output = A.output
            B_output = B.output
            D_output = D.output
            cconv_output = cconv.output

            nengo.Connection(A_input, A.input)
            nengo.Connection(B_input, B.input)

            nengo.Connection(A_output, cconv.A, synapse=synapse)
            nengo.Connection(B_output, cconv.B, synapse=synapse)
            nengo.Connection(cconv_output, D.input, synapse=synapse)

            assoc_synapse = self.assoc_params.synapse

            self.D_probe = nengo.Probe(D_output, 'output',
                                       synapse=assoc_synapse)

            self.input_probe = nengo.Probe(
                A_output, 'output', synapse=synapse)

            self.D_output = D_output

            self.A = A
            self.B = B
            self.cconv = cconv
            self.D = D

    def build_association(self, model):

        tau_rc = self.assoc_params.tau_rc
        tau_ref = self.assoc_params.tau_ref
        synapse = self.assoc_params.synapse
        radius = self.assoc_params.radius
        eval_points = self.assoc_params.eval_points
        intercepts = self.assoc_params.intercepts

        neurons_per_item = self.neurons_per_item
        threshold = self.threshold

        assoc_probes = OrderedDict()
        threshold_probes = OrderedDict()
        assoc_spike_probes = OrderedDict()

        with model:
            if self.gpus:

                if not self.identical:
                    raise NotImplementedError(
                        "Currently, can only use gpu if --identical"
                        " is also specified")

                # Add a nengo.Node which calls out to a GPU library for
                # simulating the associative memory
                self.assoc_mem = \
                    AssociativeMemoryGPU(self.gpus, self.index_vectors,
                                         self.stored_vectors,
                                         threshold=threshold,
                                         neurons_per_item=neurons_per_item,
                                         tau_ref=tau_ref, tau_rc=tau_rc,
                                         eval_points=eval_points,
                                         intercepts=intercepts,
                                         radius=radius, do_print=False,
                                         identical=self.identical,
                                         probe_keys=self.probe_keys,
                                         seed=self.seed,
                                         collect_spikes=self.collect_spikes)

                def gpu_function(t, input_vector):
                    output_vector = self.assoc_mem.step(input_vector)
                    return output_vector

                assoc = nengo.Node(
                    output=gpu_function, size_in=self.dimension,
                    size_out=self.dimension)

                nengo.Connection(self.D_output, assoc, synapse=synapse)
                nengo.Connection(assoc, self.output.input, synapse=synapse)

                for k in self.probe_keys:
                    node = nengo.Node(output=self.assoc_mem.probe_func(k))
                    probe = nengo.Probe(node, synapse=synapse)

                    threshold_probes[k] = probe

                    node = nengo.Node(output=self.assoc_mem.spike_func(k))
                    assoc_spike_probes[k] = nengo.Probe(node, synapse=None)

            else:
                self.assoc_mem = AssociativeMemory(
                    input_vocab=np.array(self.index_vectors.values()),
                    output_vocab=np.array(self.stored_vectors.values()),
                    threshold=self.threshold,
                    neuron_type=nengo.LIF(tau_rc=tau_rc, tau_ref=tau_ref),
                    n_neurons_per_ensemble=neurons_per_item)

                nengo.Connection(
                    self.D_output, self.assoc_mem.input, synapse=synapse)
                nengo.Connection(
                    self.assoc_mem.output, self.output.input, synapse=synapse)

                # for k in self.index_vectors:
                #     if k in self.probe_keys:

                #         assoc_probes[k] = nengo.Probe(
                #             assoc, 'decoded_output', synapse=synapse)

                #         threshold_probes[k] = nengo.Probe(
                #             assoc, 'decoded_output',
                #             synapse=synapse,
                #             function=self.threshold_func,
                #             seed=self.seed)

                #         assoc_spike_probes[k] = nengo.Probe(
                #             assoc, 'spikes', synapse=None)

        self.assoc_probes = assoc_probes
        self.threshold_probes = threshold_probes
        self.assoc_spike_probes = assoc_spike_probes

    def build_output(self, model):
        with model:

            self.output = EnsembleArray(
                n_neurons=self.neurons_per_dim, n_ensembles=self.dimension,
                label="output", radius=self.radius)

            output_output = self.output.output

            self.output_probe = nengo.Probe(
                output_output, 'output', synapse=0.02)

    def extract(self, item, query, target_keys=None, *args, **kwargs):
        then = datetime.datetime.now()

        if target_keys:
            self.print_instance_difficulty(item, query, target_keys)

        self.reset()

        self.A_input_vector = item
        self.B_input_vector = query

        self.simulator.run(self.timesteps * self.dt)
        self.data = self.simulator.data

        now = datetime.datetime.now()
        self.write_to_runtime_file(now - then, "unbind")

        if self.plot:
            self.plot_simulation(target_keys)

        vector = self.simulator.data[self.output_probe][-1, :]
        return [vector]

    def reset(self):
        if hasattr(self.assoc_mem, 'reset'):
            self.assoc_mem.reset()

        if hasattr(self.simulator, 'reset'):
            self.simulator.reset()
        else:
            warnings.warn("Non-GPU ensembles could not be reset")

    def build_simulator(self, model):
        if ocl_imported and self.ocl:
            platforms = pyopencl.get_platforms()

            # 0 is the Nvidia platform
            devices = platforms[0].get_devices()
            devices = [devices[i] for i in self.ocl]
            devices.sort()

            ctx = pyopencl.Context(devices=devices)

            simulator = nengo_ocl.sim_ocl.Simulator(model, context=ctx)
        else:
            if self.ocl:
                print "Failed to import nengo_ocl"

            simulator = nengo.Simulator(model)

        return simulator

    def plot_simulation(self, target_keys):
        then = datetime.datetime.now()

        correct_key = None
        if target_keys:
            correct_key = target_keys[0]

        sim = self.simulator
        t = sim.trange()

        max_val = 5.0 / np.sqrt(self.dimension)

        gs = gridspec.GridSpec(9, 2)
        fig = plt.figure(figsize=(10, 10))

        ax = plt.subplot(gs[0, 0])

        plt.plot(t, self.data[self.D_probe], label='D')
        title = 'Before Association: Vector'
        ax.text(.01, 1.20, title, horizontalalignment='left',
                transform=ax.transAxes)
        plt.ylim((-max_val, max_val))

        ax = plt.subplot(gs[0, 1])
        plt.plot(t, self.data[self.output_probe], label='Output')
        title = 'After Association: Vector'
        ax.text(.01, 1.20, title, horizontalalignment='left',
                transform=ax.transAxes)
        plt.ylim((-max_val, max_val))

        ax = plt.subplot(gs[1:3, :])

        if len(self.index_vectors) < 1000:
            for key, v in self.index_vectors.iteritems():
                input_sims = np.dot(self.data[self.D_probe], v)
                label = str(key[1])
                if key == correct_key:
                    plt.plot(t, input_sims, '--', label=label + '*')
                else:
                    plt.plot(t, input_sims, label=label)

            title = ('Dot Products Before Association.\n'
                     'Target is dashed line.\n')

            ax.text(.01, 0.80, title, horizontalalignment='left',
                    transform=ax.transAxes)
            # plt.legend(bbox_to_anchor=(-0.03, 0.5), loc='center right')
            if self.ideal_dot:
                ax.text(.01, 0.10, "Ideal dot: " + str(self.ideal_dot),
                        horizontalalignment='left', transform=ax.transAxes)
            if self.second_dot:
                ax.text(.99, 0.10, "Second dot: " + str(self.second_dot),
                        horizontalalignment='right', transform=ax.transAxes)

            plt.ylim((-1.0, 1.5))
            plt.axhline(1.0, ls=':', c='k')

        ax = plt.subplot(gs[3:5, :])
        for key, p in self.assoc_probes.iteritems():
            if key == correct_key:
                plt.plot(t, self.data[p], '--')
            else:
                plt.plot(t, self.data[p])

        title = 'Association Activation. \nTarget:' + str(correct_key)
        ax.text(.01, 0.80, title, horizontalalignment='left',
                transform=ax.transAxes)
        plt.ylim((-0.2, 1.5))
        plt.axhline(y=1.0, ls=':', c='k')

        ax = plt.subplot(gs[5:7, :])

        for key, p in self.threshold_probes.iteritems():
            if key == correct_key:
                plt.plot(t, self.data[p], '--', label=str(key))
            else:
                plt.plot(t, self.data[p], label=str(key))

        title = 'Assoc. Transfer Activation. \nTarget:' + str(correct_key)
        ax.text(.01, 0.80, title, horizontalalignment='left',
                transform=ax.transAxes)

        plt.ylim((-0.2, 1.5))
        plt.axhline(y=1.0, ls=':', c='k')

        ax = plt.subplot(gs[7:9, :])
        before_ls = '--'
        after_ls = '-'
        before_norms = [np.linalg.norm(v) for v in self.data[self.D_probe]]
        after_norms = [np.linalg.norm(v) for v in self.data[self.output_probe]]

        plt.plot(t, before_norms, before_ls, c='g', label='Norm - Before')
        plt.plot(t, after_norms, after_ls, c='g', label='Norm - After')

        if correct_key is not None:
            correct_index_hrr = hrr.HRR(data=self.index_vectors[correct_key])
            correct_stored_hrr = hrr.HRR(data=self.stored_vectors[correct_key])

            before_sims = [correct_index_hrr.compare(hrr.HRR(data=i))
                           for i in self.data[self.D_probe]]

            after_sims = [correct_stored_hrr.compare(hrr.HRR(data=o))
                          for o in self.data[self.output_probe]]

            plt.plot(t, before_sims, before_ls, c='b',
                     label='Cosine Sim - Before')
            plt.plot(t, after_sims, after_ls, c='b',
                     label='Cosine Sim - After')

        title = 'Before/After'
        ax.text(.01, 0.90, title, horizontalalignment='left',
                transform=ax.transAxes)
        plt.ylim((-1.0, 1.5))
        plt.legend(loc=3)
        plt.axhline(y=1.0, ls=':', c='k')

        date_time_string = str(datetime.datetime.now()).split('.')[0]
        date_time_string = reduce(lambda y, z: string.replace(y, z, "_"),
                                  [date_time_string, ":", ".", " ", "-"])

        plot_path = os.path.join(
            self.output_dir, 'neural_extraction_'+date_time_string+".png")

        plt.savefig(plot_path)

        now = datetime.datetime.now()
        self.write_to_runtime_file(now - then, "plot")

        if self.show:
            plt.show()

        plt.close(fig)

    def write_to_runtime_file(self, delta, label=''):
        to_print = [self.dimension, self.num_items,
                    self.neurons_per_item, self.neurons_per_dim,
                    self.timesteps, "OCL: "+str(self.ocl),
                    "GPUS: "+str(self.gpus), delta]
        print >> self.runtimes_file, label, \
            ": " ",".join([str(tp) for tp in to_print])

    def print_config(self, output_file):
        super(NeuralExtractor, self).print_config(output_file)

        output_file.write("Neural extractor config:\n")

        output_file.write("Neurons per item: " +
                          str(self.neurons_per_item) + "\n")
        output_file.write("Neurons per dimension: " +
                          str(self.neurons_per_dim) + "\n")

        output_file.write("Assoc params tau_rc: " +
                          str(self.assoc_params.tau_rc) + "\n")
        output_file.write("Assoc params tau_ref: " +
                          str(self.assoc_params.tau_ref) + "\n")
        output_file.write("Assoc params synapse: " +
                          str(self.assoc_params.synapse) + "\n")
        output_file.write("Assoc params radius: " +
                          str(self.assoc_params.radius) + "\n")
        output_file.write("Assoc params intercepts: " +
                          str(self.assoc_params.intercepts) + "\n")

        output_file.write("radius:" + str(self.radius) + "\n")
        output_file.write("synapse:" + str(self.synapse) + "\n")

        output_file.write("dimension:" + str(self.dimension) + "\n")
        output_file.write("num_items:" + str(self.num_items) + "\n")
        output_file.write("neurons_per_item:"
                          + str(self.neurons_per_item) + "\n")
        output_file.write("neurons_per_dim:"
                          + str(self.neurons_per_dim) + "\n")
        output_file.write("dt:" + str(self.dt) + "\n")
        output_file.write("timesteps:" + str(self.timesteps) + "\n")
        output_file.write("plot:" + str(self.plot) + "\n")
        output_file.write("show:" + str(self.show) + "\n")
        output_file.write("gpus:" + str(self.gpus) + "\n")
        output_file.write("ocl:" + str(self.ocl) + "\n")
        output_file.write("probe_keys:" + str(self.probe_keys) + "\n")
        output_file.write("identical:" + str(self.identical) + "\n")
        output_file.write("seed:" + str(self.seed) + "\n")

        output_file.write("threshold:" + str(self.threshold) + "\n")
        output_file.write("threshold_func:" + str(self.threshold_func) + "\n")
