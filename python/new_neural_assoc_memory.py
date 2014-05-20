# Neural Extraction Algorithm
from assoc_memory import AssociativeMemory
from gpu_assoc_memory import AssociativeMemoryGPU

import string
import datetime
from collections import OrderedDict, namedtuple
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mytools import hrr

from nengo.networks import CircularConvolution, EnsembleArray
from nengo.utils.distributions import Uniform
import nengo.utils.numpy as npext
import nengo

ocl_imported = True
try:
    import pyopencl
    import nengo_ocl
except:
    ocl_imported = False


def make_func(cls, attr):
    def f(t):
        return getattr(cls, attr)
    return f

AssocParams = namedtuple('AssocParams',
                         ['tau_rc', 'tau_ref', 'synapse',
                          'radius', 'eval_points', 'intercepts'])


class NewNeuralAssociativeMemory(AssociativeMemory):

    _type = "Neural"

    def __init__(self, index_vectors, stored_vectors, threshold=0.3,
                 neurons_per_item=20, neurons_per_dim=50, timesteps=100,
                 dt=0.001, tau_rc=0.02, tau_ref=0.002, synapse=0.005,
                 output_dir=".", probe_keys=[], plot=False, show=False,
                 ocl=[], gpus=[], identical=False):
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

        self.unitary = False
        self.bidirectional = False
        self.identity = False

        self.ideal_dot = None
        self.second_dot = None

        self.return_vec = True

        self.output_dir = output_dir

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
        self.show = show
        self.gpus = gpus
        self.ocl = ocl
        self.probe_keys = probe_keys
        self.identical = identical
        self.seed = np.random.randint(npext.maxint)

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
        self.eval_points = np.random.normal(0, 0.06, (n_eval_points, 1))
        self.radius = 5.0 / np.sqrt(self.dim)
        self.solver = nengo.decoders.lstsq_L2
        self.synapse = synapse

        self.A_input_vector = np.zeros(self.dim)
        self.B_input_vector = np.zeros(self.dim)

        self.setup_simulator()

        now = datetime.datetime.now()
        self.write_to_runtime_file(now - then, "setup")

    def setup_simulator(self):
        self.model = nengo.Network(label="Extraction", seed=self.seed)

        print "Specifiying model"
        # The order is important here
        self.build_unbind(self.model)
        self.build_output(self.model)
        self.build_association(self.model)

        print "Building simulator"
        self.simulator = self.build_simulator(self.model)

    def build_unbind(self, model):
        A_input_func = make_func(self, "A_input_vector")
        B_input_func = make_func(self, "B_input_vector")

        neurons_per_dim = self.neurons_per_dim
        radius = self.radius
        eval_points = self.eval_points
        synapse = self.synapse
        dim = self.dim

        with model:
            A_input = nengo.Node(output=A_input_func, size_out=dim)
            B_input = nengo.Node(output=B_input_func, size_out=dim)

            A = EnsembleArray(nengo.LIF(neurons_per_dim), dim, label="A",
                              radius=radius, eval_points=eval_points)

            B = EnsembleArray(nengo.LIF(neurons_per_dim), dim, label="B",
                              radius=radius, eval_points=eval_points)

            cconv = CircularConvolution(nengo.LIF(int(2 * neurons_per_dim)),
                                        dim, invert_b=True, radius=0.2)

            D = EnsembleArray(nengo.LIF(neurons_per_dim), dim, label="D",
                              radius=radius, eval_points=eval_points)

            if self.solver != nengo.decoders.lstsq_L2nz:
                solver = self.solver

                attr_name = 'lstqs_L2'
                A_output = A.add_output(attr_name,
                                        function=None,
                                        decoder_solver=solver)

                B_output = B.add_output(attr_name,
                                        function=None,
                                        decoder_solver=solver)

                p = lambda x: x[0] * x[1]
                product = cconv.product.product
                prod_output = product.add_output(attr_name,
                                                 function=p,
                                                 decoder_solver=solver)

                cconv_output = nengo.Node(size_in=dim)

                nengo.Connection(prod_output, cconv_output,
                                 transform=cconv.transform_out)

                D_output = D.add_output(attr_name,
                                        function=None,
                                        decoder_solver=solver)
            else:
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

            self.D_output = D_output

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

        with model:
            if self.gpus:

                # Add a nengo.Node which calls out to a GPU library for
                # simulating the associative memory
                self.assoc_memory = \
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
                                         seed=self.seed)

                def gpu_function(t, input_vector):
                    output_vector = self.assoc_memory.step(input_vector)
                    return output_vector

                assoc = nengo.Node(output=gpu_function,
                                   size_in=self.dim, size_out=self.dim)

                nengo.Connection(self.D_output, assoc, synapse=synapse)
                nengo.Connection(assoc, self.output.input, synapse=synapse)

                for k in self.probe_keys:
                    n = nengo.Node(output=self.assoc_memory.probe_func(k))
                    probe = nengo.Probe(n, synapse=synapse)

                    threshold_probes[k] = probe

            else:
                encoders = np.ones((neurons_per_item, 1))

                # Cuts down on synapse computation
                assoc_in = nengo.Node(size_in=self.dim, label="Assoc Input")
                assoc_out = nengo.Node(size_in=self.dim, label="Assoc Output")
                nengo.Connection(self.D_output, assoc_in, synapse=synapse)
                nengo.Connection(assoc_out, self.output.input, synapse=synapse)

                for k in self.index_vectors:
                    iv = self.index_vectors[k].reshape((1, self.dim))
                    sv = self.stored_vectors[k].reshape((self.dim, 1))

                    label = "Associate: " + str(k)
                    neurons = nengo.LIF(self.neurons_per_item,
                                        tau_rc=tau_rc, tau_ref=tau_ref)

                    assoc = nengo.Ensemble(neurons, 1,
                                           intercepts=intercepts,
                                           encoders=encoders,
                                           label=label, seed=self.seed,
                                           radius=radius,)

                    nengo.Connection(assoc_in, assoc,
                                     transform=iv, synapse=None)

                    nengo.Connection(assoc, assoc_out, transform=sv,
                                     function=self.threshold_func,
                                     synapse=None, seed=self.seed)

                    if k in self.probe_keys:

                        assoc_probes[k] = \
                            nengo.Probe(assoc, 'decoded_output',
                                        synapse=synapse)

                        threshold_probes[k] = \
                            nengo.Probe(assoc, 'decoded_output',
                                        synapse=synapse,
                                        function=self.threshold_func,
                                        seed=self.seed)

        self.assoc_probes = assoc_probes
        self.threshold_probes = threshold_probes

    def build_output(self, model):
        with model:

            self.output = EnsembleArray(nengo.LIF(self.neurons_per_dim),
                                        self.dim, label="output",
                                        radius=self.radius)

            if self.solver != nengo.decoders.lstsq_L2nz:
                attr_name = 'lstqs_L2'
                output_output = \
                    self.output.add_output(attr_name,
                                           function=None,
                                           decoder_solver=self.solver)
            else:
                output_output = self.output.output

            self.output_probe = nengo.Probe(output_output, 'output',
                                            synapse=0.02)

    def unbind_and_associate(self, item, query, *args, **kwargs):
        then = datetime.datetime.now()

        if len(self.tester.current_target_keys) > 0:
            self.print_instance_difficulty(item, query)

        self.reset()

        self.A_input_vector = item
        self.B_input_vector = query

        self.simulator.run(self.timesteps * self.dt)
        self.data = self.simulator.data

        now = datetime.datetime.now()
        self.write_to_runtime_file(now - then, "unbind")

        if self.plot:
            self.plot_cleanup_activities()

        vector = self.simulator.data[self.output_probe][-1, :]
        return [vector]

    def reset(self):
        self.assoc_memory.reset()

        if hasattr(self.simulator, 'reset'):
            self.simulator.reset()
        else:
            warnings.warn("Non-GPU ensembles could not be reset")

    def build_simulator(self, model):
        if ocl_imported and self.ocl is not None:
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

    def plot_cleanup_activities(self, item_indices=[], run_index=-1):
        """
        neither argument is currently used
        """

        then = datetime.datetime.now()

        correct_key = None
        if len(self.tester.current_target_keys) > 0:
            correct_key = self.tester.current_target_keys[0]

        sim = self.simulator
        t = sim.trange()

        max_val = 5.0 / np.sqrt(self.dim)

        gs = gridspec.GridSpec(9, 2)
        plt.figure(figsize=(10, 10))

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
        plt.savefig('../graphs/extraction_'+date_time_string+".png")

        now = datetime.datetime.now()
        self.write_to_runtime_file(now - then, "plot")

        if self.show:
            plt.show()

    def print_instance_difficulty(self, item, query):
        if len(self.tester.current_target_keys) > 0:
            # Print data about how difficult the current instance is

            correct_key = self.tester.current_target_keys[0]

            item_hrr = hrr.HRR(data=item)
            query_hrr = hrr.HRR(data=query)
            noisy_hrr = item_hrr.convolve(~query_hrr)

            correct_hrr = hrr.HRR(data=self.index_vectors[correct_key])
            sim = noisy_hrr.compare(correct_hrr)
            dot = np.dot(noisy_hrr.v, correct_hrr.v)
            norm = np.linalg.norm(noisy_hrr.v)
            print "Ideal similarity: ", sim
            print "Ideal dot: ", dot
            print "Ideal norm: ", norm

            self.ideal_dot = dot

            hrrs = [(key, hrr.HRR(data=iv))
                    for key, iv in self.index_vectors.iteritems()
                    if key != correct_key]

            sims = [noisy_hrr.compare(h) for (k, h) in hrrs]
            dots = [np.dot(noisy_hrr.v, h.v) for (k, h) in hrrs]
            sim = max(sims)
            dot = max(dots)

            print "Similarity of closest incorrect index vector ", sim
            print "Dot product of closest incorrect index vector ", dot

            self.second_dot = dot

    def write_to_runtime_file(self, delta, label=''):
        to_print = [self.dim, self.num_items,
                    self.neurons_per_item, self.neurons_per_dim,
                    self.timesteps, "OCL: "+str(self.ocl),
                    "GPUS: "+str(self.gpus), delta]
        print >> self.runtimes_file, label, \
            ": " ",".join([str(tp) for tp in to_print])

    def print_config(self, output_file):
        super(NewNeuralAssociativeMemory, self).print_config(output_file)

        output_file.write("Neurons per item: " +
                          str(self.neurons_per_item) + "\n")
        output_file.write("Neurons per dim: " +
                          str(self.neurons_per_dim) + "\n")
