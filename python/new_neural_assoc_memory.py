# Neural Extraction Algorithm
from assoc_memory import AssociativeMemory
from gpu_assoc_memory import AssociativeMemoryGPU

import string
import datetime
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mytools import hrr

from nengo.networks import CircularConvolution, EnsembleArray
from nengo.utils.distributions import Uniform
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


class NewNeuralAssociativeMemory(AssociativeMemory):

    _type = "Neural"

    def __init__(self, index_vectors, stored_vectors, threshold=0.3,
                 neurons_per_item=20, neurons_per_dim=50, timesteps=100,
                 dt=0.001, tau_rc=0.02, tau_ref=0.002, pstc=0.005,
                 output_dir=".", probe_keys=[], plot=False, ocl=[],
                 gpus=[], identical=False):
        """
        index_vectors and stored_vectors are both dictionaries mapping from
        tuples of the form (POS, number), indicating a synset, to numpy
        ndarrays containing the assigned vector
        """

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

        self.threshold = threshold
        self.transfer_func = lambda x: 1 if x > self.threshold else 0

        radius = 5.0 / np.sqrt(self.dim)

        model = nengo.Network(label="Extraction")

        self.A_input_vector = np.zeros(self.dim)
        self.B_input_vector = np.zeros(self.dim)

        synapse = pstc

        with model:
            self.A_input_func = make_func(self, "A_input_vector")
            self.B_input_func = make_func(self, "B_input_vector")

            A_input = nengo.Node(output=self.A_input_func, size_out=self.dim)
            B_input = nengo.Node(output=self.B_input_func, size_out=self.dim)

            A = EnsembleArray(nengo.LIF(neurons_per_dim), self.dim,
                              label="A", radius=radius)
            B = EnsembleArray(nengo.LIF(neurons_per_dim), self.dim,
                              label="B", radius=radius)
            D = EnsembleArray(nengo.LIF(neurons_per_dim), self.dim,
                              label="D", radius=radius)
            cconv = CircularConvolution(nengo.LIF(int(2 * neurons_per_dim)),
                                        self.dim, invert_b=True)

            nengo.Connection(A_input, A.input)
            nengo.Connection(B_input, B.input)

            nengo.Connection(A.output, cconv.A, synapse=synapse)
            nengo.Connection(B.output, cconv.B, synapse=synapse)
            nengo.Connection(cconv.output, D.input, synapse=synapse)

            output = EnsembleArray(nengo.LIF(neurons_per_dim),
                                   self.dim, label="output", radius=radius)

            assoc_encoders = np.ones((neurons_per_item, 1))
            intercepts = Uniform(0.0, 0.3)
            max_rates = Uniform(200.0, 200.0)
            #scale = 10.0
            scale = 1.0
            assoc_radius = 0.5

            if gpus:
                # Add a nengo.Node which calls out to a GPU library for
                # simulating the associative memory

                assoc_memory = \
                    AssociativeMemoryGPU(gpus, index_vectors, stored_vectors,
                                         threshold=threshold,
                                         neurons_per_item=neurons_per_item,
                                         intercepts=intercepts,
                                         max_rates=max_rates, tau_ref=tau_ref,
                                         tau_rc=tau_rc, radius=assoc_radius,
                                         do_print=False, identical=identical,
                                         probe_keys=probe_keys)

                print "done building gpu associative memory"

                def gpu_function(t, input_vector):
                    output_vector = assoc_memory.step(input_vector)
                    return output_vector

                assoc = nengo.Node(output=gpu_function,
                                   size_in=self.dim, size_out=self.dim)

                nengo.Connection(D.output, assoc, synapse=synapse)
                nengo.Connection(assoc, output.input, synapse=synapse)

                assoc_probes = OrderedDict()
                transfer_probes = OrderedDict()

                for k in probe_keys:
                    n = nengo.Node(output=assoc_memory.probe_func(k))
                    probe = nengo.Probe(n, synapse=0.02)

                    transfer_probes[k] = probe

            else:
                assoc_probes = OrderedDict()
                transfer_probes = OrderedDict()

                for k in self.index_vectors:
                    iv = self.index_vectors[k].reshape((1, self.dim))
                    sv = scale * self.stored_vectors[k].reshape((self.dim, 1))

                    assoc_label = "Associate: " + str(k)
                    assoc = nengo.Ensemble(nengo.LIF(neurons_per_item), 1,
                                           intercepts=intercepts,
                                           max_rates=max_rates,
                                           encoders=assoc_encoders,
                                           label=assoc_label,
                                           radius=assoc_radius)

                    nengo.Connection(D.output, assoc,
                                     transform=iv, synapse=synapse)

                    nengo.Connection(assoc, output.input, transform=sv,
                                     function=self.transfer_func,
                                     synapse=synapse)

                    if k in probe_keys:

                        assoc_probes[k] = \
                            nengo.Probe(assoc, 'decoded_output', synapse=0.02)

                        transfer_probes[k] = \
                            nengo.Probe(assoc, 'decoded_output', synapse=0.02,
                                        function=self.transfer_func)


            D_probe = nengo.Probe(D.output, 'output', synapse=0.02)
            output_probe = nengo.Probe(output.output, 'output', synapse=0.02)

            # end model

        self.model = model

        self.assoc_probes = assoc_probes
        self.transfer_probes = transfer_probes

        self.D_probe = D_probe
        self.output_probe = output_probe

        if ocl_imported and ocl is not None:
            pyopencl.get_platforms()
            sim_class = nengo_ocl.sim_ocl.Simulator
        else:
            if ocl:
                print "Failed to import nengo_ocl"

            sim_class = nengo.Simulator

        print "Building simulator"
        self.simulator = sim_class(model)

    def unbind_and_associate(self, item, query, *args, **kwargs):
        then = datetime.datetime.now()

        if len(self.tester.current_target_keys) > 0:
            # Print data about how difficult the current instance is

            correct_key = self.tester.current_target_keys[0]

            item_hrr = hrr.HRR(data=item)
            query_hrr = hrr.HRR(data=query)
            noisy_hrr = item_hrr.convolve(~query_hrr)

            correct_hrr = hrr.HRR(data=self.index_vectors[correct_key])
            sim = noisy_hrr.compare(correct_hrr)
            dot = np.dot(noisy_hrr.v, correct_hrr.v)
            print "Ideal similarity: ", sim
            print "Ideal dot: ", dot

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

        self.A_input_vector = item
        self.B_input_vector = query

        self.simulator.run(self.timesteps * self.dt)

        now = datetime.datetime.now()
        self.write_to_runtime_file(now - then)

        if self.plot:
            self.plot_cleanup_activities()

        vector = self.simulator.data[self.output_probe][-1, :]
        return [vector]

    def finish(self):
        pass

    def reset_nodes(self):
        pass

    def plot_cleanup_activities(self, item_indices=[], run_index=-1):
        """
        neither argument is currently used
        """
        correct_key = None
        if len(self.tester.current_target_keys) > 0:
            correct_key = self.tester.current_target_keys[0]

        sim = self.simulator
        t = sim.trange()

        max_val = 5.0 / np.sqrt(self.dim)

        gs = gridspec.GridSpec(9, 2)
        plt.figure(figsize=(10, 10))

        ax = plt.subplot(gs[0, 0])

        plt.plot(t, sim.data[self.D_probe], label='D')
        title = 'Before Association: Vector'
        ax.text(.01, 1.20, title, horizontalalignment='left',
                transform=ax.transAxes)
        plt.ylim((-max_val, max_val))

        ax = plt.subplot(gs[0, 1])
        plt.plot(t, sim.data[self.output_probe], label='Output')
        title = 'After Association: Vector'
        ax.text(.01, 1.20, title, horizontalalignment='left',
                transform=ax.transAxes)
        plt.ylim((-max_val, max_val))

        ax = plt.subplot(gs[1:3, :])

        for key, v in self.index_vectors.iteritems():
            input_sims = np.dot(sim.data[self.D_probe], v)
            label = str(key[1])
            if key == correct_key:
                plt.plot(t, input_sims, '--', label=label + '*')
            else:
                plt.plot(t, input_sims, label=label)

        title = 'Dot Products Before Association.\nTarget is dashed line.'
        ax.text(.01, 0.80, title, horizontalalignment='left',
                transform=ax.transAxes)
        # plt.legend(bbox_to_anchor=(-0.03, 0.5), loc='center right')
        if self.ideal_dot:
            ax.text(.01, 0.10, "Ideal dot: " + str(self.ideal_dot),
                    horizontalalignment='left', transform=ax.transAxes)
        if self.second_dot:
            ax.text(.99, 0.10, "Second dot: " + str(self.second_dot),
                    horizontalalignment='right', transform=ax.transAxes)

        plt.ylim((-1.0, 1.0))

        ax = plt.subplot(gs[3:5, :])
        for key, p in self.assoc_probes.iteritems():
            if key == correct_key:
                plt.plot(t, sim.data[p], '--')
            else:
                plt.plot(t, sim.data[p])

        title = 'Association Activation. \nTarget:' + str(correct_key)
        ax.text(.01, 0.80, title, horizontalalignment='left',
                transform=ax.transAxes)
        plt.ylim((-0.2, 1.0))

        ax = plt.subplot(gs[5:7, :])

        for key, p in self.transfer_probes.iteritems():
            if key == correct_key:
                plt.plot(t, sim.data[p], '--', label=str(key))
            else:
                plt.plot(t, sim.data[p], label=str(key))

        title = 'Assoc. Transfer Activation. \nTarget:' + str(correct_key)
        ax.text(.01, 0.80, title, horizontalalignment='left',
                transform=ax.transAxes)

        plt.ylim((-0.2, 1.0))

        if correct_key is not None:

            ax = plt.subplot(gs[7:9, :])

            correct_index_hrr = hrr.HRR(data=self.index_vectors[correct_key])
            correct_stored_hrr = hrr.HRR(data=self.stored_vectors[correct_key])

            input_sims = []
            output_sims = []
            probes = zip(sim.data[self.D_probe], sim.data[self.output_probe])

            for i, o in probes:
                input_sims.append(correct_index_hrr.compare(hrr.HRR(data=i)))
                output_sims.append(correct_stored_hrr.compare(hrr.HRR(data=o)))

            plt.plot(t, input_sims, label='Before')
            plt.plot(t, output_sims, label='After')
            title = 'Before/After Association: Cosine Similarity to Target'
            ax.text(.01, 0.90, title, horizontalalignment='left',
                    transform=ax.transAxes)
            plt.ylim((-1.0, 1.0))
            plt.legend(loc=4)
            plt.axhline(ls=':', c='k')

        plt.show()

        date_time_string = str(datetime.datetime.now()).split('.')[0]
        date_time_string = reduce(lambda y, z: string.replace(y, z, "_"),
                                  [date_time_string, ":", ".", " ", "-"])
        # plt.savefig('../graphs/neurons_'+date_time_string+".pdf")

    def write_to_runtime_file(self, delta):
        to_print = [self.dim, self.num_items,
                    self.neurons_per_item, self.neurons_per_dim,
                    self.timesteps, delta]
        print >> self.runtimes_file, ",".join([str(tp) for tp in to_print])

    def print_config(self, output_file):
        super(NewNeuralAssociativeMemory, self).print_config(output_file)

        output_file.write("Neurons per item: " +
                          str(self.neurons_per_item) + "\n")
        output_file.write("Neurons per dim: " +
                          str(self.neurons_per_dim) + "\n")
