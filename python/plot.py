show = True
import matplotlib as mpl
if show:
    mpl.use('Qt4Agg')
else:
    mpl.use('Agg')
import matplotlib.pyplot as plt

from corpora_management import VectorizedCorpus
from neural_extraction import NeuralExtraction

import nengo
from nengo.utils.distributions import Uniform
from nengo.utils.ensemble import tuning_curves
import numpy as np
import matplotlib.gridspec as gridspec


def plot_performance(y_vals, error_lo, error_hi, measure_labels,
                     condition_labels, filename='', show=False, colors=None,
                     **kwargs):
    """
    Plot bar chart with error bars. A measure is some specific dimension that
    we want to compare different conditions on. A condition is some
    configuration of parameters that we want to compare on different measures.

    E.g. two conditions could be 'model' and 'experiment', and the measures
    would be aspects of the performance that we want to compare them on.

    In the bar chart, each condition is assigned a color. Each measure gets a
    locations on the graph, where bars from each condition will be displayed.
    """

    import matplotlib as mpl
    if show:
        mpl.use('Qt4Agg')
    else:
        mpl.use('Agg')
    import matplotlib.pyplot as plt

    assert(len(y_vals[0]) ==
           len(error_lo[0]) ==
           len(error_hi[0]))

    assert(len(y_vals) ==
           len(error_lo) == len(error_hi))

    # this is the number of different bar locations
    # typically each location is a measure
    try:
        num_measures = len(y_vals[0])
    except IndexError:
        raise ValueError("y_vals must be a 2D array/list")

    # this is the number of bars at each location...typically a set of
    # conditions that we want to compare on multiple measures
    try:
        num_conditions = len(y_vals)
    except Exception:
        raise ValueError("y_vals must be a 2D array/list")

    if not measure_labels:
        measure_labels = [''] * num_measures

    if not condition_labels:
        condition_labels = [''] * num_conditions

    assert(len(measure_labels) == num_measures)
    assert(len(condition_labels) == num_conditions)

    # hatch = ['/', '|||||', '\\', 'xxx', '||', '--', '+', 'OO', '...', '**']
    if not colors:
        colors = [[0.33] * 3, [0.5] * 3, [0.66] * 3]

    cross_measurement_spacing = 0.4
    within_measurement_spacing = 0.0
    bar_width = 0.4

    error_lo = [[y - el for y, el in zip(yvl, ell)]
                for yvl, ell in zip(y_vals, error_lo)]
    error_hi = [[eh - y for y, eh in zip(yvl, ehl)]
                for yvl, ehl in zip(y_vals, error_hi)]

    # mpl.rc('font', family='serif', serif='Times New Roman')
    # mpl.rcParams['lines.linewidth'] = 0.3

    fig = plt.figure(figsize=(5, 2.55))

    mpl.rcParams.update({'font.size': 7})

    # plt.title("Model Performance")
    plt.ylabel("% Correct")

    bar_left_positions = [[] for b in range(num_conditions)]
    val_offset = 0
    middles = []
    for i in range(num_measures):
        val_offset += cross_measurement_spacing

        left_side = val_offset

        for j in range(num_conditions):
            if j > 0:
                val_offset += within_measurement_spacing
                val_offset += bar_width

            bar_left_positions[j].append(val_offset)

        val_offset += bar_width

        right_side = val_offset

        middles.append(float(left_side + right_side) / 2.0)

    zipped = zip(
        bar_left_positions, y_vals, colors,
        error_lo, error_hi, condition_labels)

    for blp, yv, cl, el, eh, lb in zipped:
        plt.bar(
            blp, yv, color=cl, width=bar_width, linewidth=0.2,
            yerr=[el, eh], ecolor="black", label=lb,
            error_kw={"linewidth": 0.5, "capsize": 2.0})

    plt.legend(
        loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 9},
        handlelength=.75, handletextpad=.5, shadow=False, frameon=False)

    ax = fig.axes[0]
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    handles, labels = ax.get_legend_handles_labels()

    ticks = middles

    plt.xticks(ticks, measure_labels)
    plt.ylim([0.0, 105.0])
    plt.xlim([0.0, val_offset + cross_measurement_spacing])
    plt.axhline(100.0, linestyle='--', color='black')

    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)

    if filename:
        plt.savefig(filename)

    if show:
        plt.show()


def plot_tuning_curves(filename, plot_decoding=False, show=False):
    """
    Plot tuning curves for an association population and for a standard
    subpopulation (of the neural extraction network).
    """
    import matplotlib as mpl

    if show:
        mpl.use('Qt4Agg')
    else:
        mpl.use('Agg')

    import matplotlib.pyplot as plt

    neurons_per_item = 20
    neurons_per_dim = 50
    intercepts_low = 0.29
    intercepts_range = 0.00108

    intercepts = Uniform(intercepts_low, intercepts_low + intercepts_range)

    tau_rc = 0.034
    tau_ref = 0.0026
    radius = 1.0
    encoders = np.ones((neurons_per_item, 1))

    threshold = 0.3
    threshold_func = lambda x: 1 if x > threshold else 0

    max_rates = Uniform(200, 350)

    model = nengo.Network("Associative Memory")
    with model:
        neurons = nengo.LIF(neurons_per_item,
                            tau_rc=tau_rc,
                            tau_ref=tau_ref)

        assoc = nengo.Ensemble(neurons, 1, intercepts=intercepts,
                               encoders=encoders, label="assoc",
                               radius=radius, max_rates=max_rates)

        n_eval_points = 750
        eval_points = np.random.normal(0, 0.06, (n_eval_points, 1))
        eval_points.T[0].sort()
        radius = 5.0 / np.sqrt(512)
        standard = nengo.Ensemble(nengo.LIF(neurons_per_dim), 1,
                                  eval_points=eval_points,
                                  radius=radius)

        if plot_decoding:
            dummy = nengo.Ensemble(nengo.LIF(1), 1)
            conn = nengo.Connection(assoc, dummy, function=threshold_func)
            dummy2 = nengo.Ensemble(nengo.LIF(1), 1)
            conn2 = nengo.Connection(standard, dummy2)

    sim = nengo.Simulator(model)

    if plot_decoding:
        gs = gridspec.GridSpec(3, 2)
    else:
        gs = gridspec.GridSpec(2, 2)

    plt.subplot(gs[0:2, 0])

    assoc_eval_points, assoc_activities = tuning_curves(assoc, sim)

    for neuron in assoc_activities.T:
        plt.plot(assoc_eval_points.T[0], neuron)
    plt.title("Association Subpopulation")
    plt.ylabel("Firing Rate (spikes/s)")
    plt.xlabel("ex")
    plt.ylim((0, 400))

    plt.subplot(gs[0:2, 1])

    # We want different eval points for display purposes than for
    # optimization purposes
    eval_points = Uniform(-radius, radius).sample(n_eval_points)
    eval_points.sort()
    eval_points = eval_points.reshape((n_eval_points, 1))

    # have to divide by radius on our own since tuning_curves skips that step
    _, activities = tuning_curves(standard, sim, eval_points/radius)
    for neuron in activities.T:
        plt.plot(eval_points, neuron)

    plt.title("Standard Subpopulation")
    plt.xlabel("ex")
    plt.xlim((-radius, radius))
    plt.ylim((0, 400))

    if plot_decoding:
        plt.subplot(gs[2, 0])
        decoders = sim.data[conn].decoders
        plt.plot(assoc_eval_points.T[0],
                 0.001 * np.dot(assoc_activities, decoders.T))
        plt.axhline(y=1.0, ls='--')

        plt.subplot(gs[2, 1])
        x, activities2 = tuning_curves(standard, sim, assoc_eval_points/radius)
        decoders = sim.data[conn2].decoders
        plt.plot(
            assoc_eval_points.T[0],
            0.001 * np.dot(activities2, decoders.T))
        plt.plot([-1.0, 1.0], [-1.0, 1.0], c='k', ls='--')
        plt.axvline(x=radius, c='k', ls='--')
        plt.axvline(x=-radius, c='k', ls='--')

    plt.tight_layout()

    if filename:
            plt.savefig(filename)
    if show:
            plt.show()


def hierarchical_simulation_data(dimension=128, num_synsets=50):
    input_dir = '../wordnetData/'
    unitary_relations = False
    proportion = .001

    output_dir = '../results'

    vc = VectorizedCorpus(
        dimension, input_dir, unitary_relations,
        proportion, num_synsets, create_namedict=True)

    num_links = 3

    chain = vc.find_chain(num_links, starting_keys=[]).next()
    names = [vc.key2name[c] for c in chain]
    print names

    id_vectors = vc.id_vectors
    semantic_pointers = vc.semantic_pointers

    extractor = NeuralExtraction(
        id_vectors, semantic_pointers, threshold=0.3,
        output_dir=output_dir, probe_keys=chain,
        timesteps=500, synapse=0.005,
        plot=False, show=False, ocl=False, gpus=[],
        identical=True)

    input_probe = extractor.input_probe
    D_probe = extractor.D_probe
    output_probe = extractor.output_probe

    sim = extractor.simulator

    query_vector = vc.relation_type_vectors['@']

    extractor.A_input_vector = semantic_pointers[chain[0]]
    extractor.B_input_vector = query_vector

    for i in range(num_links):
        extractor.A_input_vector = semantic_pointers[chain[i]]
        sim.run(0.1)

    t = sim.trange()
    chain_sp = [semantic_pointers[c][:, np.newaxis] for c in chain]
    chain_sp = np.concatenate(chain_sp, axis=1)
    input_similarities = np.dot(sim.data[input_probe], chain_sp)

    chain_id = [id_vectors[c][:, np.newaxis] for c in chain]
    chain_id = np.concatenate(chain_id, axis=1)
    before = np.dot(sim.data[D_probe], chain_id)
    after = np.dot(sim.data[output_probe], chain_sp)

    spike_probes = extractor.assoc_spike_probes
    spikes = [sim.data[spike_probes[key]] for key in chain]
    spikes = np.concatenate(spikes, axis=1)

    return names, t, input_similarities, before, after, spikes


def hierarchical_simulation_plot(names, t, input_similarities,
                                 before, after, spikes, show=True):
    mpl.rc('font', size=13)

    gs = gridspec.GridSpec(4, 1)

    linestyles = ['-'] * 4

    linewidth = 2.0

    # --------------------
    def do_plot(index, sims, title):
        ax = plt.subplot(gs[index, 0])

        lines = []

        for s, ls, n in zip(sims.T, linestyles, names):
            line = plt.plot(t, s, ls=ls, label=n, lw=linewidth)
            lines.extend(line)

        plt.ylabel(title)
        plt.ylim((-0.4, 1.1))

        return ax, lines

    # --------------------
    yticks = [0, 0.5, 1.0]
    title = 'Input'
    ax, lines = do_plot(0, input_similarities, title)
    plt.setp(ax, xticks=[])
    plt.yticks(yticks)

    plt.legend(lines, names, loc=4, fontsize='small')

    # --------------------
    title = 'Before Association'
    ax, lines = do_plot(1, before, title)
    plt.setp(ax, xticks=[])
    plt.yticks(yticks)

    # --------------------
    ax = plt.subplot(gs[2, 0])

    n_assoc_neurons = int(spikes.shape[1] / len(names))

    colors = [plt.getp(line, 'color') for line in lines]
    spike_colors = [colors[int(i / n_assoc_neurons)]
                    for i in range(spikes.shape[1])]

    nengo.utils.matplotlib.rasterplot(t, spikes, ax, colors=spike_colors)
    plt.setp(ax, xticks=[])
    plt.ylabel('Association Spikes')

    # --------------------
    title = 'After Association'
    ax, lines = do_plot(3, after, title)
    plt.xlabel('Time (s)')
    plt.yticks(yticks)

    plt.show()


if __name__ == "__main__":

    data = hierarchical_simulation_data()
    hierarchical_simulation_plot(*data, show=True)
