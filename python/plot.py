import argparse


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
           len(error_lo) ==
           len(error_hi))

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

    import nengo
    from nengo.utils.distributions import Uniform
    from nengo.utils.ensemble import tuning_curves
    import numpy as np
    import matplotlib.gridspec as gridspec
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw a bar graph")

    parser.add_argument(
        '-s', action='store_true',
        help='Supply this argument to save the graph')

    argvals = parser.parse_args()

    if argvals.s:
        filename = 'prgraph.pdf'
    else:
        filename = ''

    # condition (down) by measure(across)
    y_vals = [[99.25, 96.125, 86.7611, 55.24722],
              [100.0, 95.5, 99.6, 99.6]]
    error_lo = [[98.75, 94.375, 85.5888, 53.486],
                [100.0, 94.3, 99.3, 99.2]]
    error_hi = [[99.7, 97.75, 88.069, 56.986],
                [100.0, 96.9, 99.8, 99.7]]

    plot_performance(y_vals, error_lo, error_hi, filename=filename, show=True)
