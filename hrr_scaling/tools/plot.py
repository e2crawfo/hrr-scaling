import matplotlib.pyplot as plt
from nengo.utils.matplotlib import rasterplot
import numpy as np

spike_sorting = True
try:
    from scipy.cluster.vq import kmeans2
except:
    spike_sorting = False
    print "Couldn't import scipy.cluster. Can't cluster spikes."


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


def plot_tuning_curves(neuron_params, ensemble_params, filename, show=False):
    """
    Plot tuning curves a neural population.
    """

    import nengo
    from nengo.utils.ensemble import tuning_curves
    import matplotlib as mpl
    if show:
        mpl.use('Qt4Agg')
    else:
        mpl.use('Agg')

    import matplotlib.pyplot as plt

    model = nengo.Network("Dummy Network")
    with model:
        neurons = nengo.LIF(**neuron_params),

        ensemble = nengo.Ensemble(neurons=neurons, **ensemble_params)

    sim = nengo.Simulator(model)
    eval_points, activities = tuning_curves(ensemble, sim)

    for neuron in activities.T:
        plt.plot(eval_points.T[0], neuron)

    plt.title("Tuning curves")
    plt.ylabel("Firing Rate (spikes/s)")
    plt.xlabel("ex")
    plt.ylim((0, 400))

    # We want different eval points for display purposes than for
    # optimization purposes
    # eval_points = Uniform(-radius, radius).sample(n_eval_points)
    # eval_points.sort()
    # eval_points = eval_points.reshape((n_eval_points, 1))

    # # have to divide by radius on our own since tuning_curves skips that step
    # _, activities = tuning_curves(standard, sim, eval_points/radius)
    # for neuron in activities.T:
    #     plt.plot(eval_points, neuron)

    if filename:
        plt.savefig(filename)

    if show:
        plt.show()


def nengo_plot_helper(offset, t, data, label='', removex=True, yticks=None,
                      xlabel=None, spikes=False):
    if offset:
        plt.subplot(offset)

    if spikes:
        rasterplot(t, data, label=label)
    else:
        plt.plot(t, data, label=label)

    plt.ylabel(label)
    plt.xlim(min(t), max(t))
    if yticks is not None:
        plt.yticks(yticks)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if removex:
        frame = plt.gca()
        frame.axes.get_xaxis().set_ticklabels([])
    offset += 1

    ax = plt.gca()

    return ax, offset


def nengo_stack_plot(offset, t, sim, probe, func=None, label='',
                     removex=False, yticks=None, slice=None, suppl=None,
                     xlabel=None):
    """
    Accepts any nengo probe, extracts the data and creates an appropriate
    plot using matplotlib.

    offset is the subplot index to use. This is mostly deprecated, as it is
    much better to use matplotlib.gridspec for doing subplots.
    """
    if hasattr(probe, 'attr') and probe.attr == 'spikes':

        data, t = extract_probe_data(
            t, sim, probe, func, slice, suppl, spikes=True)
        return nengo_plot_helper(
            offset, t, data, label, removex, yticks,
            xlabel=xlabel, spikes=True)
    else:
        data, t = extract_probe_data(
            t, sim, probe, func, slice, suppl, spikes=False)
        return nengo_plot_helper(
            offset, t, data, label, removex, yticks,
            xlabel=xlabel, spikes=False)


def extract_probe_data(t, sim, probe, func=None, slice=None,
                       suppl=None, spikes=False):
    if spikes:
        data = sim.data(probe)
        if slice is not None:
            data = data[slice, :]
            t = t[:data.shape[0]]
        return data, t

    if isinstance(probe, list):
        data = []
        for i, p in enumerate(probe):
            if i == 0:
                data = sim.data(p)
            else:
                data = np.concatenate((data, sim.data(p)), axis=1)
    else:
        data = sim.data(probe)

    if data.ndim > 2:
        data = np.reshape(
            data, (data.shape[0], int(data.size / data.shape[0])))
    elif data.ndim == 1:
        data = data[:, np.newaxis]

    if slice is not None:
        data = data[slice, :]
        t = t[slice]

    newdata = apply_funcs(func, data, suppl)
    if newdata is not None:
        data = newdata

    if data.ndim > 1 and data.shape[1] > 600:
        mins = np.min(data, axis=1)[:, np.newaxis]
        maxs = np.max(data, axis=1)[:, np.newaxis]
        data = np.concatenate((mins, maxs), axis=1)

    return data, t


def apply_funcs(func, data, suppl=None):
    newdata = None

    if func is not None:
        if isinstance(func, list):
            func_list = func
        else:
            func_list = [func]

        for i, func in enumerate(func_list):
            if callable(func):
                if suppl:
                    fdata = np.array(
                        [func(d, s) for d, s
                         in zip(data, suppl)])[:, np.newaxis]
                else:
                    fdata = np.array([func(d) for d in data])[:, np.newaxis]

                if newdata is None:
                    newdata = fdata
                else:
                    newdata = np.concatenate((newdata, fdata), axis=1)
    return newdata


def spike_sorter(spikes, k=None, slice=None, binsize=None):
    """
    Sort neurons according to their spiking profile.

    If k is None, neurons sorted based on time of first spike.
    Otherwise, neurons sorted using kmeans clustering algorithm, with
    k clusters.

    Args:
    spikes -- an array of spikes as returned by a nengo 2.0 probe.

    k -- the number of clusters. Sorting instead of clustering if k is None.

    slice -- slice along the first dimension of the spike array.

    binsize -- if > 1, binsize contiguous time points are summed to create a
               single time point. Allows clustering based on lower frequency
               activity. Used only if k is not None.
    """

    if slice is not None:
        cluster_data = spikes[slice]
    else:
        cluster_data = spikes

    if binsize is not None:
        new_data = np.zeros((0, cluster_data.shape[1]))

        num_bins = int(np.ceil(np.true_divide(cluster_data.shape[0], binsize)))

        for i in range(num_bins):
            lo = i * binsize
            hi = min((i+1) * binsize, cluster_data.shape[0])
            binned_data = np.sum(cluster_data[lo:hi, :], axis=0)[np.newaxis, :]

            new_data = np.concatenate((new_data, binned_data), axis=0)

        cluster_data = new_data

    cluster_data = cluster_data.T

    if k is None or not spike_sorting:
        firsts = []
        for row in cluster_data:
            first = np.nonzero(row)[0]
            if first.size > 0:
                first = np.min(first)
            else:
                first = row.size
            firsts.append(first)

        firsts = np.array(firsts)
        labels = firsts[:, np.newaxis]
    else:
        # NB: There is a bug in scipy to the effect that if initial clusters
        # are not supplied (the second argument) then this call will fail.
        # Something to do with Cholesky decomposition, and a matrix not being
        # pos-def.
        centroids, labels = kmeans2(
            cluster_data, cluster_data[:k, :], iter=100)

    order = range(spikes.shape[1])
    order.sort(key=lambda l: labels[l])
    spikes = spikes[:, order]

    return spikes
