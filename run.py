try:
    import matplotlib as mpl
    # mpl.use('Qt4Agg')
    mpl.use('Agg')
    can_plot = True
except ImportError:
    can_plot = False

from hrr_scaling.tools import parse_args
from hrr_scaling.main import run

argvals = parse_args(True)

# specify tests
num_runs = argvals.num_runs

jump_trials = argvals.jump
hier_trials = argvals.hier
sent_trials = argvals.sent
deep_trials = argvals.deep
expr = argvals.expr

unitary_roles = argvals.unitary_roles
short_sentence = argvals.shortsent
do_neg = not argvals.noneg

# seeds
corpus_seed = argvals.corpus_seed
extractor_seed = argvals.extractor_seed
test_seed = argvals.test_seed
seed = argvals.seed

# corpus args
dimension = argvals.d
num_synsets = argvals.num_synsets
proportion = argvals.p
unitary_relations = argvals.unitary_relations
id_vecs = not argvals.no_ids
sp_noise = argvals.sp_noise
normalize = not argvals.no_norm

# extractor args
abstract = argvals.abstract
synapse = argvals.synapse
timesteps = argvals.steps
threshold = argvals.t
probe_all = argvals.probe_all
identical = argvals.identical
fast = argvals.fast
plot = argvals.plot and can_plot and not abstract

gpus = argvals.gpus
ocl = argvals.ocl

name = argvals.name

run(num_runs, jump_trials, hier_trials, sent_trials, deep_trials, expr,
    unitary_roles, short_sentence, do_neg, corpus_seed,
    extractor_seed, test_seed, seed, dimension, num_synsets,
    proportion, unitary_relations, id_vecs, sp_noise, normalize,
    abstract, synapse, timesteps, threshold, probe_all, identical, fast,
    plot, gpus, ocl, name)
