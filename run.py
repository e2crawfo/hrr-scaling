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
n_runs = argvals.runs

jump_trials = argvals.jump
hier_trials = argvals.hier
sent_trials = argvals.sent
deep_trials = argvals.deep
expr = argvals.expr

unitary_roles = argvals.unitary_roles
short_sentence = argvals.short_sent
do_neg = not argvals.no_neg

# seeds
corpus_seed = argvals.corpus_seed
extractor_seed = argvals.extractor_seed
test_seed = argvals.test_seed
seed = argvals.seed

# corpus args
dimension = argvals.d
n_synsets = argvals.num_synsets
proportion = argvals.p
unitary_relations = argvals.unitary_relations
id_vecs = not argvals.no_ids
sp_noise = argvals.sp_noise
normalize = not argvals.no_norm

# extractor args
abstract = argvals.abstract
synapse = argvals.synapse
timesteps = argvals.steps
threshold = argvals.threshold
probe_all = argvals.probe_all
fast = argvals.fast
plot = not argvals.no_plot and can_plot and not abstract

gpu = argvals.gpu
use_gpus = argvals.use_gpus

if gpu and not use_gpus:
    use_gpus = [0]

identical = bool(use_gpus)

name = argvals.name

run(n_runs, jump_trials, hier_trials, sent_trials, deep_trials, expr,
    unitary_roles, short_sentence, do_neg, corpus_seed,
    extractor_seed, test_seed, seed, dimension, n_synsets,
    proportion, unitary_relations, id_vecs, sp_noise, normalize,
    abstract, synapse, timesteps, threshold, probe_all, identical, fast,
    plot, use_gpus, name)
