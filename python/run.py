try:
    import matplotlib as mpl
    mpl.use('Qt4Agg')
    can_plot = True
except ImportError:
    can_plot = False

import numpy as np
import utilities
import symbol_definitions
from wordnet_extraction_tester import WordnetExtractionTester
from extraction import Extraction
from neural_extraction import NeuralExtraction
from fast_neural_extraction import FastNeuralExtraction
import random

argvals = utilities.parse_args(True)

steps = argvals.steps
corpus_seed = argvals.corpus_seed
model_seed = argvals.model_seed
test_seed = argvals.test_seed
seed = argvals.seed
dim = argvals.d
proportion = argvals.p
threshold = argvals.t
do_relation_stats = argvals.r
use_bi_relations = argvals.b
linalg = argvals.l
algorithm = argvals.a and linalg
neural = not linalg
quick = argvals.q and neural
plot = argvals.plot and can_plot and neural
show = argvals.show and plot
num_words = argvals.numwords
pstc = argvals.pstc
noneg = argvals.noneg
shortsent = argvals.shortsent
num_synsets = argvals.num_synsets
ocl = argvals.ocl
probeall = argvals.probeall
fast = argvals.fast
identical = argvals.identical
verbose = argvals.v
use_pure_cleanup = argvals.i
unitary = argvals.u

gpus = argvals.gpus
if gpus is not None:
    gpus.sort()

ocl = argvals.ocl
if ocl is not None:
    ocl.sort()

outfile_suffix = \
    utilities.create_outfile_suffix(neural, unitary, use_pure_cleanup,
                                    use_bi_relations, algorithm)

if corpus_seed == -1:
    corpus_seed = random.randrange(1000)

if model_seed == -1:
    model_seed = random.randrange(1000)

if test_seed == -1:
    test_seed = random.randrange(1000)

if seed != -1:
    random.seed(seed)
    corpus_seed = random.randrange(1000)
    model_seed = random.randrange(1000)
    test_seed = random.randrange(1000)


np.random.seed(corpus_seed)
random.seed(corpus_seed)

use_bi_relations = use_bi_relations and not use_pure_cleanup

if use_bi_relations:
    relation_symbols = symbol_definitions.bi_relation_symbols()
else:
    relation_symbols = symbol_definitions.uni_relation_symbols()

input_dir, output_dir = utilities.read_config()

(corpus_dict, id_vectors, semantic_pointers, relation_type_vectors) = \
    utilities.setup_corpus(input_dir, relation_symbols, dim,
                           use_pure_cleanup, unitary, proportion, num_synsets)

# change these to use specific words/relations
probe_keys = []
words = []
relations = []

if num_words > 0:
    probe_keys, words, relations = \
        utilities.gen_probe_keys(corpus_dict, num_words,
                                 relation_symbols, words, relations)

if not (len(relations) == len(words) and len(words) > 0):
    words = []
    relations = []


test = argvals.test[0] if len(argvals.test) > 0 else 'j'

if test != 'f':
    num_runs = int(argvals.test[1]) if len(argvals.test) > 1 else 1
    num_trials = int(argvals.test[2]) if len(argvals.test) > 2 else 1
    print test, num_runs, num_trials
else:
    expression = argvals.test[1]

if probeall:
    probe_keys = id_vectors.keys()

np.random.seed(model_seed)
random.seed(model_seed)

# pick an associator
if neural:
    if fast and gpus:
        associator = \
            FastNeuralExtraction(id_vectors, semantic_pointers,
                                 threshold=threshold,
                                 output_dir=output_dir,
                                 probe_keys=probe_keys,
                                 timesteps=steps, synapse=pstc,
                                 plot=plot, show=show, ocl=ocl,
                                 gpus=gpus, identical=identical)
    else:
        associator = \
            NeuralExtraction(id_vectors, semantic_pointers,
                             threshold=threshold,
                             output_dir=output_dir,
                             probe_keys=probe_keys,
                             timesteps=steps, synapse=pstc,
                             plot=plot, show=show, ocl=ocl,
                             gpus=gpus, identical=identical)
else:
    associator = Extraction(id_vectors, semantic_pointers,
                            use_pure_cleanup, unitary,
                            use_bi_relations, threshold,
                            algorithm)

np.random.seed(test_seed)
random.seed(test_seed)

# get symbols for the different tests
h_test_symbols = symbol_definitions.hierarchical_test_symbols()
sentence_symbols = symbol_definitions.sentence_role_symbols()

tester = \
    WordnetExtractionTester(corpus_dict, id_vectors, semantic_pointers,
                            relation_type_vectors, associator,
                            test_seed, output_dir, h_test_symbols,
                            sentence_symbols, unitary, verbose,
                            outfile_suffix)

if len(words) > 0:
    tester.set_jump_plan(words, relations)

if test == 'j':
    tester.runBootstrap_jump(num_runs, num_trials)
elif test == 'h':
    tester.runBootstrap_hierarchical(num_runs, num_trials, do_neg=not noneg)
elif test == 's':
    tester.runBootstrap_sentence(num_runs, num_trials)
elif test == 'd':
    tester.runBootstrap_sentence(num_runs, num_trials,
                                 deep=True, short=shortsent)
elif test == 'f':
    tester.runBootstrap_single(1, 1, expression=expression)
elif test == 'c':
    tester.get_similarities()
