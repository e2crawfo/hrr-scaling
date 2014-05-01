try:
  import matplotlib as mpl
  mpl.use('Qt4Agg')
  import matplotlib.pyplot as plt
  can_plot = True
except ImportError:
  can_plot = False

import numpy as np
from mytools import nf, hrr
import utilities
from vector_operations import *
import symbol_definitions
from wordnet_assoc_memory_tester import WordnetAssociativeMemoryTester
from assoc_memory import AssociativeMemory
from neural_assoc_memory import NeuralAssociativeMemory
from new_neural_assoc_memory import NewNeuralAssociativeMemory
import random
from collections import OrderedDict

argvals = utilities.parse_args(True)

steps = argvals.steps
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
num_gpus = max(argvals.gpus, 0)
num_words = argvals.numwords
pick_devices = argvals.pick_devices
pstc = argvals.pstc
noneg = argvals.noneg
shortsent = argvals.shortsent
num_synsets = argvals.num_synsets
new = argvals.new
ocl = argvals.ocl
probeall = argvals.probeall
if pick_devices is not None: pick_devices = list(OrderedDict.fromkeys(pick_devices))
else: pick_devices = range(num_gpus)

verbose = argvals.v

use_pure_cleanup = argvals.i
unitary = argvals.u

outfile_suffix = utilities.create_outfile_suffix(neural, unitary, use_pure_cleanup, use_bi_relations, algorithm)

if model_seed == -1:
  model_seed = random.randrange(1000)

if test_seed == -1:
  test_seed = random.randrange(1000)

if seed != -1:
  random.seed(seed)
  model_seed = random.randrange(1000)
  test_seed = random.randrange(1000)

np.random.seed(model_seed)
random.seed(model_seed)

use_bi_relations = use_bi_relations and not use_pure_cleanup

if use_bi_relations:
  relation_symbols = symbol_definitions.bi_relation_symbols()
else:
  relation_symbols = symbol_definitions.uni_relation_symbols()

input_dir, output_dir = utilities.read_config()

vector_factory = VectorFactory()

(corpus_dict, id_vectors, semantic_pointers, relation_type_vectors) = \
    utilities.setup_corpus(input_dir, relation_symbols, dim, vector_factory,
                           use_pure_cleanup, unitary, proportion, num_synsets)

#change these to use specific words/relations
probe_indices = []
words = []
relations = []

if num_words > 0:
  probe_indices, words, relations = utilities.gen_probe_indices(corpus_dict, num_words, relation_symbols, words, relations)

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
    probe_indices = id_vectors.keys()

#pick an associator
if neural:
  if new:
      associator = NewNeuralAssociativeMemory(id_vectors, semantic_pointers, threshold,
                                              output_dir = output_dir,
                                              probe_indices=probe_indices,
                                              timesteps=steps, pstc=pstc, plot=plot,
                                              ocl = ocl)

  else:
      associator = NeuralAssociativeMemory(id_vectors, semantic_pointers, use_pure_cleanup, unitary,
                                           use_bi_relations, threshold, output_dir = output_dir,
                                           probe_indices=probe_indices, timesteps=steps, quick=quick,
                                           devices=pick_devices, pstc=pstc, plot=plot)
else:
  associator = AssociativeMemory(id_vectors, semantic_pointers, use_pure_cleanup, unitary,
                                 use_bi_relations, threshold, algorithm)


np.random.seed(test_seed)
random.seed(test_seed)

#get symbols for the different tests
h_test_symbols = symbol_definitions.hierarchical_test_symbols()
sentence_symbols = symbol_definitions.sentence_role_symbols()



tester = WordnetAssociativeMemoryTester(corpus_dict, id_vectors, semantic_pointers,
                    relation_type_vectors, associator, test_seed, output_dir, h_test_symbols,
                    sentence_symbols, vector_factory, unitary, verbose, outfile_suffix)

if len(words) > 0:
  tester.set_jump_plan(words, relations)

if test == 'j':
  tester.runBootstrap_jump(num_runs, num_trials)
elif test == 'h':
  tester.runBootstrap_hierarchical(num_runs, num_trials, do_neg=not noneg)
elif test == 's':
  tester.runBootstrap_sentence(num_runs, num_trials)
elif test == 'd':
  tester.runBootstrap_sentence(num_runs, num_trials, deep=True, short=shortsent)
elif test == 'f': #f as in free form
  tester.runBootstrap_single(1, 1, expression=expression)
elif test == 'c':
  tester.get_similarities()
else:
  pass

