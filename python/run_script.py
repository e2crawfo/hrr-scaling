try:
  import matplotlib as mpl
  mpl.use('Agg')
  import matplotlib.pyplot as plt
  can_plot = True
except ImportError:
  can_plot = False

import utilities
from vector_operations import *
import symbol_definitions
from wordnet_assoc_memory_tester import WordnetAssociativeMemoryTester
from assoc_memory import AssociativeMemory
from neural_assoc_memory import NeuralAssociativeMemory
import random
from collections import OrderedDict

argvals = utilities.parse_args(True)

steps = argvals.steps
vector_seed = argvals.vector_seed
test_seed = argvals.test_seed
save = argvals.save
dim = argvals.d
proportion = argvals.p
threshold = argvals.t
config_name = argvals.c
do_relation_stats = argvals.r
use_bi_relations = argvals.b
algorithm = argvals.a
neural = argvals.n and not algorithm
quick = argvals.q and neural
graph = argvals.g and can_plot
num_gpus = max(argvals.gpus, 0)
num_words = argvals.numwords
pick_devices = argvals.pick_devices
if pick_devices is not None: pick_devices = list(OrderedDict.fromkeys(pick_devices))
else: pick_devices = range(num_gpus)

verbose = argvals.v

id_vecs = argvals.i
unitary = argvals.u

outfile_suffix = utilities.create_outfile_suffix(neural, unitary, id_vecs, use_bi_relations, algorithm)

if vector_seed == -1:
  vector_seed = random.randrange(1000)

if test_seed == -1:
  test_seed = random.randrange(1000)

use_bi_relations = use_bi_relations and not id_vecs

if use_bi_relations:
  relation_symbols = symbol_definitions.bi_relation_symbols()
else:
  relation_symbols = symbol_definitions.uni_relation_symbols()

test = argvals.test[0] if len(argvals.test) > 0 else 'j'
num_runs = int(argvals.test[1]) if len(argvals.test) > 1 else 1
num_trials = int(argvals.test[2]) if len(argvals.test) > 2 else 1
print test, num_runs, num_trials

input_dir, output_dir = utilities.read_config(config_name)

vector_factory = VectorFactory(vector_seed)

(corpus_dict, id_vectors, semantic_pointers) = \
    utilities.setup_corpus(input_dir, relation_symbols, dim, vector_factory, test_seed, id_vecs, unitary, proportion)

#change these to use specific words/relations
probes = []
words = []
relations = []

if num_words > 0:
  (probes, words, relations) = utilities.gen_probes(corpus_dict, num_words, relation_symbols, words, relations)

  if not (len(relations) == len(words) and len(words) > 0):
    words = []
    relations = []

#if do_relation_stats:
#  kwargs["relation_stats"] = utilities.setup_relation_stats()


if neural:
  associator = NeuralAssociativeMemory(id_vectors, semantic_pointers, id_vecs, unitary, use_bi_relations, threshold,
                                       output_dir = output_dir, probes=probes, timesteps=steps, quick=quick, num_gpus=num_gpus)
=======
                                       output_dir = output_dir, probes=probes, timesteps=steps, quick=quick, devices=pick_devices)
>>>>>>> Added --pick-devices command line arg
else:
  associator = AssociativeMemory(id_vectors, semantic_pointers, id_vecs, unitary, use_bi_relations, threshold, algorithm)

h_test_symbols = symbol_definitions.hierarchical_test_symbols()

sentence_symbols = symbol_definitions.sentence_role_symbols()

tester = WordnetAssociativeMemoryTester(corpus_dict, id_vectors, semantic_pointers,
                    relation_symbols, associator, test_seed, output_dir, h_test_symbols, 
                    sentence_symbols, vector_factory, unitary, verbose, outfile_suffix)

if len(words) > 0:
  tester.set_jump_plan(words, relations)

if graph:
  data_display = utilities.draw_associator_graph
else:
  data_display = lambda x: x

if test == 'j':
  tester.runBootstrap_jump(num_runs, num_trials, dataFunc = data_display)
elif test == 'h':
  tester.runBootstrap_hierarchical(num_runs, num_trials, dataFunc = data_display)
elif test == 's':
  tester.runBootstrap_sentence(num_runs, num_trials, dataFunc = data_display)
elif test == 'c':
  tester.get_similarities()
else:
  pass

