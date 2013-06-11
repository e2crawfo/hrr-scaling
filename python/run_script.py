
try:
  import matplotlib as mpl
  mpl.use('Agg')
  import matplotlib.pyplot as plt
  can_plot = True
except ImportError:
  can_plot = False

import startup_utils
from vector_operations import *
import symbol_definitions
from wordnet_assoc_memory_tester import WordnetAssociativeMemoryTester
from assoc_memory import AssociativeMemory
from neural_assoc_memory import NeuralAssociativeMemory
import random

argvals = startup_utils.parse_args(True)

steps = argvals.steps
seed = argvals.seed
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

verbose = argvals.v

id_vecs = argvals.i
unitary = argvals.u

outfile_suffix = startup_utils.create_outfile_suffix(neural, unitary, id_vecs, use_bi_relations, algorithm)

if seed == -1:
  seed = random.randrange(1000)
print seed

use_bi_relations = use_bi_relations and not id_vecs

if use_bi_relations:
  relation_symbols = symbol_definitions.bi_relation_symbols()
else:
  relation_symbols = symbol_definitions.uni_relation_symbols()

test = argvals.test[0] if len(argvals.test) > 0 else 'j'
num_runs = int(argvals.test[1]) if len(argvals.test) > 1 else 1
num_trials = int(argvals.test[2]) if len(argvals.test) > 2 else 1
print test, num_runs, num_trials

input_dir, output_dir = startup_utils.read_config(config_name)

(corpus_dict, id_vectors, semantic_pointers) = \
    startup_utils.setup_corpus(input_dir, relation_symbols, seed, dim, id_vecs, unitary, proportion)

#change these to use specific words/relations
num_words = 0
probes = []
#words = [('n', 2606384)]
#relations = [0]
words  = []
relations = []

if num_words > 0:
  (probes, words, relations) = startup_utils.gen_probes(corpus_dict, num_words, relation_symbols, words, relations)

  if not (len(relations) == len(words) and len(words) > 0):
    words = []
    relations = []

#if do_relation_stats:
#  kwargs["relation_stats"] = startup_utils.setup_relation_stats()


if neural:
  associator = NeuralAssociativeMemory(id_vectors, semantic_pointers, id_vecs, unitary, use_bi_relations, threshold, output_dir = output_dir, probes=probes, timesteps=steps, quick=quick)
else:
  associator = AssociativeMemory(id_vectors, semantic_pointers, id_vecs, unitary, use_bi_relations, threshold, algorithm)

isA_symbols = symbol_definitions.isA_symbols()
partOf_symbols = symbol_definitions.partOf_symbols()

sentence_symbols = symbol_definitions.sentence_role_symbols()

tester = WordnetAssociativeMemoryTester(corpus_dict, id_vectors, semantic_pointers,
                    relation_symbols, associator, seed, output_dir, isA_symbols, partOf_symbols, sentence_symbols, unitary, verbose, outfile_suffix)

if len(words) > 0:
  tester.set_jump_plan(words, relations)

if graph:
  data_display = startup_utils.draw_associator_graph
else:
  data_display = lambda x: x

#short tests
if test == 'j':
  tester.runBootstrap_jump(num_runs, num_trials, dataFunc = data_display)
elif test == 'h':
  tester.runBootstrap_hierarchical(num_runs, num_trials, dataFunc = data_display)
elif test == 'm':
  tester.runBootstrap_hierarchical(num_runs, num_trials, dataFunc = data_display, symbols=partOf_symbols)
elif test == 's':
  tester.runBootstrap_sentence(num_runs, num_trials, dataFunc = data_display)
elif test == 'c':
  tester.get_similarities()
else:
  pass


#For paper:
#tester.runBootstrap_jump(20, 100, dataFunc = data_display, **kwargs)
#tester.runBootstrap_hierarchical(20, 20, dataFunc = data_display, **kwargs)
#tester.runBootstrap_sentence(20, 30, dataFunc = data_display, **kwargs)

