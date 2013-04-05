import startup_utils
from vector_operations import *
import symbol_definitions
from assoc_memory_tester import AssociativeMemoryTester
from assoc_memory import AssociativeMemory
from neural_assoc_memory import NeuralAssociativeMemory

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

id_vecs = argvals.i
unitary_vecs = argvals.u

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

(corpusDict, idVectors, structuredVectors) = \
    startup_utils.setup_corpus(input_dir, relation_symbols, seed, save, dim, proportion, id_vecs=id_vecs, unitary_vecs=unitary_vecs)

num_words = 0
probes = []
words = []
relations = []

kwargs = {}

if num_words > 0:
  (probes, words, relations) = startup_utils.gen_probes(num_words, relation_symbols)

  if len(relations) == len(words) and len(words) > 0:
    kwargs["planned_relations"] = relation_indices
    kwargs["planned_words"] = words

if do_relation_stats:
  kwargs["relation_stats"] = startup_utils.setup_relation_stats()


#associator = NeuralAssociativeMemory(idVectors, structuredVectors, id_vecs, unitary_vecs, output_dir = output_dir, probes=probes, thresh=threshold)
associator = AssociativeMemory(idVectors, structuredVectors, threshold, id_vecs, unitary_vecs)

isA_symbols = symbol_definitions.isA_symbols()
sentence_symbols = symbol_definitions.sentence_role_symbols()

tester = AssociativeMemoryTester(corpusDict, idVectors, structuredVectors,
                    relation_symbols, associator, True, output_dir = output_dir, isA_symbols=isA_symbols, sentence_symbols=sentence_symbols, seed=seed)

data_display = startup_utils.draw_associator_graph

#short tests
if test == 'j':
  tester.runBootstrap_jump(num_runs, num_trials, dataFunc = data_display)
elif test == 'h':
  tester.runBootstrap_hierarchical(num_runs, num_trials, dataFunc = data_display)
elif test == 's':
  tester.runBootstrap_sentence(num_runs, num_trials, dataFunc = data_display)
else:
  pass

#For paper:
#tester.runBootstrap_jump(20, 100, dataFunc = data_display, **kwargs)
#tester.runBootstrap_hierarchical(20, 20, dataFunc = data_display, **kwargs)
#tester.runBootstrap_sentence(20, 30, dataFunc = data_display, **kwargs)

