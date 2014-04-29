try:
  import matplotlib as mpl
  mpl.use('Qt4Agg')
  import matplotlib.pyplot as plt
  can_plot = True
except ImportError:
  can_plot = False

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
vector_seed = argvals.vector_seed
test_seed = argvals.test_seed
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
probeall = argvals.probeall
if pick_devices is not None: pick_devices = list(OrderedDict.fromkeys(pick_devices))
else: pick_devices = range(num_gpus)

verbose = argvals.v

use_pure_cleanup = argvals.i
unitary = argvals.u

outfile_suffix = utilities.create_outfile_suffix(neural, unitary, use_pure_cleanup, use_bi_relations, algorithm)

if vector_seed == -1:
  vector_seed = random.randrange(1000)

if test_seed == -1:
  test_seed = random.randrange(1000)

use_bi_relations = use_bi_relations and not use_pure_cleanup

if use_bi_relations:
  relation_symbols = symbol_definitions.bi_relation_symbols()
else:
  relation_symbols = symbol_definitions.uni_relation_symbols()


input_dir, output_dir = utilities.read_config()

vector_factory = VectorFactory(vector_seed)

(corpus_dict, id_vectors, semantic_pointers, relation_type_vectors) = \
    utilities.setup_corpus(input_dir, relation_symbols, dim, vector_factory, test_seed, use_pure_cleanup, unitary, proportion, num_synsets)


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
    #create a sentence
    expression = argvals.test[1]
    expression = expression.replace('!id', 'p0')

    num_ids = expression.count('id')
    expression = expression.replace('id', '%s')
    temp_names = ['id'+str(i) for i in range(num_ids)]
    expression = expression % tuple(temp_names)

    chosen_id_keys = random.sample(id_vectors, expression.count('id') + 1)
    chosen_id_vectors = [hrr.HRR(data=id_vectors[key]) for key in chosen_id_keys]
    target_key = chosen_id_keys[0]

    names_dict = dict(zip(['p0'] + temp_names, chosen_id_vectors))
    names_keys_dict = dict(zip(['p0'] + temp_names, chosen_id_keys))

    query_vectors = nf.find_query_vectors(expression, 'p0')
    query_expression = '*'.join(query_vectors)

    temp_names = expression.replace('*', '+').split('+')
    temp_names = [tn.strip() for tn in temp_names]
    unitary_names = [u for u in temp_names if u[-1:] == "u"]

    vocab = hrr.Vocabulary(dim, unitary=unitary_names)
    for n, v in names_dict.iteritems():
        vocab.add(n, v)

    print "expression:", expression
    print "query_expression:", query_expression
    print "unitary_names:", unitary_names
    print "target_key:", target_key
    print "name_keys_dict:", names_keys_dict

    test_vector = eval(expression, {}, vocab)
    test_vector.normalize()

    query_vector = eval(query_expression, {}, vocab)
    probe_indices.extend(chosen_id_keys)

    #probe_indices = id_vectors.keys()

if probeall:
    probe_indices = id_vectors.keys()

#pick an associator
if neural:
  if new:
      associator = NewNeuralAssociativeMemory(id_vectors, semantic_pointers, threshold,
                                              output_dir = output_dir,
                                              probe_indices=probe_indices,
                                              timesteps=steps, pstc=pstc, plot=plot)

  else:
      associator = NeuralAssociativeMemory(id_vectors, semantic_pointers, use_pure_cleanup, unitary,
                                           use_bi_relations, threshold, output_dir = output_dir,
                                           probe_indices=probe_indices, timesteps=steps, quick=quick,
                                           devices=pick_devices, pstc=pstc, plot=plot)
else:
  associator = AssociativeMemory(id_vectors, semantic_pointers, use_pure_cleanup, unitary,
                                 use_bi_relations, threshold, algorithm)

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
  tester.runBootstrap_single(1, 1, test_vector=test_vector.v, query_vector=query_vector.v,
                             target_key=target_key)
elif test == 'c':
  tester.get_similarities()
else:
  pass

