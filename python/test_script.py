import startup_utils
from VectorOperations import *
from associativeMemoryTester import AssociativeMemoryTester
from neuralAssociativeMemory import NeuralAssociativeMemory

argvals = startup_utils.parse_args(True)

steps = argvals.steps
seed = argvals.seed
save = argvals.save
use_corpus = argvals.u
dim = argvals.d
proportion = argvals.p
config_name = argvals.c
do_relation_stats = argvals.r

input_dir, output_dir = startup_utils.read_config(config_name)

(corpusDict, idVectors, structuredVectors) = \
    startup_utils.setup_corpus(input_dir, seed, save, use_corpus, dim, proportion)

num_words = 0
probes = []
words = []
relations = []

kwargs = {}

if num_words > 0:
  (probes, words, relations) = startup_utils.gen_probes(num_words, vocab_symbols)

  if len(relations) == len(words) and len(words) > 0:
    kwargs["planned_relations"] = relation_indices
    kwargs["planned_words"] = words

if do_relation_stats:
  kwargs["relation_stats"] = startup_utils.setup_relation_stats()


associator = NeuralAssociativeMemory(idVectors, structuredVectors, 
                                     output_dir = output_dir, probes=probes)

tester = AssociativeMemoryTester(corpusDict, idVectors, 
                    structuredVectors, associator, True, output_dir = output_dir)

data_display = startup_utils.draw_associator_graph

#short tests
#tester.runBootstrap_jump(1, 1, dataFunc = data_display, **kwargs)
#tester.runBootstrap_hierarchical(1, 1, dataFunc = data_display, **kwargs)
tester.runBootstrap_sentence(1, 1, dataFunc = data_display, **kwargs)

#For paper:
#tester.runBootstrap_jump(20, 100, dataFunc = data_display, **kwargs)
#tester.runBootstrap_hierarchical(20, 20, dataFunc = data_display, **kwargs)
#tester.runBootstrap_sentence(20, 30, dataFunc = data_display, **kwargs)

