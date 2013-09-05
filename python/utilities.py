__author__ = 'e2crawfo'

from corpora_management import CorpusHandler
from probe import Probe

import ConfigParser
import argparse
import random


def parse_args(print_args=False):
  parser = argparse.ArgumentParser(description='Test a cleanup memory.')

  parser.add_argument('test', nargs='*', help="Specify the test type (one of 'j', 'h' or 's'), the number of runs and the number of trials per run (e.g. python run_script.py j 10 100).")

  #parameters of the experiment
  parser.add_argument('-u', action='store_true', help='Supply this argument to use unitary vectors.')
  parser.add_argument('-i', action='store_true', help='Supply this argument to require that the semantic pointers be used as the index vectors.')
  parser.add_argument('-d', default=512, type=int, help='Specify the number of dimensions to use.')
  parser.add_argument('-p', default=1.0, type=float, help='Specify the proportion of Wordnet synsets to use. A float between 0 and 1.')
  parser.add_argument('--vector-seed', default=-1, type=int, help='Seed for the random number generator that creates the vectors.')
  parser.add_argument('--test-seed', default=-1, type=int, help='Seed for the random number generator that creates the tests.')

  #picking the type of cleanup memory
  parser.add_argument('-l', action='store_true', help='Supply this argument do cleanup using pure linear algebra rather than neurons. If neither -a nor -l is specified, cleanup is performed by a neural network.')
  parser.add_argument('-a', action='store_true', help='Supply this argument to use the neural cleanup algorithm, but without neurons. If neither -a nor -l is specified, cleanup is performed by a neural network.')

  #parameters for the neural network
  parser.add_argument('-t', default=0.3, type=float, help='Specify the cleanup threshold. A float between 0 and 1.')
  parser.add_argument('--pstc', default=0.02, type=float, help='Post-synaptic time constant. Controls the shape of the post-synaptic current.')
  parser.add_argument('--steps', default=100, type=int, help='Number of steps to run the neural model for.')

  #configuring gpus
  parser.add_argument('--gpus', default=1, type=int, help='Number of gpus to use to run the neural model.')
  parser.add_argument('--pick-devices', nargs='+', type=int, help='Specify the devices (gpus) to use. Specified as a list of integers' 
      ' (e.g. "python run_script.py j 10 100 --pick-devices 0 2 3" would use 3 devices, skipping the device with index 1).')

  parser.add_argument('-v', action='store_true', help='Supply this argument to print the data that is printed to the file')
  parser.add_argument('--numwords', default=0, type=int, help='Number of planned words. Only has an effect on jump tests.')

  #not used very often
  parser.add_argument('-b', action='store_true', help='Supply this argument to use bidirectional relations.')
  parser.add_argument('-r', action='store_true', help='Supply this argument to collect the relation stats.')
  parser.add_argument('-g', action='store_true', help='Supply this argument to display graphs (only works in the neural case).')
  parser.add_argument('-q', action='store_true', help='Supply this argument to do an accelerated (quick) neural run.')

  argvals = parser.parse_args()

  if print_args:
    print argvals

  return argvals

def create_outfile_suffix(neural, unitary, identity, bidirectional, algorithm):
  suff = "_"

  if neural: suff += "n"
  if unitary: suff += "u"
  if identity: suff += "i"
  if bidirectional: suff += "b"
  if algorithm: suff += "a"

  return suff

#Read config
def read_config(config_name="config"):
  configParser = ConfigParser.SafeConfigParser()
  configParser.readfp( open(config_name) )

  input_dir = configParser.get("Directories", "input_dir")
  output_dir = configParser.get("Directories", "output_dir")
  return (input_dir, output_dir)


#Setup corpus - just calls the functions in corpora_management.py which do all the heavy lifting.
def setup_corpus(input_dir, relation_symbols, dim, vf, seed, id_vecs=False, unitary_vecs = False, proportion = 1.0):

  corpus = CorpusHandler(D=dim, input_dir = input_dir, relation_symbols=relation_symbols, vf=vf, seed=seed)

  corpus.parseWordnet()

  if proportion < 1.0:
    corpus.createCorpusSubset(proportion)

  print "Wordnet data parsed."
  corpus.formKnowledgeBase(id_vecs, unitary_vecs)
  print "Knowledge base formed."

  corpusDict = corpus.corpusDict
  id_vectors = corpus.cleanupMemory
  semantic_pointers = corpus.knowledgeBase

  return (corpusDict, id_vectors, semantic_pointers)


#pick some key that i want to test...note this has to be the TARGET of something i decode
# then add a word/relation combo to the jump test that will decode to that key
# then add a probe on that key so we can see what happens!

def gen_probes(corpus_dict, num_words, relation_symbols, words=[], relations=[], seed=1):
  """Generate probes for an associative memory test.

  Specify links to be tested ahead of time, and put probes on the
  populations that will be activated.

  param int num_words: the number of populations to monitor
  param list relation_symbols : the usable relation symbols
  """

  probes = []

  rng = random.Random(seed)

  n = 0
  while n < num_words:
    if n < len(words):
      testableLinks = [r for r in corpus_dict[words[n]] if r[0] in relation_symbols]
      link = testableLinks[relations[n]]
    else:
      word = rng.sample(corpus_dict, 1)[0]
      testableLinks = [r for r in corpus_dict[word] if r[0] in relation_symbols]

      if len(testableLinks) > 0:
        index = rng.sample(range(len(testableLinks)), 1)[0]
        link = testableLinks[index]
      else:
        continue

      words.append(word)
      relations.append(index)

    probes.append(Probe(link[1], None, "identity"))
    probes.append(Probe(link[1], None, "transfer"))

    n+=1

  return (probes, words, relations)

def setup_relation_stats(largest_degree = 1000):
  relation_stats = dict(zip(range(largest_degree), [[[],[],[],0] for i in range(largest_degree)]))
  return relation_stats

def draw_associator_graph(associator):
  associator.drawCombinedGraph()

def print_header(output_file, string, char='*', width=15, left_newline=True):
  line = char * width
  string = line + " " + string + " " + line + "\n"

  if left_newline:
    string = "\n" + string

  output_file.write(string)

def print_footer(output_file, string, char='*', width=15):
  print_header(output_file, "End " + string, char=char, width=width, left_newline=False)
