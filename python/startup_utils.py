from vector_operations import *

from corpora_management import CorpusHandler
from probe import Probe

import sys
import ConfigParser
import argparse
import random
import numpy
import time
import argparse

#Parse args
def parse_args(print_args=False):
  parser = argparse.ArgumentParser(description='Test an associative memory.')
  parser.add_argument('--steps', default=100, type=int, help='Number of steps to run the neural model for.')
  parser.add_argument('--seed', default=-1, type=int, help='Seed for the random number generator.')
  parser.add_argument('--save', default=False, type=bool, help='Whether to save the results of parsing the Wordnet files')
  parser.add_argument('-p', default=1.0, type=float, help='Specify the proportion of Wordnet synsets to use.')
  parser.add_argument('-t', default=0.3, type=float, help='Specify the cleanup threshold.')
  parser.add_argument('-c', default='config', help='Specifiy the name of the config file')
  parser.add_argument('-d', default=512, type=int, help='Specify the number of dimensions to use')
  parser.add_argument('-r', action='store_true', help='Supply this argument to collect the relation stats')
  parser.add_argument('-b', action='store_true', help='Supply this argument to use bidirectional relations')
  parser.add_argument('-u', action='store_true', help='Supply this argument to use unitary vectors')
  parser.add_argument('-i', action='store_true', help='Supply this argument to use identity vectors')
  parser.add_argument('-n', action='store_true', help='Supply this argument to use a neural cleanup')
  parser.add_argument('-g', action='store_true', help='Supply this argument to display graphs (only makes a difference if -n is also supplied)')
  parser.add_argument('-v', action='store_true', help='Supply this argument to print the data the is printed to the file')

  parser.add_argument('test', nargs='*', help='Specify the test type, the number of runs and the number of trials')


  argvals = parser.parse_args()

  if print_args:
    print argvals

  return argvals


#Read config
def read_config(config_name="config"):
  configParser = ConfigParser.SafeConfigParser()
  configParser.readfp( open(config_name) )

  input_dir = configParser.get("Directories", "input_dir")
  output_dir = configParser.get("Directories", "output_dir")
  return (input_dir, output_dir)


#Setup corpus
def setup_corpus(input_dir, relation_symbols, seed, save, dim, proportion, id_vecs=False, unitary_vecs = False, use_corpus=True):
  if use_corpus:
    if dim != -1:
      corpus = CorpusHandler(True, D=dim, input_dir = input_dir, relation_symbols=relation_symbols, seed=seed+1)
    else:
      corpus = CorpusHandler(True, input_dir = input_dir, relation_symbols = relation_symbols, seed=seed+1)

    corpus.parseWordnet()

    if proportion < 1.0:
      corpus.createCorpusSubset(proportion)

    print "Wordnet data parsed."
    corpus.formKnowledgeBase(id_vecs, unitary_vecs)
    print "Knowledge base formed."

    corpusDict = corpus.corpusDict
    idVectors = corpus.cleanupMemory
    structuredVectors = corpus.knowledgeBase

    if save:
      print "Saving..."
      corpus.saveCorpusDict(input_dir+'/cd1.data')
      print "corpus saved"
      corpus.saveCleanup(input_dir+'/clean1.data')
      print "cleanup saved"
      corpus.saveKnowledgeBase(input_dir+'/kb1.data')
      print "knowledge base saved"
  else:
    corpus = CorpusHandler(True, D=dim, input_dir = input_dir)
    corpus.corpusDict = None
    idVectors = None
    structuredVectors = None

  return (corpusDict, idVectors, structuredVectors)


#pick some key that i want to test...note this has to be the TARGET of something i decode
# then add a word/relation combo to the jump test that will decode to that key
# then add a probe on that key so we can see what happens!

def gen_probes(corpus_dict, num_words, relation_symbols):
  """Generate probes for an associative memory test.

  Specify links to be tested ahead of time, and put probes on the
  populations that will be activated.

  param int num_words: the number of populations to monitor
  param list relation_symbols : the usable relation symbols
  """

  probes = []
  words = []
  relations = []

  n = 0
  while n < num_words: 
    word = random.sample(corpus_dict, 1)[0]
    testableLinks = [r for r in corpus_dict[word] if r[0] in relation_symbols]

    if len(testableLinks) > 0:
      index = random.sample(range(len(testableLinks)), 1)[0]
      link = testableLinks[index]

      words.append(word)
      probes.append(Probe(link[1], "identity"))
      probes.append(Probe(link[1], "transfer"))
      relations.append(index)
      n+=1 

  return (probes, words, relations)

def setup_relation_stats(largest_degree = 1000):
  relation_stats = dict(zip(range(largest_degree), [[[],[],[],0] for i in range(largest_degree)]))
  return relation_stats

def draw_associator_graph(associator):
  associator.drawCombinedGraph()


