from vector_operations import *

from corpora_manager import CorpusHandler
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
  parser.add_argument('--seed', default=1, type=int, help='Seed for the random number generator.')
  parser.add_argument('--save', default=False, type=bool, help='Whether to save the results of parsing the Wordnet files')
  parser.add_argument('-p', default=1.0, type=float, help='Specify the proportion of Wordnet synsets to use.')
  parser.add_argument('-c', default='config', help='Specifiy the name of the config file')
  parser.add_argument('-d', default=512, type=int, help='Specify the number of dimensions to use')
  parser.add_argument('-r', nargs='?', const=True, default=False, help='Supply this argument to collect the relation stats')
  parser.add_argument('-b', nargs='?', const=True, default=False, help='Supply this argument to use bidirectional relations')

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
def setup_corpus(input_dir, relation_symbols, seed, save, use_corpus, dim, proportion):
  if seed is not None:
    random.seed(seed)
    numpy.random.seed(seed)

  if use_corpus:
    if dim != -1:
      corpus = CorpusHandler(True, D=dim, input_dir = input_dir, relation_symbols=relation_symbols)
    else:
      corpus = CorpusHandler(True, input_dir = input_dir, relation_symbols = relation_symbols)

    corpus.parseWordnet()

    if proportion < 1.0:
      corpus.createCorpusSubset(proportion,1)

    print "Wordnet data parsed."
    corpus.formKnowledgeBase()
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

def gen_probes(num_words, relation_symbols):
  """Generate probes for an associative memory test.

  Specify links to be tested ahead of time, and put probes on the
  populations that will be activated.

  param int num_words: the number of populations to monitor
  param list relation_symbols : the usable relation symbols
  """

  probes = []
  words = random.sample(corpus.corpusDict, num_words)
  relations = []
  for word in words:
    testableLinks = [r for r in corpus.corpusDict[word] if r[0] in relation_symbols]

    index = random.sample(range(len(testableLinks)), 1)[0]
    link = testableLinks[index]

    probes.append(Probe(link[1], "identity"))
    probes.append(Probe(link[1], "transfer"))
    #relation_keys.append(link[1])
    relations.append(index)

  return (probes, words, relations)

def setup_relation_stats(largest_degree = 1000):
  relation_stats = dict(zip(range(largest_degree), [[[],[],[],0] for i in range(largest_degree)]))
  return relation_stats

def draw_associator_graph(associator):
  associator.drawCombinedGraph()


