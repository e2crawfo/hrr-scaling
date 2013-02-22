from SymbolDefinitions import *
from VectorOperations import *

from CorporaManagement import CorpusHandler
#from cleanupTester import CleanupTester
#from neuralCleanupTester import NeuralCleanupTester
from associativeMemoryTester import AssociativeMemoryTester
from neuralAssociativeMemory import NeuralAssociativeMemory
from associativeMemory import AssociativeMemory
from probe import Probe

import sys
import ConfigParser 
import random
import numpy

timesteps = 30
save = False
use_corpus = False
subset = False
proportion = 1.0

i = 1
dim = -1
seed = None
threads = 1
while len(sys.argv) > i:
  arg = sys.argv[i]

  if arg == "seed":
    #hrr.set_random_seed(int(sys.argv[i+1]))
    seed = int(sys.argv[i+1])
    i=i+2
  elif arg == "steps":
    timesteps = int(sys.argv[i+1])
    i=i+2
  elif arg == "save":
    save = True
    i=i+1
  elif arg == "dim":
    dim = int(sys.argv[i+1])
    i=i+2
  elif arg == "subset":
    subset = True
    proportion = float(sys.argv[i+1])
    i=i+2
  elif arg == "threads":
    threads = int(sys.argv[i+1])
    if threads < 1: threads = 1
    i=i+2
  else:
    i=i+1

use_corpus = True

configParser = ConfigParser.SafeConfigParser()
configParser.readfp( open("config") )

input_dir = configParser.get("Directories", "input_dir")
output_dir = configParser.get("Directories", "output_dir")

if seed is not None:
  random.seed(seed)
  numpy.random.seed(seed)

if use_corpus:
  if dim != -1:
    corpus = CorpusHandler(True, D=dim, input_dir = input_dir)
  else:
    corpus = CorpusHandler(True, input_dir = input_dir)

  corpus.parseWordnet()

  if subset:
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
  #corpusDict = corpus.corpusDict
  #idVectors = corpus.cleanupMemory
  #structuredVectors = corpus.knowledgeBase
  corpus = CorpusHandler(True, D=dim, input_dir = input_dir)
  corpus.corpusDict = None
  idVectors = None
  structuredVectors = None

import time

#pick some key that i want to test...note this has to be the TARGET of something i decode
# then add a word/relation combo to the jump test that will decode to that key
# then add a probe on that key so we can see what happens!

probes = []
words = []
relation_indices = []
relation_keys = []

num_words = 0
words = random.sample(corpus.corpusDict, num_words)
for word in words:
  testableLinks = [r for r in corpus.corpusDict[word] if r[0] in vocab_symbols]
  index = random.sample(range(len(testableLinks)), 1)[0]

  link = testableLinks[index]
  print word
  print link

  #changing the probes to accept keys rather then key indices
  probes.append(Probe(link[1], "identity"))
  probes.append(Probe(link[1], "transfer"))
  relation_keys.append(link[1])
  relation_indices.append(index)

kwargs = {}
if len(relation_indices) > 0:
  kwargs["relation_indices"] = relation_indices

if len(words) > 0:
  kwargs["planned_words"] = words

largest_degree = 10000
relation_stats = dict(zip(range(largest_degree), [[[],[],[],0] for i in range(largest_degree)]))
kwargs["relation_stats"] = relation_stats

#associator = AssociativeMemory(idVectors, structuredVectors, 0.3)
associator = NeuralAssociativeMemory(idVectors, structuredVectors, output_dir = output_dir, probes=probes)
tester = AssociativeMemoryTester(corpus.corpusDict, idVectors, structuredVectors, associator, True, output_dir = output_dir)
#tester.test_key_integrity()

def dataDisplay(associator):
  #associator.drawCombinedGraph()
  pass

tester.runBootstrap_jump(1, 5, dataFunc = dataDisplay, **kwargs)
#tester.runBootstrap_jump(1, num_words, dataFunc = dataDisplay, **kwargs)


