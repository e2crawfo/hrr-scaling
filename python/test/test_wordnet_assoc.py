from ..wordnet_assoc_memory_tester import WordnetAssociativeMemoryTester
from ..assoc_memory import AssociativeMemory
from ..neural_assoc_memory import NeuralAssociativeMemory

from .. import startup_utils
from .. import symbol_definitions

import unittest
from nose.plugins.attrib import attr

#for test selecting using Attrib nose plugin
#@attr(speed='fast')
class TestWordnetAssoc(unittest.TestCase):

  def createAssociator(self, id_vectors, semantic_pointers):
    pass

  def setUp(self):
    self.id = False
    self.uv = False
    self.ub = False
    self.threshold = 0.8
    self.assoc_threshold = 0.3
    seed = 10
    dim = 512
    prop = 0.1
    input_dir, output_dir = startup_utils.read_config()
    relation_symbols = symbol_definitions.uni_relation_symbols()

    isA_symbols = symbol_definitions.isA_symbols()
    sentence_symbols = symbol_definitions.sentence_role_symbols()

    (self.corpus_dict, self.id_vectors, self.semantic_pointers) = \
      startup_utils.setup_corpus(input_dir, relation_symbols, seed, dim, self.id, self.uv, prop) 

    self.createAssociator(self.id_vectors, self.semantic_pointers)
    self.tester = WordnetAssociativeMemoryTester(self.corpus_dict, self.id_vectors, self.semantic_pointers,
        relation_symbols, self.associator, seed, output_dir, isA_symbols, sentence_symbols, self.uv, True)

  def test_jump(self):
    self.tester.runBootstrap_jump(1, 1)

  def test_hierarchical(self):
    self.tester.runBootstrap_hierarchical(1, 1)

  def test_sentence(self):
    self.tester.runBootstrap_sentence(1, 1)

#for test selecting using Attrib nose plugin
@attr(speed='fast')
class TestWordnetAssocNonVector(TestWordnetAssoc):

  def createAssociator(self, id_vectors, semantic_pointers):
    self.associator = AssociativeMemory(id_vectors, semantic_pointers, self.id, self.uv, self.ub, self.assoc_threshold)

#for test selecting using Attrib nose plugin
@attr(speed='fast')
class TestWordnetAssocVector(TestWordnetAssoc):

  def createAssociator(self, id_vectors, semantic_pointers):
    self.associator = AssociativeMemory(id_vectors, semantic_pointers, self.id, self.uv, self.ub, self.assoc_threshold, return_vec=True)

#for test selecting using Attrib nose plugin
@attr(speed='slow')
class TestWordnetAssocNeural(TestWordnetAssoc):

  def createAssociator(self, id_vectors, semantic_pointers):
    self.associator = NeuralAssociativeMemory(id_vectors, semantic_pointers, self.id, self.uv, self.ub, self.assoc_threshold, print_output=False)
