from hrr_scaling.vector_operations import VectorFactory
from hrr_scaling.wordnet_assoc_memory_tester import WordnetAssociativeMemoryTester
from hrr_scaling.assoc_memory import AssociativeMemory

from hrr_scaling import tools
from hrr_scaling import symbol_definitions

import unittest


class TestWordnetAssocNonVector(unittest.TestCase):

    def setUp(self):
        self.id = False
        self.uv = False
        self.ub = False
        self.threshold = 0.8
        self.assoc_threshold = 0.3
        seed = 10
        dim = 512
        prop = 0.1
        input_dir, output_dir = tools.read_config()
        relation_symbols = symbol_definitions.uni_relation_symbols()
        vector_factory = VectorFactory(seed)

        isA_symbols = symbol_definitions.isA_symbols()
        sentence_symbols = symbol_definitions.sentence_role_symbols()

        self.corpus_dict, self.id_vectors, self.semantic_pointers = \
            tools.setup_corpus(
                input_dir, relation_symbols, dim, vector_factory,
                seed, self.id, self.uv, prop)

        self.createAssociator(self.id_vectors, self.semantic_pointers)
        self.tester = WordnetAssociativeMemoryTester(
            self.corpus_dict, self.id_vectors, self.semantic_pointers,
            relation_symbols, self.associator, seed, output_dir, isA_symbols,
            sentence_symbols, VectorFactory(), self.uv, True)

    def test_jump(self):
        self.tester.runBootstrap_jump(1, 1)

    def test_hierarchical(self):
        self.tester.runBootstrap_hierarchical(1, 1)

    def test_sentence(self):
        self.tester.runBootstrap_sentence(1, 1)

    def createAssociator(self, id_vectors, semantic_pointers):
        self.associator = AssociativeMemory(
            id_vectors, semantic_pointers, self.id, self.uv,
            self.ub, self.assoc_threshold)


class TestWordnetAssocVector(TestWordnetAssocNonVector):

    def createAssociator(self, id_vectors, semantic_pointers):
        self.associator = AssociativeMemory(
            id_vectors, semantic_pointers, self.id, self.uv, self.ub,
            self.assoc_threshold, return_vec=True)
