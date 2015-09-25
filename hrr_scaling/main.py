from hrr_scaling.tools import read_config
from hrr_scaling.tools.file_helpers import make_filename, make_sym_link
from hrr_scaling.extraction_tester import ExtractionTester
from hrr_scaling.corpora_management import VectorizedCorpus

from hrr_scaling.extractor import Extractor
from hrr_scaling.neural_extractor import NeuralExtractor
from hrr_scaling.fast_neural_extractor import FastNeuralExtractor

from hrr_scaling.wordnet_tests import ExpressionTest, JumpTest
from hrr_scaling.wordnet_tests import HierarchicalTest
from hrr_scaling.wordnet_tests import SentenceTest

import random
import os

import numpy as np


def run(num_runs, jump_trials, hier_trials, sent_trials, deep_trials, expr,
        unitary_roles, short_sentence, do_neg, corpus_seed,
        extractor_seed, test_seed, seed, dimension, num_synsets,
        proportion, unitary_relations, id_vecs, sp_noise, normalize,
        abstract, synapse, timesteps, threshold, probe_all, identical,
        fast, plot, gpus, name):

    input_dir, output_dir = read_config()

    if not name:
        name = "hrr_scaling_results"

    directory = make_filename(name, output_dir)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    make_sym_link(
        os.path.split(directory)[1], os.path.join(output_dir, 'latest'))

    neural = not abstract

    if gpus is not None:
        gpus.sort()

    def make_corpus_factory(dimension, input_dir, unitary_relations, id_vecs,
                            proportion, num_synsets, sp_noise, normalize):

        def make_corpus():
            corpus = VectorizedCorpus(
                dimension=dimension, input_dir=input_dir,
                unitary_relations=unitary_relations, id_vecs=id_vecs,
                sp_noise=sp_noise, normalize=normalize,
                num_synsets=num_synsets, proportion=proportion)

            return corpus

        return make_corpus

    def make_extractor_factory(neural, fast, gpus, plot, threshold,
                               timesteps, synapse):

        # pick an extraction algorithm
        def make_extractor(id_vectors, semantic_pointers,
                           probe_keys, output_dir):
            if neural:
                if fast and gpus:
                    extractor = FastNeuralExtractor(
                        id_vectors, semantic_pointers,
                        threshold=threshold, probe_keys=probe_keys,
                        timesteps=timesteps, synapse=synapse,
                        plot=plot, gpus=gpus, identical=identical,
                        output_dir=output_dir)
                else:
                    extractor = NeuralExtractor(
                        id_vectors, semantic_pointers, threshold=threshold,
                        probe_keys=probe_keys, timesteps=timesteps,
                        synapse=synapse, plot=plot, gpus=gpus,
                        identical=identical, output_dir=output_dir)
            else:
                extractor = Extractor(
                    id_vectors, semantic_pointers,
                    threshold, output_dir=output_dir)

            return extractor

        return make_extractor

    corpus_factory = make_corpus_factory(
        dimension, input_dir, unitary_relations, id_vecs,
        proportion, num_synsets, sp_noise, normalize)

    extractor_factory = make_extractor_factory(
        neural, fast, gpus, plot, threshold, timesteps, synapse)

    if seed != -1:
        random.seed(seed)
        corpus_seed = random.randrange(1000)
        extractor_seed = random.randrange(1000)
        test_seed = random.randrange(1000)
    else:
        if corpus_seed == -1:
            corpus_seed = random.randrange(1000)

        if extractor_seed == -1:
            extractor_seed = random.randrange(1000)

        if test_seed == -1:
            test_seed = random.randrange(1000)

    np.random.seed(test_seed)
    random.seed(test_seed)

    test_runner = ExtractionTester(
        corpus_factory, extractor_factory, corpus_seed,
        extractor_seed, test_seed, probe_all, directory)

    if jump_trials > 0:
        test = JumpTest(jump_trials)
        test_runner.add_test(test)

    if hier_trials > 0:
        test = HierarchicalTest(hier_trials, do_neg=do_neg)
        test_runner.add_test(test)

    if sent_trials > 0:
        test = SentenceTest(
            sent_trials, deep=False,
            unitary=unitary_roles, short=short_sentence)

        test_runner.add_test(test)

    if deep_trials > 0:
        test = SentenceTest(
            deep_trials, deep=True,
            unitary=unitary_roles, short=short_sentence)

        test_runner.add_test(test)

    if expr:
        expr_trials = expr[0]
        expr = expr[1]

        test = ExpressionTest(expr_trials, expression=expr)
        test_runner.add_test(test)

    test_runner.run_bootstrap(num_runs)
