from hrr_scaling.tools import read_config
from hrr_scaling.extraction_tester import ExtractionTester
from hrr_scaling.corpora_management import VectorizedCorpus

from hrr_scaling.extraction import Extraction
from hrr_scaling.neural_extraction import NeuralExtraction
from hrr_scaling.fast_neural_extraction import FastNeuralExtraction

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
        abstract, synapse, timesteps, threshold, probeall, identical,
        fast, plot, show, gpus, ocl, name):

    input_dir, output_dir = read_config()

    output_file = os.path.join(output_dir, name)

    neural = not abstract

    if gpus is not None:
        gpus.sort()

    if ocl is not None:
        ocl.sort()

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

    def make_extractor_factory(neural, fast, gpus, ocl, plot, show, threshold,
                               timesteps, synapse):

        # pick an extraction algorithm
        def make_extractor(id_vectors, semantic_pointers,
                           probe_keys, output_dir):
            if neural:
                if fast and gpus:
                    extractor = FastNeuralExtraction(
                        id_vectors, semantic_pointers,
                        threshold=threshold, probe_keys=probe_keys,
                        timesteps=timesteps, synapse=synapse,
                        plot=plot, show=show, ocl=ocl,
                        gpus=gpus, identical=identical, output_dir=output_dir)
                else:
                    extractor = NeuralExtraction(
                        id_vectors, semantic_pointers, threshold=threshold,
                        probe_keys=probe_keys, timesteps=timesteps,
                        synapse=synapse, plot=plot, show=show, ocl=ocl,
                        gpus=gpus, identical=identical, output_dir=output_dir)
            else:
                extractor = Extraction(
                    id_vectors, semantic_pointers,
                    threshold, output_dir=output_dir)

            return extractor

        return make_extractor

    corpus_factory = make_corpus_factory(
        dimension, input_dir, unitary_relations, id_vecs,
        proportion, num_synsets, sp_noise, normalize)

    extractor_factory = make_extractor_factory(
        neural, fast, gpus, ocl, plot, show, threshold, timesteps, synapse)

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

    test_runner = ExtractionTester(corpus_factory, extractor_factory,
                                   corpus_seed, extractor_seed, test_seed,
                                   probeall, output_file)

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
