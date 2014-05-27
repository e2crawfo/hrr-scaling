try:
    import matplotlib as mpl
    mpl.use('Qt4Agg')
    can_plot = True
except ImportError:
    can_plot = False

import random
import numpy as np

from extraction_tester import ExtractionTester
import utilities

from corpora_management import VectorizedCorpus

from extraction import Extraction
from neural_extraction import NeuralExtraction
from fast_neural_extraction import FastNeuralExtraction

from wordnet_tests import ExpressionTest, JumpTest
from wordnet_tests import HierarchicalTest
from wordnet_tests import SentenceTest


def run(num_runs, jump_trials, hier_trials, sent_trials, deep_trials, expr,
        unitary_roles, short_sentence, do_neg, corpus_seed,
        extractor_seed, test_seed, seed, dimension, num_synsets,
        proportion, unitary_relations, abstract, synapse, timesteps,
        threshold, probeall, identical, fast, plot, show, gpus, ocl,
        outfile_format=""):

    input_dir, output_dir = utilities.read_config()

    neural = not abstract

    if gpus is not None:
        gpus.sort()

    if ocl is not None:
        ocl.sort()

    def make_corpus_factory(dimension, input_dir, unitary_relations,
                            proportion, num_synsets):

        def make_corpus():
            corpus = VectorizedCorpus(
                dimension, input_dir, unitary_relations,
                proportion, num_synsets)

            return corpus

        return make_corpus

    def make_extractor_factory(neural, fast, gpus, ocl, plot, show, threshold,
                               output_dir, timesteps, synapse):

        # pick an extraction algorithm
        def make_extractor(id_vectors, semantic_pointers, probe_keys):
            if neural:
                if fast and gpus:
                    extractor = FastNeuralExtraction(
                        id_vectors, semantic_pointers,
                        threshold=threshold,
                        output_dir=output_dir,
                        probe_keys=probe_keys,
                        timesteps=timesteps, synapse=synapse,
                        plot=plot, show=show, ocl=ocl,
                        gpus=gpus, identical=identical)
                else:
                    extractor = NeuralExtraction(
                        id_vectors, semantic_pointers, threshold=threshold,
                        output_dir=output_dir, probe_keys=probe_keys,
                        timesteps=timesteps, synapse=synapse,
                        plot=plot, show=show, ocl=ocl, gpus=gpus,
                        identical=identical)
            else:
                extractor = Extraction(
                    id_vectors, semantic_pointers, threshold)

            return extractor

        return make_extractor

    corpus_factory = make_corpus_factory(
        dimension, input_dir, unitary_relations, proportion, num_synsets)

    extractor_factory = make_extractor_factory(
        neural, fast, gpus, ocl, plot, show, threshold,
        output_dir, timesteps, synapse)

    outfile_suffix = \
        utilities.create_outfile_suffix(neural, unitary_relations)

    if corpus_seed == -1:
        corpus_seed = random.randrange(1000)

    if extractor_seed == -1:
        extractor_seed = random.randrange(1000)

    if test_seed == -1:
        test_seed = random.randrange(1000)

    if seed != -1:
        random.seed(seed)
        corpus_seed = random.randrange(1000)
        extractor_seed = random.randrange(1000)
        test_seed = random.randrange(1000)

    np.random.seed(test_seed)
    random.seed(test_seed)

    test_runner = ExtractionTester(corpus_factory, extractor_factory,
                                   corpus_seed, extractor_seed, test_seed,
                                   probeall, output_dir, outfile_suffix,
                                   outfile_format)

    if jump_trials > 0:

        test = JumpTest(test_runner, jump_trials)

        test_runner.add_test(test)

    if hier_trials > 0:

        test = HierarchicalTest(test_runner, hier_trials, do_neg=do_neg)

        test_runner.add_test(test)

    if sent_trials > 0:

        test = SentenceTest(
            test_runner, sent_trials, deep=False,
            unitary=unitary_roles, short=short_sentence)

        test_runner.add_test(test)

    if deep_trials > 0:

        test = SentenceTest(
            test_runner, deep_trials, deep=True,
            unitary=unitary_roles, short=short_sentence)

        test_runner.add_test(test)

    if expr:
        expr_trials = expr[0]
        expr = expr[1]

        test = ExpressionTest(test_runner, expr_trials, expression=expr)

        test_runner.add_test(test)

    test_runner.run_bootstrap(num_runs)

if __name__ == "__main__":
    argvals = utilities.parse_args(True)

    # specify tests
    num_runs = argvals.num_runs

    jump_trials = argvals.jump
    hier_trials = argvals.hier
    sent_trials = argvals.sent
    deep_trials = argvals.deep
    expr = argvals.expr

    unitary_roles = argvals.unitary_roles
    short_sentence = argvals.shortsent
    do_neg = not argvals.noneg

    # seeds
    corpus_seed = argvals.corpus_seed
    extractor_seed = argvals.extractor_seed
    test_seed = argvals.test_seed
    seed = argvals.seed

    # corpus args
    dimension = argvals.d
    num_synsets = argvals.num_synsets
    proportion = argvals.p
    unitary_relations = argvals.unitary_relations

    # extractor args
    abstract = argvals.abstract
    synapse = argvals.synapse
    timesteps = argvals.steps
    threshold = argvals.t
    probeall = argvals.probeall
    identical = argvals.identical
    fast = argvals.fast
    plot = argvals.plot and can_plot and not abstract
    show = argvals.show and plot

    gpus = argvals.gpus
    ocl = argvals.ocl

    run(num_runs, jump_trials, hier_trials, sent_trials, deep_trials, expr,
        unitary_roles, short_sentence, do_neg, corpus_seed,
        extractor_seed, test_seed, seed, dimension, num_synsets,
        proportion, unitary_relations, abstract, synapse, timesteps,
        threshold, probeall, identical, fast, plot, show, gpus, ocl)
