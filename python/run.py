try:
    import matplotlib as mpl
    # mpl.use('Qt4Agg')
    mpl.use('Agg')
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
        proportion, unitary_relations, id_vecs, sp_noise, normalize,
        abstract, synapse, timesteps, threshold, probeall, identical,
        fast, plot, show, gpus, ocl, output_file):

    input_dir, _ = utilities.read_config()

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

if __name__ == "__main__":
    args = {'num_runs': 2, 'jump_trials': 2, 'hier_trials': 2,
            'sent_trials': 0,
            'deep_trials': 2, 'expr': 0, 'unitary_roles': True,
            'short_sentence': False, 'do_neg': True, 'corpus_seed': -1,
            'extractor_seed': -1, 'test_seed': -1, 'seed': 1000,
            'dimension': 512, 'num_synsets': 5000, 'proportion': 1.0,
            'unitary_relations': False, 'id_vecs': True, 'abstract': False,
            'synapse': 0.005, 'timesteps': 75, 'threshold': 0.3,
            'probeall': False, 'identical': True, 'fast': False, 'plot': True,
            'show': False, 'gpus': [], 'ocl': [], 'output_file': ""}
    if 1:
        run(**args)
        import sys
        sys.exit()

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
    id_vecs = not argvals.no_ids
    sp_noise = argvals.sp_noise
    normalize = not argvals.no_norm

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
        proportion, unitary_relations, id_vecs, sp_noise, normalize,
        abstract, synapse, timesteps, threshold, probeall, identical, fast,
        plot, show, gpus, ocl)
