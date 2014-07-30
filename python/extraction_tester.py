import numpy as np
import random
import gc
import psutil
import os
from mytools import bootstrap


class ExtractionTester(object):

    def __init__(self, corpus_factory, extractor_factory,
                 corpus_seed, extractor_seed, test_seed,
                 probeall=False, output_file="."):

        self.corpus_factory = corpus_factory
        self.extractor_factory = extractor_factory

        self.corpus_seed = corpus_seed
        self.extractor_seed = extractor_seed
        self.test_seed = test_seed

        self.output_file = output_file

        self.output_dir = output_file + "_data"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.probeall = probeall

        self.corpus_rng = random.Random()
        self.corpus_rng.seed(self.corpus_seed)

        self.extractor_rng = random.Random()
        self.extractor_rng.seed(self.extractor_seed)

        self.tests = []
        self.bootstrapper = bootstrap.Bootstrapper(
            verbose=True, write_raw_data=True)

    def next_extractor_seed(self):
        return self.extractor_rng.randint(0, np.iinfo(np.int32).max)

    def next_corpus_seed(self):
        return self.corpus_rng.randint(0, np.iinfo(np.int32).max)

    def add_test(self, test):
        self.tests.append(test)
        test.bootstrapper = self.bootstrapper
        test.output_dir = self.output_dir
        test.seed = self.test_seed

    def memory_usage_psutil(self):
        # return the memory usage in MB
        process = psutil.Process(os.getpid())
        mem = process.get_memory_info()[0] / float(2 ** 20)
        return mem

    def initialize(self):
        corpus_seed = self.next_corpus_seed()
        np.random.seed(corpus_seed)
        random.seed(corpus_seed)

        self.corpus = self.corpus_factory()

        id_vectors = self.corpus.id_vectors
        semantic_pointers = self.corpus.semantic_pointers

        if self.probeall:
            probe_keys = id_vectors.keys()
        else:
            probe_keys = []

        extractor_seed = self.next_extractor_seed()
        np.random.seed(extractor_seed)
        random.seed(extractor_seed)

        self.extractor = self.extractor_factory(
            id_vectors, semantic_pointers, probe_keys, self.output_dir)

        return self.corpus, self.extractor

    # Run a series of bootstrap runs, then combine the success rate from each
    # individual run into a total mean success rate with confidence intervals
    # the extractor on the run to be displayed
    def run_bootstrap(self, num_runs):

        for test in self.tests:
            test.bootstrap_start(num_runs)

        for i in range(num_runs):

            corpus, extractor = self.initialize()

            for test in self.tests:
                test.corpus = corpus
                test.extractor = extractor

            for test in self.tests:
                test.bootstrap_step(i)

            corpus = None
            extractor = None

            for test in self.tests:
                test.corpus = None
                test.extractor = None

            gc.collect()

            self.bootstrapper.add_data(
                'memory_usage_in_mb',
                self.memory_usage_psutil())

            self.bootstrapper.print_summary(self.output_file, flush=True)

        for test in self.tests:
            test.bootstrap_end()
