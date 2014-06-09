# extraction tester
import numpy as np

import datetime
import string
import random
import gc
import psutil
import os


class ExtractionTester(object):

    def __init__(self, corpus_factory, extractor_factory,
                 corpus_seed, extractor_seed, test_seed,
                 probeall=False, output_dir=".", outfile_suffix="",
                 outfile_format=""):

        self.corpus_factory = corpus_factory
        self.extractor_factory = extractor_factory

        self.corpus_seed = corpus_seed
        self.extractor_seed = extractor_seed
        self.test_seed = test_seed

        self.probeall = probeall

        date_time_string = str(datetime.datetime.now()).split('.')[0]
        date_time_string = reduce(
            lambda y, z: string.replace(y, z, "_"),
            [date_time_string, ":", " ", "-"])

        if outfile_format:
            self.filename_format = outfile_format
        else:
            self.filename_format = (output_dir + '/%s_results_' +
                                    date_time_string + outfile_suffix)

        self.corpus_rng = random.Random()
        self.corpus_rng.seed(self.corpus_seed)

        self.extractor_rng = random.Random()
        self.extractor_rng.seed(self.extractor_seed)

        self.tests = []

    def next_extractor_seed(self):
        return self.extractor_rng.randint(0, np.iinfo(np.int32).max)

    def next_corpus_seed(self):
        return self.corpus_rng.randint(0, np.iinfo(np.int32).max)

    def add_test(self, test):
        self.tests.append(test)

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
            id_vectors, semantic_pointers, probe_keys)

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
                if i == 0:
                    test.print_config()

                test.bootstrap_step(i)

            corpus = None
            extractor = None

            for test in self.tests:
                test.corpus = None
                test.extractor = None

            gc.collect()

            for test in self.tests:
                test.add_data(
                    'memory_usage_in_mb', self.memory_usage_psutil())

        for test in self.tests:
            test.bootstrap_end()
