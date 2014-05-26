# extraction tester
import numpy as np

import datetime
import string
import random


class ExtractionTester(object):

    def __init__(self, corpus_factory, extractor_factory,
                 corpus_seed, extractor_seed, test_seed,
                 probeall=False, output_dir=".", outfile_suffix=""):

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

        self.filename_format = (output_dir + '/%s_results_' +
                                date_time_string + outfile_suffix)

        self.tests = []

    def add_test(self, test):
        self.tests.append(test)

    def initialize(self):
        np.random.seed(self.corpus_seed)
        random.seed(self.corpus_seed)

        self.corpus = self.corpus_factory()

        id_vectors = self.corpus.id_vectors
        semantic_pointers = self.corpus.semantic_pointers

        if self.probeall:
            probe_keys = id_vectors.keys()
        else:
            probe_keys = []

        np.random.seed(self.extractor_seed)
        random.seed(self.extractor_seed)

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

                if i == 0:
                    test.print_config()

                test.bootstrap_step(i)

        for test in self.tests:
            test.bootstrap_end()

    def print_config(self, output_file):
        pass
