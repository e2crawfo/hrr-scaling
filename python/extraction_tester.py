# extraction tester
import utilities as util

from mytools import hrr, bootstrap
import numpy as np

import datetime
import string
import sys
import shutil


class ExtractionTester(object):
    def __init__(self, id_vectors, semantic_pointers, extractor,
                 seed, output_dir=".", unitary=False, verbose=False,
                 outfile_suffix=""):

        self.num_jumps = 0

        self.output_dir = output_dir

        date_time_string = str(datetime.datetime.now()).split('.')[0]
        self.date_time_string = reduce(
            lambda y, z: string.replace(y, z, "_"),
            [date_time_string, ":", " ", "-"])

        self.sentence_results_file = None
        self.jump_results_file = None
        self.hierarchical_results_file = None

        self.extractor = extractor
        self.extractor.set_tester(self)

        self.id_vectors = id_vectors
        self.semantic_pointers = semantic_pointers

        self.bootstrapper = None
        self.verbose = verbose

        self.unitary = unitary

        self.D = len(semantic_pointers.values()[0])

        self.current_start_key = None
        self.current_target_keys = None
        self.current_relation_keys = None
        self.current_num_relations = None

        self.test_threshold = 0.7
        self.soft_threshold = 0.4

        self.jump_plan_words = []
        self.jump_plan_relation_indices = []

        self.outfile_suffix = outfile_suffix

        self.seed = seed

    def extract(self, item, query):
        self.num_jumps += 1
        result = self.extractor.extract(item, query)

        return result

    def testLink(self, query_vector, word_vec=None, word_key=None, goal=None,
                 output_file=None, return_vec=False, answers=[],
                 num_relations=-1, depth=0, threshold=0.0):

        util.print_header(output_file, "Testing link", char='-')

        if word_vec is None:
            # should be an error here if neither is supplied
            word_vec = self.semantic_pointers[word_key]

        if word_key:
            print >> output_file, "start :", word_key
            util.print_header(sys.stdout, "Start")
            print(word_key)

        if goal:
            util.print_header(sys.stdout, "Target")
            print(goal)

        print >> output_file, "goal: ", goal
        print >> output_file, "depth: ", depth

        self.current_target_keys = answers
        self.current_num_relations = num_relations

        cleanResult = self.extract(word_vec, query_vector)

        if goal:
            if self.extractor.return_vec:
                cleanResultVector = cleanResult[0]

                if goal in answers:
                    answers.remove(goal)

                stats = self.getStats(
                    cleanResultVector, goal, answers, output_file)

                cleanResult = stats[0]
                target_match = stats[1]
                second_match = stats[2]
                size = stats[3]
                highest_invalid_match = stats[4]

                print >> output_file, "target match: ", target_match
                print >> output_file, "second match : ", second_match
                print >> output_file, "size : ", size
                print >> output_file, "highest_invalid_match : ", \
                    highest_invalid_match
                print >> output_file, "num_relations: ", num_relations

                self.add_data("d_"+str(depth)+"_target_match", target_match)
                self.add_data("d_"+str(depth)+"_second_match", second_match)
                self.add_data("d_"+str(depth)+"_size", size)

                self.add_data("d_"+str(depth)+"_hinv_match",
                              highest_invalid_match)

                if num_relations > 0:
                    self.add_data("r_"+str(num_relations)+"_target_match",
                                  target_match)

                    self.add_data("r_"+str(num_relations)+"_second_match",
                                  second_match)

                    self.add_data("r_"+str(num_relations)+"_size", size)

                    self.add_data("r_"+str(num_relations)+"_hinv_match",
                                  highest_invalid_match)

                jump_correct = (target_match > self.test_threshold and
                                target_match > highest_invalid_match)

                if return_vec:
                    return cleanResultVector
                else:
                    return (cleanResult, jump_correct, False, False)

            if return_vec:
                # here there should be an error about trying to return vectors
                # even though we don't get them back from the extractor
                pass

            # here we are returning keys
            if answers:
                # clean result assumed to be sorted by activation
                front = []
                for r in cleanResult:
                    if r in answers:
                        front.append(r)
                    else:
                        break

                correct = goal in front
                validAnswers = all([r in answers for r in cleanResult])
                exactGoal = len(cleanResult) > 0 and cleanResult[0] == goal

                return (cleanResult, correct, validAnswers, exactGoal)
            else:
                return cleanResult

        else:
            if self.extractor.return_vec:
                cleanResultVector = cleanResult[0]

                cleanResult, largest, size = self.getStats(
                    cleanResultVector, None, None, output_file)

                cleanResult = []

                norm = np.linalg.norm(word_vec)

                print >> output_file, "negInitialVecSize: ", norm
                print >> output_file, "negLargestDotProduct: ", largest
                print >> output_file, "negSize: ", size

                self.add_data("negInitialVecSize", norm)
                self.add_data("negLargestDotProduct", largest)
                self.add_data("negSize", size)

                if return_vec:
                    return cleanResultVector

            return cleanResult

    def test_vector(self, vector, target):
        """
        Used to decide whether a given vector is similar enough to a reference
        vector, using the soft threshold - usually set to maximize performance

        vector is the vector we are testing
        target is the vector we are comparing to
        """
        hrr_vec = hrr.HRR(data=vector)
        return hrr_vec.compare(hrr.HRR(data=target)) > self.soft_threshold

    def sufficient_norm(self, vector):
        return np.linalg.norm(vector) >= 0.1

    def get_key_from_vector(self, vector, vector_dict, return_comps=False):
        if not self.sufficient_norm(vector):
            return None

        comparisons = self.find_matches(vector, vector_dict)
        max_key, max_val = max(comparisons, key=lambda x: x[1])

        if max_val > self.soft_threshold:
            return max_key
        else:
            return None

    def find_matches(self, vector, vector_dict, exempt=[]):
        hrr_vec = hrr.HRR(data=vector)

        for key in vector_dict.keys():
            if key not in exempt:
                yield (key, hrr_vec.compare(hrr.HRR(data=vector_dict[key])))

    def getStats(self, cleanResultVector, goal, other_answers, fp):
        size = np.linalg.norm(cleanResultVector)

        if not goal:
            comparisons = self.find_matches(cleanResultVector,
                                            self.semantic_pointers)

            largest_match = max(comparisons, key=lambda x: x[1])
            return (largest_match[0], largest_match[1], size)
        else:
            comparisons = self.find_matches(cleanResultVector,
                                            self.semantic_pointers,
                                            exempt=[goal])

            if other_answers:
                invalids = []
                valids = []
                for c in comparisons:
                    if c[0] in other_answers:
                        valids.append(c)
                    else:
                        invalids.append(c)

                max_invalid = max(invalids, key=lambda x: x[1])
                max_invalid_key, max_invalid_match = max_invalid

                if len(valids) == 0:
                    second_key, second_match = max_invalid
                else:
                    max_valid = max(valids, key=lambda x: x[1])
                    max_valid_key, max_valid_match = max_valid

                    if max_invalid_match > max_valid_match:
                        second_key, second_match = max_invalid
                    else:
                        second_key, second_match = max_valid

            else:
                second_key, second_match = max(comparisons, key=lambda x: x[1])
                max_invalid_match = second_match

            hrr_vec = hrr.HRR(data=self.semantic_pointers[goal])
            target_match = hrr_vec.compare(hrr.HRR(data=cleanResultVector))

            if target_match > second_match:
                cleanResult = goal
            else:
                cleanResult = second_key

            return (cleanResult, target_match,
                    second_match, size, max_invalid_match)

    def copyFile(self, fp, func=None):
        if func:
            name = fp.name
            mangled_name = name + "_copy"

            fp.close()
            shutil.copyfile(name, mangled_name)
            func('a')

    def add_data(self, index, data):
        if self.bootstrapper:
            self.bootstrapper.add_data(index, data)

    # Run a series of bootstrap runs, then combine the success rate from each
    # individual run into a total mean success rate with confidence intervals
    # the extractor on the run to be displayed
    def runBootstrap(self, sample_size, num_trials_per_sample,
                     num_bootstrap_samples, output_file, func, statNames=None,
                     file_open_func=None, write_raw_data=True):
        start_time = datetime.datetime.now()

        self.bootstrapper = bootstrap.Bootstrapper(
            self.verbose, write_raw_data, seed=self.seed)

        # Now start running the tests
        self.num_jumps = 0
        output_file.write("Begin series of " + str(sample_size) + " runs with "
                          + str(num_trials_per_sample) + " trials each.\n")
        self.extractor.print_config(output_file)

        for i in range(sample_size):

            output_file.write("Begin run " + str(i + 1) + " out of " +
                              str(sample_size) + ":\n")

            func("", num_trials_per_sample)

            self.print_bootstrap_summary(i + 1, sample_size, output_file)
            output_file.flush()

        self.finish()

        end_time = datetime.datetime.now()
        delta_time = end_time - start_time

        self.print_bootstrap_runtime_summary(output_file, delta_time)
        self.print_relation_stats(output_file)

    def print_bootstrap_summary(self, sample_index, sample_size, output_file):

        output_file.write("After " + str(sample_index) + " samples out of " +
                          str(sample_size) + "\n")

        self.bootstrapper.print_summary(output_file)

    def print_bootstrap_runtime_summary(self, output_file, time):
        util.print_header(output_file, "Runtime Summary")
        output_file.write("Total elapsed time for bootstrap runs: "
                          + str(time) + "\n")
        output_file.write("Total num jumps: " + str(self.num_jumps) + "\n")

        if self.num_jumps != 0:
            seconds_per_jump = time.seconds / float(self.num_jumps)
            output_file.write("Average time per jump: " +
                              str(seconds_per_jump) + "\n")

        util.print_footer(output_file, "Runtime Summary")

    def finish(self):
        pass

    def get_similarities(self):
        self.similarities_file = open(self.output_dir+'/similarities_' +
                                      self.date_time_string, 'w')

        self.runBootstrap(1, 1, 999, self.similarities_file,
                          self.extractor.get_similarities_sample,
                          write_raw_data=False)
