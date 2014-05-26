# wordnet extraction test_runner

import random
import datetime
import sys
import shutil
from collections import defaultdict

import numpy as np

from mytools import hrr, nf, bootstrap

import symbol_definitions
import utilities as util


def to_seconds(delta):
    seconds = delta.seconds
    milliseconds = (delta.microseconds / 1000) / 1000.0
    return seconds + milliseconds


class WordnetTest(object):

    def __init__(self, test_runner, test_name, num_trials):

        self.num_trials = num_trials

        self.test_runner = test_runner

        filename = test_runner.filename_format % test_name
        mode = 'w'

        self.output_file = open(filename, mode)

        self.current_start_key = None
        self.current_target_keys = None
        self.current_relation_keys = None
        self.current_num_relations = None

        self.test_threshold = 0.7

        self.num_jumps = 0
        self.bootstrapper = None

        self.rng = random.Random(self.test_runner.test_seed)

    @property
    def extractor(self):
        return self._extractor

    @extractor.setter
    def extractor(self, _extractor):
        self._extractor = _extractor

    @property
    def corpus(self):
        return self._corpus

    @corpus.setter
    def corpus(self, _corpus):

        self._corpus = _corpus

        self.corpus_dict = _corpus.corpus_dict
        self.id_vectors = _corpus.id_vectors
        self.semantic_pointers = _corpus.semantic_pointers
        self.relation_type_vectors = _corpus.relation_type_vectors

    def add_data(self, index, data):
        if self.bootstrapper:
            self.bootstrapper.add_data(index, data)

    def start_run(self):
        self.extractor.set_tester(self)

    def bootstrap_start(self, num_runs):

        self.bootstrapper = bootstrap.Bootstrapper(
            verbose=True, write_raw_data=True)

        self.output_file.write("Begin series of " + str(num_runs)
                               + " runs with " + str(self.num_trials)
                               + " trials each.\n")

    def bootstrap_step(self, index):

        then = datetime.datetime.now()

        self.output_file.write("Begin run " + str(index) + "\n")

        self.run()

        self.output_file.write("After " + str(index) + " runs.")

        now = datetime.datetime.now()
        delta_time = now - then

        self.add_data("step_runtime", to_seconds(delta_time))

        self.bootstrapper.print_summary(self.output_file)
        self.output_file.flush()

    def bootstrap_end(self):
        if self.output_file:
            self.output_file.close()

    def extract(self, item, query):
        self.num_jumps += 1
        result = self.extractor.extract(item, query)

        return result

    def test_link(self, query_vector, word_vec=None, word_key=None, goal=None,
                  output_file=None, return_vec=False, answers=[],
                  num_relations=-1, depth=0):

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

        clean_result = self.extract(word_vec, query_vector)

        if goal:
            if self.extractor.return_vec:
                clean_result_vector = clean_result[0]

                if goal in answers:
                    answers.remove(goal)

                stats = self.get_stats(
                    clean_result_vector, goal, answers, output_file)

                clean_result = stats[0]
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
                    return clean_result_vector
                else:
                    return (clean_result, jump_correct, False, False)

            if return_vec:
                # here there should be an error about trying to return vectors
                # even though we don't get them back from the extractor
                pass

            # here we are returning keys
            if answers:
                # clean result assumed to be sorted by activation
                front = []
                for r in clean_result:
                    if r in answers:
                        front.append(r)
                    else:
                        break

                correct = goal in front
                validAnswers = all([r in answers for r in clean_result])
                exactGoal = len(clean_result) > 0 and clean_result[0] == goal

                return (clean_result, correct, validAnswers, exactGoal)
            else:
                return clean_result

        else:
            if self.extractor.return_vec:
                clean_result_vector = clean_result[0]

                clean_result, largest, size = self.get_stats(
                    clean_result_vector, None, None, output_file)

                clean_result = []

                norm = np.linalg.norm(word_vec)

                print >> output_file, "negInitialVecSize: ", norm
                print >> output_file, "negLargestDotProduct: ", largest
                print >> output_file, "negSize: ", size

                self.add_data("negInitialVecSize", norm)
                self.add_data("negLargestDotProduct", largest)
                self.add_data("negSize", size)

                if return_vec:
                    return clean_result_vector

            return clean_result

    def get_key_from_vector(self, vector, vector_dict, return_comps=False):
        if not np.linalg.norm(vector) >= 0.1:
            return None

        comparisons = self.find_matches(vector, vector_dict)
        max_key, max_val = max(comparisons, key=lambda x: x[1])

        if max_val > self.decision_threshold:
            return max_key
        else:
            return None

    def find_matches(self, vector, vector_dict, exempt=[]):
        hrr_vec = hrr.HRR(data=vector)

        for key in vector_dict.keys():
            if key not in exempt:
                yield (key, hrr_vec.compare(hrr.HRR(data=vector_dict[key])))

    def get_stats(self, clean_result_vector, goal, other_answers, fp):
        size = np.linalg.norm(clean_result_vector)

        if not goal:
            comparisons = self.find_matches(clean_result_vector,
                                            self.semantic_pointers)

            largest_match = max(comparisons, key=lambda x: x[1])
            return (largest_match[0], largest_match[1], size)
        else:
            comparisons = self.find_matches(clean_result_vector,
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
            target_match = hrr_vec.compare(hrr.HRR(data=clean_result_vector))

            if target_match > second_match:
                clean_result = goal
            else:
                clean_result = second_key

            return (clean_result, target_match,
                    second_match, size, max_invalid_match)

    def copy_file(self, fp, func=None):
        if func:
            name = fp.name
            mangled_name = name + "_copy"

            fp.close()
            shutil.copyfile(name, mangled_name)
            func('a')

    def print_config(self):
        title = "WordnetTest Config"
        util.print_header(self.output_file, title)

        self.output_file.write("num_trials : " + str(self.num_trials) + "\n")
        self.output_file.write("test_threshold : " +
                               str(self.test_threshold) + "\n")
        self.corpus.print_config(self.output_file)
        self.extractor.print_config(self.output_file)

        self.output_file.write(self.__class__.__name__)
        self._print_config()

        util.print_footer(self.output_file, title)

    def _print_config(self):
        pass


class ExpressionTest(WordnetTest):

    def __init__(self, test_runner, num_trials, expression):

        test_name = 'expression'

        super(ExpressionTest, self).__init__(
            test_runner, test_name, num_trials)

        self.expression = expression

    def run(self):
        self.start_run()

        expression = self.expression

        dimension = len(self.id_vectors.values()[0])
        expression = expression.replace('!id', 'p0')

        num_ids = expression.count('id')
        expression = expression.replace('id', '%s')
        temp_names = ['id'+str(i) for i in range(num_ids)]
        expression = expression % tuple(temp_names)

        chosen_id_keys = self.rng.sample(self.id_vectors,
                                         expression.count('id') + 1)
        chosen_id_vectors = [hrr.HRR(data=self.id_vectors[key])
                             for key in chosen_id_keys]
        target_key = chosen_id_keys[0]

        names_dict = dict(zip(['p0'] + temp_names, chosen_id_vectors))
        names_keys_dict = dict(zip(['p0'] + temp_names, chosen_id_keys))

        query_vectors = nf.find_query_vectors(expression, 'p0')
        query_expression = '*'.join(query_vectors)

        temp_names = expression.replace('*', '+').split('+')
        temp_names = [tn.strip() for tn in temp_names]
        unitary_names = [u for u in temp_names if u[-1:] == "u"]

        vocab = hrr.Vocabulary(dimension, unitary=unitary_names)
        for n, v in names_dict.iteritems():
            vocab.add(n, v)

        print "expression:", expression
        print "query_expression:", query_expression
        print "unitary_names:", unitary_names
        print "target_key:", target_key
        print "name_keys_dict:", names_keys_dict

        test_vector = eval(expression, {}, vocab)
        test_vector.normalize()

        query_vector = eval(query_expression, {}, vocab)

        result, correct, valid, exact = self.test_link(
            query_vector.v, test_vector.v, None, target_key,
            self.output_file, return_vec=False, answers=[target_key])

    def print_config(self):
        self.output_file.write("ExpressionTest config:\n")

        super(ExpressionTest, self).print_config()

        self.output_file.write("expression : " +
                               str(self.expression) + "\n")


class JumpTest(WordnetTest):

    def __init__(self, test_runner, num_trials):

        test_name = 'jump'

        super(JumpTest, self).__init__(test_runner, test_name, num_trials)

    def run(self):
        # select a key, follow a link, record success / failure

        self.start_run()

        testNumber = 0

        correct_score = 0
        valid_score = 0
        exact_score = 0

        while testNumber < self.num_trials:
            words = self.rng.sample(self.corpus_dict,
                                    self.num_trials-testNumber)

            for word in words:
                testableLinks = [r for r in self.corpus_dict[word]
                                 if r[0] in self.relation_type_vectors]

                if len(testableLinks) > 0:
                    prompt = self.rng.sample(testableLinks, 1)[0]

                    util.print_header(self.output_file, "New Jump Test")

                    answers = [r[1] for r in self.corpus_dict[word]
                               if r[0] == prompt[0]]
                    relation_vec = self.relation_type_vectors[prompt[0]]

                    result, correct, valid, exact = self.test_link(
                        relation_vec, None, word, prompt[1],
                        self.output_file,
                        num_relations=len(testableLinks),
                        answers=answers)

                    print >> self.output_file, "Correct goal? ", correct
                    print >> self.output_file, "Valid answers? ", valid
                    print >> self.output_file, "Exact goal? ", exact

                    testNumber += 1

                    if correct:
                        correct_score += 1
                    if valid:
                        valid_score += 1
                    if exact:
                        exact_score += 1

        # print the score
        title = "Jump Test Summary"
        util.print_header(self.output_file, title)
        self.output_file.write("valid_score,"+str(valid_score)+":\n")
        self.output_file.write("totaltests,"+str(testNumber)+":\n")
        util.print_footer(self.output_file, title)

        correct_score = float(correct_score) / float(testNumber)
        valid_score = float(valid_score) / float(testNumber)
        exact_score = float(exact_score) / float(testNumber)

        print "score,"+str(correct_score)

        self.add_data("jump_score_correct", correct_score)
        self.add_data("jump_score_valid", valid_score)
        self.add_data("jump_score_exact", exact_score)

    def print_config(self):
        self.output_file.write("JumpTest config:\n")

        super(JumpTest, self).print_config()


class HierarchicalTest(WordnetTest):

    def __init__(self, test_runner, num_trials, do_neg):

        test_name = 'hierarchical'

        super(HierarchicalTest, self).__init__(
            test_runner, test_name, num_trials)

        self.relation_types = symbol_definitions.hierarchical_test_symbols()

        self.stat_depth = 10
        self.do_neg = do_neg
        self.decision_threshold = 0.4

    def run(self):
        """Check whether word A is a type of word B. Test with n cases in
        which word A IS NOT a descendant of word B and m cases where word
        A IS a descendent of word B. The rtype parameter specifies which
        relationships to use in the search (by default, only the isA
        relationships)."""

        self.start_run()

        rtype = self.relation_types

        p = self.num_trials

        if self.do_neg:
            n = p
        else:
            n = 0

        p_count = 0
        n_count = 0

        p_score = 0
        n_score = 0

        negative_pairs = []
        positive_pairs = []

        # find positive and negative pairs
        while n_count < n:
            start = self.rng.sample(self.corpus_dict, 1)[0]
            target = self.rng.sample(self.corpus_dict, 1)[0]

            parent_list = self.findAllParents(
                start, None, rtype, False, stat_depth=0, print_output=False)

            pair = (start, target)
            if target in parent_list and p_count < p:
                positive_pairs.append(pair)
                p_count += 1
            elif not (target in parent_list):
                negative_pairs.append(pair)
                n_count += 1

        while p_count < p:
            start = self.rng.sample(self.corpus_dict, 1)[0]
            parent_list = self.findAllParents(
                start, None, rtype, False, stat_depth=0, print_output=False)

            if len(parent_list) == 0:
                continue

            target = self.rng.sample(parent_list, 1)[0]
            positive_pairs.append((start, target))
            p_count += 1

        # now run the tests
        title = "New Hierarchical Test - Negative"
        for pair in negative_pairs:
            util.print_header(self.output_file, title)

            # do it symbolically first, for comparison
            self.findAllParents(
                pair[0], pair[1], rtype, False, stat_depth=self.stat_depth,
                print_output=True)

            result = self.findAllParents(
                pair[0], pair[1], rtype, True, stat_depth=self.stat_depth,
                print_output=True)

            if result == -1:
                n_score += 1

        title = "New Hierarchical Test - Positive"
        for pair in positive_pairs:
            util.print_header(self.output_file, title)

            # do it symbolically first, for comparison
            self.findAllParents(
                pair[0], pair[1], rtype, False, stat_depth=self.stat_depth,
                print_output=True)

            result = self.findAllParents(
                pair[0], pair[1], rtype, True, stat_depth=self.stat_depth,
                print_output=True)

            if result > -1:
                p_score += 1

        # print the score
        title = "Hierarchical Test Summary"
        util.print_header(self.output_file, title)
        self.output_file.write("Start trial:\n")
        self.output_file.write("FP,"+str(n - n_score)+"\n")
        self.output_file.write("CR,"+str(n_score)+"\n")
        self.output_file.write("hits,"+str(p_score)+"\n")
        self.output_file.write("misses,"+str(p - p_score)+"\n")
        self.output_file.write(
            "TS,"+str(n_score + p_score)+" out of "+str(n+p)+"\n")
        self.output_file.write("NT,"+str(n)+"\n")
        self.output_file.write("PT,"+str(p)+"\n")
        util.print_footer(self.output_file, title)

        print "Start trial:\n"
        print "FP,"+str(n-n_score)+"\n"
        print "CR,"+str(n_score)+"\n"
        print "hits,"+str(p_score)+"\n"
        print "misses,"+str(p-p_score)+"\n"
        print "TS,"+str(n_score+p_score)+" out of "+str(n+p)+"\n"
        print "NT,"+str(n)+"\n"
        print "PT,"+str(p)+"\n"

        overall_score = float(n_score + p_score) / float(p + n)
        self.add_data("hierarchical_score", overall_score)

        return result

    def findAllParents(self, start_key, target_key=None, rtype=[],
                       use_HRR=False, stat_depth=0, print_output=False):

        if print_output:
            print >> self.output_file, \
                "In find all parents, useHRR=", use_HRR

            print >> self.output_file, "Start:", start_key

            if target_key is not None:
                print >> self.output_file, "Target:", target_key

        use_vecs = use_HRR and self.extractor.return_vec

        level = 0
        if use_vecs:
            layerA = [self.semantic_pointers[start_key]]

            if target_key:
                target_vector = self.semantic_pointers[target_key]
                target_hrr = hrr.HRR(data=target_vector)
        else:
            layerA = [start_key]

        layerB = []
        parents = set()

        while len(layerA) > 0:
            word = layerA.pop()

            # test whether we've found the target
            found = False
            if use_vecs:
                word_hrr = hrr.HRR(data=word)
                found = target_hrr.compare(word_hrr) > self.decision_threshold
            else:
                found = word == target_key

            if found:
                if print_output:
                    print >> self.output_file, target_key, \
                        "found at level ", level

                return level

            if use_vecs:
                key = self.get_key_from_vector(word, self.semantic_pointers)
            else:
                key = word

            if key:
                if key in parents:
                    continue

                if level > 0:
                    parents.add(key)

                    if print_output:
                        print >> self.output_file, key, \
                            "found at level ", level

                links = []

                if not use_HRR:
                    links = [r[1] for r in self.corpus_dict[word]
                             if r[0] in rtype]
                else:

                    for symbol in rtype:
                        answers = [r[1] for r in self.corpus_dict[key]
                                   if r[0] == symbol]
                        relation_vec = self.relation_type_vectors[symbol]

                        if len(answers) == 0:
                            target = None
                        else:
                            target = answers[0]

                        relations = filter(
                            lambda x: x[0] in self.relation_type_vectors,
                            self.corpus_dict[key])

                        num_relations = len(relations)

                        if use_vecs:
                            result = self.test_link(
                                relation_vec, word, key, target,
                                self.output_file,
                                return_vec=True, depth=level,
                                num_relations=num_relations,
                                answers=answers)

                            links.append(result)

                        else:
                            results = self.test_link(
                                relation_vec, None, key, target,
                                self.output_file,
                                return_vec=False, depth=level,
                                num_relations=num_relations, answers=answers)

                            if answers:
                                results = results[0]

                            links.extend(results)

                if len(links) > 0:
                    layerB.extend(links)

            if len(layerA) == 0:
                level = level + 1
                layerA = layerB
                layerB = []

        if target_key is None:
            return list(parents)
        else:
            return -1

    def print_config(self):
        self.output_file.write("HierarchicalTest config:\n")

        super(HierarchicalTest, self).print_config()

        self.output_file.write("relation_types : " +
                               str(self.relation_types) + "\n")
        self.output_file.write("stat_depth : " + str(self.stat_depth) + "\n")
        self.output_file.write("do_neg : " + str(self.do_neg) + "\n")
        self.output_file.write("decision_threshold : " +
                               str(self.decision_threshold) + "\n")


class SentenceTest(WordnetTest):

    def __init__(self, test_runner, num_trials, deep, short, unitary):

        test_name = 'sentence'

        super(SentenceTest, self).__init__(test_runner, test_name, num_trials)

        self.deep = deep
        self.short = short
        self.unitary = unitary

        self.sentence_symbols = symbol_definitions.sentence_role_symbols()

        self.role_hrrs = None

    def run(self):
        self.start_run()

        if not self.role_hrrs:
            self.create_role_hrrs()

        score = defaultdict(float)

        for i in range(self.num_trials):
            title = "New Sentence Test"
            if self.deep:
                title += "- Deep"

            util.print_header(self.output_file, title)

            sentence = self.generate_sentence()

            if self.deep:
                embed = self.rng.sample(sentence.keys(), 1)[0]

                embedded_sentence = self.generate_sentence()

                del sentence[embed]

                for role in embedded_sentence.keys():
                    sentence[embed + role] = embedded_sentence[role]

            tag_vectors = {}
            sentence_hrr = hrr.HRR(data=np.zeros(self.dimension))

            # Pick role-fillers and create HRR representing the sentence
            # Also store the hrr to use as the query to extract each synset
            # included in the sentence.
            for role in sentence:
                tag_hrr = [self.role_hrrs[x] for x in role]
                tag_hrr = reduce(lambda x, y: x * y, tag_hrr)

                synset = sentence[role]

                sentence_hrr += tag_hrr * hrr.HRR(data=self.id_vectors[synset])

                tag_vectors[role] = tag_hrr.v

            sentence_hrr.normalize()

            sentence_vector = sentence_hrr.v

            print >> self.output_file, "Roles in sentence:"
            print >> self.output_file, sentence

            # ask about parts of the sentence
            sentence_score = defaultdict(float)
            sentence_length = defaultdict(float)
            for role in sentence.keys():

                answer = sentence[role]

                self.current_start_key = None
                self.current_target_keys = [answer]
                self.current_num_relations = len(sentence)

                print >> self.output_file, "\nTesting ", role

                result, correct, valid, exact = self.test_link(
                    tag_vectors[role], sentence_vector, None, answer,
                    output_file=self.output_file, return_vec=False,
                    num_relations=len(sentence), answers=[answer])

                depth = len(role)
                if correct:
                    sentence_score[depth] += 1
                    print >> self.output_file, "Correct."
                else:
                    print >> self.output_file, "Incorrect."

                sentence_length[depth] += 1

                if self.short:
                    break

            for d in sentence_score:
                sentence_percent = sentence_score[d] / sentence_length[d]

                print >> self.output_file, \
                    "Percent correct for current sentence at depth %d: %f" \
                    % (d, sentence_percent)

                score[d] = score[d] + sentence_percent

        for d in score:
            print "Sentence test score at depth %d: %f out of %d" \
                % (d, score[d], self.num_trials)

            percent = score[d] / self.num_trials

            title = "Sentence Test Summary - Depth = %d" % d
            util.print_header(self.output_file, title)
            print >> self.output_file, "Correct: ", score[d]
            print >> self.output_file, "Total: ", self.num_trials
            print >> self.output_file, "Percent: ", percent
            util.print_footer(self.output_file, title)

            self.add_data("sentence_score_%d" % d, percent)

    def create_role_hrrs(self):

        self.dimension = len(self.id_vectors.values()[0])

        nouns = []
        adjectives = []
        adverbs = []
        verbs = []

        for synset in self.corpus_dict.keys():

            pos, offset = synset

            if pos == 'n':
                nouns.append(offset)
            elif pos == 'a':
                adjectives.append(offset)
            elif pos == 'r':
                adverbs.append(offset)
            elif pos == 'v':
                verbs.append(offset)
            else:
                raise Exception('Unexpected POS token: '+pos)

        self.role_hrrs = {}
        for role in self.sentence_symbols:

            self.role_hrrs[role] = hrr.HRR(self.dimension)

            if self.unitary:
                self.role_hrrs[role].make_unitary()

        self.posmap = {'n': nouns, 'a': adjectives,
                       'r': adverbs, 'v': verbs}

    def generate_sentence(self):
        """Returns a dictionary mapping role symbols to synsets."""

        sentence = {}

        for role in self.sentence_symbols:
            valid_pos = self.posmap[self.sentence_symbols[role][1]]
            include = self.rng.random() < self.sentence_symbols[role][0]

            if valid_pos and include:
                pos = self.sentence_symbols[role][1]
                synset = (pos, self.rng.sample(self.posmap[pos], 1)[0])
                sentence[(role,)] = synset

        return sentence

    def _print_config(self):

        self.output_file.write("deep : " + str(self.deep) + "\n")
        self.output_file.write("unitary : " + str(self.unitary) + "\n")
        self.output_file.write("short : " + str(self.short) + "\n")
        self.output_file.write("sentence_symbols : " +
                               str(self.sentence_symbols) + "\n")
