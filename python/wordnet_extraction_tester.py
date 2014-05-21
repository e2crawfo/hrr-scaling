# wordnet extraction tester
from extraction_tester import ExtractionTester
import utilities as util

import random
from collections import defaultdict
import numpy as np
from mytools import hrr, nf


class WordnetExtractionTester(ExtractionTester):
    def __init__(self, corpus, id_vectors, semantic_pointers,
                 relation_type_vectors, extractor, seed, output_dir=".",
                 h_test_symbols=None, sentence_symbols=None,
                 unitary=False, verbose=False, outfile_suffix=""):

        super(WordnetExtractionTester, self).__init__(
            id_vectors, semantic_pointers, extractor, seed,
            output_dir, unitary, verbose, outfile_suffix)

        self.sentence_results_file = None
        self.jump_results_file = None
        self.hierarchical_results_file = None

        self.corpus = corpus

        self.relation_type_vectors = relation_type_vectors

        h_test_symbols = [] if not h_test_symbols else h_test_symbols
        sentence_symbols = [] if not sentence_symbols else sentence_symbols

        self.h_test_symbols = h_test_symbols
        self.sentence_symbols = sentence_symbols

        self.role_hrrs = None

        self.jump_plan_words = []
        self.jump_plan_relation_indices = []

        self.rng = random.Random(self.seed)

    def set_jump_plan(self, w, ri):
        self.jump_plan_words = w
        self.jump_plan_relation_indices = ri

    def singleTest(self, testName, n, expression):
        # create a sentence
        dim = len(self.id_vectors.values()[0])
        expression = expression.replace('!id', 'p0')

        num_ids = expression.count('id')
        expression = expression.replace('id', '%s')
        temp_names = ['id'+str(i) for i in range(num_ids)]
        expression = expression % tuple(temp_names)

        chosen_id_keys = random.sample(self.id_vectors,
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

        vocab = hrr.Vocabulary(dim, unitary=unitary_names)
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

        result, correct, valid, exact = self.testLink(
            query_vector.v, test_vector.v, None, target_key,
            self.jump_results_file, return_vec=False, answers=[target_key],
            threshold=self.test_threshold)

    def jumpTest(self, testName, n):
        # select a key, follow a hyp/hol link, record success / failure

        testNumber = 0

        correct_score = 0
        valid_score = 0
        exact_score = 0

        while testNumber < n:
            if testNumber < len(self.jump_plan_words):
                words = self.jump_plan_words[
                    testNumber: min(n, len(self.jump_plan_words))
                    ]
            else:
                words = self.rng.sample(self.corpus, n-testNumber)

            for word in words:
                testableLinks = [r for r in self.corpus[word]
                                 if r[0] in self.relation_type_vectors]

                if len(testableLinks) > 0:
                    if testNumber < len(self.jump_plan_relation_indices):
                        prompt = testableLinks[
                            self.jump_plan_relation_indices[testNumber]
                            ]
                    else:
                        prompt = self.rng.sample(testableLinks, 1)[0]

                    util.print_header(self.jump_results_file, "New Jump Test")

                    answers = [r[1] for r in self.corpus[word]
                               if r[0] == prompt[0]]
                    relation_vec = self.relation_type_vectors[prompt[0]]

                    result, correct, valid, exact = self.testLink(
                        relation_vec, None, word, prompt[1],
                        self.jump_results_file,
                        num_relations=len(testableLinks),
                        answers=answers, threshold=self.test_threshold)

                    print >> self.jump_results_file, "Correct goal? ", correct
                    print >> self.jump_results_file, "Valid answers? ", valid
                    print >> self.jump_results_file, "Exact goal? ", exact

                    testNumber += 1

                    if correct:
                        correct_score += 1
                    if valid:
                        valid_score += 1
                    if exact:
                        exact_score += 1

        # print the score
        title = "Jump Test Summary"
        util.print_header(self.jump_results_file, title)
        self.jump_results_file.write("valid_score,"+str(valid_score)+":\n")
        self.jump_results_file.write("totaltests,"+str(testNumber)+":\n")
        util.print_footer(self.jump_results_file, title)

        correct_score = float(correct_score) / float(testNumber)
        valid_score = float(valid_score) / float(testNumber)
        exact_score = float(exact_score) / float(testNumber)

        print "score,"+str(correct_score)

        self.add_data("jump_score_correct", correct_score)
        self.add_data("jump_score_valid", valid_score)
        self.add_data("jump_score_exact", exact_score)

    def hierarchicalTest(self, testName, p, stat_depth=0, n=None,
                         rtype=[], startFromParent=False, do_neg=True):
        """Check whether word A is a type of word B. Test with n cases in
        which word A IS NOT a descendant of word B and m cases where word
        A IS a descendent of word B. The rtype parameter specifies which
        relationships to use in the search (by default, only the isA
        relationships)."""

        if n is None:
            n = p

        if not do_neg:
            n = 0

        p_count = 0
        n_count = 0

        p_score = 0
        n_score = 0

        negative_pairs = []
        positive_pairs = []

        # find positive and negative pairs
        while n_count < n:
            start = self.rng.sample(self.corpus, 1)[0]
            target = self.rng.sample(self.corpus, 1)[0]

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
            start = self.rng.sample(self.corpus, 1)[0]
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
            util.print_header(self.hierarchical_results_file, title)

            # for printing
            self.findAllParents(
                pair[0], pair[1], rtype, False, stat_depth=stat_depth,
                print_output=True)

            result = self.findAllParents(
                pair[0], pair[1], rtype, True, stat_depth=stat_depth,
                print_output=True)

            if result == -1:
                n_score += 1

        title = "New Hierarchical Test - Positive"
        for pair in positive_pairs:
            util.print_header(self.hierarchical_results_file, title)

            self.findAllParents(
                pair[0], pair[1], rtype, False, stat_depth=stat_depth,
                print_output=True)

            result = self.findAllParents(
                pair[0], pair[1], rtype, True, stat_depth=stat_depth,
                print_output=True)

            if result > -1:
                p_score += 1

        # print the score
        title = "Hierarchical Test Summary"
        util.print_header(self.hierarchical_results_file, title)
        self.hierarchical_results_file.write("Start trial:\n")
        self.hierarchical_results_file.write("FP,"+str(n - n_score)+"\n")
        self.hierarchical_results_file.write("CR,"+str(n_score)+"\n")
        self.hierarchical_results_file.write("hits,"+str(p_score)+"\n")
        self.hierarchical_results_file.write("misses,"+str(p - p_score)+"\n")
        self.hierarchical_results_file.write(
            "TS,"+str(n_score + p_score)+" out of "+str(n+p)+"\n")
        self.hierarchical_results_file.write("NT,"+str(n)+"\n")
        self.hierarchical_results_file.write("PT,"+str(p)+"\n")
        util.print_footer(self.hierarchical_results_file, title)

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
            print >> self.hierarchical_results_file, \
                "In find all parents, useHRR=", use_HRR

            print >> self.hierarchical_results_file, "Start:", start_key

            if target_key is not None:
                print >> self.hierarchical_results_file, "Target:", target_key

        use_vecs = use_HRR and self.extractor.return_vec

        level = 0
        if use_vecs:
            layerA = [self.semantic_pointers[start_key]]

            if target_key:
                target_vector = self.semantic_pointers[target_key]
        else:
            layerA = [start_key]

        layerB = []
        parents = set()

        while len(layerA) > 0:
            word = layerA.pop()

            # test whether we've found the target
            found = False
            if use_vecs:
                found = self.test_vector(word, target_vector)
            else:
                found = word == target_key

            if found:
                if print_output:
                    print >> self.hierarchical_results_file, target_key, \
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
                        print >> self.hierarchical_results_file, key, \
                            "found at level ", level

                links = []

                if not use_HRR:
                    links = [r[1] for r in self.corpus[word] if r[0] in rtype]
                else:

                    for symbol in rtype:
                        answers = [r[1] for r in self.corpus[key]
                                   if r[0] == symbol]
                        relation_vec = self.relation_type_vectors[symbol]

                        if len(answers) == 0:
                            target = None
                        else:
                            target = answers[0]

                        relations = filter(
                            lambda x: x[0] in self.relation_type_vectors,
                            self.corpus[key])

                        num_relations = len(relations)

                        if use_vecs:
                            result = self.testLink(
                                relation_vec, word, key, target,
                                self.hierarchical_results_file,
                                return_vec=True, depth=level,
                                num_relations=num_relations,
                                answers=answers)

                            links.append(result)

                        else:
                            results = self.testLink(
                                relation_vec, None, key, target,
                                self.hierarchical_results_file,
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

    def sentenceTest(self, testName, n, deep=False, short=False):
        # check that POS lists exist (form them if required)
        if self.role_hrrs is None:
            self.nouns = []
            self.adjectives = []
            self.adverbs = []
            self.verbs = []
            for word in self.corpus.keys():

                pos, offset = word

                if pos == 'n':
                    self.nouns.append(offset)
                elif pos == 'a':
                    self.adjectives.append(offset)
                elif pos == 'r':
                    self.adverbs.append(offset)
                elif pos == 'v':
                    self.verbs.append(offset)
                else:
                    raise Exception('Unexpected POS token: '+pos)

            self.role_hrrs = {}
            for symbol in self.sentence_symbols:

                self.role_hrrs[symbol] = hrr.HRR(self.D)

                if self.unitary:
                    self.role_hrrs[symbol].make_unitary()

        posmap = {'n': self.nouns, 'a': self.adjectives,
                  'r': self.adverbs, 'v': self.verbs}

        score = defaultdict(float)

        for i in range(n):
            title = "New Sentence Test"
            if deep:
                title += "- Deep"

            util.print_header(self.sentence_results_file, title)

            included_roles = []

            for symbol in self.sentence_symbols:
                valid_pos = posmap[self.sentence_symbols[symbol][1]]
                include = self.rng.random() < self.sentence_symbols[symbol][0]

                if valid_pos and include:
                    included_roles.append((symbol,))

            if deep:
                embed = self.rng.sample(included_roles, 1)[0]
                included_roles.remove(embed)
                for symbol in self.sentence_symbols:
                    if self.rng.random() < self.sentence_symbols[symbol][0]:
                        included_roles.append((embed[0], symbol))

            sentence = {}
            tag_vectors = {}
            sentence_hrr = hrr.HRR(data=np.zeros(self.D))

            # Pick role-fillers and create HRR representing the sentence
            # Also store the hrr to use as the query to extract each synset
            # included in the sentence.
            for role in included_roles:
                print >> self.sentence_results_file, role
                pos = self.sentence_symbols[role[-1]][1]
                word = (pos, self.rng.sample(posmap[pos], 1)[0])

                sentence[role] = word

                tag_hrr = [self.role_hrrs[x] for x in role]

                tag_hrr = reduce(lambda x, y: x * y, tag_hrr)

                sentence_hrr += tag_hrr * hrr.HRR(data=self.id_vectors[word])

                tag_vectors[role] = tag_hrr.v

            sentence_hrr.normalize()

            sentence_vector = sentence_hrr.v

            print >> self.sentence_results_file, "Roles in sentence:"
            print >> self.sentence_results_file, sentence

            # ask about parts of the sentence
            sentence_score = defaultdict(float)
            sentence_length = defaultdict(float)
            for role in sentence.keys():

                answer = sentence[role]

                self.current_start_key = None
                self.current_target_keys = [answer]
                self.current_num_relations = len(sentence)

                print >> self.sentence_results_file, "\nTesting ", role

                result, correct, valid, exact = self.testLink(
                    tag_vectors[role], sentence_vector, None, answer,
                    output_file=self.sentence_results_file, return_vec=False,
                    num_relations=len(sentence), answers=[answer],
                    threshold=self.test_threshold)

                depth = len(role)
                if correct:
                    sentence_score[depth] += 1
                    print >> self.sentence_results_file, "Correct."
                else:
                    print >> self.sentence_results_file, "Incorrect."

                sentence_length[depth] += 1

                if short:
                    break

            for d in sentence_score:
                sentence_percent = sentence_score[d] / sentence_length[d]

                print >> self.sentence_results_file, \
                    "Percent correct for current sentence at depth %d: %f" \
                    % (d, sentence_percent)

                score[d] = score[d] + sentence_percent

        for d in score:
            print "Sentence test score at depth %d: %f out of %d" \
                % (d, score[d], n)

            percent = score[d] / n

            title = "Sentence Test Summary - Depth = %d" % d
            util.print_header(self.sentence_results_file, title)
            print >> self.sentence_results_file, "Correct: ", score[d]
            print >> self.sentence_results_file, "Total: ", n
            print >> self.sentence_results_file, "Percent: ", percent
            util.print_footer(self.sentence_results_file, title)

            self.add_data("sentence_score_%d" % d, percent)

    def openJumpResultsFile(self, mode='w'):
        if not self.jump_results_file:
            self.jump_results_file = open(
                self.output_dir+'/jump_results_' +
                self.date_time_string + self.outfile_suffix, mode)

    def openHierarchicalResultsFile(self, mode='w'):
        if not self.hierarchical_results_file:
            self.hierarchical_results_file = open(
                self.output_dir+'/hierarchical_results_' +
                self.date_time_string + self.outfile_suffix, mode)

    def openSentenceResultsFile(self, mode='w'):
        if not self.sentence_results_file:
            self.sentence_results_file = open(
                self.output_dir+'/sentence_results_' +
                self.date_time_string + self.outfile_suffix, mode)

    def closeFiles(self):
        if self.sentence_results_file:
            self.sentence_results_file.close()

        if self.jump_results_file:
            self.jump_results_file.close()

        if self.hierarchical_results_file:
            self.hierarchical_results_file.close()

    def runBootstrap_single(self, sample_size, num_trials_per_sample,
                            num_bootstrap_samples=999, expression=''):

        single_test = lambda x, y: self.singleTest(x, y, expression)

        self.openJumpResultsFile()

        self.runBootstrap(
            sample_size, num_trials_per_sample, num_bootstrap_samples,
            self.jump_results_file, single_test)

    def runBootstrap_jump(self, sample_size, num_trials_per_sample,
                          num_bootstrap_samples=999):

        self.openJumpResultsFile()

        self.runBootstrap(
            sample_size, num_trials_per_sample, num_bootstrap_samples,
            self.jump_results_file, self.jumpTest)

    def runBootstrap_hierarchical(self, sample_size, num_trials_per_sample,
                                  num_bootstrap_samples=999, stats_depth=0,
                                  symbols=None, do_neg=True):

        file_open_func = self.openHierarchicalResultsFile
        file_open_func()

        if not symbols:
            symbols = self.h_test_symbols

        htest = lambda x, y: self.hierarchicalTest(
            x, y, stats_depth, rtype=symbols, do_neg=do_neg)

        self.runBootstrap(
            sample_size, num_trials_per_sample, num_bootstrap_samples,
            self.hierarchical_results_file, htest, file_open_func)

    def runBootstrap_sentence(self, sample_size, num_trials_per_sample,
                              num_bootstrap_samples=999, deep=False,
                              short=False):

        self.openSentenceResultsFile()

        stest = lambda x, y: self.sentenceTest(x, y, deep=deep, short=short)

        self.runBootstrap(
            sample_size, num_trials_per_sample, num_bootstrap_samples,
            self.sentence_results_file, stest)

    def print_relation_stats(self, output_file):
        relation_counts = {}
        relation_count = 0
        relation_hist = {}

        for key in self.corpus:
            lst = self.corpus[key]
            length = len(lst)

            if length not in relation_hist:
                relation_hist[length] = 1
            else:
                relation_hist[length] += 1

            for relation in lst:
                relation_count += 1
                if not relation[0] in relation_counts:
                    relation_counts[relation[0]] = 1
                else:
                    relation_counts[relation[0]] += 1

        title = "Relation Distribution"
        util.print_header(output_file, title)
        output_file.write("relation_counts: " + str(relation_counts) + " \n")
        output_file.write("relation_count: " + str(relation_count) + " \n")
        output_file.write("relation_hist: " + str(relation_hist) + " \n")
        print float(sum(relation_hist.values())) / float(len(relation_hist))
        util.print_footer(output_file, title)
