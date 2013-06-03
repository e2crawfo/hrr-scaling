#wordnet assoc memory tester
from vector_operations import *
import symbol_definitions
from assoc_memory_tester import AssociativeMemoryTester

from ccm.lib import hrr

import copy

import random
import datetime
import string
import sys


class WordnetAssociativeMemoryTester(AssociativeMemoryTester):
  def __init__(self, corpus, id_vectors, semantic_pointers, relation_symbols, associator, seed=1, output_dir=".", isA_symbols = [], partOf_symbols = [], sentence_symbols = [], unitary=False, verbose=False):

        super(WordnetAssociativeMemoryTester, self).__init__(id_vectors,
            semantic_pointers, associator, seed, output_dir, unitary, verbose)

        self.sentence_results_file=None
        self.jump_results_file=None
        self.hierarchical_results_file=None

        self.corpus = corpus

        self.relation_symbols = relation_symbols
        self.isA_symbols = isA_symbols
        self.sentence_symbols = sentence_symbols

        self.sentence_vocab = None

        self.jump_plan_words = []
        self.jump_plan_relation_indices = []

  def set_jump_plan(self, w, ri):
      self.jump_plan_words = w
      self.jump_plan_relation_indices = ri

  def jumpTest(self, testName, n, dataFunc=None):
        # select a key, follow a hyp/hol link, record success / failure

        testNumber = 0

        correct_score = 0
        valid_score = 0
        exact_score = 0

        while testNumber < n:
            if testNumber < len(self.jump_plan_words):
              words = self.jump_plan_words[testNumber: min(n, len(self.jump_plan_words))]
            else:
              words = self.rng.sample(self.corpus, n-testNumber)

            for word in words:
                testableLinks = [r for r in self.corpus[word] if r[0] in self.relation_symbols]

                if len(testableLinks) > 0:
                    if testNumber < len(self.jump_plan_relation_indices):
                      prompt = testableLinks[self.jump_plan_relation_indices[testNumber]]
                    else:
                      prompt = self.rng.sample(testableLinks, 1)[0]

                    self.print_header(self.jump_results_file, "New Jump Test")

                    answers = [r[1] for r in self.corpus[word] if r[0]==prompt[0]]

                    result, correct, valid, exact = self.testLink(prompt[0], None, word, prompt[1], self.jump_results_file, num_relations = len(testableLinks), answers=answers, threshold=self.test_threshold)

                    print >> sys.stderr, "Correct goal? ",correct
                    print >> sys.stderr, "Valid answers? ",valid
                    print >> sys.stderr, "Exact goal? ",exact
                    print >> self.jump_results_file, "Correct goal? ",correct
                    print >> self.jump_results_file, "Valid answers? ",valid
                    print >> self.jump_results_file, "Exact goal? ",exact

                    if dataFunc:
                      dataFunc(self.associator)

                    testNumber += 1

                    if correct: correct_score += 1
                    if valid: valid_score += 1
                    if exact: exact_score += 1

        # print the score
        title = "Jump Test Summary"
        self.print_header(self.jump_results_file, title)
        self.jump_results_file.write("valid_score,"+str(valid_score)+":\n")
        self.jump_results_file.write("totaltests,"+str(testNumber)+":\n")
        self.print_footer(self.jump_results_file, title)

        correct_score = float(correct_score) / float(testNumber)
        valid_score = float(valid_score) / float(testNumber)
        exact_score = float(exact_score) / float(testNumber)

        print "score,"+str(valid_score)

        self.add_data("jump_score_correct", correct_score)
        self.add_data("jump_score_valid", valid_score)
        self.add_data("jump_score_exact", exact_score)

  def hierarchicalTest(self, testName, n, stat_depth = 0, m=None, rtype=[], startFromParent=False, dataFunc=None):
        """Check whether word A is a type of word B. Test with n cases in which
        word A IS NOT a descendant of word B and m cases where word A IS a
        descendent of word B. The rtype parameter specifies which relationships
        to use in the search (by default, only the isA relationships)."""

        if m is None:
          m = n

        n_count = 0
        m_count = 0
        p_score = 0
        n_score = 0

        negative_pairs = []
        positive_pairs = []

        #find positive and negative pairs
        while n_count < n:
          start = self.rng.sample(self.corpus, 1)[0]
          target = self.rng.sample(self.corpus, 1)[0]

          parent_list = self.findAllParents(start, None, rtype, False, stat_depth=0, print_output=False)

          pair = (start, target)
          if target in parent_list and m_count < m:
            positive_pairs.append(pair)
            m_count += 1
          elif not (target in parent_list):
            negative_pairs.append(pair)
            n_count += 1

        while m_count < m:
          start = self.rng.sample(self.corpus, 1)[0]
          parent_list = self.findAllParents(start, None, rtype, False, stat_depth=0, print_output=False)

          if len(parent_list) == 0: continue

          target = self.rng.sample(parent_list, 1)[0]
          positive_pairs.append((start, target))
          m_count += 1

        #now run the tests!
        title = "New Hierarchical Test - Negative"
        for pair in negative_pairs:
          self.print_header(self.hierarchical_results_file, title)

          #for printing
          self.findAllParents(pair[0], pair[1], rtype, False, stat_depth=stat_depth, print_output=True)
          result = self.findAllParents(pair[0], pair[1], rtype, True, stat_depth=stat_depth, print_output=True)
          if result == -1: n_score += 1

        title = "New Hierarchical Test - Positive"
        for pair in positive_pairs:
          self.print_header(self.hierarchical_results_file, title)

          self.findAllParents(pair[0], pair[1], rtype, False, stat_depth=stat_depth, print_output=True)
          result = self.findAllParents(pair[0], pair[1], rtype, True, stat_depth=stat_depth, print_output=True)
          if result > -1: p_score += 1


        # print the score
        title = "Hierarchical Test Summary"
        self.print_header(self.hierarchical_results_file, title)
        self.hierarchical_results_file.write("Start trial:\n")
        self.hierarchical_results_file.write("FP,"+str(n-n_score)+"\n")#false positive
        self.hierarchical_results_file.write("CR,"+str(n_score)+"\n")#correct rejections
        self.hierarchical_results_file.write("hits,"+str(p_score)+"\n")
        self.hierarchical_results_file.write("misses,"+str(m-p_score)+"\n")
        self.hierarchical_results_file.write("TS,"+str(n_score+p_score)+" out of "+str(n+m)+"\n")#successful tests, out of total
        self.hierarchical_results_file.write("NT,"+str(n)+"\n")#neg tests
        self.hierarchical_results_file.write("PT,"+str(m)+"\n")#pos tests
        self.print_footer(self.hierarchical_results_file, title)

        print "Start trial:\n"
        print "FP,"+str(n-n_score)+"\n"#false positive
        print "CR,"+str(n_score)+"\n"#correct negative
        print "hits,"+str(p_score)+"\n"#correct positive
        print "misses,"+str(m-p_score)+"\n"#false negative
        print "TS,"+str(n_score+p_score)+" out of "+str(n+m)+"\n"#successful tests, out of total
        print "NT,"+str(n)+"\n"#neg tests
        print "PT,"+str(m)+"\n"#pos tests

        overall_score = float(n_score + p_score) / float(m + n)
        self.add_data("hierarchical_score", overall_score)

        return result

  def findAllParents(self, start_key, target_key=None, rtype=[], use_HRR=False, stat_depth=0, print_output = False):

        if print_output:
          print >> self.hierarchical_results_file, "In find all parents, useHRR=", use_HRR
          print >> self.hierarchical_results_file, "Start:", start_key

          if target_key is not None:
            print >> self.hierarchical_results_file, "Target:", target_key

        use_vecs = use_HRR and self.associator.return_vec

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

            #test whether we've found the target
            found = False
            if use_vecs:
              found = self.test_vector(word, target_vector)
            else:
              found = word == target_key

            if found:
              if print_output:
                print >> self.hierarchical_results_file, target_key, "found at level ", level
              return level
            
            if use_vecs:
              key = self.get_key_from_vector(word, self.semantic_pointers)
            else:
              key = word

            if key:
              if key in parents: continue
              if level > 0:
                parents.add(key)

                if print_output:
                  print >> self.hierarchical_results_file, key, "found at level ", level

              links =  []
              if not use_HRR:
                links = [r[1] for r in self.corpus[word] if r[0] in rtype]
              else:
                for symbol in rtype:
                  answers = [r[1] for r in self.corpus[key] if r[0] == symbol]

                  if len(answers) == 0:
                    target = None
                  else:
                    target = answers[0]

                  num_relations = len(filter(lambda x: x[0] in self.relation_symbols, self.corpus[key]))

                  if use_vecs:
                    result = self.testLink(symbol, word, key, target, self.hierarchical_results_file, return_vec=True, depth=level, num_relations=num_relations, answers=answers)
                    links.append( result )
                  else:
                    results = self.testLink(symbol, None, key, target, self.hierarchical_results_file, return_vec=False, depth=level, num_relations=num_relations, answers=answers)
                    if answers:
                      results=results[0]
                    links.extend( results )


              if len(links) > 0:
                  layerB.extend(links)

            if len(layerA)==0:
                level = level + 1
                layerA = layerB
                layerB = []

        if target_key is None:
            return list(parents)
        else:
            return -1

  def sentenceTest(self, testName, n, dataFunc=None):
        # check that POS lists exist (form them if required)
        if self.sentence_vocab is None:
            self.nouns = []
            self.adjectives = []
            self.adverbs = []
            self.verbs = []
            for word in self.corpus.keys():
                pos, offset = word
                if pos == 'n': self.nouns.append(offset)
                elif pos == 'a' : self.adjectives.append(offset)
                elif pos == 'r' : self.adverbs.append(offset)
                elif pos == 'v' : self.verbs.append(offset)
                else: raise Exception('Unexpected POS token: '+pos)
            self.sentence_vocab = {}
            for symbol in self.sentence_symbols:
                if self.unitary:
                  self.sentence_vocab[symbol] = genUnitaryVec(self.D)
                else:
                  self.sentence_vocab[symbol] = genVec(self.D)

        posmap = {'n':self.nouns, 'a':self.adjectives, 'r':self.adverbs, 'v':self.verbs}

        score = 0

        for i in range(n):
            title = "New Sentence Test"
            self.print_header(self.sentence_results_file, title)
            sentence = {}
            sentenceVector = numpy.zeros(self.D)

            print >> self.sentence_results_file, "Roles in sentence:"
            for symbol in self.sentence_symbols:

                if self.rng.random() < self.sentence_symbols[symbol][0]: # choose lexical items to include
                    print >> self.sentence_results_file, symbol
                    pos = self.sentence_symbols[symbol][1] # determine the POS for this lexical item
                    word = (pos, self.rng.sample(posmap[pos], 1)[0]) # choose words
                    sentence[symbol] = word    # build the sentence in a python dictionary
                    sentenceVector = sentenceVector + cconv(self.sentence_vocab[symbol],
                                                            self.id_vectors[word]) # build the sentence as an HRR vector
            # ask about parts of the sentence
            sentence_score = 0
            for symbol in sentence.keys():

                answer = sentence[symbol]

                self.current_start_key = None
                self.current_target_keys = [answer]
                self.current_num_relations = len(sentence)

                print >> self.sentence_results_file, "Testing ", symbol

                result, correct, valid, exact = self.testLink(self.sentence_vocab[symbol], sentenceVector, None, answer,
                    output_file = self.sentence_results_file, return_vec=False,
                    relation_is_vec=True, num_relations=len(sentence), answers=[answer], threshold = self.test_threshold)

                if exact:
                    sentence_score += 1
                    print >> self.sentence_results_file, "Correct."
                else:
                    print >> self.sentence_results_file, "Incorrect."

            sentence_percent = float(sentence_score) / float(len(sentence))
            print >> self.sentence_results_file, "Percent correct for current sentence: "
            score = score + sentence_percent

        print "sentence test score:", score, "out of", n

        percent = float(score) / float(n)
        title = "Sentence Test Summary"
        self.print_header(self.sentence_results_file, title)
        print >> self.sentence_results_file, "Correct: ", score
        print >> self.sentence_results_file, "Total: ", n
        print >> self.sentence_results_file, "Percent: ", percent
        self.print_footer(self.sentence_results_file, title)

        self.add_data("sentence_score", percent)

  def openJumpResultsFile(self, mode='w'):
    if not self.jump_results_file:
      self.jump_results_file=open(self.output_dir+'/jump_results_' + self.date_time_string , mode)

  def openHierarchicalResultsFile(self, mode='w'):
    if not self.hierarchical_results_file:
      self.hierarchical_results_file=open(self.output_dir+'/hierarchical_results_' + self.date_time_string , mode)

  def openSentenceResultsFile(self, mode='w'):
    if not self.sentence_results_file:
      self.sentence_results_file=open(self.output_dir+'/sentence_results_' + self.date_time_string , mode)

  def closeFiles(self):
    if self.sentence_results_file:
      self.sentence_results_file.close()

    if self.jump_results_file:
      self.jump_results_file.close()

    if self.hierarchical_results_file:
      self.hierarchical_results_file.close()

  def runBootstrap_jump(self, sample_size, num_trials_per_sample, num_bootstrap_samples=999, dataFunc=None):

    self.openJumpResultsFile()

    self.runBootstrap(sample_size, num_trials_per_sample, num_bootstrap_samples, self.jump_results_file, self.jumpTest, dataFunc=dataFunc)


  def runBootstrap_hierarchical(self, sample_size, num_trials_per_sample, num_bootstrap_samples=999, stats_depth=0, dataFunc=None, symbols=None):

    file_open_func = self.openHierarchicalResultsFile
    file_open_func()

    if not symbols:
      symbols = self.isA_symbols

    htest = lambda x, y, dataFunc=None: self.hierarchicalTest(x,y, stats_depth, rtype=symbols, dataFunc=dataFunc)

    self.runBootstrap(sample_size, num_trials_per_sample, num_bootstrap_samples, self.hierarchical_results_file, htest, file_open_func, dataFunc=dataFunc)


  def runBootstrap_sentence(self, sample_size, num_trials_per_sample, num_bootstrap_samples=999, dataFunc=None):

    self.openSentenceResultsFile()

    self.runBootstrap(sample_size, num_trials_per_sample, num_bootstrap_samples, self.sentence_results_file, self.sentenceTest, dataFunc=dataFunc)

  def print_relation_stats(self, output_file):
    relation_counts = {}
    relation_count = 0
    relation_hist = {}

    for key in self.corpus:
      if not len(self.corpus[key]) in relation_hist:
        relation_hist[len(self.corpus[key])] = 1
      else:
        relation_hist[len(self.corpus[key])] += 1

      for relation in self.corpus[key]:
        if relation[0] not in self.relation_symbols: continue

        relation_count += 1
        if not relation[0] in relation_counts:
          relation_counts[relation[0]] = 1
        else:
          relation_counts[relation[0]] += 1

    title = "Relation Distribution"
    self.print_header(output_file, title)
    output_file.write("relation_counts: " + str(relation_counts) + " \n")
    output_file.write("relation_count: " + str(relation_count) + " \n")
    output_file.write("relation_hist: " + str(relation_hist) + " \n")
    self.print_footer(output_file, title)

