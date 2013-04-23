from vector_operations import *
import symbol_definitions

from bootstrap import Bootstrapper

import copy
import heapq

import cPickle as pickle
import random
import datetime
import math
import string
import sys
import shutil
from ccm.lib import hrr

class AssociativeMemoryTester(object):
  def __init__(self, corpus, idVectors, structuredVectors, relation_symbols, associator, vector_indexing, cleanLevel = 0.3, seed=1, output_dir=".", isA_symbols = [], sentence_symbols = [], unitary=False, verbose=False):

        self.num_jumps = 0

        self.output_dir = output_dir

        date_time_string = str(datetime.datetime.now())
        self.date_time_string = reduce(lambda y,z: string.replace(y,z,"_"), [date_time_string,":","."," ","-"])

        self.sentence_results_file=None
        self.jump_results_file=None
        self.hierarchical_results_file=None

        self.seed = seed
        self.rng = random.Random()
        self.rng.seed(seed)
        self.np_rng = numpy.random.RandomState(seed)

        self.associator = associator
        self.associator.set_tester(self)
        self.vector_indexing =vector_indexing

        self.corpus = corpus
        self.idVectors = idVectors
        self.structuredVectors = structuredVectors

        self.relation_symbols = relation_symbols
        self.isA_symbols = isA_symbols
        self.sentence_symbols = sentence_symbols

        self.bootstrapper = None
        self.verbose = verbose

        self.key_indices = {}
        self.sentence_vocab = None
        self.unitary = unitary

        i = 0
        for key in self.structuredVectors.keys():
          self.key_indices[key] = i
          i += 1

        self.D = len(idVectors.values()[0])

        self.current_start_key = None
        self.current_target_keys = None
        self.current_relation_keys = None
        self.current_num_relations = None

        self.test_threshold = 0.8
        self.soft_threshold = 0.4

        self.jump_plan_words = []
        self.jump_plan_relation_indices = []

  def unbind_and_associate(self, item, query):
      self.num_jumps += 1
      result = self.associator.unbind_and_associate(item, query)

      return result

  def sufficient_norm(self, vector):
      return numpy.linalg.norm(vector) >= 0.1

  def set_jump_plan(self, w, ri):
      self.jump_plan_words = w
      self.jump_plan_relation_indices = ri

  def testLink(self, relation, word_vec=None, word_key=None, goal=None, output_file = None, return_vec=False, relation_is_vec=False, answers=[], num_relations = -1, depth=0, threshold=0.0):

        self.print_header(output_file, "Testing link", char='-')

        if word_vec is None:
          #should be an error here if neither is supplied
          word_vec = self.structuredVectors[word_key]

        if word_key:
          print >> output_file, "start :", word_key
          self.print_header(sys.stdout, "Start")
          print(word_key)

        if goal:
          self.print_header(sys.stdout, "Target")
          print(goal)

        if relation_is_vec:
          print >> output_file, "relation is a vector"
        else:
          print >> output_file, "relation: ", relation
          relation_key = relation
          relation = self.idVectors[relation]

        print >> output_file, "goal: ", goal
        print >> output_file, "depth: ", depth

        self.current_target_keys = answers
        self.current_num_relations = num_relations

        #cleanResultVectors = self.unbind_and_associate(word, relation, True, urn_agreement=goal)
        cleanResult = self.unbind_and_associate(word_vec, relation)

        if goal:
          if self.associator.return_vec:
            cleanResultVector = cleanResult[0]
            cleanResult, target_match, second_match, size = self.getStats(cleanResultVector, goal, output_file)

            cleanResult = [cleanResult] if (target_match > threshold) else []

            print >> output_file, "target match: ", target_match
            print >> output_file, "second match : ", second_match
            print >> output_file, "num_relations: ", num_relations

            self.add_data("depth_"+str(depth)+"_target_match", target_match)
            self.add_data("depth_"+str(depth)+"_second_match", second_match)
            self.add_data("depth_"+str(depth)+"_size", size)

            if num_relations > 0:
              self.add_data("rel_"+str(num_relations)+"_target_match", target_match)
              self.add_data("rel_"+str(num_relations)+"_second_match", second_match)
              self.add_data("rel_"+str(num_relations)+"_size", size)
            
            if return_vec:
              return cleanResultVector

          if return_vec:
            #here there should be an error about trying to return vectors even though we 
            #don't get them back from the associator
            pass

          #here we are returning keys
          if answers:
            #clean result assumed to be sorted by activation
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
          if self.associator.return_vec:
            cleanResultVector = cleanResult[0]
            cleanResult, largest, size = self.getStats(cleanResultVector, None, output_file)
            cleanResult = []

            norm = numpy.linalg.norm(word_vec)

            print >> output_file, "negInitialVecSize: ", norm
            print >> output_file, "negLargestDotProduct: ", largest
            print >> output_file, "negSize: ", size

            self.add_data("negInitialVecSize", norm)
            self.add_data("negLargestDotProduct", largest)
            self.add_data("negSize", size)

            if return_vec:
              return cleanResultVector

          return cleanResult


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
          layerA = [self.structuredVectors[start_key]]

          if target_key:
            target_vector = self.structuredVectors[target_key]
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
              key = self.get_key_from_vector(word, self.structuredVectors)
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
                                                            self.idVectors[word]) # build the sentence as an HRR vector
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

  def test_vector(self, vector, target):
    """
    This is used on the hierarchical test to simply decide whether we have found the vector or not
    vector is the vector we are testing
    target is the vector we are comparing to 
    """
    hrr_vec = hrr.HRR(data=vector)
    return hrr_vec.compare(hrr.HRR(data=target)) > self.soft_threshold 
    

  def get_key_from_vector(self, vector, vector_dict, return_comps = False):
      if not self.sufficient_norm(vector):
        return None

      comparisons = self.find_matches(vector, vector_dict)
      max_key, max_val = max(comparisons, key = lambda x: x[1])

      if max_val > self.soft_threshold:
        return max_key
      else:
        return None

  def find_matches(self, vector, vector_dict, exempt=[]):
    hrr_vec = hrr.HRR(data=vector)
    keys = vector_dict.keys()

    for key in vector_dict.keys():
      if key not in exempt:
        yield (key, hrr_vec.compare(hrr.HRR(data=vector_dict[key])))
    
  def getStats(self, cleanResultVector, answer, fp, do_matches=True, threshold = 0.0):

    size = numpy.linalg.norm(cleanResultVector)

    if not answer:
      comparisons = self.find_matches(cleanResultVector, self.structuredVectors)
      largest_match = max(comparisons, key = lambda x: x[1])
      return (largest_match[0], largest_match[1], size)

    else:
      comparisons = self.find_matches(cleanResultVector, self.structuredVectors, exempt=[answer])
    
      second_key, second_match = max(comparisons, key = lambda x: x[1])

      hrr_vec = hrr.HRR(data=self.structuredVectors[answer])
      target_match = hrr_vec.compare(hrr.HRR(data=cleanResultVector))

      if target_match > second_match:
        cleanResult = answer
      else:
        cleanResult = second_key

      return (cleanResult, target_match, second_match, size)


  def openJumpResultsFile(self, mode='w'):
    if not self.jump_results_file:
      self.jump_results_file=open(self.output_dir+'/jump_results_' + self.date_time_string , mode)

  def openHierarchicalResultsFile(self, mode='w'):
    if not self.hierarchical_results_file:
      self.hierarchical_results_file=open(self.output_dir+'/hierarchical_results_' + self.date_time_string , mode)

  def openSentenceResultsFile(self, mode='w'):
    if not self.sentence_results_file:
      self.sentence_results_file=open(self.output_dir+'/sentence_results_' + self.date_time_string , mode)

  def copyFile(self, fp, func=None):
    if func:
      name = fp.name
      mangled_name = name + "_copy"

      fp.close()
      shutil.copyfile(name, mangled_name)
      func('a')

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


  def runBootstrap_hierarchical(self, sample_size, num_trials_per_sample, num_bootstrap_samples=999, stats_depth=0, dataFunc=None):

    file_open_func = self.openHierarchicalResultsFile
    file_open_func()

    htest = lambda x, y, dataFunc=None: self.hierarchicalTest(x,y, stats_depth, rtype=self.isA_symbols, dataFunc=dataFunc)

    self.runBootstrap(sample_size, num_trials_per_sample, num_bootstrap_samples, self.hierarchical_results_file, htest, file_open_func, dataFunc=dataFunc)


  def runBootstrap_sentence(self, sample_size, num_trials_per_sample, num_bootstrap_samples=999, dataFunc=None):

    self.openSentenceResultsFile()

    self.runBootstrap(sample_size, num_trials_per_sample, num_bootstrap_samples, self.sentence_results_file, self.sentenceTest, dataFunc=dataFunc)

  #a general function for adding data to our data object, with arbitrary index depth.
  #note we are always appending to a list. This is the new bookkeeping mechanism.
  #everything just throws the data in a common dictionary, data (its mostly done in testLink)
  #and after every bootstrap run we just bootstrap on each of the named objects in that dictionary

  #assume self.data is  just a flat dictionary pointing to lists of numbers to be bootstrapped
  def add_data(self, index, data):
    if self.bootstrapper:
      self.bootstrapper.add_data(index, data)

#Run a series of bootstrap runs, then combine the success rate from each
#individual run into a total mean success rate with confidence intervals
#dataFunc is a function that takes an associator, and is called after every trial. Allows data about
#the associator on the run to be displayed
  def runBootstrap(self, sample_size, num_trials_per_sample, num_bootstrap_samples, output_file,
      func, statNames=None, file_open_func=None, dataFunc=None):
    start_time = datetime.datetime.now()

    self.bootstrapper = Bootstrapper(self.verbose)

    #Now start running the tests
    self.num_jumps = 0
    output_file.write("Begin series of " + str(sample_size) + " runs, with " + str(num_trials_per_sample) + " trials each.\n")
    self.associator.print_config(output_file)

    for i in range(sample_size):
      output_file.write("Begin run " + str(i + 1) + " out of " + str(sample_size) + ":\n")
      func("", num_trials_per_sample, dataFunc=dataFunc)

      self.print_bootstrap_summary(i + 1, sample_size, output_file)
      output_file.flush()

    self.finish()

    end_time = datetime.datetime.now()
    self.print_bootstrap_runtime_summary(output_file, end_time - start_time)
    self.print_relation_stats(output_file)

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

  def print_bootstrap_summary(self, sample_index, sample_size, output_file):

    title = "Bootstrap Summary"
    self.print_header(output_file, title)
    output_file.write("After " + str(sample_index) + " samples out of " + str(sample_size) + "\n")
    self.bootstrapper.print_summary(output_file)
    self.print_footer(output_file, title)

  def print_bootstrap_runtime_summary(self, output_file, time):
    self.print_header(output_file, "Runtime Summary")
    output_file.write("Total elapsed time for bootstrap runs: " + str(time) + "\n")
    output_file.write("Total num jumps: " + str(self.num_jumps) + "\n")

    if self.num_jumps != 0:
      output_file.write("Average time per jump: " + str(float(time.seconds) / float(self.num_jumps)) + "\n")

    self.print_footer(output_file, "Runtime Summary")


  def print_header(self, output_file, string, char='*', width=15, left_newline=True):
    line = char * width
    string = line + " " + string + " " + line + "\n"

    if left_newline:
      string = "\n" + string

    output_file.write(string)

  def print_footer(self, output_file, string, char='*', width=15):
    self.print_header(output_file, "End " + string, char=char, width=width, left_newline=False)

  #this will show that the keys don't match up. however, that is fixed when the vectors are given to the GPU.
  #we fix it by using the order of they keys in the items for both the indices and the items
  def test_key_integrity(self):
    valid = self.structuredVectors.keys() == self.idVectors.keys()

    if valid:
      print "keys check out"
    else:
      print "somethings wrong"
      for i, pair in enumerate(zip(self.structuredVectors.keys(), self.idVectors.keys())):
        if pair[0] != pair[1]:
          print "bad pair, index: ", i, " pair: ", pair


  def finish(self):
    pass


