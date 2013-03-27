from vector_operations import *
import symbol_definitions

import bootstrap

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
  def __init__(self, corpus, idVectors, structuredVectors, relation_symbols, associator, vector_indexing, cleanLevel = 0.3, seed=None, output_dir=".", isA_symbols = [], sentence_symbols = []):

        self.num_jumps = 0

        self.output_dir = output_dir

        date_time_string = str(datetime.datetime.now())
        self.date_time_string = reduce(lambda y,z: string.replace(y,z,"_"), [date_time_string,":","."," ","-"])

        self.sentence_results_file=None
        self.jump_results_file=None
        self.hierarchical_results_file=None

        if seed is not None:
          random.seed(seed)
          numpy.random.seed(seed)

        self.associator = associator
        self.vector_indexing =vector_indexing
        self.corpus = corpus
        self.idVectors = idVectors
        self.structuredVectors = structuredVectors
        self.vocab = [idVectors]
        self.sentenceVocab = None
        self.relation_symbols = relation_symbols
        self.isA_symbols = isA_symbols
        self.sentence_symbols = sentence_symbols
        self.data = None

        self.key_indices = {}
        i = 0
        for key in self.structuredVectors.keys():
          self.key_indices[key] = i
          i += 1

        self.D = len(idVectors.values()[0])
        print "self.D : ", self.D

  def loadVocab(self, filename):
        with open(filename, 'r') as vfile:
            self.vocab = pickle.load(vfile)

  def saveVocab(self, filename):
        with open(filename, 'w') as vfile:
            pickle.dump(self.vocab, vfile)

  def setupRandomVocab(self, numVocab):
        """split knowledge base into distinct vocabularies (randomly)"""
        keys = self.idVectors.keys()
        if numVocab > len(keys):
            raise Exception('Number of vocabularies outstrips number of vectors!')
        random.shuffle(keys)
        vocabSize = int(len(keys)/numVocab)
        vocabRemainder = int(len(keys)%numVocab)
        vocab = [keys[i*vocabSize:(i+1)*vocabSize] for i in range(numVocab)]
        for i in range(vocabRemainder):
            vocab[i].append(keys[numVocab*vocabSize+i])
        self.vocab = vocab

  def unbind_and_associate(self, item, query, return_as_vector, *args, **kwargs):
      self.num_jumps += 1
      result = self.associator.unbind_and_associate(item, query, *args, **kwargs)

      if not return_as_vector:
        result = [self.get_key_from_vector(r, self.structuredVectors) for r in result]
        result = filter(None, result)

      return result

  def sufficient_norm(self, vector):
      return numpy.linalg.norm(vector) >= 0.1

  def get_key_from_vector(self, vector, vector_dict):
      if not self.sufficient_norm(vector):
        return None

      hrr_vec = hrr.HRR(data=vector)
      comparisons=[(key, hrr_vec.compare(hrr.HRR(data=vector_dict[key]))) for key in vector_dict.keys()]

      max_key = comparisons[0][0]
      max_val = comparisons[0][1]

      for pair in comparisons:
        if pair[1] > max_val:
          max_val = pair[1]
          max_key = pair[0]

      if max_val > 0.8:
        return max_key
      else:
        return None


  def testLink(self, word, relation, goal=None, output_file = None, start_from_vec=False, return_vec=False, relation_is_vec=False, answers=[], num_relations = -1, depth=0):

        self.print_header(output_file, "Testing link", char='-')

        if start_from_vec:
          word_key = self.get_key_from_vector(word, self.structuredVectors)
          print >> output_file, "start :", word_key
        else:
          print >> output_file, "start: ", word
          word_key = word
          word = self.structuredVectors[word]

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

        #cleanResultVectors = self.unbind_and_associate(word, relation, True, urn_agreement=goal)
        cleanResultVectors = self.unbind_and_associate(word, relation, True)

        if goal:
          cleanResult, target_match, second_match, size = self.getStats(cleanResultVectors, goal, output_file)

          print >> output_file, "target match: ", target_match
          print >> output_file, "second match : ", second_match
          print >> output_file, "size: ", size
          print >> output_file, "num_relations: ", num_relations

          self.add_data("depth_"+str(depth)+"_target_match", target_match)
          self.add_data("depth_"+str(depth)+"_second_match", second_match)
          self.add_data("depth_"+str(depth)+"_size", size)

          if num_relations > 0:
            self.add_data("rel_"+str(num_relations)+"_target_match", target_match)
            self.add_data("rel_"+str(num_relations)+"_second_match", second_match)
            self.add_data("rel_"+str(num_relations)+"_size", size)

          if return_vec:
            return cleanResultVectors
          else:
            if answers:
              validAnswers = all([r in answers for r in cleanResult])
              exactGoal = target_match > second_match

              return (cleanResult, validAnswers, exactGoal)
            else:
              return cleanResult

        else:
          largest, size = self.getStats(cleanResultVectors, None, self.hierarchical_results_file)
          norm = numpy.linalg.norm(word)

          print >> output_file, "negInitialVecSize: ", norm
          print >> output_file, "negLargestDotProduct: ", largest
          print >> output_file, "negSize: ", size

          self.add_data("negInitialVecSize", norm)
          self.add_data("negLargestDotProduct", largest)
          self.add_data("negSize", size)

          return cleanResultVectors


  def jumpTest(self, testName, n, dataFunc=None, *args, **kwargs):
        # select a key, follow a hyp/hol link, record success / failure

        testNumber = 0

        score = 0
        exactGoals = 0

        planned_words = []

        if "planned_words" in kwargs:
          planned_words = kwargs["planned_words"]

        relation_indices = []
        if "planned_relations" in kwargs:
          relation_indices = kwargs["relation_indices"]

        while testNumber < n:
            if testNumber < len(planned_words):
              words = planned_words[testNumber: min(n, len(planned_words))]
            else:
              words = random.sample(self.corpus, n-testNumber)

            for word in words:
                testableLinks = [r for r in self.corpus[word] if r[0] in self.relation_symbols]

                if len(testableLinks) > 0:
                    if testNumber < len(relation_indices):
                      prompt = testableLinks[relation_indices[testNumber]]
                    else:
                      prompt = random.sample(testableLinks, 1)[0]

                    self.print_header(self.jump_results_file, "New Jump Test")

                    answers = [r[1] for r in self.corpus[word] if r[0]==prompt[0]]

                    (result, valid, exact) = self.testLink(word, prompt[0], prompt[1], self.jump_results_file, num_relations = len(testableLinks), answers=answers)

                    print >> sys.stderr, "Valid answers? ",valid
                    print >> sys.stderr, "Exact goal? ",exact
                    print >> self.jump_results_file, "Valid answers? ",valid
                    print >> self.jump_results_file, "Exact goal? ",exact

                    if dataFunc:
                      dataFunc(self.associator)

                    testNumber += 1

                    if valid: score += 1

                    if exact: exactGoals += 1

        # print the score
        title = "Jump Test Summary"
        self.print_header(self.jump_results_file, title)
        self.jump_results_file.write("score,"+str(score)+":\n")
        self.jump_results_file.write("totaltests,"+str(testNumber)+":\n")
        self.print_footer(self.jump_results_file, title)

        valid_score = float(score) / float(testNumber)
        exact_score = float(exactGoals) / float(testNumber)

        print "score,"+str(valid_score)

        self.add_data("jump_score_valid", valid_score)
        self.add_data("jump_score_exact", exact_score)

  def hierarchicalTest(self, testName, n, stat_depth = 0, m=None, rtype=[], startFromParent=False, dataFunc=None, *args, **kwargs):
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
          start = random.sample(self.corpus, 1)[0]
          target = random.sample(self.corpus, 1)[0]

          parent_list = self.findAllParents(start, None, rtype, False, stat_depth=0, print_output=False, *args, **kwargs)

          pair = (start, target)
          if target in parent_list and m_count < m:
            positive_pairs.append(pair)
            m_count += 1
          elif not (target in parent_list):
            negative_pairs.append(pair)
            n_count += 1

        while m_count < m:
          start = random.sample(self.corpus, 1)[0]
          parent_list = self.findAllParents(start, None, rtype, False, stat_depth=0, print_output=False, *args, **kwargs)

          if len(parent_list) == 0: continue

          target = random.sample(parent_list, 1)[0]
          positive_pairs.append((start, target))
          m_count += 1

        #now run the tests!
        title = "New Hierarchical Test - Negative"
        for pair in negative_pairs:
          self.print_header(self.hierarchical_results_file, title)

          #for printing
          self.findAllParents(pair[0], pair[1], rtype, False, stat_depth=stat_depth, print_output=True, *args, **kwargs)
          result = self.findAllParents(pair[0], pair[1], rtype, True, stat_depth=stat_depth, print_output=True, *args, **kwargs)
          if result == -1: n_score += 1

        title = "New Hierarchical Test - Positive"
        for pair in positive_pairs:
          self.print_header(self.hierarchical_results_file, title)

          self.findAllParents(pair[0], pair[1], rtype, False, stat_depth=stat_depth, print_output=True, *args, **kwargs)
          result = self.findAllParents(pair[0], pair[1], rtype, True, stat_depth=stat_depth, print_output=True, *args, **kwargs)
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

  def findAllParents(self, start_key, target_key=None, rtype=[], use_HRR=False, stat_depth=0, print_output = False, *arg, **kwargs):

        if print_output:
          print >> self.hierarchical_results_file, "In find all parents, useHRR=", use_HRR
          print >> self.hierarchical_results_file, "Start:", start_key

          if target_key is not None:
            print >> self.hierarchical_results_file, "Target:", target_key


        level = 0
        if use_HRR:
          layerA = [self.structuredVectors[start_key]]
        else:
          layerA = [start_key]

        layerB = []
        parents = set()

        while len(layerA) > 0:
            word = layerA.pop()

            if use_HRR:
              key = self.get_key_from_vector(word, self.structuredVectors)
            else:
              key = word

            if key:
              if key in parents: continue
              if level > 0:
                parents.add(key)

                if print_output:
                  print >> self.hierarchical_results_file, key, "found at level ", level

                if target_key and target_key == key:
                  return level

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

                results = self.testLink(word, symbol, target, self.hierarchical_results_file, start_from_vec=True, return_vec=True, depth=level, num_relations=num_relations, answers=answers)

                links.extend( filter(self.sufficient_norm, results) )


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

  def sentenceTest(self, testName, n, dataFunc=None, *args, **kwargs):
        # check that POS lists exist (form them if required)
        if self.sentenceVocab is None:
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
            self.sentenceVocab = {}
            for symbol in self.sentence_symbols:
                self.sentenceVocab[symbol] = genVec(self.D)

        posmap = {'n':self.nouns, 'a':self.adjectives, 'r':self.adverbs, 'v':self.verbs}

        score = 0

        for i in range(n):
            title = "New Sentence Test"
            self.print_header(self.sentence_results_file, title)
            sentence = {}
            sentenceVector = numpy.zeros(self.D)

            print >> self.sentence_results_file, "Roles in sentence:"
            for symbol in self.sentence_symbols:

                if random.random() < self.sentence_symbols[symbol][0]: # choose lexical items to include
                    print >> self.sentence_results_file, symbol
                    pos = self.sentence_symbols[symbol][1] # determine the POS for this lexical item
                    word = (pos, random.sample(posmap[pos], 1)[0]) # choose words
                    sentence[symbol] = word    # build the sentence in a python dictionary
                    sentenceVector = sentenceVector + cconv(self.sentenceVocab[symbol],
                                                            self.idVectors[word]) # build the sentence as an HRR vector
            # ask about parts of the sentence
            sentence_score = 0
            for symbol in sentence.keys():

                answer = sentence[symbol]

                print >> self.sentence_results_file, "Testing ", symbol
                result, valid, exact = self.testLink(sentenceVector, self.sentenceVocab[symbol], answer,
                    output_file = self.sentence_results_file, return_vec=False,
                    start_from_vec=True, relation_is_vec=True, num_relations=len(sentence), answers=[answer])

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

  def getStats(self, cleanResultVectors, answer, fp):

    cleanResult = [self.get_key_from_vector(v, self.structuredVectors) for v in cleanResultVectors]

    size = numpy.linalg.norm(cleanResultVectors[0])
    matches = [(key, hrr.HRR(data=cleanResultVectors[0]).compare(hrr.HRR(data=self.structuredVectors[key]))) for key in self.structuredVectors.keys()]

    largest = heapq.nlargest(2, matches, key = lambda x: x[1])

    if not answer:
      return (largest[0][1], size)

    target_match = None
    second_match = None
    if largest[0][0] == answer:
      target_match = largest[0][1]
      second_match = largest[1][1]
    else:
      target_match = matches[ self.key_indices[answer] ][1]
      second_match = largest[0][1]

    print >> sys.stderr, "Guess keys: ", cleanResult
    print >> fp, "Guess keys: ", cleanResult

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

  def runBootstrap_jump(self, sample_size, num_trials_per_sample, num_bootstrap_samples=999, dataFunc=None, *args, **kwargs):

    self.openJumpResultsFile()

    self.runBootstrap(sample_size, num_trials_per_sample, num_bootstrap_samples, self.jump_results_file, self.jumpTest, dataFunc=dataFunc, *args, **kwargs)


  def runBootstrap_hierarchical(self, sample_size, num_trials_per_sample, num_bootstrap_samples=999, stats_depth=0, dataFunc=None, *args, **kwargs):

    file_open_func = self.openHierarchicalResultsFile
    file_open_func()

    htest = lambda x, y, dataFunc=None, *args, **kwargs: self.hierarchicalTest(x,y, stats_depth, rtype=self.isA_symbols, dataFunc=dataFunc, *args, **kwargs)

    self.runBootstrap(sample_size, num_trials_per_sample, num_bootstrap_samples, self.hierarchical_results_file, htest, file_open_func, dataFunc=dataFunc, *args, **kwargs)


  def runBootstrap_sentence(self, sample_size, num_trials_per_sample, num_bootstrap_samples=999, dataFunc=None, *args, **kwargs):

    self.openSentenceResultsFile()

    self.runBootstrap(sample_size, num_trials_per_sample, num_bootstrap_samples, self.sentence_results_file, self.sentenceTest, dataFunc=dataFunc, *args, **kwargs)

  #a general function for adding data to our data object, with arbitrary index depth.
  #note we are always appending to a list. This is the new bookkeeping mechanism.
  #everything just throws the data in a common dictionary, data (its mostly done in testLink)
  #and after every bootstrap run we just bootstrap on each of the named objects in that dictionary

  #assume self.data is  just a flat dictionary pointing to lists of numbers to be bootstrapped
  def add_data(self, index, data):
    if not (index in self.data):
      self.data[index] = []

    self.data[index].append(data)

#Run a series of bootstrap runs, then combine the success rate from each
#individual run into a total mean success rate with confidence intervals
#dataFunc is a function that takes an associator, and is called after every trial. Allows data about
#the associator on the run to be displayed
  def runBootstrap(self, sample_size, num_trials_per_sample, num_bootstrap_samples, output_file,
      func, statNames=None, file_open_func=None, dataFunc=None, *args, **kwargs):
    start_time = datetime.datetime.now()

    self.data = {}

    #Now start running the tests
    self.num_jumps = 0
    output_file.write("Begin series of " + str(sample_size) + " runs, with " + str(num_trials_per_sample) + " trials each.\n")

    for i in range(sample_size):
      output_file.write("Begin run " + str(i + 1) + " out of " + str(sample_size) + ":\n")
      func("", num_trials_per_sample, dataFunc=dataFunc, *args, **kwargs)

      self.print_bootstrap_summary(i + 1, sample_size, output_file)
      output_file.flush()

    self.finish()

    end_time = datetime.datetime.now()
    self.print_bootstrap_runtime_summary(output_file, end_time - start_time)
    self.print_relation_stats(output_file, **kwargs)

  def print_relation_stats(self, output_file, **kwargs):
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
    mean = lambda x: float(sum(x)) / float(len(x))

    title = "Bootstrap Summary"
    self.print_header(output_file, title)
    
    output_file.write("After " + str(sample_index) + " samples out of " + str(sample_size) + "\n")

    data_keys = self.data.keys()
    data_keys.sort()

    for n in data_keys:
      s = self.data[n]
      CI = bootstrap.bootstrap_CI(0.05, mean, s, 999)

      output_file.write("\nmean " + n + ": " + str(mean(s)) + "\n")
      output_file.write("lower 95% CI bound: " + str(CI[0]) + "\n")
      output_file.write("upper 95% CI bound: " + str(CI[1]) + "\n")
      output_file.write("raw data: " + str(s) + "\n")

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


