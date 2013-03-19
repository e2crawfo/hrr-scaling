from SymbolDefinitions import *
from VectorOperations import *

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
  def __init__(self, corpus, idVectors, structuredVectors, associator, vector_indexing, cleanLevel = 0.3, seed=None, output_dir="."):
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

  def get_key_from_vector(self, vector, vector_dict):
      if numpy.linalg.norm(vector) < 0.1:
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

  def testLink(self, word, relation, goal):
        allAnswers = [r[1] for r in self.corpus[word] if r[0]==relation]
        #result = cconv(self.knowledgeBase[word], pInv(self.knowledgeBase[prompt[0]]))

        print >> self.jump_results_file, "************New Jump test*************"
        print >> self.jump_results_file, "word relations: ", [r for r in self.corpus[word]]
        print >> self.jump_results_file, "current relation: ", relation
        print >> self.jump_results_file, "target index"
        print >> self.jump_results_file, self.key_indices[goal]

        print >> sys.stderr, "Printing answers:"

        for ans in allAnswers:
          print >> sys.stderr, ans
          print >> sys.stderr, "index:" + str(self.key_indices[ans]) + ", final index is " + str(len(self.structuredVectors) - 1)

        print >> sys.stderr, "Done printing answers"

        cleanResultVectors = self.unbind_and_associate(self.structuredVectors[word], self.idVectors[relation], True, urn_agreement=goal)
        target_match, second_match, size = self.getStats(cleanResultVectors, goal, self.jump_results_file)

        print >> self.jump_results_file, "target match: ", target_match
        print >> self.jump_results_file, "second match : ", second_match
        print >> self.jump_results_file, "size: ", size

        cleanResult = [self.get_key_from_vector(vec, self.structuredVectors) for vec in cleanResultVectors]
        if len(cleanResult) == 0:
          return (False, target_match, second_match, size, False)

        #validAnswers is whether the results we returned were all valid answers
        validAnswers = all([r in allAnswers for r in cleanResult])

        #exact goal is whether the goal answer was in the results we returned  
        exactGoal = target_match > second_match

        print >> sys.stderr, "Valid answers? ",validAnswers
        print >> sys.stderr, "Exact goal? ",exactGoal
        print >> self.jump_results_file, "Valid answers? ",validAnswers
        print >> self.jump_results_file, "Exact goal? ",exactGoal

        return (validAnswers, target_match, second_match, size, exactGoal)
    
  def jumpTest(self, testName, n, dataFunc=None, *args, **kwargs):
        # select a key, follow a hyp/hol link, record success / failure

        testNumber = 0
        score = 0

        target_matches = []
        second_matches = []
        sizes = []
        exactGoals = 0

        planned_words = [] 
        
        if "planned_words" in kwargs:
          planned_words = kwargs["planned_words"]
        
        relation_indices = []
        if "relation_indices" in kwargs:
          relation_indices = kwargs["relation_indices"]

        relation_stats = {}
        if "relation_stats" in kwargs:
          relation_stats = kwargs["relation_stats"]

        while testNumber < n:
            if testNumber < len(planned_words):
              words = planned_words[testNumber: min(n, len(planned_words))]
            else:
              words = random.sample(self.corpus, n-testNumber)
             
            for word in words:
                testableLinks = [r for r in self.corpus[word] if r[0] in vocab_symbols]

                if len(testableLinks) > 0:
                    if testNumber < len(relation_indices):
                      prompt = testableLinks[relation_indices[testNumber]]
                    else:
                      prompt = random.sample(testableLinks, 1)[0]

                    result = self.testLink(word, prompt[0], prompt[1])
                    if dataFunc:
                      dataFunc(self.associator)

                    testNumber = testNumber + 1

                    success = result[0]
                    if success: score = score + 1
                    
                    exactGoal = result[4]
                    if exactGoal:
                      exactGoals += 1

                    target_matches.append(result[1])
                    second_matches.append(result[2])
                    sizes.append(result[3])

                    if relation_stats:
                      relation_stats[len(testableLinks)][0].append(result[1])
                      relation_stats[len(testableLinks)][1].append(result[2])
                      relation_stats[len(testableLinks)][2].append(result[3])
                      relation_stats[len(testableLinks)][3] += 1

        # print the score
        print >> self.jump_results_file, "************ Jump test summary *************"
        self.jump_results_file.write("score,"+str(score)+":\n")
        self.jump_results_file.write("totaltests,"+str(testNumber)+":\n")

        print "Start trial:\n"
        print "score,"+str(score)+":\n"
        print "totaltests,"+str(testNumber)+":\n"

        #print testName, 'results:'
        #print score, "links successful out of", testNumber, "tests."
        return [[float(score) / float(testNumber)], target_matches, second_matches, sizes, [float(exactGoals) / float(testNumber)]]
    
  def hierarchicalTest(self, testName, n, stat_depth = 0, m=None, rtype=isA_symbols, startFromParent=False, dataFunc=None, *args, **kwargs):
        if m is None:
          m = n
        """Check whether word A is a type of word B. Test with n cases in which
        word A IS NOT a descendant of word B and m cases where word A IS a
        descendent of word B. The rtype parameter specifies which relationships
        to use in the search (by default, only the isA relationships)."""
        n_count = 0
        m_count = 0
        p_score = 0
        n_score = 0
        
        totalTargetMatches = [[] for i in range(stat_depth)]
        totalSecondMatches = [[] for i in range(stat_depth)]
        totalSizes = [[] for i in range(stat_depth)]

        negInitialVecSize = []
        negLargestDotProduct = []
        negSizes = []


        # choose pairs of words randomly, test heritage
        while n_count < n:
            samples_A = random.sample(self.corpus, n-n_count)
            samples_B = random.sample(self.corpus, n-n_count)
            for i in range(len(samples_A)):
                word = samples_A[i]
                target = samples_B[i]


                print >> self.hierarchical_results_file, "************New hierarchical test*************"

                parentList = self.findAllParents(word, rtype, stat_depth=stat_depth)[0]
                if target in parentList and m_count < m:
                    print >> self.hierarchical_results_file, "   *****Positive test*****"

                    m_count = m_count + 1
                    if startFromParent:
                        result = self.findAllParents(target, rtype, True, word, True, stat_depth=stat_depth)[0]
                    else:
                        result = self.findAllParents(word, rtype, True, target, stat_depth=stat_depth, *args, **kwargs)
                        target_matches = result[4]
                        second_matches = result[5]
                        sizes = result[6]

                        negInitialVecSize.extend(result[1])
                        negLargestDotProduct.extend(result[2])
                        negSizes.extend(result[3])

                        result = result[0]

                        for i, l in enumerate(target_matches):
                          totalTargetMatches[i].extend( l )

                        for i, l in enumerate(second_matches):
                          totalSecondMatches[i].extend( l )

                        for i, l in enumerate(sizes):
                          totalSizes[i].extend( l )

                    if result > -1: p_score = p_score + 1
                elif target not in parentList:
                    print >> self.hierarchical_results_file, "   *****Negative test*****"

                    n_count = n_count + 1
                    if startFromParent:
                        result = self.findAllParents(target, rtype, True, word, True, stat_depth=stat_depth)[0]
                    else:
                        result = self.findAllParents(word, rtype, True, target, stat_depth=stat_depth, *args, **kwargs)
                        target_matches = result[4]
                        second_matches = result[5]
                        sizes = result[6]

                        negInitialVecSize.extend(result[1])
                        negLargestDotProduct.extend(result[2])
                        negSizes.extend(result[3])

                        result = result[0]

                        for i, l in enumerate(target_matches):
                          totalTargetMatches[i].extend( l )

                        for i, l in enumerate(second_matches):
                          totalSecondMatches[i].extend( l )

                        for i, l in enumerate(sizes):
                          totalSizes[i].extend( l )

                    if result > -1: n_score = n_score + 1
                else:
                    print >> self.hierarchical_results_file, "************Hierarchical test aborted*************"

        # choose single words randomly and select parents to fill test quota
        while m_count < m:
            words = random.sample(self.corpus, m-m_count)
            for word in words:

                print >> self.hierarchical_results_file, "************New hierarchical test*************"
                
                parentList = self.findAllParents(word, rtype, stat_depth=stat_depth)[0]
                if len(parentList) > 0:
                    print >> self.hierarchical_results_file, "   *****Positive test*****"

                    m_count = m_count + 1
                    target = random.sample(parentList, 1)[0]
                    #print >> self.hierarchical_results_file, "Start word: ", word, ", index: ", self.key_indices[word]
                    #print >> self.hierarchical_results_file, "Target word: ", target, ", index: ", self.key_indices[target]
                    if startFromParent:
                        result = self.findAllParents(target, rtype, True, word, True, stat_depth = stat_depth)[0]
                    else:
                        result = self.findAllParents(word, rtype, True, target, stat_depth=stat_depth, *args, **kwargs)
                        target_matches = result[4]
                        second_matches = result[5]
                        sizes = result[6]

                        negInitialVecSize.extend(result[1])
                        negLargestDotProduct.extend(result[2])
                        negSizes.extend(result[3])

                        result = result[0]

                        for i, l in enumerate(target_matches):
                          totalTargetMatches[i].extend( l )

                        for i, l in enumerate(second_matches):
                          totalSecondMatches[i].extend( l )

                        for i, l in enumerate(sizes):
                          totalSizes[i].extend( l )

                    if result > -1: p_score = p_score + 1

                else:
                    print >> self.hierarchical_results_file, "************Hierarchical test aborted*************"
          
        # print the score
        print >> self.hierarchical_results_file, "************ Hierarchical test summary *************"
        self.hierarchical_results_file.write("Start trial:\n")
        self.hierarchical_results_file.write("FP,"+str(n_score)+"\n")#false positive
        self.hierarchical_results_file.write("CR,"+str(n-n_score)+"\n")#correct rejections
        self.hierarchical_results_file.write("hits,"+str(p_score)+"\n")
        self.hierarchical_results_file.write("misses,"+str(m-p_score)+"\n")
        self.hierarchical_results_file.write("TS,"+str(n-n_score+p_score)+" out of "+str(n+m)+"\n")#successful tests, out of total
        self.hierarchical_results_file.write("NT,"+str(n)+"\n")#neg tests
        self.hierarchical_results_file.write("PT,"+str(m)+"\n")#pos tests

        print "Start trial:\n"
        print "FP,"+str(n_score)+"\n"#false positive
        print "CR,"+str(n-n_score)+"\n"#correct negative
        print "hits,"+str(p_score)+"\n"#correct positive
        print "misses,"+str(m-p_score)+"\n"#false negative
        print "TS,"+str(n-n_score+p_score)+" out of "+str(n+m)+"\n"#successful tests, out of total
        print "NT,"+str(n)+"\n"#neg tests
        print "PT,"+str(m)+"\n"#pos tests

        result = [[float(n-n_score + p_score) / float(m + n)], negInitialVecSize, negLargestDotProduct, negSizes] 
        for i in range(stat_depth):
          result.append(totalTargetMatches[i])
          result.append(totalSecondMatches[i])
          result.append(totalSizes[i])

        return result


  def findAllParents(self, word, rtype=isA_symbols, useHRR=False, target=None, findChildren=False, stat_depth = 0, *arg, **kwargs):

        print >> self.hierarchical_results_file, "In find all parents, useHRR=", useHRR

        print >> self.hierarchical_results_file, "Start:", word, ", index: ", self.key_indices[word]
        #if self.vector_indexing and useHRR:
        if target is not None:
          print >> self.hierarchical_results_file, "Target:", target, ", index: ", self.key_indices[target]

    #this outputs a list of keys, regardless of whether its in vector_indexing mode or not
    
        level = 0
        if self.vector_indexing and useHRR:
          layerA = [self.structuredVectors[word]]
        else:
          layerA = [word]

        layerB = []
        parents = set()

        relation_stats = {} 
        if "relation_stats" in kwargs:
          relation_stats = kwargs["relation_stats"]	

        target_matches = [[] for i in range(stat_depth)]
        second_matches = [[] for i in range(stat_depth)]
        sizes = [[] for i in range(stat_depth)]

        #for links that shouldnt be successful aren't in wordNet)
        negInitialVecSize = []
        negLargestDotProduct = []
        negSizes = []

        #print >> self.hierarchical_results_file, "Start word: ", word, ", index: ", self.key_indices[word]

        while len(layerA) > 0:
            word = layerA.pop()
            
            # add word to parents (unless it's the initial word)
            # find the parents of this word
            if useHRR:
                
                if self.vector_indexing:
                  print >> self.hierarchical_results_file, "Norm of link: ", numpy.linalg.norm(word)
                  key=self.get_key_from_vector(word, self.structuredVectors)
                  if key is not None and key in parents: continue
                  if key is not None and level > 0:
                    parents.add(key)
                  #in the vector_indexing case, word will be an actual vector
                  #only temporary, to see if it works in principle
                  #vector = self.structuredVectors[key]
                  vector = word

                else:
                  #in the non-vector_indexing(autovector_indexing) case, word is a key that 
                  #indexes both structuredVectors and idVectors
                  if word in parents: continue
                  if level > 0:
                    parents.add(word)
                  vector = self.structuredVectors[word]

                links = []
                for symbol in rtype:
                  #have to fix this
                  results = self.unbind_and_associate(word, self.idVectors[symbol], True)
                  word_key = self.get_key_from_vector(word, self.structuredVectors)
                  answers = [r[1] for r in self.corpus[word_key] if r[0] == symbol]

                  if len(answers) > 0:
                    if level < stat_depth:
                      target_match, second_match, size = self.getStats(results, answers[0], self.hierarchical_results_file)

                      target_matches[level].append(target_match)
                      second_matches[level].append(second_match)
                      sizes[level].append(size)

                      index = len([r[1] for r in self.corpus[word_key] if r[0] in vocab_symbols])
                      if relation_stats:
                        relation_stats[index][0].append(target_match)
                        relation_stats[index][1].append(second_matche)
                        relation_stats[index][2].append(size)
                        relation_stats[index][3] += 1
                  else:
                    [largest, size] = self.getStats(results, None, self.hierarchical_results_file)

                    negInitialVecSize.append(numpy.linalg.norm(word))
                    negLargestDotProduct.append(largest)
                    negSizes.append(size)

                  #results = self.unbind_and_associate(vector, self.idVectors[symbol], self.vector_indexing)
                  links.extend(copy.deepcopy(results))

            else:
                if word in parents: continue
                if level > 0:
                  parents.add(word)
                links = [r[1] for r in self.corpus[word] if r[0] in rtype]

            #remove any links that are not similar enough to any vectors in the vocab
            #mainly for vectors with 0 norm
            #this step is sort of worrisome, shouldn't really be filtering stuff out like this...though it should be simple enough to neurally filter out things that are too small
            if self.vector_indexing and useHRR:

              adjusted_links=[]
              keys=[]
              for link in links:
                key=self.get_key_from_vector(link, self.structuredVectors)

                if key is not None:
                  adjusted_links.append(link)
                  keys.append(key)
              
              if len(adjusted_links) > 0:
                layerB.extend(adjusted_links)

              links=keys
            else:
              # add the parent words to layer B
              if len(links) > 0:
                  layerB.extend(links)

            #turn the links into keys for the sake of giving feedback on screen

            for link in links:
              print >> self.hierarchical_results_file, link, "found at level ", level + 1, ", index: ", self.key_indices[link]
            
            if len(layerA)==0:
                level = level + 1
                layerA = layerB
                layerB = []

            # return the level at which the target was found, if it was found
            if target is not None and target in links:
                return [level, negInitialVecSize, negLargestDotProduct, negSizes, target_matches, second_matches, sizes]

        # return the list of parents
        if target is None:
            return [parents]

        # ...or return -1 if a target was specified (and not found)
        else:
            return [-1, negInitialVecSize, negLargestDotProduct, negSizes, target_matches, second_matches, sizes]

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
            for symbol in sentence_symbols:
                self.sentenceVocab[symbol] = genVec(self.D)

        posmap = {'n':self.nouns, 'a':self.adjectives, 'r':self.adverbs, 'v':self.verbs}

        score = 0
        target_matches = []
        second_matches = []
        sizes = []
        
        for i in range(n):
            print >> self.sentence_results_file, "************New sentence test*************"
            sentence = {}
            sentenceVector = numpy.zeros(self.D)

            print >> self.sentence_results_file, "Roles in sentence:"
            for symbol in sentence_symbols:

                if random.random() < sentence_symbols[symbol][0]: # choose lexical items to include
                    print >> self.sentence_results_file, symbol
                    pos = sentence_symbols[symbol][1] # determine the POS for this lexical item
                    word = (pos, random.sample(posmap[pos], 1)[0]) # choose words
                    sentence[symbol] = word    # build the sentence in a python dictionary
                    sentenceVector = sentenceVector + cconv(self.sentenceVocab[symbol],
                                                            self.idVectors[word]) # build the sentence as an HRR vector
            # ask about parts of the sentence
            sentence_score = 0
            for symbol in sentence.keys():

                answer = sentence[symbol]

                print >> self.sentence_results_file, "Testing ", symbol

                cleanResultVectors = self.unbind_and_associate(sentenceVector, self.sentenceVocab[symbol], True)

                target_match, second_match, size = self.getStats(cleanResultVectors, answer, self.sentence_results_file)

                target_matches.append(target_match)
                second_matches.append(second_match)
                sizes.append(size)

                print >> sys.stderr, "Guess keys: ", cleanResult
                print >> sys.stderr, "Guess indices: ", [self.key_indices.get(key) for key in cleanResult]
                print >> self.sentence_results_file, "Guess keys: ", cleanResult
                print >> self.sentence_results_file, "Guess indices: ", [self.key_indices.get(key) for key in cleanResult]

                #query = pInv(self.sentenceVocab[symbol])
                #result = self.cleanup(cconv(sentenceVector, query), self.cleanupMemory.keys())
                if len(cleanResult)>0 and cleanResult[0] == answer:
                    sentence_score = sentence_score + 1
                    print >> self.sentence_results_file, "correct!"
                else:
                    print >> self.sentence_results_file, "incorrect!"

            sentence_percent = float(sentence_score) / float(len(sentence))
            print >> self.sentence_results_file, "Percent correct for current sentence: " 
            score = score + sentence_percent

        print "sentence test score:", score, "out of", n

        percent = float(score) / float(n)
        print >> self.sentence_results_file, "************ Sentence test summary *************"
        print >> self.sentence_results_file, "Correct: ", score
        print >> self.sentence_results_file, "Total: ", n
        print >> self.sentence_results_file, "Percent: ", percent

        return [[percent], target_matches, second_matches, sizes]

  def getStats(self, cleanResultVectors, answer, fp):

    cleanResult = [self.get_key_from_vector(v, self.structuredVectors) for v in cleanResultVectors]

    size = numpy.linalg.norm(cleanResultVectors[0])
    matches = [(key, hrr.HRR(data=cleanResultVectors[0]).compare(hrr.HRR(data=self.structuredVectors[key]))) for key in self.structuredVectors.keys()]

    largest = heapq.nlargest(2, matches, key = lambda x: x[1])

    if not answer:
      return [largest[0][1], size]

    target_match = None
    second_match = None
    if largest[0][0] == answer:
      target_match = largest[0][1]
      second_match = largest[1][1]
    else:
      target_match = matches[ self.key_indices[answer] ][1]
      second_match = largest[0][1]

    print >> sys.stderr, "Guess keys: ", cleanResult
    print >> sys.stderr, "Guess indices: ", [self.key_indices.get(key) for key in cleanResult]
    print >> fp, "Guess keys: ", cleanResult
    print >> fp, "Guess indices: ", [self.key_indices.get(key) for key in cleanResult]

    return [target_match, second_match, size]

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

    self.runBootstrap(sample_size, num_trials_per_sample, num_bootstrap_samples, self.jump_results_file, self.jumpTest, ["score", "target dot product", "largest non-target dot product", "norm", "exactGoal"], dataFunc=dataFunc, *args, **kwargs)


  def runBootstrap_hierarchical(self, sample_size, num_trials_per_sample, num_bootstrap_samples=999, stats_depth=0, dataFunc=None, *args, **kwargs):

    file_open_func = self.openHierarchicalResultsFile
    file_open_func()

    htest = lambda x, y: self.hierarchicalTest(x,y, stats_depth)
    s = lambda x: [x + "target dot product", x + "largest non-target dot product", x + "norm"]
    strings = [ s( str(i + 1) + " ") for i in range(stats_depth)]
    strings = [i for l in strings for i in l]
    strings = ["score", "negInitialVecSize", "negLargestDotProduct", "negSizes"] + strings

    self.runBootstrap(sample_size, num_trials_per_sample, num_bootstrap_samples, self.hierarchical_results_file, htest, strings, file_open_func, dataFunc=dataFunc, *args, **kwargs)


  def runBootstrap_sentence(self, sample_size, num_trials_per_sample, num_bootstrap_samples=999, dataFunc=None, *args, **kwargs):

    self.openSentenceResultsFile()

    self.runBootstrap(sample_size, num_trials_per_sample, num_bootstrap_samples, self.sentence_results_file, self.sentenceTest, ["score", "target dot product", "largest non-target dot product", "norm"], dataFunc=dataFunc, *args, **kwargs)


#Run a series of bootstrap runs, then combine the success rate from each
#individual run into a total mean success rate with confidence intervals
#dataFunc is a function that takes an associator, and is called after every trial. Allows data about
#the associator on the run to be displayed
  def runBootstrap(self, sample_size, num_trials_per_sample, num_bootstrap_samples, output_file,
      func, statNames=None, file_open_func=None, dataFunc=None, *args, **kwargs):
    start_time = datetime.datetime.now()

    relation_counts = {}
    relation_count = 0
    relation_hist = {}

    for key in self.corpus:
      if not len(self.corpus[key]) in relation_hist:
        relation_hist[len(self.corpus[key])] = 1
      else:
        relation_hist[len(self.corpus[key])] += 1

      for relation in self.corpus[key]:
        if relation[0] not in vocab_symbols: continue

        relation_count += 1
        if not relation[0] in relation_counts:
          relation_counts[relation[0]] = 1
        else:
          relation_counts[relation[0]] += 1

    output_file.write("************ Relation stats *************\n")
    output_file.write("relation_counts: " + str(relation_counts) + " \n")
    output_file.write("relation_count: " + str(relation_count) + " \n")
    output_file.write("relation_hist: " + str(relation_hist) + " \n")
    output_file.write("************ End relation stats *************\n")

    #Now start running the tests
    self.num_jumps = 0
    output_file.write("Begin series of " + str(sample_size) + " runs, with " + str(num_trials_per_sample) + " trials each.\n")

    stats = None

    for i in range(sample_size):
      output_file.write("Begin run " + str(i + 1) + " out of " + str(sample_size) + ":\n")
      result = func("", num_trials_per_sample, dataFunc=dataFunc, *args, **kwargs)

      if len(result) > 0:
        if not stats:
          stats = [[] for j in result]

        for (r,S) in zip(result, stats):
          S.extend(r)

      self.print_bootstrap_summary(statNames, stats, i + 1, sample_size, output_file)
      output_file.flush()

    self.finish()

    end_time = datetime.datetime.now()
    self.print_bootstrap_runtime_summary(output_file, end_time - start_time)
    self.print_relation_stats(output_file, **kwargs)




  def print_relation_stats(self, output_file, **kwargs):
    if "relation_stats" in kwargs:
      relation_stats = kwargs["relation_stats"]
      for r in relation_stats:
        if len(relation_stats[r][0]) > 0:
          printout = [sum(relation_stats[r][0]) / len(relation_stats[r][0]),\
                     sum(relation_stats[r][1]) / len(relation_stats[r][1]),\
                     sum(relation_stats[r][2]) / len(relation_stats[r][2]),\
                     relation_stats[r][3]]

          output_file.write( str(r) + ", " + str(printout) + "\n")



  def print_bootstrap_summary(self, statNames, stats, sample_index, sample_size, output_file):
    mean = lambda x: float(sum(x)) / float(len(x))

    output_file.write("************ Bootstrap summary *************\n")
    output_file.write("After " + str(sample_index) + "samples out of " + str(sample_size) + "\n")

    if stats:
      i = 0
      for s in stats:
        CI = bootstrap.bootstrap_CI(0.05, mean, s, 999)

        output_file.write("mean " + statNames[i] + ": " + str(mean(s)) + "\n")
        output_file.write("lower 95% CI bound: " + str(CI[0]) + "\n")
        output_file.write("upper 95% CI bound: " + str(CI[1]) + "\n")
        output_file.write("raw data: " + str(s) + "\n")
        i += 1




  def print_bootstrap_runtime_summary(self, output_file, time):
    output_file.write("************ Runtime summary *************\n")
    output_file.write("Total elapsed time for bootstrap runs: " + str(time) + "\n")
    output_file.write("Total num jumps: " + str(self.num_jumps) + "\n")

    if self.num_jumps != 0:
      output_file.write("Average time per jump: " + str(float(time.seconds) / float(self.num_jumps)) + "\n")



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


