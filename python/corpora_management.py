import sys
import random
import Queue
import bootstrap
import collections

#temporary
from ccm.lib import hrr

from vector_operations import *

class CorpusHandler:
    #D = 512 # number of dimensions per vocab vector
    corpusDict = None

    def __init__(self, D=512, input_dir=".", relation_symbols=[], vf=VectorFactory(), seed=1):
      self.D = D
      self.input_dir = input_dir
      self.relation_symbols = relation_symbols
      self.bootstrapper = bootstrap.Bootstrapper()

      self.seed = seed
      self.rng = random.Random(seed)
      self.vector_factory = vf

    def parseWordnet(self):

        fileposmap = {self.input_dir+'/data.noun': 'n', self.input_dir+'/data.verb': 'v', self.input_dir+'/data.adj' : 'a', self.input_dir+'/data.adv' : 'r'}
        if self.corpusDict is not None:
            print "Warning: overwriting existing corpus dictionary."

        self.corpusDict = {}

        # See http://wordnet.princeton.edu/wordnet/man/wndb.5WN.html
        # and http://wordnet.princeton.edu/wordnet/man/wninput.5WN.html
        # for more information about what's going on here (parsing files).
        for filename in fileposmap:
            with open(filename,'rb') as f:
                pos = fileposmap[filename]
                self.skipNotice(f)

                line = f.readline()
                while line:
                    parse = line.split()
                    tag = (pos, int(parse[0])) # this tag uniquely identifies a synset (word sense)
                    self.corpusDict[tag] = []
                    w_cnt = int(parse[3], 16)
                    p_i = 4+w_cnt*2
                    p_cnt = int(parse[p_i])

                    for i in range(p_cnt):
                        ptr = parse[p_i+1]
                        offset = int(parse[p_i+2])
                        ss_type = parse[p_i+3]
                        if ss_type == 's': ss_type = 'a' # adjective Satellites are just Adjectives
                        pointerTag = (ss_type, offset)
                        self.corpusDict[tag].append((ptr, pointerTag))
                        # ignoring parse[p_i+4] (word/word mappings)
                        p_i = p_i + 4

                    line = f.readline()

    def createCorpusSubset(self, proportion):
      # randomly pick a starting point, following all links from that node,
      # do the same recursively for each node we just added. 

      # just have to make sure we remove all the dangling links...
      # so once we've determined that we have enough nodes, have to go through
      # all the nodes we have added to the graph but have yet to recurse on 
      # and delete all their links that point to something which isn't
      # in the subset

      subset_dict = {}
      proportion = max(0.0,min(1.0, proportion))

      size = 0
      target_size = proportion * len(self.corpusDict)

      queue = Queue.Queue()

      while size < target_size:
        if queue.empty():
          queue.put( self.rng.choice(self.corpusDict.keys()))

        next_entry = queue.get()

        if next_entry in subset_dict:
          continue

        subset_dict[next_entry] = self.corpusDict[next_entry]
        new_entries = self.corpusDict[next_entry]
        size=size+1

        for item in new_entries:
          queue.put(item[1])

      for key in subset_dict:
        removal_list = []

        for item in subset_dict[key]:
          if item[1] not in subset_dict:
            removal_list.append(item)

        for item in removal_list:
          subset_dict[key].remove(item)

      self.corpusDict = subset_dict

    #currently this never gets called
    def processCorpus(self):
        '''Remove excessive relationships, handle circularity, etc.'''
        processed = set()
        stack = []
        for item in self.corpusDict:
            if item in processed: continue

            activeItem = item
            localStack = []
            localProcessed = set()

            while activeItem is not None:
                # find all relations of current item
                if activeItem not in localProcessed:
                    for relation in self.corpusDict[activeItem]:
                        if relation[0] in self.relation_symbols:
                            localStack.append(relation)
                            localProcessed.add(activeItem)

                # select an item and update stack (for branching trees)
                activeRelation = None
                if len(localStack) > 1:
                    activeRelation = localStack.pop()
                    stack.append((activeItem, localProcessed.copy(), localStack))
                    localStack = []
                elif len(localStack) == 1:
                    activeRelation = localStack[0]
                    localStack = []
                elif len(stack) > 0:
                    processed.update(localProcessed)
                    activeItem, localProcessed, localStack = stack.pop()
                else:
                    activeItem = None

                # check whether the link completes a circle
                if activeRelation is not None:
                    if activeRelation[1] in localProcessed:
                        # delete a relation if a circular definition was found
                        self.corpusDict[activeItem].remove(activeRelation)
                        print 'deleting', activeRelation, 'from', activeItem
                    elif activeRelation[1] not in processed:
                        activeItem = activeRelation[1]
                    else:
                        localStack = []

        processed.update(localProcessed)

    def generate_relation_type_vectors(self, use_unitary=False):

        self.relation_type_vectors = {}

        for symbol in self.relation_symbols:
          if use_unitary:
              self.relation_type_vectors[symbol] = self.vector_factory.genUnitaryVec(self.D)
          else:
              self.relation_type_vectors[symbol] = self.vector_factory.genVec(self.D)

    def formKnowledgeBase(self, identityCleanup=False, useUnitary=False):
        # Check existence of corpus
        if self.corpusDict is None:
            raise Exception("Attempted to form the knowledge base without a corpus.")

        print "Number of items in knowledge base:", len(self.corpusDict)
        if identityCleanup:
          print "Processing Corpus"
          self.processCorpus()

        print "Generating relation type symbols"
        self.generate_relation_type_vectors(useUnitary)

        # Order words by the dependencies of their definitions - only have to do it
        # if we're forming an identity cleanup
        if not identityCleanup:
            keyOrder = self.corpusDict.keys()
        else:
            keyOrder = []
            stuck = False
            resolved = set(self.relation_symbols)

            dependencies = {}
            for key in self.corpusDict.keys():
                dependencies[key] = set([tag[1] for tag in self.corpusDict[key]
                                         if tag[0] in self.relation_symbols])

            while len(keyOrder) < len(self.corpusDict)+len(self.relation_symbols) and not stuck:
                resolvable = set()
                for key in dependencies:
                  if dependencies[key].issubset(resolved):
                        resolvable.add(key)

                # add the resolvable keys to the order list and resolved set
                keyOrder.extend(resolvable)
                resolved = resolved.union(resolvable)

                # remove resolved tags from the dependency dictionary
                for r in resolvable:
                    del dependencies[r]

                # if no items are resolvable, we're stuck
                if len(resolvable)==0:
                    stuck=True

            del resolved
            del resolvable
            if len(keyOrder) < len(self.corpusDict):
                raise Exception("Dependency resolution failed.")

        if useUnitary:
          vector_function = self.vector_factory.genUnitaryVec
        else:
          vector_function = self.vector_factory.genVec

        self.semantic_pointers = collections.OrderedDict()

        if identityCleanup:
            self.id_vectors = self.semantic_pointers
        else:
            self.id_vectors = collections.OrderedDict()
            for key in keyOrder:
                self.id_vectors[key] = vector_function(self.D)

        for key in keyOrder:
            semantic_pointer = vector_function(self.D)

            for relation in self.corpusDict[key]:
                if relation[0] not in self.relation_type_vectors: continue

                vector = self.id_vectors[relation[1]]

                pair = cconv(self.relation_type_vectors[relation[0]], vector)
                semantic_pointer += pair

            normalize(semantic_pointer)

            self.semantic_pointers[key] = semantic_pointer

    # File parsing utilities
    def skipNotice(self, f):
        '''Seeks to the beginning of actual data in data files
        (enabling the parser to skip the copyright notice).'''
        c = f.read(1)
        while c==' ':
            f.readline()
            c = f.read(1)
        f.seek(-1, 1)

