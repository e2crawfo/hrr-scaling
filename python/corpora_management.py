import random
import Queue
import collections
from mytools import hrr, bootstrap


class CorpusHandler:

    corpusDict = None

    def __init__(self, D=512, input_dir=".", relation_symbols=[]):
        self.D = D
        self.input_dir = input_dir
        self.relation_symbols = relation_symbols
        self.bootstrapper = bootstrap.Bootstrapper()

    def parseWordnet(self):

        fileposmap = {self.input_dir+'/data.noun': 'n',
                      self.input_dir+'/data.verb': 'v',
                      self.input_dir+'/data.adj': 'a',
                      self.input_dir+'/data.adv': 'r'}

        if self.corpusDict is not None:
            print "Warning: overwriting existing corpus dictionary."

        self.corpusDict = {}

        # See http://wordnet.princeton.edu/wordnet/man/wndb.5WN.html
        # and http://wordnet.princeton.edu/wordnet/man/wninput.5WN.html
        # for more information about what's going on here (parsing files).
        for filename in fileposmap:
            with open(filename, 'rb') as f:
                pos = fileposmap[filename]
                self.skipNotice(f)

                line = f.readline()
                while line:
                    parse = line.split()
                    tag = (pos, int(parse[0]))
                    self.corpusDict[tag] = []
                    w_cnt = int(parse[3], 16)
                    p_i = 4+w_cnt*2
                    p_cnt = int(parse[p_i])

                    for i in range(p_cnt):
                        ptr = parse[p_i+1]
                        offset = int(parse[p_i+2])
                        ss_type = parse[p_i+3]

                        # adjective Satellites are just Adjectives
                        if ss_type == 's':
                            ss_type = 'a'

                        pointerTag = (ss_type, offset)
                        self.corpusDict[tag].append((ptr, pointerTag))
                        # ignoring parse[p_i+4] (word/word mappings)
                        p_i = p_i + 4

                    line = f.readline()

    def createCorpusSubset(self, proportion):
        # randomly pick a starting point, following all links from that node,
        # do the same recursively for each node we just added.

        # just have to make sure we remove all the dangling links...
        # so once we've determined that we have enough nodes, have to go
        # through all the nodes we have added to the graph but have yet to
        # recurse on and delete all their links that point to something
        # which isn't in the subset

        subset_dict = {}
        proportion = max(0.0, min(1.0, proportion))

        size = 0
        target_size = proportion * len(self.corpusDict)

        queue = Queue.Queue()

        while size < target_size:
            if queue.empty():
                queue.put(random.choice(self.corpusDict.keys()))

            next_entry = queue.get()

            if next_entry in subset_dict:
                continue

            subset_dict[next_entry] = self.corpusDict[next_entry]
            new_entries = self.corpusDict[next_entry]
            size = size + 1

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

    def processCorpus(self):
        '''Remove excessive relationships, handle circularity, etc.'''

        processed = set()
        stack = []

        for item in self.corpusDict:
            if item in processed:
                continue

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
                    stack.append((activeItem,
                                  localProcessed.copy(),
                                  localStack))
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

    def formKnowledgeBase(self, identity_cleanup=False, unitary=False):
        # Check existence of corpus
        if self.corpusDict is None:
            raise Exception("Attempted to form the knowledge"
                            "base without a corpus.")

        print "Number of items in knowledge base:", len(self.corpusDict)
        if identity_cleanup:
            print "Processing Corpus"
            self.processCorpus()

        print "Generating relation type vectors"
        print "Using relation types: ", self.relation_symbols

        self.relation_type_vectors = {symbol: hrr.HRR(self.D)
                                      for symbol in self.relation_symbols}
        if unitary:
            for k, h in self.relation_type_vectors:
                h.make_unitary()

        # Order words by the dependencies of their definitions
        # Only have to do this if we're forming an identity cleanup
        if not identity_cleanup:
            key_order = self.corpusDict.keys()
        else:
            key_order = []
            resolved = set(self.relation_symbols)

            dependencies = {}
            for key in self.corpusDict.keys():
                dependencies[key] = set([tag[1] for tag in self.corpusDict[key]
                                         if tag[0] in self.relation_symbols])

            while len(key_order) < (len(self.corpusDict)
                                    + len(self.relation_symbols)):

                resolvable = set()

                for key in dependencies:
                    if dependencies[key].issubset(resolved):
                        resolvable.add(key)

                # add the resolvable keys to the order list and resolved set
                key_order.extend(resolvable)
                resolved = resolved.union(resolvable)

                # remove resolved tags from the dependency dictionary
                for r in resolvable:
                    del dependencies[r]

                # if no items are resolvable, we're stuck
                if len(resolvable) == 0:
                    break

            del resolved
            del resolvable
            if len(key_order) < len(self.corpusDict):
                raise Exception("Dependency resolution failed.")

        self.semantic_pointers = collections.OrderedDict()

        print "Generating ID-vectors"
        if identity_cleanup:
            self.id_vectors = self.semantic_pointers
        else:
            self.id_vectors = collections.OrderedDict()

            for key in key_order:
                self.id_vectors[key] = hrr.HRR(self.D)

        print "Generating HRR vectors"
        for key in key_order:
            semantic_pointer = hrr.HRR(self.D)

            for relation in self.corpusDict[key]:
                if relation[0] not in self.relation_type_vectors:
                    continue

                id_vector = self.id_vectors[relation[1]]

                relation_type_vector = self.relation_type_vectors[relation[0]]

                pair = id_vector * relation_type_vector

                semantic_pointer += pair

            semantic_pointer.normalize()

            self.semantic_pointers[key] = semantic_pointer

        for k in key_order:
            h = self.semantic_pointers[k]
            self.semantic_pointers[k] = h.v

            h = self.id_vectors[k]
            self.id_vectors[k] = h.v

        for k in self.relation_type_vectors:
            h = self.relation_type_vectors[k]
            self.relation_type_vectors[k] = h.v

    # File parsing utilities
    def skipNotice(self, f):
        '''Seeks to the beginning of actual data in data files
        (enabling the parser to skip the copyright notice).'''
        c = f.read(1)
        while c == ' ':
            f.readline()
            c = f.read(1)
        f.seek(-1, 1)
