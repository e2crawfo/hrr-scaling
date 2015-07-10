from hrr_scaling.tools.hrr import HRR
from hrr_scaling import symbol_definitions
from hrr_scaling import tools

import random
import Queue
import collections

import numpy as np


class VectorizedCorpus(object):

    corpus_dict = None

    def __init__(self, dimension=512, input_dir="wordnet_data",
                 unitary_relations=False, proportion=1.0, num_synsets=-1,
                 id_vecs=True, relation_symbols=None, create_namedict=False,
                 dry_run=False, sp_noise=0, normalize=True):

        self.dimension = dimension
        self.input_dir = input_dir

        self.unitary_relations = unitary_relations
        self.create_namedict = create_namedict
        self.normalize = normalize

        if sp_noise < 0:
            sp_noise = 0
        self.sp_noise = sp_noise

        if relation_symbols is None:
            self.relation_symbols = symbol_definitions.uni_relation_symbols()
        else:
            self.relation_symbols = relation_symbols

        self.parse_wordnet()

        self.proportion = (float(num_synsets)/len(self.corpus_dict)
                           if num_synsets > 0 else proportion)

        if self.proportion < 1.0:
            self.create_corpus_subset(self.proportion)

        print "Wordnet data parsed."

        if dry_run:
            print "Dry run. Skipping vectorization."
        else:
            self.form_knowledge_base(id_vecs, unitary_relations)
            print "Vectorization of WordNet complete"

    def parse_wordnet(self):

        fileposmap = {self.input_dir+'/data.noun': 'n',
                      self.input_dir+'/data.verb': 'v',
                      self.input_dir+'/data.adj': 'a',
                      self.input_dir+'/data.adv': 'r'}

        if self.corpus_dict is not None:
            print "Warning: overwriting existing corpus dictionary."

        self.corpus_dict = {}
        self.name2key = collections.defaultdict(list)
        self.key2name = {}

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
                    description = line.strip().split('|')[-1]
                    tag = (pos, int(parse[0]))
                    self.corpus_dict[tag] = []
                    w_cnt = int(parse[3], 16)
                    p_i = 4+w_cnt*2
                    p_cnt = int(parse[p_i])

                    if self.create_namedict:
                        self.name2key[parse[4]].append((tag, description))
                        self.key2name[tag] = parse[4]

                    for i in range(p_cnt):
                        ptr = parse[p_i+1]
                        offset = int(parse[p_i+2])
                        ss_type = parse[p_i+3]

                        # adjective Satellites are just Adjectives
                        if ss_type == 's':
                            ss_type = 'a'

                        pointerTag = (ss_type, offset)
                        self.corpus_dict[tag].append((ptr, pointerTag))
                        # ignoring parse[p_i+4] (word/word mappings)
                        p_i = p_i + 4

                    line = f.readline()

    def has_valid_relations(self, key):
        valid_relations = filter(
            lambda x: x[0] in self.relation_symbols,
            self.corpus_dict[key])

        return len(valid_relations) > 0

    def create_corpus_subset(self, proportion):
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
        target_size = proportion * len(self.corpus_dict)

        queue = Queue.Queue()

        while size < target_size:
            if queue.empty():
                key = random.choice(self.corpus_dict.keys())

                while key in subset_dict or not self.has_valid_relations(key):
                    key = random.choice(self.corpus_dict.keys())

                queue.put(key)

            next_entry = queue.get()

            if next_entry in subset_dict:
                continue

            subset_dict[next_entry] = self.corpus_dict[next_entry]
            new_entries = self.corpus_dict[next_entry]
            size = size + 1

            for item in new_entries:
                if item[0] in self.relation_symbols:
                    queue.put(item[1])

        for key in subset_dict:
            removal_list = []

            for item in subset_dict[key]:
                if item[1] not in subset_dict:
                    removal_list.append(item)

            for item in removal_list:
                subset_dict[key].remove(item)

        self.corpus_dict = subset_dict

    def processCorpus(self):
        '''Remove excessive relationships, handle circularity, etc.'''

        processed = set()
        stack = []

        for item in self.corpus_dict:
            if item in processed:
                continue

            activeItem = item
            localStack = []
            localProcessed = set()

            while activeItem is not None:
                # find all relations of current item
                if activeItem not in localProcessed:
                    for relation in self.corpus_dict[activeItem]:
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
                        self.corpus_dict[activeItem].remove(activeRelation)
                        print 'deleting', activeRelation, 'from', activeItem
                    elif activeRelation[1] not in processed:
                        activeItem = activeRelation[1]
                    else:
                        localStack = []

        processed.update(localProcessed)

    def form_knowledge_base(self, id_vecs=True, unitary=False):

        # Check existence of corpus
        if self.corpus_dict is None:
            raise Exception("Attempted to form the knowledge "
                            "base without a corpus.")

        print "Number of items in knowledge base:", len(self.corpus_dict)

        if not id_vecs:
            print "Processing Corpus"
            self.processCorpus()

        print "Generating relation type vectors"
        print "Using relation types: ", self.relation_symbols

        self.relation_type_vectors = {symbol: HRR(self.dimension)
                                      for symbol in self.relation_symbols}
        if unitary:
            for k, h in self.relation_type_vectors.iteritems():
                h.make_unitary()

        if id_vecs:
            key_order = self.corpus_dict.keys()
        else:
            # Order words by the dependencies of their definitions
            # Only have to do this if we're not using ID-vectors
            key_order = []
            resolved = set(self.relation_symbols)

            dependencies = {}
            for key in self.corpus_dict.keys():
                dependencies[key] = set(
                    [tag[1] for tag in self.corpus_dict[key]
                     if tag[0] in self.relation_symbols])

            while len(key_order) < (len(self.corpus_dict)
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
            if len(key_order) < len(self.corpus_dict):
                raise Exception("Dependency resolution failed.")

        self.semantic_pointers = collections.OrderedDict()

        print "Generating ID-vectors"

        if id_vecs:
            self.id_vectors = collections.OrderedDict()

            for key in key_order:
                self.id_vectors[key] = HRR(self.dimension)
        else:
            self.id_vectors = self.semantic_pointers

        print "Generating HRR vectors"
        for key in key_order:
            relations = filter(
                lambda x: x[0] in self.relation_symbols,
                self.corpus_dict[key])

            if len(relations) == 0:
                self.semantic_pointers[key] = HRR(self.dimension)
                continue

            semantic_pointer = HRR(data=np.zeros(self.dimension))

            for n in range(self.sp_noise):
                semantic_pointer += HRR(self.dimension)

            for relation in relations:

                id_vector = self.id_vectors[relation[1]]

                relation_type_vector = self.relation_type_vectors[relation[0]]

                pair = id_vector * relation_type_vector

                semantic_pointer += pair

            if self.normalize:
                semantic_pointer.normalize()

            self.semantic_pointers[key] = semantic_pointer

        # convert all vectors from hrrs to numpy ndarrays
        for k in key_order:
            h = self.semantic_pointers[k]
            self.semantic_pointers[k] = h.v

        if id_vecs:
            for k in key_order:
                h = self.id_vectors[k]
                self.id_vectors[k] = h.v

        for k in self.relation_type_vectors:
            h = self.relation_type_vectors[k]
            self.relation_type_vectors[k] = h.v

    def find_chain(self, chain_length, relation_symbol='@', exclusive=True,
                   starting_keys=None):

        if isinstance(starting_keys, str):
            starting_keys = [starting_keys]

        keys = []
        for key in starting_keys:
            if isinstance(key, str):
                if self.create_namedict:
                    keys.extend([t[0] for t in self.name2key[key]])
                else:
                    raise ValueError("starting_keys must be keys (not strings)"
                                     "since no name dict was created")
            else:
                keys.append(key)

        print keys
        starting_keys = keys

        if not starting_keys:
            starting_keys = self.corpus_dict

        for key in starting_keys:
            chain = [key]
            while len(chain) < chain_length + 1:
                relations = filter(lambda x: x[0] == relation_symbol,
                                   self.corpus_dict[chain[-1]])

                valid = not exclusive and len(relations) > 0
                valid = valid or (exclusive and len(relations) == 1)

                if valid:
                    chain.append(relations[0][1])
                else:
                    break

            if len(chain) == chain_length + 1:
                yield chain

    def name2relations(self, name):
        if self.create_namedict:
            keys = (t[0] for t in self.name2key[name])
            return {key: [(t[0], self.key2name[t[1]])
                          for t in self.corpus_dict[key]
                          if t[0] in self.relation_symbols]
                    for key in keys}

    # File parsing utilities
    def skipNotice(self, f):
        '''Seeks to the beginning of actual data in data files
        (enabling the parser to skip the copyright notice).'''
        c = f.read(1)
        while c == ' ':
            f.readline()
            c = f.read(1)
        f.seek(-1, 1)

    def print_config(self, output_file):
        output_file.write("VectorizedCorpus config:\n")
        output_file.write("Dimension: " + str(self.dimension) + "\n")
        output_file.write("Unitary relations: " +
                          str(self.unitary_relations) + "\n")
        output_file.write("Proportion: " + str(self.proportion) + "\n")
        output_file.write("Num items: " + str(len(self.id_vectors)) + "\n")

        self.print_relation_stats(output_file)

    def print_relation_stats(self, output_file):
        relation_counts = {}
        relation_count = 0
        relation_hist = {}

        for key in self.corpus_dict:
            lst = self.corpus_dict[key]
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
        tools.print_header(output_file, title)

        output_file.write("relation_counts: " + str(relation_counts) + " \n")
        output_file.write("relation_count: " + str(relation_count) + " \n")
        output_file.write("relation_hist: " + str(relation_hist) + " \n")

        tools.print_footer(output_file, title)
