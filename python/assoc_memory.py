try:
  import matplotlib as plt
except:
  pass

from ccm.lib import hrr
from vector_operations import *
from bootstrap import Bootstrapper
import heapq
import random

class AssociativeMemory(object):

  _type = "Direct"
  tester = None

  # the associative memory maps from indices to items
  # indices and items must both be dictionaries, whose values are vectors
  def __init__(self, indices, items, identity, unitary, bidirectional=False, threshold=0.3, return_vec=False):
    self.indices = indices
    self.items = items
    self.threshold = threshold
    self.dim= len(items.values()[1])
    self.hrr_vecs = dict([(key, hrr.HRR(data=self.indices[key])) for key in self.indices.keys()])
    self.unitary = unitary
    self.identity = identity
    self.similarities = dict(zip(indices.keys(), [0 for i in range(len(indices))]))
    self.return_vec = return_vec
    self.bidirectional = bidirectional

  def set_tester(self, tester):
      self.tester = tester

  def unbind_and_associate(self, item, query, key=None):
      messy = cconv(item, pInv(query))
      return self.associate(messy)

  def associate(self, messyVector):

      print("********In Associate*********")

      keys = self.indices.keys()

      messy_hrr = hrr.HRR(data=messyVector)

      for key in keys:
        self.similarities[key] = messy_hrr.compare(self.hrr_vecs[key])

      results = filter(lambda item: item[1] > self.threshold, self.similarities.iteritems())
      result_keys = [item[0] for item in results]

      #collect stats
      target_keys = self.tester.current_target_keys
      num_correct_relations = len(target_keys)
      num_relations = self.tester.current_num_relations

      #Correct dot products are the dot products of the things specified in target_keys.
      #Largest incorrect is the largest member of nlargest whose key is not in targey_keys.
      for key in target_keys:
        self.tester.add_data(str(num_relations) + "_correct_dot_product", self.similarities[key])

      nlargest = heapq.nlargest(num_correct_relations + 1, self.similarities.iteritems(), key=lambda x: x[1])
      largest_incorrect = filter(lambda x: x[0] not in target_keys, nlargest)[0]

      self.tester.add_data(str(num_relations) + "_largest_incorrect_dot_product", largest_incorrect[1])
      self.tester.add_data("num_reaching_threshold", len(result_keys))

      if self.return_vec:
        result = zeroVec(self.dim)

        for key in result_keys:
          result += self.similarities[key] * self.items[key]

        result = normalize(result)

        results = [result]

      else:
        #now return something useful
        if len(result_keys) == 0:
            if self.return_vec:
              results = [zeroVec(self.dim)]
            else:
              results = []

            print("Nothing reached threshold")
        else:
            print(str(len(result_keys)) + " reached threshold")

            #Here we are getting the at most 10 largest from result keys, in order from greatest to smallest dot product.
            #and in general we are returning the keys
            max_passes = 10
            if len(result_keys) > max_passes:
              results = heapq.nlargest(max_passes, results, key=lambda x: x[1])

            results.sort(reverse=True, key=lambda x:x[1])
            if self.return_vec:
              results = [self.items[r[0]] for r in results]
            else:
              results = [r[0] for r in results]

      return results

  def drawCombinedGraph(self, indices=None):
    pass

  def print_config(self, output_file):
    output_file.write("Unitary: " + str(self.unitary) + "\n")
    output_file.write("Identity: " + str(self.identity) + "\n")
    output_file.write("Bidirectional: " + str(self.bidirectional) + "\n")
    output_file.write("Num items: " + str(len(self.items)) + "\n")
    output_file.write("Num indices: " + str(len(self.indices)) + "\n")
    output_file.write("Dimension: " + str(self.dim) + "\n")
    output_file.write("Threshold: " + str(self.threshold) + "\n")
    output_file.write("Test Seed: " + str(self.tester.seed) + "\n")
    output_file.write("Associator type: " + str(self._type) + "\n")

  def get_similarities_random(self, s, n, dataFunc=None):
    samples_per_vec = 500
    i = 0
    print "In get_similarities_random"
    for idkey1 in self.hrr_vecs.keys():

      key_sample = random.sample(self.hrr_vecs, samples_per_vec)
      vec1 = self.hrr_vecs[idkey1]

      if i % 100 == 0:
        print "Sampling for vector: ", i

      for idkey2 in key_sample:
        if idkey1 == idkey2:
          continue
        vec2 = self.hrr_vecs[idkey2]

        similarity = vec1.compare(vec2)
        self.tester.add_data(idkey1, similarity)
        self.tester.add_data(idkey2, similarity)
        self.tester.add_data("all", similarity)

      i += 1

  def get_similarities_sample(self, s, n, dataFunc=None):
    num_samples = 2 * len(self.hrr_vecs)
    threshold = 0.1
    print "In get_similarities_sample"
    print "Num samples:" + str(num_samples)
    print "Threshold:" + str(threshold)

    for i in range(num_samples):
      idkey1, idkey2 = random.sample(self.hrr_vecs, 2)
      vec1 = self.hrr_vecs[idkey1]
      vec2 = self.hrr_vecs[idkey2]

      similarity = vec1.compare(vec2)

      if similarity > threshold:
        self.tester.add_data("all", similarity)

      if i % 1000 == 0:
        print "Trial: ", i


  def get_similarities(self, s, n, dataFunc=None):
    """get similarities of idvectors"""
    remaining_keys = self.hrr_vecs.keys()
    for idkey2 in self.hrr_vecs.keys():
      vec1 = self.hrr_vecs[idkey1]

      for idkey2 in remaining_keys:
        if idkey1 == idkey2:
          continue
        vec2 = self.hrr_vecs[idkey2]

        similarity = vec1.compare(vec2)
        self.tester.add_data(idkey1, similarity)
        self.tester.add_data(idkey2, similarity)
        self.tester.add_data("all", similarity)

      remaining_keys.remove(idkey1)

