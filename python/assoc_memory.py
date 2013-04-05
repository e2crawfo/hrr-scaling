try:
  import matplotlib as plt
except:
  pass

from ccm.lib import hrr
from vector_operations import *
from bootstrap import Bootstrapper
import heapq

class AssociativeMemory(object):

  _type = "Direct"
  tester = None

  # the associative memory maps from indices to items
  # indices and items must both be dictionaries, whose values are vectors
  def __init__(self, indices, items, threshold, identity, unitary , corpusDict=None):
    self.indices = indices
    self.items = items
    self.threshold = threshold
    self.dim= len(items.values()[1])
    self.hrr_vecs = dict([(key, hrr.HRR(data=self.indices[key])) for key in self.indices.keys()])
    self.unitary = unitary
    self.identity = identity
    self.corpusDict = corpusDict
    self.similarities = dict(zip(indices.keys(), [0 for i in range(len(indices))]))
    self.return_vec = False

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

      for key in target_keys:
        self.tester.add_data(str(num_relations) + "_correct_dot_product", self.similarities[key])

      nlargest = heapq.nlargest(num_correct_relations + 1, self.similarities.iteritems(), key=lambda x: x[1])
      largest_incorrect = filter(lambda x: x[0] not in target_keys, nlargest)[0]

      self.tester.add_data(str(num_relations) + "_largest_incorrect_dot_product", largest_incorrect[1])

      #now return something useful
      if len(result_keys) == 0:
          if self.return_vec:
            results = [zeroVec(self.dim)]
          else:
            results = []

          print("Nothing reached threshold")
      else:
          print(str(len(result_keys)) + " reached threshold")
          #plt.pyplot.hist([x[0] for x in result], 100)
          #plt.pyplot.show()
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
    output_file.write("Num items: " + str(len(self.indices)) + "\n")
    output_file.write("Dimension: " + str(self.dim) + "\n")
    output_file.write("Threshold: " + str(self.threshold) + "\n")
    output_file.write("Seed: " + str(self.tester.seed) + "\n")
    output_file.write("Associator type: " + str(self._type) + "\n")

