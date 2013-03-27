
from ccm.lib import hrr
from VectorOperations import *

class AssociativeMemory(object):

  # the associative memory maps from indices to items
  # indices and items must both be dictionaries, whose values are vectors
  def __init__(self, indices, items, threshold):
    self.indices = indices
    self.items = items
    self.threshold = threshold
    self.dim= len(items.values()[1])
    self.hrr_vecs = dict([(key, hrr.HRR(data=self.indices[key])) for key in self.indices.keys()])

  def unbind_and_associate(self, item, query, *arg, **kwargs):
      messy = cconv(item, pInv(query))
      return self.associate(messy)

  def associate(self, messyVector):

      print("********In Associate*********")
      keys = self.indices.keys()

      messy_hrr = hrr.HRR(data=messyVector)

      similarity = lambda key: messy_hrr.compare(self.hrr_vecs[key])
      similarities = [(similarity(key), key) for key in keys]
      result = filter(lambda x: x[0] > self.threshold, similarities)

      if len(result) == 0:
          print("max:")
          print(max(similarities, key=lambda x:x[0]))
          result = [zeroVec(self.dim)]
          print("Nothing reached threshold")
          print(self.threshold)
      else:
          print(str(len(result)) + " reached threshold")
          result.sort(reverse=True, key=lambda x:x[0])
          result = [self.items[r[1]] for r in result]

      return result

  def drawCombinedGraph(self, indices=None):
    pass



