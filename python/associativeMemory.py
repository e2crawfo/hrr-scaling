
from VectorOperations import *

class AssociativeMemory(object):

  # the associative memory maps from indices to items
  # indices and items must both be dictionaries, whose values are vectors
  def __init__(self, indices, items, threshold):
    self.indices = indices
    self.items = items
    self.threshold = threshold

  def unbind_and_associate(self, item, query):
      messy = cconv(item, pInv(query))
      return self.associate(messy)

  def associate(self, messyVector):
      keys = self.indices.keys()

      result = []
      for key in keys:
          vector = self.indices[key]
          d = numpy.dot(messyVector, vector)
          if d > self.threshold:
              result.append((d, key))

      result.sort(reverse=True)

      for i in range(len(result)):
          result[i] = self.items[result[i][1]]

      return result



