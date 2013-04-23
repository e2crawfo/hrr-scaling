from ..neural_assoc_memory import NeuralAssociativeMemory
from ..vector_operations import *
from ..ccm.lib import hrr
import unittest
import random

class TestNeuralCleanup(unittest.TestCase):

  def setUp(self):
    pass

  def test_medium(self):
    D = 512
    num_vecs = 4
    seed = 10
    rng = make_rng(seed)
    threshold = 0.8

    roles = [genVec(D, rng) for i in range(num_vecs)]
    items = [genVec(D, rng) for i in range(num_vecs)]

    items_dict = dict(zip(range(num_vecs), items))

    cconv_func = lambda v, t: cconv(t[0], t[1]) + v
    semantic_pointer = reduce(cconv_func, zip(roles, items), genVec(D))
    semantic_pointer = normalize(semantic_pointer)
    nam = NeuralAssociativeMemory(items_dict, items_dict, False, False)

    for r, i in zip(roles, items):
      vec = nam.unbind_and_associate(semantic_pointer, r)

      hrr_vec = hrr.HRR(data=vec)
      similarity = hrr_vec.compare(hrr.HRR(data=i))
      print "Neural Cleanup Test - Medium : Similarity = ", similarity
      assert(similarity > threshold)

  def test_many_relations(self):
    D = 512
    num_vecs = 10
    seed = 10
    rng = make_rng(seed)
    threshold = 0.8

    roles = [genVec(D, rng) for i in range(num_vecs)]
    items = [genVec(D, rng) for i in range(num_vecs)]

    items_dict = dict(zip(range(num_vecs), items))

    cconv_func = lambda v, t: cconv(t[0], t[1]) + v
    semantic_pointer = reduce(cconv_func, zip(roles, items), genVec(D))
    semantic_pointer = normalize(semantic_pointer)
    nam = NeuralAssociativeMemory(items_dict, items_dict, False, False)

    for r, i in zip(roles, items):
      vec = nam.unbind_and_associate(semantic_pointer, r)

      hrr_vec = hrr.HRR(data=vec)
      similarity = hrr_vec.compare(hrr.HRR(data=i))
      print "Neural Cleanup Test - Many Relations : Similarity = ", similarity
      assert(similarity > threshold)

  def test_repeated_roles(self):
    D = 512
    num_vecs = 8
    ratio = 2 #vecs to roles
    seed = 10
    rng = make_rng(seed)
    threshold = 0.8

    items = [genVec(D, rng) for i in range(num_vecs)]
    roles = [genVec(D, rng) for i in range(num_vecs * ratio)]

    items_dict = dict(zip(range(num_vecs), items))

    cconv_func = lambda v, t: cconv(t[0], t[1]) + v

    expanded_roles = reduce(lambda x, y: x.extend(y), [roles for i in range(ratio)], [])
    semantic_pointer = reduce(cconv_func, zip(roles, items), genVec(D))
    semantic_pointer = normalize(semantic_pointer)
    nam = NeuralAssociativeMemory(items_dict, items_dict, False, False)

    for r, i in zip(roles, items):
      vec = nam.unbind_and_associate(semantic_pointer, r)

      hrr_vec = hrr.HRR(data=vec)
      similarity = hrr_vec.compare(hrr.HRR(data=i))
      print "Neural Cleanup Test - Medium : Similarity = ", similarity
      assert(similarity > threshold)
