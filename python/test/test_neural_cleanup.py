from ..neural_assoc_memory import NeuralAssociativeMemory
from ..vector_operations import *
from ..ccm.lib import hrr

import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import unittest
from nose.plugins.attrib import attr

class DummyTester(object):
  current_target_keys = None

@attr(slow=1)
class TestNeuralCleanup(unittest.TestCase):

  def setUp(self):
    self.cconv_func = lambda v, t: cconv(t[0], t[1]) + v
    pass

  def test_neural_medium(self):
    D = 512
    num_vecs = 4
    seed = 10
    threshold = 0.8
    vf = VectorFactory(seed)

    roles = [vf.genVec(D) for i in range(num_vecs)]
    items = [vf.genVec(D) for i in range(num_vecs)]

    items_dict = dict(zip(range(num_vecs), items))

    semantic_pointer = reduce(self.cconv_func, zip(roles, items), vf.genVec(D))
    semantic_pointer = normalize(semantic_pointer)

    nam = NeuralAssociativeMemory(indices_dict, items_dict, False, False, False, 0.3, print_output=False)

    for r, i in zip(roles, items):

      vec = nam.unbind_and_associate(semantic_pointer, r)

      hrr_vec = hrr.HRR(data=vec)
      similarity = hrr_vec.compare(hrr.HRR(data=i))
      print "Neural Cleanup Test - Medium : Similarity = ", similarity
      assert(similarity > threshold)

  def test_neural_many_relations(self):
    D = 512
    num_vecs = 10
    seed = 10
    threshold = 0.8

    vf = VectorFactory(seed)

    roles = [vf.genVec(D) for i in range(num_vecs)]
    items = [vf.genVec(D) for i in range(num_vecs)]

    items_dict = dict(zip(range(num_vecs), items))

    semantic_pointer = reduce(self.cconv_func, zip(roles, items), vf.genVec(D))
    semantic_pointer = normalize(semantic_pointer)
    nam = NeuralAssociativeMemory(indices_dict, items_dict, False, False, False, 0.3, print_output=False)

    for r, i in zip(roles, items):
      vec = nam.unbind_and_associate(semantic_pointer, r)

      hrr_vec = hrr.HRR(data=vec)
      similarity = hrr_vec.compare(hrr.HRR(data=i))
      print "Neural Cleanup Test - Many Relations : Similarity = ", similarity
      assert(similarity > threshold)

  def test_neural_repeated_roles(self):
    D = 512
    num_vecs = 8
    ratio = 2 #vecs to roles
    seed = 10
    vf = VectorFactory(seed)
    threshold = 0.8

    items = [vf.genVec(D) for i in range(num_vecs)]
    roles = [vf.genVec(D) for i in range(num_vecs * ratio)]

    items_dict = dict(zip(range(num_vecs), items))

    expanded_roles = reduce(lambda x, y: x.extend(y), [roles for i in range(ratio)], [])
    semantic_pointer = reduce(self.cconv_func, zip(roles, items), vf.genVec(D))
    semantic_pointer = normalize(semantic_pointer)
    nam = NeuralAssociativeMemory(indices_dict, items_dict, False, False, False, 0.3, print_output=False)

    for r, i in zip(roles, items):
      vec = nam.unbind_and_associate(semantic_pointer, r)

      hrr_vec = hrr.HRR(data=vec)
      similarity = hrr_vec.compare(hrr.HRR(data=i))
      print "Neural Cleanup Test - Medium : Similarity = ", similarity
      assert(similarity > threshold)

  def get_similarities(self, test_vec, vecs, exceptions=[]):
    hrr_vec = hrr.HRR(data=test_vec)
    similarities = [hrr_vec.compare(hrr.HRR(data=vecs[v])) for v in vecs if v not in exceptions]
    return similarities

  def test_neural_hierarchical(self):
    D = 512
    seed = 10
    vf = VectorFactory(seed)
    threshold = 0.8
    second_threshold = 0.6
    size_threshold = 0.5

    #Create a graph using networkx library, and create semantic pointers
    # and id-vectors to match the graph structure
    num_vertices = 40
    graph = nx.fast_gnp_random_graph(num_vertices, .1, seed, True)
    #nx.draw_circular(graph)
    #plt.savefig("test_network.png")

    max_out_degree_vertex = max(graph, key=lambda x: graph.out_degree(x))
    max_out_degree = graph.out_degree(max_out_degree_vertex)
    queries = [vf.genVec(D) for i in range(max_out_degree)]

    id_vecs = [vf.genVec(D) for i in range(num_vertices)]
    semantic_pointers = [vf.genVec(D) for i in range(num_vertices)]

    for u in graph:
      i = 0

      for v in graph.neighbors_iter(u):
        semantic_pointers[u] += cconv(queries[i], id_vecs[v])
        graph.edge[u][v]["r"] = i
        i+=1

      semantic_pointers[u] = normalize(semantic_pointers[u])

    # now run some tests - for now each test consists of starting at the vector with the highest out degree,
    # randomly picking a valid edge, and neurally moving along that edge .
    num_tests = 5
    max_depth = 5

    indices_dict = dict( zip(range(num_vertices), id_vecs))
    items_dict = dict( zip(range(num_vertices), semantic_pointers))

    nam = NeuralAssociativeMemory(indices_dict, items_dict, False, False, False, 0.3, print_output=False)

    rng = random.Random(seed)

    dummy_tester = DummyTester()
    nam.set_tester(dummy_tester)

    second_failures = 0
    sim_failures = 0
    size_failures = 0

    for t in range(num_tests):

      cur_vertex = max_out_degree_vertex
      cur_sp = semantic_pointers[max_out_degree_vertex]

      depth = 0

      while graph.out_degree(cur_vertex) > 0 and depth < max_depth:
        target_vertex = rng.sample( graph.successors(cur_vertex), 1)[0]

        dummy_tester.current_target_keys = [target_vertex]
        query_index = graph.edge[cur_vertex][target_vertex]["r"]
        vec = nam.unbind_and_associate(cur_sp, queries[query_index])[0]

        hrr_vec = hrr.HRR(data=vec)
        similarity = hrr_vec.compare(hrr.HRR(data=semantic_pointers[target_vertex]))
        size = np.linalg.norm(vec)
        second = max(self.get_similarities(vec, items_dict, [target_vertex]))

        print "Neural Cleanup Test - Hierarchical : Test # ", t, ", Depth = ", depth, ", Query = ", query_index, ", Num Queries = ", len(graph.successors(cur_vertex))
        print " ... Current Vertex = ", cur_vertex, ", Target Vertex = ", target_vertex
        print " ... Similarity = ", similarity
        print " ... Second = ", second
        print " ... Size of result = ", size

        if similarity < threshold: sim_failures += 1
        if second > second_threshold: second_failures += 1
        if size < size_threshold: size_failures += 1

        cur_sp = vec
        cur_vertex = target_vertex
        depth += 1

    assert(sim_failures < 3)
    assert(size_failures < 3)
    assert(second_failures < 3)



