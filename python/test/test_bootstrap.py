from ..bootstrap import *
import unittest
import random

class TestBootstrapFunctions(unittest.TestCase):

  def setUp(self):
    self.seed = 10
    self.rng = random.Random(self.seed)
 
  def test_specific_draw_bootstrap_samples(self):
    lst = range(5)
    num_samples = 3

    sample = draw_bootstrap_samples(lst, num_samples, self.rng)
    reference_sample = [[2, 2, 2, 1, 4], [4, 3, 0, 2, 1], [1, 4, 4, 0, 4]]
    self.assertEqual(sample, reference_sample)

    simple_sample = draw_bootstrap_samples([1], 3, self.rng)
    reference_sample = [[1], [1], [1]]
    self.assertEqual(simple_sample, reference_sample)

  def test_gist_draw_bootstrap_samples(self):
    lst = range(6)
    num_samples = 4

    sample = draw_bootstrap_samples(lst, num_samples, self.rng)
    assert(isinstance(sample, list))
    self.assertEqual(len(sample), num_samples)

    for s in sample:
      self.assertEqual(len(s), len(lst))

  def test_base_draw_bootstrap_samples(self):
    sample = draw_bootstrap_samples([], 2, self.rng)
    self.assertEqual(sample, [[], []])

    sample = draw_bootstrap_samples([0, 1], 0, self.rng)
    self.assertEqual(sample, [])

    sample = draw_bootstrap_samples([], 0, self.rng)
    self.assertEqual(sample, [])

  def test_bootstrap_CI(self):
    reference_CI = (0.0, 1.0)
    


if __name__ == '__main__':
  unittest.main()
