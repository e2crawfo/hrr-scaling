from scipy import integrate
import numpy as np


# Base class taken from nengo 2.0
class Distribution(object):
    """
    A base class for probability distributions.

    The only thing that a probabilities distribution needs is a
    ``sample`` function. This base class ensures that all distributions
    accept the same arguments for the sample function.
    """

    def sample(self, n, rng=np.random):
        raise NotImplementedError("Distributions should implement sample.")

    def p(self, c):
        return self.func(c)

    def F(self, low=-1.0, high=1.0, error=False):
        result = integrate.quad(self.func, low, high)

        if not error:
            result = result[0]

        return result


class NormalizedVectorElement(Distribution):
    """
    The distribution of an element of a vector chosen at random
    from the unit hypersphere of the given dimension.

    It can be shown that this is also the distribution of the dot product
    between two randomly chosen vectors from the unit hypersphere.
    To see why, note that we can simply choose the first vector (randomly),
    rotate it so that its the vector [1, 0, ..., 0], then choose the second
    vector (randomly), and finally take the dot-product. Thus the dot-product
    of the two vectors is equal to the first element of the second vector,
    which was chosen randomly from the unit circle.
    """

    def __init__(self, dimensions):
        self.dimensions = dimensions
        raw_function = lambda c: (1 - c**2)**((dimensions-3)/2.0)
        self.normalizing_constant = integrate.quad(raw_function, -1.0, 1.0)[0]
        self.func = lambda c: raw_function(c) / self.normalizing_constant
