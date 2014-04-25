import numpy
from numpy.fft import fft, ifft
import random

class VectorFactory:
  def __init__(self, seed=1):
    self.np_rng = numpy.random.RandomState(seed)

  def genUnitaryVec(self, d):
    return genUnitaryVec(d, self.np_rng)

  def genVec(self, d, selfReflect=False):
    return genVec(d, self.np_rng, selfReflect)

def make_rng(seed):
    return (rng, np_rng)

def normalize(x):
    norm = numpy.linalg.norm(x)

    if norm > 0.00001:
      return x / norm
    else:
      return x

def zeroVec(dim):
    return numpy.zeros(dim)

def genVec(dim, np_rng=None, selfReflect = False):

    if not np_rng:
      np_rng = numpy.random

    if selfReflect:
        v = np_rng.normal(0,1,int(dim/2)+1)
        v = numpy.concatenate((v, v[-2+(dim%2):0:-1]))
    else:
        v = np_rng.normal(0,1,dim)
    return normalize(v)

def genHRRVec(dim):
    v = HRR(dim)
    v.normalize()
    return v.v

def genUnitaryVec(d, np_rng=None):

    if not np_rng:
      np_rng = numpy.random

    uvec_f = [1] + [numpy.e**(1j*numpy.pi*np_rng.uniform(-1, 1)) for i in range((d-1)/2)]
    if d%2==0: uvec_f.append(1)
    uvec_f = numpy.concatenate((uvec_f, numpy.conjugate(uvec_f[-2+(d%2):0:-1])))
    return normalize(numpy.real(numpy.fft.ifft(uvec_f)))

def cconv(a, b):
    return numpy.real(ifft(fft(a)*fft(b)))

def pInv(a):
    return numpy.concatenate(([a[0]], a[:0:-1]))
