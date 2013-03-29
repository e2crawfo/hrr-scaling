import numpy
from numpy.fft import fft, ifft
import random

def make_rng(seed):
    rng = random.Random()
    rng.seed(seed)
    np_rng = numpy.random.RandomState(seed)
    return (rng, np_rng)

def normalize(x):
    return x/numpy.linalg.norm(x)

def zeroVec(dim):
    return numpy.zeros(dim)

def genVec(dim, rng=None, selfReflect = False):
    if not rng:
      np_rng = numpy.random
    else:
      np_rng = rng[1]

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

def genUnitaryVec(d, rng=None):
    if not rng:
      rng = numpy.random
    else:
      rng = rng[0]

    uvec_f = [1] + [numpy.e**(1j*numpy.pi*rng.uniform(-1, 1)) for i in range((d-1)/2)]
    if d%2==0: uvec_f.append(1)
    uvec_f = numpy.concatenate((uvec_f, numpy.conjugate(uvec_f[-2+(d%2):0:-1])))
    return normalize(numpy.real(numpy.fft.ifft(uvec_f)))

def cconv(a, b):
    return numpy.real(ifft(fft(a)*fft(b)))

def pInv(a):
    return numpy.concatenate(([a[0]], a[:0:-1]))
