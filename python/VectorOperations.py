import numpy
from numpy.fft import fft, ifft
import random

def normalize(x):
    return x/numpy.linalg.norm(x)

def genVec(dim, selfReflect = False):
    if selfReflect:
        v = numpy.random.normal(0,1,int(dim/2)+1)
        v = numpy.concatenate((v, v[-2+(dim%2):0:-1]))
    else:
        v = numpy.random.normal(0,1,dim)
    return normalize(v)

def genHRRVec(dim):
    v = HRR(dim)
    v.normalize()
    return v.v

def genUnitaryVec(d):
    uvec_f = [1] + [numpy.e**(1j*numpy.pi*random.uniform(-1, 1)) for i in range((d-1)/2)]
    if d%2==0: uvec_f.append(1)
    uvec_f = numpy.concatenate((uvec_f, numpy.conjugate(uvec_f[-2+(d%2):0:-1])))
    return normalize(numpy.real(numpy.fft.ifft(uvec_f)))

def cconv(a, b):
    return numpy.real(ifft(fft(a)*fft(b)))

def pInv(a):
    return numpy.concatenate(([a[0]], a[:0:-1]))
