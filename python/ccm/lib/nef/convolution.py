
import math
import numpy
from .array import make_array_HRR
from .connect import connect

def cconv(a, b):
    return numpy.real(numpy.ifft(numpy.fft(a)*numpy.fft(b)))

# DxD discrete fourier transform matrix            
def discrete_fourier_transform(D):
    m=[]
    for i in range(D):
        row=[]
        for j in range(D):            
            row.append(complex_exp((-2*math.pi*1.0j/D)*(i*j)))
        m.append(row)
    return m

# DxD discrete inverse fourier transform matrix            
def discrete_fourier_transform_inverse(D):
    m=[]
    for i in range(D):
        row=[]
        for j in range(D):            
            row.append(complex_exp((2*math.pi*1.0j/D)*(i*j))/D)
        m.append(row)
    return m

# formula for e^z for complex z
def complex_exp(z):
    a=z.real
    b=z.imag
    return math.exp(a)*(math.cos(b)+1.0j*math.sin(b))

def product(x):
    return x[0]*x[1]

def make_array(name,N_per_D,dimensions,quick=True,encoders=[[1,1],[1,-1],[-1,1],[-1,-1]],radius=3):
    return make_array_HRR(name,N_per_D,(dimensions/2+1)*4,dimensions=2,quick=quick,encoders=encoders,radius=radius)


def output_transform(dimensions):
    ifft=numpy.array(discrete_fourier_transform_inverse(dimensions))

    def makeifftrow(D,i):
        if i==0 or i*2==D: return ifft[i]
        if i<=D/2: return ifft[i]+ifft[-i].real-ifft[-i].imag*1j
        return numpy.zeros(dimensions)
    ifftm=numpy.array([makeifftrow(dimensions,i) for i in range(dimensions/2+1)])
    
    ifftm2=[]
    for i in range(dimensions/2+1):
        ifftm2.append(ifftm[i].real)
        ifftm2.append(-ifftm[i].real)
        ifftm2.append(-ifftm[i].imag)
        ifftm2.append(-ifftm[i].imag)
    ifftm2=numpy.array(ifftm2)

    return ifftm2.T

def input_transform(dimensions,first,invert=False):
    fft=numpy.array(discrete_fourier_transform(dimensions))

    M=[]
    for i in range((dimensions/2+1)*4):
        if invert: row=fft[-(i/4)]
        else: row=fft[i/4]
        if first:
            if i%2==0:
                row2=numpy.array([row.real,numpy.zeros(dimensions)])
            else:
                row2=numpy.array([row.imag,numpy.zeros(dimensions)])
        else:
            if i%4==0 or i%4==3:
                row2=numpy.array([numpy.zeros(dimensions),row.real])
            else:    
                row2=numpy.array([numpy.zeros(dimensions),row.imag])
        M.extend(row2)
    return M
    
    
        
               
def make_convolution(name,A,B,C,N_per_D,quick=False,encoders=[[1,1],[1,-1],[-1,1],[-1,-1]],radius=3,pstc_out=0.01,pstc_in=0.01,pstc_gate=0.01,invert_first=False,invert_second=False,mode='default',output_scale=1):
#    if isinstance(A,str):
#        A=self.network.getNode(A)
#    if isinstance(B,str):
#        B=self.network.getNode(B)
#    if isinstance(C,str):
#        C=self.network.getNode(C)

    dimensions=C.dimensions
    #if (B is not None and B.dimension!=dimensions) or (A is not None and A.dimension!=dimensions):
#        raise Exception('Dimensions not the same for convolution (%d,%d->%d)'%(A.dimension,B.dimension,C.dimension))
        
    #if mode=='direct':
    #    D=DirectConvolution(name,dimensions,invert_first,invert_second)
    #    self.add(D)
    #    D.getTermination('A').setTau(pstc_in)
    #    D.getTermination('B').setTau(pstc_in)
    #    D.getTermination('gate').setTau(pstc_gate)
    #    if A is not None:
    #        self.connect(A,D.getTermination('A'))
    #    if B is not None:
    #        self.connect(B,D.getTermination('B'))
    #    self.connect(D.getOrigin('C'),C,pstc=pstc_out,weight=output_scale)
    #else:

    D=make_array(name,N_per_D,dimensions,quick=quick,encoders=encoders,radius=radius)
    #D=make_array(self,name,N_per_D,dimensions,quick=quick,encoders=encoders,radius=radius)

    A2=input_transform(dimensions,True,invert_first)
    B2=input_transform(dimensions,False,invert_second)

    connect(A,D,weight=A2,tau=pstc_in)
    connect(B,D,weight=B2,tau=pstc_in)
    
    #if A is not None:
    #    self.connect(A,D.getTermination('A'))
    #if B is not None:
    #    self.connect(B,D.getTermination('B'))


    ifftm2=output_transform(dimensions)

    connect(D,C,func=product,weight=ifftm2*output_scale,tau=pstc_out)
    
    #self.connect(D,C,func=product,transform=ifftm2*output_scale,pstc=pstc_out)
        
    return D

    
