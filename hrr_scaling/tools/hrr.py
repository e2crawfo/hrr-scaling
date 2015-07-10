import numpy as np
from numpy import array,sqrt,dot
from numpy.linalg import norm
from numpy.fft import fft, ifft
import random

def circconv(a, b):
    return np.real(ifft(fft(a)*fft(b)))

class HRR:
    def __init__(self, N=None, data=None, unitary=False):
        if data is not None:
            self.v=array(data)
        elif N is not None:
            self.randomize(N)
        else:
            raise Exception('Must specify size or data for HRR')

        if unitary:
            self.make_unitary()

    def length(self):
        return norm(self.v)

    def normalize(self):
        nrm=norm(self.v)
        if nrm>0: self.v/=nrm

    def __str__(self):
        return str(self.v)

    def randomize(self,N=None):
        if N is None: N=len(self.v)
        self.v=array([random.gauss(0,1) for i in range(N)])
        self.normalize()

    def make_unitary(self):
        fft_val = np.fft.fft(self.v)
        fft_imag = fft_val.imag
        fft_real = fft_val.real
        fft_norms = [sqrt(fft_imag[n]**2 + fft_real[n]**2) for n in range(len(self.v))]
        fft_unit = fft_val / fft_norms
        self.v = (np.fft.ifft(fft_unit)).real

    def __add__(self,other):
        return HRR(data=self.v+other.v)

    def __iadd__(self,other):
        self.v+=other.v
        return self

    def __neg__(self):
        return HRR(data=-self.v)

    def __sub__(self,other):
        return HRR(data=self.v-other.v)

    def __isub__(self,other):
        self.v-=other.v
        return self

    def __mul__(self,other):
        if isinstance(other,HRR):
            return self.convolve(other)
        else:
            return HRR(data=self.v*other)

    def convolve(self,other):
        x=circconv(self.v,other.v)
        #x=ifft(fft(self.v)*fft(other.v)).real
        return HRR(data=x)

    def __rmul__(self,other):
        if isinstance(other,HRR):
            return self.convolve(other)
        else:
            return HRR(data=self.v*other)

    def __imul__(self,other):
        self.v=circconv(self.v,other.v)
        #self.v=ifft(fft(self.v)*fft(other.v))
        return self

    def compare(self,other):
        scale=norm(self.v)*norm(other.v)
        if scale==0: return 0
        return dot(self.v,other.v)/(scale)

    def dot(self,other):
        return dot(self.v,other.v)

    def distance(self,other):
        return 1-self.compare(other)

    def __invert__(self):
        N=len(self.v)
        return HRR(data=[self.v[0]]+[self.v[N-i] for i in range(1,N)])

    def __len__(self):
        return len(self.v)

    def copy(self):
        return HRR(data=self.v)

    def mse(self,other):
        err=0
        for i in range(len(self.v)):
            err+=(self.v[i]-other.v[i])**2
        return err/len(self.v)

    def get_transform_matrix(self):
        D=len(self.v)
        T=[]
        for i in range(D):
            T.append([self.v[(i-j)%D] for j in range(D)])
        return np.array(T)



                       
                       
from math import sin,pi,acos
class Vocabulary:
    defaults={}
    registered={}
    prob_normal=None
    last_val=None
    
    def register(self,node):
        # This will pose a memory problem in the future. If scripts are re-run and the node id is not the same,
        # then vocabularies will accumulate
        Vocabulary.registered[id(node)]=self
    
    def __init__(self,dimensions,randomize=True,unitary=False,max_similarity=0.1,include_pairs=False):
        self.dimensions=dimensions
        self.randomize=randomize
        self.unitary=unitary
        self.max_similarity=max_similarity
        self.hrr={}
        self.hrr['I']=HRR(data=np.eye(dimensions)[0])
        self.keys=[]
        self.key_pairs=[]
        self.vectors=None
        self.vector_pairs=None
        self.include_pairs=include_pairs
        Vocabulary.defaults[dimensions]=self

    def __getitem__(self,key, do_warning=False):
        if key not in self.hrr:
            if self.randomize:  
                count=0
                v=HRR(self.dimensions)
                while count<100 and self.vectors is not None:
                    similarity=np.dot(self.vectors,v.v)
                    if max(similarity)<self.max_similarity:
                        break
                    v=HRR(self.dimensions)
                    count+=1
                if count>=100:        
                    if do_warning:
                        print 'Warning: Could not create an HRR with max_similarity=%1.2f (D=%d, M=%d)'%(self.max_similarity,self.dimensions,len(self.hrr))
                
                # Check and make HRR vector unitary if needed
                if self.unitary is True or (isinstance(self.unitary,list) and key in self.unitary):
                    v.make_unitary()
            else:
                ov=[0]*self.dimensions
                ov[len(self.hrr)]=1.0
                v = HRR(data = ov)

            self.add(key,v)
        return self.hrr[key]        

    def add(self,key,v):
        # Perform checks
        if(isinstance(v,HRR)):
            self.hrr[key] = v
            self.keys.append(key)
            if self.vectors is None:
                self.vectors=np.array([self.hrr[key].v])
            else:
                self.vectors=np.resize(self.vectors,(len(self.keys),self.dimensions))
                self.vectors[-1,:]=self.hrr[key].v
            
            # Generate vector pairs 
            if(self.include_pairs or self.vector_pairs is not None):
                for k in self.keys[:-1]:
                    self.key_pairs.append('%s*%s'%(k,key))
                    v=(self.hrr[k]*self.hrr[key]).v
                    if self.vector_pairs is None:
                        self.vector_pairs=np.array([v])
                    else:    
                        self.vector_pairs=np.resize(self.vector_pairs,(len(self.key_pairs),self.dimensions))
                        self.vector_pairs[-1,:]=v
        else:
            raise TypeError('hrr.Vocabulary.add() Type error: Argument provided not of HRR type')
        
    def generate_pairs(self):
        """ This function is intended to be used in situations where a vocabulary has already been
        created without including pairs, but it becomes necessary to have the pairs (for graphing
        in interactive plots, for example). This is essentially identical to the add function above,
        except that it makes all the pairs in one pass (and without adding new vectors).
        """
        self.key_pairs = []
        self.vector_pairs = None
        for i in range(1, len(self.keys)):
            for k in self.keys[:i]:
                key = self.keys[i]
                self.key_pairs.append('%s*%s'%(k,key))
                v=(self.hrr[k]*self.hrr[key]).v
                if self.vector_pairs is None:
                    self.vector_pairs=np.array([v])
                else:    
                    self.vector_pairs=np.resize(self.vector_pairs,(len(self.key_pairs),self.dimensions))
                    self.vector_pairs[-1,:]=v

    def parse(self,text):
        return eval(text,{},self)
        
    def text(self,v,threshold=0.1,minimum_count=1,include_pairs=True,join='+',maximum_count=5,terms=None,normalize=False):
        if isinstance(v,HRR): v=v.v
        if v is None or self.vectors is None: return ''        
        if normalize:
            nrm=norm(v)
            if nrm>0: v/=nrm
        
        m=np.dot(self.vectors,v)
        matches=[(m[i],self.keys[i]) for i in range(len(m))]
        if include_pairs:
            if self.vector_pairs is None: self.generate_pairs()
            # vector_pairs may still be none after generate_pairs (if there is only 1 vector)
            if self.vector_pairs is not None:
                m2=np.dot(self.vector_pairs,v)
                matches.extend([(m2[i],self.key_pairs[i]) for i in range(len(m2))])
        if terms is not None:
            matches=[m for m in matches if m[1] in terms]
        matches.sort()
        matches.reverse()

        r=[]        
        for m in matches:
            if threshold is None or (m[0]>threshold and len(r)<maximum_count): r.append(m)
            elif len(r)<minimum_count: r.append(m)
            else: break
            
        return join.join(['%0.2f%s'%(c,k) for (c,k) in r])

    def dot(self,v):
        if isinstance(v,HRR): v=v.v
        return np.dot(self.vectors,v)

    def dot_pairs(self,v):
        if len(self.keys)<2: return None # There are no pairs.
        if isinstance(v,HRR): v=v.v
        if self.vector_pairs is None: self.generate_pairs()
        return np.dot(self.vector_pairs,v)

    def transform_to(self,other,keys=None):
        if keys is None:
            keys=list(self.keys)
            for k in other.keys:
                if k not in keys: keys.append(k)
        t=np.zeros((other.dimensions,self.dimensions),typecode='f')
        for k in keys:
            a=self[k].v
            b=other[k].v
            t+=array([a*bb for bb in b])
        return t

    def prob_angle(self, angle):
      return sin(angle)**(self.dimensions-2)

    def prob_within_angle(self, angle, steps=100000, use_cache=True):
      #steps is actually steps per some angle

      if self.prob_normal is not None:
        denom = self.prob_normal
      else:
        denom=0
        ddenom=1/float(steps)

        for i in range(int(steps * pi)):
          denom += self.prob_angle(ddenom*i)

        denom*=ddenom
        self.prob_normal = denom

      #format of last val is (angle, value)
      dnum= 1/float(steps)
      if self.last_val is not None and use_cache:

        if self.last_val[0] < angle:
          start, end = angle, self.last_val[0]
        else:
          start, end = self.last_val[0], angle

        remaining_angle = start - end

        added_num = 0
        for i in range(int(steps * remaining_angle)):
          added_num += self.prob_angle(pi-start+dnum*i)
        added_num *= dnum

        if self.last_val[0] < angle:
          num = self.last_val[1] + added_num
        else:
          num = self.last_val[1] - added_num

        self.last_val = (angle, num)

      else:
        num=0
        for i in range(int(steps * angle)):
          num += self.prob_angle(pi-angle+dnum*i)
        num*=dnum
        self.last_val = (angle, num)

      prob_within=num/denom

      return prob_within

    def prob_cleanup(self,compare,vocab_size):
        angle=acos(compare)
        perror = self.prob_within_angle(angle)
        pcorrect=(1-perror)**vocab_size
        return pcorrect


