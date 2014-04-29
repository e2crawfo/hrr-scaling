import numpy
import ccm
import copy

import sys
import Queue
import threading

from .connect import connect
from .accumulator import Accumulator

class ArrayNode:
    _set_array=None
    _value=None
    _array=None
#    _input=None
    _output=None
    _threads=None

    def __init__(self,dimensions,min=-1,max=1,noise=0):
        self.inputs=[]
        self.outputs=[]
        self.dimensions=dimensions
        self.min=min
        self.max=max
        self.array_noise=noise

        self.mode='direct'
        self._all_nodes=None
        self.accumulator=Accumulator(dimensions)

    def value_to_array(self,value):
      return value

    def array_to_value(self,array):
      return array

    def set(self,value,calc_output=True):
        if value is None:
            self._set_array=None
            self._array=None
            self._value=None
            if calc_output: self._calc_output()
        else:
            array=numpy.array(self.value_to_array(value))
            if len(array.shape)>1: array.shape=array.shape[0]
            assert len(array)==self.dimensions
            self._set_array=array
            if calc_output: self._calc_output()
            self._array=None
            self._value=None

    def array(self):
        if self._array is None:
            if self._output is None: self._output=numpy.zeros(self.dimensions)
            #self._array=self._output
            return self._output
        return self._array

    def value(self):
        if self._value is None:
            self._value=self.array_to_value(self.array())
        return self._value



    def _calc_output(self):
        if self._output is None: self._output=numpy.zeros(self.dimensions)
        else: self._output[:]=0
        if self._set_array is not None:
            self._output+=self._set_array
        else:
            self._output+=self.accumulator.value()
        if self.array_noise>0:
            self._output+=numpy.random.randn(self.dimensions)*self.array_noise

    def _clear_inputs(self):
        pass

    def tick_accumulator(self, dt):
      for i in self.inputs:
        i.transmit_in(dt)

      self.accumulator.tick(dt)

    def clone(self):
      clone=copy.copy(self)
      clone.inputs=[]
      clone.outputs=[]
      clone.accumulator=Accumulator(self.dimensions)
      return clone

    def clear_state(self):
      self._value=None
      self._array=None

    def get_output_array_direct(self, conn, dt):
        return conn.apply_func(self._output)

    def add_input_array_direct(self, conn, tau, dt):
        self.accumulator.add(conn.apply_weight(conn.array), tau, dt)

    def tick(self,dt=None):
        if dt is None: dt=getattr(self,'dt',None)
        nodes=self._all_nodes
        if nodes is None:
            nodes=self.all_nodes()

        for n in nodes:
            n._clear_inputs()
            n.tick_accumulator(dt)

        for n in nodes:
            for conn in n.outputs:
                conn.transmit_out(dt)

        for n in nodes:
            n._calc_output()
            n.clear_state()

    def connect(self,other,func=None,weight=None,tau=None):
        return connect(self,other,func=func,weight=weight,tau=tau)

    """
    def all_nodes(self,list=None):
        if list is None: list=[]
        list.append(self)
        for c in self.inputs:
            if c.pop1 not in list:
                c.pop1.all_nodes(list)
        for c in self.outputs:
            if c.pop2 not in list:
                c.pop2.all_nodes(list)
        return list
    """
    def all_nodes(self):
        all=[]
        done=set()
        work=set([self])
        while len(work)>0:
            n=work.pop()
            done.add(n)
            all.append(n)
            for a in n.inputs:
                if a.pop1 not in done: work.add(a.pop1)
            for a in n.outputs:
                if a.pop2 not in done: work.add(a.pop2)
        return all

    def all_connections(self):
        conns = set()
        done=set()
        work=set([self])
        while len(work)>0:
            n=work.pop()
            done.add(n)

            for a in n.inputs:
                if a.pop1 not in done: work.add(a.pop1)
                if a not in conns: conns.add(a)
            for a in n.outputs:
                if a.pop2 not in done: work.add(a.pop2)
                if a not in conns: conns.add(a)

        return conns;

    def reset(self):
      self.accumulator=Accumulator(self.dimensions)
      self._output=None
      self._array=None
      self._value=None



