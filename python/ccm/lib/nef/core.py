import numpy
import ccm
import copy

import sys
import Queue
import threading

from .connect import connect
from .accumulator import Accumulator

nodeLock = threading.Lock()
nodeCV = threading.Condition(nodeLock)
controlCV = threading.Condition(nodeLock)

threads_complete = 0
threads_waiting_for_signal = 0
signal = False
exitFlag = False

class NodeThread (threading.Thread):
  def __init__(self, threadID, name, total_threads, nodes, dt):
    threading.Thread.__init__(self)
    self.threadID = threadID
    self.name = name
    self.nodes = nodes
    self.dt = dt
    self.total_threads = total_threads
  
  def wait_for_signal(self):
    global signal, nodeCV, controlCV, threads_waiting_for_signal

    nodeCV.acquire()
    #print self.name + " about to wait for signal"
    threads_waiting_for_signal = threads_waiting_for_signal+1

    if threads_waiting_for_signal == self.total_threads:
      #print self.name + " about to notify main"
      controlCV.notify()

    while not signal:
      nodeCV.wait()

    #print self.name + " done waiting for signal"
    threads_waiting_for_signal = threads_waiting_for_signal-1
    #print self.name + " threads waiting for signal: " + str(threads_waiting_for_signal)

    if threads_waiting_for_signal == 0:
      signal=False

    nodeCV.release()

  def wait_for_other_threads(self, send_done_signal=False):
    global nodeCV, threads_complete, controlCV

    nodeCV.acquire()
    #print self.name + " about to wait for other threads"
    threads_complete=threads_complete+1

    #print self.name + "threads complete = " + str(threads_complete)
    if threads_complete == self.total_threads:
      #print self.name + " notifying"
      nodeCV.notifyAll()
      threads_complete = 0
      if send_done_signal:
        controlCV.notify()
      #print self.name + " finishing the wait"
    else:
      #print self.name + " waiting"
      nodeCV.wait()

    #print self.name + " returning"
    nodeCV.release()

  def check_for_kill(self):
    global nodeLock, exitFlag
    nodeLock.acquire()
    exit_val = exitFlag
    nodeLock.release()
    return exit_val

  def run(self):
    while True:
      self.wait_for_signal()

      if self.check_for_kill():
        return

      for n in self.nodes:
        ##print self.name + " ticking accumulator" + str(n)
        n._clear_inputs()
        n.tick_accumulator(self.dt)

      self.wait_for_other_threads()

      for n in self.nodes:
        ##print self.name + " ticking connection" + str(n)
        for conn in n.outputs:
          conn.transmit_out(self.dt)

      self.wait_for_other_threads()

      for n in self.nodes:
          ##print self.name + " ticking output" + str(n)
          n._calc_output()
          n.clear_state()

      self.wait_for_other_threads(send_done_signal=True)

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
        
    def kill_multithreaded(self):
        global signal, exitFlag, nodeCV, controlCV, threads_waiting_for_signal, threads_complete

        nodeCV.acquire()
        signal = True
        exitFlag = True
        nodeCV.notifyAll()
        nodeCV.release()

        self._threads=None

    def tick_multithreaded(self, dt, threads=2):
        global signal, nodeCV, controlCV, threads_waiting_for_signal, threads_complete
        total_threads=threads

        if self._threads is None:
          nodes=self._all_nodes
          if nodes is None:
              nodes=self.all_nodes()

          # assuming we have at least as many nodes as threads
          nodes_per_thread = int(numpy.floor( len(nodes) / total_threads ))
          leftover = int(numpy.mod(len(nodes), total_threads))
          self._threads = []

          end = 0


          for i in range(total_threads):
            start = end
            if leftover > i:
              end = start + nodes_per_thread+1
            else:
              end = start + nodes_per_thread

            nodes_for_thread = nodes[start:end]

            new_thread = NodeThread(i, "NodeThread_" + str(i), total_threads, nodes_for_thread, dt)
            self._threads.append(new_thread)

            new_thread.start()


        nodeCV.acquire()
        
        print "in control, threads_waiting_for_signal: ", threads_waiting_for_signal, " total_threads: ", total_threads, ", signal: ", signal
        #make sure all threads are the at beginning of the loop
        if threads_waiting_for_signal != total_threads:
          controlCV.wait()

        #now tell all the threads to take a step 
        signal=True
        nodeCV.notifyAll()

        #wait for the threads to finish the step
        controlCV.wait()

        nodeCV.release()
        print "in control, resuming main thread: threads_waiting_for_signal: ", threads_waiting_for_signal, " total_threads: ", total_threads, ", signal: ", signal


    def tick(self,dt=None):
        if dt is None: dt=getattr(self,'dt',None)
        nodes=self._all_nodes
        if nodes is None:
            nodes=self.all_nodes()
            #for n in nodes:
            #    n._all_nodes=nodes
        for n in nodes:
            n._clear_inputs()
            #n.accumulator.tick(dt)
            n.tick_accumulator(dt)
        #print >> sys.stderr, "Printing conn.array"
        for n in nodes:
            for conn in n.outputs:
                conn.transmit_out(dt)
                #print >> sys.stderr, conn.pop2
                #print >> sys.stderr, conn.apply_weight(conn.array)

                #f=getattr(n,'_transmit_%s'%conn.type())
                #f(conn,dt)
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
                    


                    


