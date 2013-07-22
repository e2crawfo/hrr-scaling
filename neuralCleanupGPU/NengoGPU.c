

#ifdef __cplusplus
extern "C"{
#endif

#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <sys/timeb.h>
#include <assert.h>

#include "NengoGPU.h"
#include "NengoGPU_CUDA.h"
#include "NengoGPUData.h"



NengoGPUData** nengoDataArray;
float startTime = 0, endTime = 0;
int do_print;
volatile int myCVsignal = 0;
int numDevices = 0;

pthread_cond_t* cv_GPUThreads = NULL;
pthread_cond_t* cv_JNI = NULL;
pthread_mutex_t* mutex = NULL;

FILE* fp;

// 0 - initialize
// 1 - check
// 2 - increment
// 3 - set to value
// 4 - add value
// Keeps track of how many nodes have been processed. Implemented as a function like this for the sake of encapsulation and synchronization.
int manipulateNumDevicesFinished(int action, int value)
{
  static int numDevicesFinished;
  static pthread_mutex_t* myMutex;

  if(action == 0)
  {
    myMutex = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
    
    if(!myMutex)
    {
      printf("bad malloc\n");
      exit(EXIT_FAILURE);
    }

    pthread_mutex_init(myMutex, NULL);
    numDevicesFinished = 0;
    return numDevicesFinished;
  }

  if(action == -1)
  {
    pthread_mutex_destroy(myMutex);
    free(myMutex);
    return numDevicesFinished;
  }

  pthread_mutex_lock(myMutex);

  int temp = 0;

  switch(action)
  {
    case 1:
      temp = numDevicesFinished ;break;
    case 2:
      temp = ++numDevicesFinished; break;
    case 3:
      temp = numDevicesFinished = value; break;
    case 4:
      numDevicesFinished += value;
      temp = numDevicesFinished; break;
  }

  pthread_mutex_unlock(myMutex);

  return temp;
}

int manipulateReset(int action)
{
  static int reset;
  switch(action)
  {
    case -1: reset = 0; break;
    case 0: break;
    case 1: reset = 1; break;
  }

  return reset;
}

// -1 - initialize
// 0 - check
// 1 - kill 
// Keeps track of whether the signal to end the run has been issued.
int manipulateKill(int action)
{
  static int kill;

  switch(action)
  {
    case -1: kill = 0; break;
    case 0: break;
    case 1: kill = 1; break;
  }

  return kill;
}

// Called by the function nativeSetupRun in NengoGPU_JNI.c. By the time this is called, the NengoGPUData structure for each device should have all its static data set
// (but not yet loaded onto a device, since it should't have access to a device yet).
// This function initializes the synchronization primitives and creates a new thread for each GPU in use.
void run_start()
{
  if(do_print)
    printf("NengoGPU: RUN_START\n");

  manipulateReset(-1);
  manipulateKill(-1);
  manipulateNumDevicesFinished(0, 0);
  myCVsignal = 0;

  pthread_t* current_thread = (pthread_t*) malloc(sizeof(pthread_t));
  if(!current_thread)
  {
    printf("bad malloc\n");
    exit(EXIT_FAILURE);
  }

  // Initialize the mutex and condition variable. Must be done before we create the threads since the threads use them.
  mutex = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
  cv_GPUThreads = (pthread_cond_t*) malloc(sizeof(pthread_cond_t));
  cv_JNI = (pthread_cond_t*) malloc(sizeof(pthread_cond_t));
  if(!mutex || !cv_GPUThreads || !cv_JNI)
  {
    printf("bad malloc\n");
    exit(EXIT_FAILURE);
  }

  pthread_mutex_init(mutex, NULL);
  pthread_cond_init(cv_GPUThreads, NULL);
  pthread_cond_init(cv_JNI, NULL);

  NengoGPUData* currentData;

  // Start the node-processing threads. Their starting function is start_GPU_thread.
  int i = 0;
  for(;i < numDevices; i++)
  {
    currentData = nengoDataArray[i];

    pthread_create(current_thread, NULL, &start_GPU_thread, (void*)currentData);
  }

  // Wait for the threads to do their initializing (signalled by myCVSignal == numDevices), then return.
  pthread_mutex_lock(mutex);
  while(myCVsignal < numDevices)
  {
    pthread_cond_wait(cv_JNI, mutex);
  }
  myCVsignal = 0;
  pthread_cond_broadcast(cv_GPUThreads);
  pthread_mutex_unlock(mutex);
  
  free(current_thread);

  sched_yield();
}

// Called once per GPU device per simulation run. This is the entry point for each processing thread. Its input is the
// NengoGPUData structure that it is to process. The behaviour of this function is: wait until we get the signal to step
// (from nativeStep in NengoGPU_JNI.c), process the NengoGPUData structure for one step with run_NEFEnsembles, wait again.
// Eventually manipulateKill(0) will return true, meaning the run is finished and the function will break out of the loop 
// and free its resources.
void* start_GPU_thread(void* arg)
{
  NengoGPUData* nengoData = (NengoGPUData*) arg;

  int numDevicesFinished;

  if (nengoData->do_print)
    printf("GPU Thread %d: about to acquire device\n", nengoData->device);

  initGPUDevice(nengoData->device);

  if (nengoData->do_print)
    printf("GPU Thread %d: done acquiring device\n", nengoData->device);
  
  if (nengoData->do_print)
    printf("GPU Thread %d: about to move simulation data to device\n", nengoData->device);

  moveToDeviceNengoGPUData(nengoData);

  if (nengoData->do_print)
    printf("GPU Thread %d: done moving simulation data to device\n", nengoData->device);

  //printVecs(nengoData);

  // signal to parent thread that initialization is complete, then wait for the other threads to finish initialization.
  pthread_mutex_lock(mutex);
  myCVsignal++;
  if(myCVsignal == numDevices)
  {
    pthread_cond_broadcast(cv_JNI);
  }
  pthread_cond_wait(cv_GPUThreads, mutex);
  pthread_mutex_unlock(mutex);
  
  // Wait for the signal to step. If that signal has already come, then myCVsignal == 1. In that case, we don't wait (if we did, we'd wait forever).
  pthread_mutex_lock(mutex);
  if(myCVsignal == 0)
  {
    pthread_cond_wait(cv_GPUThreads, mutex);
  }
  pthread_mutex_unlock(mutex);

  // The main loop for the processing threads. The thread is either processing nodes on the GPU or it is waiting for the call to step.
  while(!manipulateKill(0))
  {
    if(manipulateReset(0))
    {
      reset_NEFEnsembles(nengoData);
    }
    else
    {
      run_NEFEnsembles(nengoData, startTime, endTime);
    }

    // signal that this device is finished processing for the step
    numDevicesFinished = manipulateNumDevicesFinished(2, 0);

    pthread_mutex_lock(mutex);
    // Wakeup the main thread if all devices are finished running
    if(numDevicesFinished == numDevices)
    {
      pthread_cond_broadcast(cv_JNI);
      manipulateNumDevicesFinished(3, 0);
    }
    // Wait for call from main thread to step
    pthread_cond_wait(cv_GPUThreads, mutex);
    pthread_mutex_unlock(mutex);
  }

  // Should only get here after run_kill has been called
  freeNengoGPUData(nengoData);
  shutdownGPUDevice();

  // if this is the last thread to finish, we wake up the main thread, it has to free some things before we finish
  pthread_mutex_lock(mutex);
  myCVsignal++;
  if(myCVsignal == numDevices)
  {
    pthread_cond_broadcast(cv_GPUThreads);
  }
  pthread_mutex_unlock(mutex);
  return NULL;
}

// Free everything - should only be called when the run is over
void run_kill()
{
  // now when the threads check kill, the answer will be yes
  manipulateKill(1);
  manipulateNumDevicesFinished(-1, 0);

  // Wakeup GPU threads so they can free their resources
  pthread_mutex_lock(mutex);
  myCVsignal = 0;
  pthread_cond_broadcast(cv_GPUThreads);
  pthread_cond_wait(cv_GPUThreads, mutex);
  pthread_mutex_unlock(mutex);

  // Once the GPU threads are done, free shared resources and return
  free(nengoDataArray);

  pthread_mutex_destroy(mutex);
  pthread_cond_destroy(cv_GPUThreads);
  pthread_cond_destroy(cv_JNI);

  free(mutex);
  free(cv_GPUThreads);
  free(cv_JNI);
}

// another entry point, distinct from the External entry point. Intended to by called from python
// with ctypes (but can also, of course, be called from c)
// sizes store number of ensembles, number of network arrays and number of projections

void setup(int numDevicesRequested, int* devicesToUse, float dt, int numVectors, int dimension, int autoassociative, int** index_vectors, int** result_vectors, float tau, float* encoder, float* decoder, int num_neurons, float* alpha, float* Jbias, float tau_ref, float tau_rc, int* return_spikes, int print_data, int stop_early)
{

  int i, j, k;
  
  int numAvailableDevices = getGPUDeviceCount();

  do_print = print_data;
  if(do_print)
    printf("NengoGPU: SETUP\n"); 

  numDevices = numDevicesRequested > numAvailableDevices ? numAvailableDevices : numDevicesRequested;

  if(do_print)
    printf("Using %d devices. %d available\n", numDevices, numAvailableDevices);

  nengoDataArray = (NengoGPUData**) malloc(sizeof(NengoGPUData*) * numDevices);

  NengoGPUData* currentData;

  // Create the NengoGPUData structs, one per device.
  for(i = 0; i < numDevices; i++)
  {
    nengoDataArray[i] = getNewNengoGPUData();
  }

  if(do_print)
    printf("About to create the NengoGPUData structures\n");

  int items_per_device = numVectors / numDevices;
  int leftover = numVectors % numDevices;
  int item_index = 0;
  int items_for_current_device = 0;

  // Now we start to load the data into the NengoGPUData struct for each device. 
  // (though the data doesn't get put on the actual device just yet).
  // Because of the CUDA architecture, we have to do some weird things to get a good speedup. 
  // These arrays that store the transforms, decoders, are setup in a non-intuitive way so 
  // that memory accesses can be parallelized in CUDA kernels. For more information, see the NengoGPU user manual.
  for(i = 0; i < numDevices; i++)
  {
    currentData = nengoDataArray[i];
    
    currentData->device = devicesToUse[i];
    currentData->stop_early = stop_early;

    currentData->do_print = do_print;
    currentData->numNeuronsPerItem = num_neurons;
    currentData->dimension = dimension;
    currentData->autoassociative = autoassociative;

    currentData->tau = tau;
    currentData->tau_ref = tau_ref;
    currentData->tau_rc = tau_rc;
    currentData->dt = dt;

    items_for_current_device = items_per_device + (leftover > 0 ? 1 : 0);
    leftover--;

    currentData->numItems = items_for_current_device;
    for(j = 0; j < items_for_current_device; j++)
    {
      if(return_spikes[item_index + j])
      {
        currentData->numSpikesToReturn += num_neurons;
      }
    }

    initializeNengoGPUData(currentData);

    for(j = 0; j < items_for_current_device; j++)
    {
      memcpy(currentData->index_vectors->array + j * dimension, index_vectors[item_index + j], dimension * sizeof(float));
      memcpy(currentData->result_vectors->array + j * dimension, result_vectors[item_index + j], dimension * sizeof(float));
    }

    memcpy(currentData->encoder->array, encoder, num_neurons * sizeof(float));
    memcpy(currentData->decoder->array, decoder, num_neurons * sizeof(float));
    memcpy(currentData->alpha->array, alpha, num_neurons * sizeof(float));
    memcpy(currentData->Jbias->array, Jbias, num_neurons * sizeof(float));

    if(currentData->numSpikesToReturn)
    {
      int toIndex = 0;
      for(j = 0; j < items_for_current_device; j++)
      {
        if(return_spikes[item_index + j])
        {
          for(k = 0; k < num_neurons; k++)
          {
            intArraySetElement(currentData->spikeMap, toIndex, j * num_neurons + k);
            toIndex++;
          }
        }
      }
    }

    item_index += items_for_current_device;

    //printf("printing nengo gpu data\n"); 
    //printNengoGPUData(currentData, 1);
  }

  // we have all the data we need, now start the worker threads which control the GPU's directly.
  run_start();
}

// Called once per step from the External code. Puts the representedInputValues in the proper form for processing, then tells each GPU thread
// to take a step. Once they've finished the step, this function puts the representedOutputValues and spikes in the appropriate External
// arrays so that they can be read on the External side when this call returns.
void step(float* input, float* output, float* spikes, float start, float end, float* decoded_values)
{
  startTime = start;
  endTime = end;

  if(do_print && ((int) (startTime * 1000)) % 10 == 0)
    printf("NengoGPU: STEP %f\n", start);

  NengoGPUData* currentData;

  int i, j, k;

  for( i = 0; i < numDevices; i++)
  {
    currentData = nengoDataArray[i];
    memcpy(currentData->inputHost->array, input, currentData->dimension * sizeof(float));
  }

  // tell the runner threads to run and then wait for them to finish. The last of them to finish running will wake this thread up. 
  pthread_mutex_lock(mutex);
  myCVsignal = 1;
  pthread_cond_broadcast(cv_GPUThreads);
  pthread_cond_wait(cv_JNI, mutex);
  pthread_mutex_unlock(mutex);

  memset(output, 0, nengoDataArray[0]->dimension * sizeof(float));

  for(k=0; k < currentData->dimension; k++)
  {
    output[k] = 0.0;
  }

  for(i = 0; i < numDevices; i++)
  {
    currentData = nengoDataArray[i];

    if(currentData->stop_early)
    {
      int num_non_zero = 0;
      float threshold = 0.000001;
      for(j = 0; j < currentData->numItems; j++)
      {
        float val = currentData->decodedValuesHost->array[j];
        decoded_values[j] = val;

        if(val > threshold || val < -threshold)
        {
          for(k=0; k < currentData->dimension; k++)
          {
            output[k] += currentData->result_vectors->array[j * currentData->dimension + k] * val;
          }

          num_non_zero++;
        }
      }

      if(num_non_zero > 0)
        printf("GPU Thread %d: num nonzero decoded values: %d\n", currentData->device, num_non_zero);
    }
    else
    {
      for(j = 0; j < currentData->dimension; j++)
      {
        output[j] += currentData->outputHost->array[j];
      }
    }
  }

  // spikes should be exactly as big as the number of spikes we have requested be sent back
  int item_index = 0;
  for(i = 0; i < numDevices; i++)
  {
    currentData = nengoDataArray[i];

    for(j = 0; j < currentData->numSpikesToReturn; j++)
    {
      spikes[item_index + j] = currentData->spikesHost->array[j];
    }

    item_index += currentData->numSpikesToReturn;
  }
}

void kill()
{
  if(do_print)
    printf("NengoGPU: KILL\n");

  run_kill();
}

void reset()
{
  if(do_print)
    printf("NengoGPU: RESET\n");

  manipulateReset(1);

  pthread_mutex_lock(mutex);
  myCVsignal = 1;
  pthread_cond_broadcast(cv_GPUThreads);
  pthread_cond_wait(cv_JNI, mutex);
  pthread_mutex_unlock(mutex);

  manipulateReset(-1);
}




#ifdef __cplusplus
}

#endif
