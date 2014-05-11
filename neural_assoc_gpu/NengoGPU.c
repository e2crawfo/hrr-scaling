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

NengoGPUData** nengo_data_array;
float start_time = 0, end_time = 0;
int do_print;
volatile int myCVsignal = 0;
int num_devices = 0;

pthread_cond_t* cv_GPUThreads = NULL;
pthread_cond_t* cv_JNI = NULL;
pthread_mutex_t* mutex = NULL;

FILE* fp;

// 0 - initialize
// 1 - check
// 2 - increment
// 3 - set to value
// 4 - add value
// Keeps track of how many nodes have been processed.
// Implemented as a function like this for the sake of
// encapsulation and synchronization.
int manipulateNumDevicesFinished(int action, int value)
{
  static int num_devicesFinished;
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
    num_devicesFinished = 0;
    return num_devicesFinished;
  }

  if(action == -1)
  {
    pthread_mutex_destroy(myMutex);
    free(myMutex);
    return num_devicesFinished;
  }

  pthread_mutex_lock(myMutex);

  int temp = 0;

  switch(action)
  {
    case 1:
      temp = num_devicesFinished ;break;
    case 2:
      temp = ++num_devicesFinished; break;
    case 3:
      temp = num_devicesFinished = value; break;
    case 4:
      num_devicesFinished += value;
      temp = num_devicesFinished; break;
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

// Called by the function setup. By the time this is called, the NengoGPUData structure
// for each device should have all its static data set (but not yet loaded onto a device,
// since it should't have access to a device yet). This function initializes the
// synchronization primitives and creates a new thread for each GPU in use
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

  // Initialize the mutex and condition variable. Must be done
  // before we create the threads since the threads use them.
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

  NengoGPUData* current_data;

  // Start the node-processing threads. Their starting function is start_GPU_thread.
  int i = 0;
  for(;i < num_devices; i++)
  {
    current_data = nengo_data_array[i];

    pthread_create(current_thread, NULL, &start_GPU_thread, (void*)current_data);
  }

  // Wait for the threads to do their initializing (signalled by
  // myCVSignal == num_devices), then return.
  pthread_mutex_lock(mutex);
  while(myCVsignal < num_devices)
  {
    pthread_cond_wait(cv_JNI, mutex);
  }
  myCVsignal = 0;
  pthread_cond_broadcast(cv_GPUThreads);
  pthread_mutex_unlock(mutex);

  free(current_thread);

  sched_yield();
}

// Called once per GPU device per simulation run. This is the entry point for each processing
// thread. Its input is the NengoGPUData structure that it is to process. The behaviour of
// this function is: wait until we get the signal to step (from the step function),
// process the NengoGPUData structure for one step with run_neural_associative_memory,
// wait again. Eventually manipulateKill(0) will return true, meaning the run is finished and
// the function will break out of the loop and free its resources.
void* start_GPU_thread(void* arg)
{
  NengoGPUData* nengo_data = (NengoGPUData*) arg;

  int num_devicesFinished;

  if (nengo_data->do_print)
    printf("GPU Thread %d: about to acquire device\n", nengo_data->device);

  initGPUDevice(nengo_data->device);

  if (nengo_data->do_print)
    printf("GPU Thread %d: done acquiring device\n", nengo_data->device);

  if (nengo_data->do_print)
    printf("GPU Thread %d: about to move simulation data to device\n", nengo_data->device);

  moveToDeviceNengoGPUData(nengo_data);

  if (nengo_data->do_print)
    printf("GPU Thread %d: done moving simulation data to device\n", nengo_data->device);

  //printVecs(nengo_data);

  // signal to parent thread that initialization is complete, then wait for the other threads to finish initialization.
  pthread_mutex_lock(mutex);
  myCVsignal++;
  if(myCVsignal == num_devices)
  {
    pthread_cond_broadcast(cv_JNI);
  }
  pthread_cond_wait(cv_GPUThreads, mutex);
  pthread_mutex_unlock(mutex);

  // Wait for the signal to step. If that signal has already come, then myCVsignal == 1.
  // In that case, we don't wait (if we did, we'd wait forever).
  pthread_mutex_lock(mutex);
  if(myCVsignal == 0)
  {
    pthread_cond_wait(cv_GPUThreads, mutex);
  }
  pthread_mutex_unlock(mutex);

  // The main loop for the processing threads. The thread is either processing nodes on 
  // the GPU or it is waiting for the call to step.
  while(!manipulateKill(0))
  {
    if(manipulateReset(0))
    {
      reset_neural_associative_memory(nengo_data);
    }
    else
    {
      run_neural_associative_memory(nengo_data, start_time, end_time);
    }

    // signal that this device is finished processing for the step
    num_devicesFinished = manipulateNumDevicesFinished(2, 0);

    pthread_mutex_lock(mutex);
    // Wakeup the main thread if all devices are finished running
    if(num_devicesFinished == num_devices)
    {
      pthread_cond_broadcast(cv_JNI);
      manipulateNumDevicesFinished(3, 0);
    }
    // Wait for call from main thread to step
    pthread_cond_wait(cv_GPUThreads, mutex);
    pthread_mutex_unlock(mutex);
  }

  // Should only get here after run_kill has been called
  freeNengoGPUData(nengo_data);
  shutdownGPUDevice();

  // if this is the last thread to finish, we wake up the main thread,
  // it has to free some things before we finish
  pthread_mutex_lock(mutex);
  myCVsignal++;
  if(myCVsignal == num_devices)
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
  free(nengo_data_array);

  pthread_mutex_destroy(mutex);
  pthread_cond_destroy(cv_GPUThreads);
  pthread_cond_destroy(cv_JNI);

  free(mutex);
  free(cv_GPUThreads);
  free(cv_JNI);
}

// Main entry point, distinct from the External entry point. Intended to by
// called from python with ctypes (but can also, of course, be called from c)
void setup(int num_devices_requested, int* devices_to_use, float dt, int num_items,
           int dimension, int** index_vectors, int** stored_vectors, float tau,
           float* decoders, int neurons_per_item, float* gain, float* bias,
           float tau_ref, float tau_rc, float radius, int identical_ensembles,
           int print_data, int* probe_indices, int num_probes)
{

  int i, j;

  int num_available_devices = getGPUDeviceCount();

  do_print = print_data;
  //if(do_print)
  printf("NeuralAssocGPU: SETUP\n");

  num_devices = num_devices_requested > num_available_devices ? num_available_devices : num_devices_requested;

  if(do_print)
    printf("Using %d devices. %d available\n", num_devices, num_available_devices);

  nengo_data_array = (NengoGPUData**) malloc(sizeof(NengoGPUData*) * num_devices);

  NengoGPUData* current_data;

  // Create the NengoGPUData structs, one per device.
  for(i = 0; i < num_devices; i++)
  {
    nengo_data_array[i] = getNewNengoGPUData();
  }

  if(do_print)
    printf("About to create the NengoGPUData structures\n");

  int items_per_device = num_items / num_devices;
  int leftover = num_items % num_devices;
  int item_index = 0;
  int items_for_current_device = 0;
  int probe_count = 0;

  // Now we start to load the data into the NengoGPUData struct for each device. 
  // (though the data doesn't get put on the actual device just yet).
  // Because of the CUDA architecture, we have to do some weird things to get a good speedup
  // These arrays that store the transforms, decoders, are setup in a non-intuitive way so
  // that memory accesses can be parallelized in CUDA kernels. For more information, see 
  // the NengoGPU user manual.
  for(i = 0; i < num_devices; i++)
  {
    // set values
    current_data = nengo_data_array[i];

    current_data->device = devices_to_use[i];

    current_data->do_print = do_print;
    current_data->neurons_per_item = neurons_per_item;
    current_data->dimension = dimension;

    current_data->tau = tau;
    current_data->tau_ref = tau_ref;
    current_data->tau_rc = tau_rc;
    current_data->radius = radius;
    current_data->dt = dt;

    items_for_current_device = items_per_device + (leftover > 0 ? 1 : 0);
    leftover--;

    current_data->num_items = items_for_current_device;
    current_data->identical_ensembles = identical_ensembles;

    probe_count = 0;
    for(j = 0; j < num_probes; j++)
    {
        if(probe_indices[j] >= item_index &&
           probe_indices[j] < item_index + items_for_current_device)
        {
            probe_count++;
        }
    }

    current_data->num_probes = probe_count;

    // create the arrays
    initializeNengoGPUData(current_data);

    // populate the arrays
    for(j = 0; j < items_for_current_device; j++)
    {
      memcpy(current_data->index_vectors->array + j * dimension,
             index_vectors[item_index + j], dimension * sizeof(float));
      memcpy(current_data->stored_vectors->array + j * dimension,
             stored_vectors[item_index + j], dimension * sizeof(float));
    }

    memcpy(current_data->decoders->array, decoders, neurons_per_item * sizeof(float));
    memcpy(current_data->gain->array, gain, neurons_per_item * sizeof(float));
    memcpy(current_data->bias->array, bias, neurons_per_item * sizeof(float));

    // populate the probe map
    probe_count = 0;
    for(j = 0; j < num_probes; j++)
    {
        if(probe_indices[j] >= item_index &&
           probe_indices[j] < item_index + items_for_current_device)
        {
            current_data->probe_map->array[probe_count] = probe_indices[j] - item_index;
            probe_count++;
        }
    }

    item_index += items_for_current_device;

    //printf("printing nengo gpu data\n");
    //printNengoGPUData(current_data, 1);
  }

  // We have all the data we need, now start the worker threads which control
  // the GPU's directly.
  run_start();
}


// Called once per step from python code. Puts the representedInputValues in the proper
// form for processing, then tells each GPU thread to take a step. Once they've finished
// the step, this function puts the representedOutputValues and spikes in the appropriate
// python arrays so that they can be read on the python side when this call returns
void step(float* input, float* output, float* probes, float start, float end, int n_steps)
{
  start_time = start;
  end_time = end;

  //if(do_print && ((int) (start_time * 1000)) % 10 == 0)
  if(n_steps % 10 == 0) 
      printf("NeuralAssocGPU: STEP %f\n", start);

  NengoGPUData* current_data;

  int i, j, k;

  for( i = 0; i < num_devices; i++)
  {
    current_data = nengo_data_array[i];
    memcpy(current_data->input_host->array, input, current_data->dimension * sizeof(float));
  }

  // Tell the runner threads to run and then wait for them to finish.
  // The last of them to finish running will wake this thread up.
  pthread_mutex_lock(mutex);
  myCVsignal = 1;
  pthread_cond_broadcast(cv_GPUThreads);
  pthread_cond_wait(cv_JNI, mutex);
  pthread_mutex_unlock(mutex);

  memset(output, 0, nengo_data_array[0]->dimension * sizeof(float));

  for(k=0; k < current_data->dimension; k++)
  {
    output[k] = 0.0;
  }

  for(i = 0; i < num_devices; i++)
  {
    current_data = nengo_data_array[i];

    for(j = 0; j < current_data->dimension; j++)
    {
      output[j] += current_data->output_host->array[j];
    }
  }

  for(i = 0; i < num_devices; i++)
  {
    current_data = nengo_data_array[i];

    for(j = 0; j < current_data->dimension; j++)
    {
      output[j] += current_data->output_host->array[j];
    }
  }

  int probe_index = 0;
  for(i = 0; i < num_devices; i++)
  {
    current_data = nengo_data_array[i];

    for(j = 0; j < current_data->num_probes; j++)
    {
      probes[probe_index] = current_data->probes_host->array[j];
      probe_index++;
    }
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
