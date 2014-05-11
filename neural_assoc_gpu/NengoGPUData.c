#ifdef __cplusplus
extern "C"{
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
//#include <cuda_runtime.h>
//#include <cublas.h>

#include "NengoGPUData.h"
#include "NengoGPU_CUDA.h"
#include "NengoGPU.h"

extern FILE* fp;


///////////////////////////////////////////////////////
// intArray and floatArray allocating and freeing
///////////////////////////////////////////////////////
intArray* newIntArray(int size, const char* name)
{
  intArray* new = (intArray*)malloc(sizeof(intArray));
  if(!new)
  {
    printf("Failed to allocate memory for intArray. name: %s, attemped size: %d\n", name, size);
    exit(EXIT_FAILURE);
  }

  new->array = (int*)malloc(size * sizeof(int));
  if(!new->array)
  {
    printf("Failed to allocate memory for intArray. name: %s, attemped size: %d\n", name, size);
    exit(EXIT_FAILURE);
  }

  new->size = size;
  new->name = strdup(name);
  new->on_device = 0;

  return new;
}

void freeIntArray(intArray* a)
{
  if(!a)
    return;

  if(a->array)
    a->on_device ? cudaFree(a->array) : free(a->array);

  if(a->name)
    free(a->name);

  free(a);
}

intArray* newIntArrayOnDevice(int size, const char* name)
{
  intArray* new = (intArray*)malloc(sizeof(intArray));

  new->array = allocateCudaIntArray(size);
  new->size = size;
  new->name = strdup(name);
  new->on_device = 1;

  return new;
}

floatArray* newFloatArray(int size, const char* name)
{
  floatArray* new = (floatArray*)malloc(sizeof(floatArray));
  if(!new)
  {
    printf("Failed to allocate memory for floatArray. name: %s, attemped size: %d\n", name, size);
    exit(EXIT_FAILURE);
  }

  new->array = (float*)malloc(size * sizeof(float));
  if(!new->array)
  {
    printf("Failed to allocate memory for floatArray. name: %s, attemped size: %d\n", name, size);
    exit(EXIT_FAILURE);
  }

  new->size = size;
  new->name = strdup(name);
  new->on_device = 0;

  return new;
}

void freeFloatArray(floatArray* a)
{
  //printf("freeing %s, size: %d, on_device: %d, address: %d\n", a->name, a->size, a->on_device, (int)a->array);

  if(!a)
    return;

  if(a->array)
    a->on_device ? cudaFree(a->array) : free(a->array);
  
  if(a->name)
    free(a->name);

  free(a);
}

floatArray* newFloatArrayOnDevice(int size, const char* name)
{
  floatArray* new = (floatArray*)malloc(sizeof(floatArray));

  new->array = allocateCudaFloatArray(size);
  new->size = size;
  new->name = strdup(name);
  new->on_device = 1;

  return new;
}
  

///////////////////////////////////////////////////////
// intArray and floatArray safe getters and setters
///////////////////////////////////////////////////////

void checkBounds(char* verb, char* name, int size, int index)
{
  if(index >= size || index < 0)
  {
    printf("%s safe array out of bounds, name: %s, size: %d, index:%d\n", verb, name, size, index);
    exit(EXIT_FAILURE);
  }
}

void checkLocation(char* verb, char* name, int on_device, int size, int index)
{
  if(on_device)
  {
    printf("%s safe array that is not on the host, name: %s, size: %d, index:%d\n", verb, name, size, index);
    exit(EXIT_FAILURE);
  }
}

void intArraySetElement(intArray* a, int index, int value)
{
  checkBounds("Setting", a->name, a->size, index);
  checkLocation("Setting", a->name, a->on_device, a->size, index);

  a->array[index] = value;
}

void floatArraySetElement(floatArray* a, int index, float value)
{
  checkBounds("Setting", a->name, a->size, index);
  checkLocation("Setting", a->name, a->on_device, a->size, index);

  a->array[index] = value;
}

int intArrayGetElement(intArray* a, int index)
{
  checkBounds("Getting", a->name, a->size, index);
  checkLocation("Getting", a->name, a->on_device, a->size, index);

  return a->array[index];
}

float floatArrayGetElement(floatArray* a, int index)
{
  checkBounds("Getting", a->name, a->size, index);
  checkLocation("Getting", a->name, a->on_device, a->size, index);

  return a->array[index];
}

void intArraySetData(intArray* a, int* data, int dataSize)
{
  if(dataSize > a->size)
  {
    printf("Warning: calling intArraySetData with a data set that is too large; truncating data. name: %s, size: %d, dataSize: %d", a->name, a->size, dataSize);
  }
  
  memcpy(a->array, data, dataSize * sizeof(int));
}

void floatArraySetData(floatArray* a, float* data, int dataSize)
{
  if(dataSize > a->size)
  {
    printf("Warning: calling floatArraySetData with a data set that is too large; truncating data. name: %s, size: %d, dataSize: %d", a->name, a->size, dataSize);
  }
  
  memcpy(a->array, data, dataSize * sizeof(float));
}

///////////////////////////////////////////////////////
// int_list functions 
///////////////////////////////////////////////////////
int_list* cons_int_list(int_list* list, int item)
{
  int_list* new = (int_list*)malloc(sizeof(int_list));
  new->first = item;
  new->next = list;
  return new;
}

void free_int_list(int_list* list)
{
  if(list)
  {
    int_list* temp = list->next;
    free(list);
    free_int_list(temp);
  }
}

///////////////////////////////////////////////////////
// int_queue functions
///////////////////////////////////////////////////////
int_queue* new_int_queue()
{
  int_queue* new = (int_queue*) malloc(sizeof(int_queue));

  new->size = 0;
  new->head = NULL;
  new->tail = NULL;

  return new;
}

int pop_int_queue(int_queue* queue)
{
  if(queue )
  {
    int val;
    switch(queue->size)
    {
      case 0:
        fprintf(stderr, "Error \"int_queue\": accessing empty queue\n");
        exit(EXIT_FAILURE);
        break;
      case 1:
        val = queue->head->first;
        free_int_list(queue->head);
        queue->head = NULL;
        queue->tail = NULL;
        queue->size--;
        return val;
        break;
      default:
        val = queue->head->first;
        int_list* temp = queue->head;
        queue->head = temp->next;
        temp->next = NULL;
        free_int_list(temp);
        queue->size--;
        return val;
    }
  }
  else
  {
    fprintf(stderr, "Error \"int_queue\": accessing null queue\n");
    exit(EXIT_FAILURE);
  }
}

void add_int_queue(int_queue* queue, int val)
{
  if(queue)
  {
    queue->tail->next = cons_int_list(NULL, val);
    queue->tail = queue->tail->next;
    queue->size++;
  }
  else
  {
    fprintf(stderr, "Error \"int_queue\": accessing null queue\n");
    exit(EXIT_FAILURE);
  }
}

void free_int_queue(int_queue* queue)
{
  if(queue)
  {
    free_int_list(queue->head);
    free(queue);
  }
}

///////////////////////////////////////////////////////
// NengoGPUData functions
///////////////////////////////////////////////////////

// return a fresh NengoGPUData object with all numerical values zeroed out and all pointers set to null
NengoGPUData* getNewNengoGPUData()
{
  NengoGPUData* new = (NengoGPUData*)malloc(sizeof(NengoGPUData));
  
  new->fp = NULL;
  new->on_device = 0;
  new->initialized = 0;
  new->device = 0;
  new->do_print = 0;

  new->start_time = 0.0;
  new->end_time = 0.0;

  new->identical_ensembles = 0;
  new->neurons_per_item = 0;
  new->dimension = 0;
  new->num_items = 0;

  new->tau = 0;
  new->pstc = 0;
  new->tau_ref = 0;
  new->tau_rc = 0;
  new->dt = 0;
  new->radius = 0;

  //new->handle = NULL;
  new->handle_initialized = 0;

  new->input_host = NULL;
  new->input_device = NULL;

  new->encode_result = NULL;
  new->decoded_values = NULL;

  new->output_host = NULL;
  new->output_device = NULL;

  new->index_vectors = NULL;
  new->stored_vectors = NULL;

  new->decoders = NULL;
  new->gain = NULL;
  new->bias = NULL;
  new->voltage = NULL;
  new->reftime = NULL;
  new->spikes = NULL;

  new->probes_host = NULL;
  new->probes_device = NULL;
  new->probe_map = NULL;

  return new;
}


// Should only be called once the NengoGPUData's numerical values have been set. This function
// allocates memory of the approprate size for each pointer. Memory is allocated on the host.
// The idea is to call this before we load the data in from the JNI structures, so we have
// somewhere to put that data. Later, we will move most of the data to the device.
void initializeNengoGPUData(NengoGPUData* new)
{
  if(new == NULL)
  {
     return;
  }

  char filename[50];
  sprintf(filename, "../neuralCleanupGPU/gpuOutput/gpuOutput%d.txt", new->device);
  new->fp = fopen(filename, "w");

  char* name; 

  name = "input_host";
  new->input_host = newFloatArray(new->dimension * new->num_steps, name);
  name = "output_host";
  new->output_host = newFloatArray(new->dimension * new->num_steps, name);

  name = "index_vectors";
  new->index_vectors = newFloatArray(new->dimension * new->num_items, name);
  name = "stored_vectors";
  new->stored_vectors = newFloatArray(new->dimension * new->num_items, name);

  name = "decoders";
  new->decoders = newFloatArray(new->neurons_per_item, name);
  name = "gain";
  new->gain = newFloatArray(new->neurons_per_item, name);
  name = "bias";
  new->bias = newFloatArray(new->neurons_per_item, name);

  name = "probe_map";
  new->probe_map = newIntArray(new->num_probes, name);
  name = "probes_host";
  new->probes_host = newFloatArray(new->num_probes * new->num_steps, name);
}


// Called at the end of initializeNengoGPUData to determine whether any of the mallocs failed.
void checkNengoGPUData(NengoGPUData* currentData)
{
  int status = 0;

  if(status)
  {
    printf("bad NengoGPUData\n");
    exit(EXIT_FAILURE);
  }
}

// Move data that has to be on the device to the device
void moveToDeviceNengoGPUData(NengoGPUData* nengo_data)
{
  if(!nengo_data->on_device)
  {
    // this function is in NengoGPU_CUDA.cu
    initializeDeviceInputAndOutput(nengo_data);

    moveToDeviceFloatArray(nengo_data->index_vectors);
    moveToDeviceFloatArray(nengo_data->stored_vectors);

    moveToDeviceFloatArray(nengo_data->decoders);
    moveToDeviceFloatArray(nengo_data->gain);
    moveToDeviceFloatArray(nengo_data->bias);
    moveToDeviceIntArray(nengo_data->probe_map);

    nengo_data->on_device = 1;
  }
}

// Free the NengoGPUData. Makes certain assumptions about where each array is (device or host).
void freeNengoGPUData(NengoGPUData* nengo_data)
{
  freeFloatArray(nengo_data->input_host);
  freeFloatArray(nengo_data->input_device);
  freeFloatArray(nengo_data->encode_result);
  freeFloatArray(nengo_data->decoded_values);
  freeFloatArray(nengo_data->output_host);
  freeFloatArray(nengo_data->output_device);

  freeFloatArray(nengo_data->index_vectors);
  freeFloatArray(nengo_data->stored_vectors);

  freeFloatArray(nengo_data->decoders);

  freeFloatArray(nengo_data->gain);
  freeFloatArray(nengo_data->bias);
  freeFloatArray(nengo_data->voltage);
  freeFloatArray(nengo_data->reftime);
  freeFloatArray(nengo_data->spikes);

  freeFloatArray(nengo_data->probes_host);
  freeFloatArray(nengo_data->probes_device);
  freeIntArray(nengo_data->probe_map);

  if(nengo_data->fp)
    fclose(nengo_data->fp);

  if(nengo_data->handle_initialized)
    cublasDestroy(nengo_data->handle);

  free(nengo_data);
};


void printVecs(NengoGPUData* nengo_data)
{
  nengo_data->fp ? fprintf(nengo_data->fp, "printing index vectors:\n") : printf("printing index vectors:\n");
  printFloatArray(nengo_data->fp, nengo_data->index_vectors, nengo_data->dimension, nengo_data->num_items);
  nengo_data->fp ? fprintf(nengo_data->fp, "printing stored vectors:\n") : printf("printing stored vectors:\n");
  printFloatArray(nengo_data->fp, nengo_data->stored_vectors, nengo_data->dimension, nengo_data->num_items);
}

// print the NengoGPUData. Should only be called once the data has been set.
void printNengoGPUData(NengoGPUData* nengo_data, int printArrays)
{
  
  nengo_data->fp ? fprintf(nengo_data->fp, "printing NengoGPUData:\n") : printf("printing NengoGPUData:\n");

  nengo_data->fp ? fprintf(nengo_data->fp, "start_time; %f\n", nengo_data->start_time) : printf("start_time; %f\n", nengo_data->start_time);
  nengo_data->fp ? fprintf(nengo_data->fp, "end_time; %f\n", nengo_data->end_time) : printf("end_time; %f\n", nengo_data->end_time);

  nengo_data->fp ? fprintf(nengo_data->fp, "on_device; %d\n", nengo_data->on_device) : printf("on_device; %d\n", nengo_data->on_device);
  nengo_data->fp ? fprintf(nengo_data->fp, "initialized; %d\n", nengo_data->initialized) : printf("initialized; %d\n", nengo_data->initialized);
  nengo_data->fp ? fprintf(nengo_data->fp, "device; %d\n", nengo_data->device) : printf("device; %d\n", nengo_data->device);

  nengo_data->fp ? fprintf(nengo_data->fp, "neurons_per_item: %d\n", nengo_data->neurons_per_item) : printf("neurons_per_item: %d\n", nengo_data->neurons_per_item);
  nengo_data->fp ? fprintf(nengo_data->fp, "dimension: %d\n", nengo_data->dimension) : printf("dimension: %d\n", nengo_data->dimension);
  nengo_data->fp ? fprintf(nengo_data->fp, "num_items: %d\n", nengo_data->num_items) : printf("num_items: %d\n", nengo_data->num_items);
  nengo_data->fp ? fprintf(nengo_data->fp, "tau: %f\n", nengo_data->tau) : printf("tau: %f\n", nengo_data->tau);
  nengo_data->fp ? fprintf(nengo_data->fp, "pstc: %f\n", nengo_data->pstc) : printf("pstc: %f\n", nengo_data->pstc);
  nengo_data->fp ? fprintf(nengo_data->fp, "tau_ref: %f\n", nengo_data->tau_ref) : printf("tau_ref: %f\n", nengo_data->tau_ref);
  nengo_data->fp ? fprintf(nengo_data->fp, "tau_rc: %f\n", nengo_data->tau_rc) : printf("tau_rc: %f\n", nengo_data->tau_rc);
  nengo_data->fp ? fprintf(nengo_data->fp, "dt: %f\n", nengo_data->dt) : printf("dt: %f\n", nengo_data->dt);

  if(printArrays)
  {
    printFloatArray(nengo_data->fp, nengo_data->input_host, nengo_data->dimension, 1);

    printFloatArray(nengo_data->fp, nengo_data->input_device, nengo_data->dimension, 1);

    printFloatArray(nengo_data->fp, nengo_data->encode_result, nengo_data->num_items, 1);
    printFloatArrayFromDevice(nengo_data->fp, nengo_data->decoded_values, 1, nengo_data->num_items, 1);
    printFloatArray(nengo_data->fp, nengo_data->output_host, nengo_data->dimension, 1);
    printFloatArray(nengo_data->fp, nengo_data->output_device, nengo_data->dimension, 1);

    printFloatArray(nengo_data->fp, nengo_data->decoders, nengo_data->neurons_per_item, 1);

    printFloatArray(nengo_data->fp, nengo_data->gain, nengo_data->neurons_per_item, 1);
    printFloatArray(nengo_data->fp, nengo_data->bias, nengo_data->neurons_per_item, 1);
    printFloatArray(nengo_data->fp, nengo_data->voltage, nengo_data->neurons_per_item, nengo_data->num_items);
    printFloatArray(nengo_data->fp, nengo_data->reftime, nengo_data->neurons_per_item, nengo_data->num_items);
    printFloatArray(nengo_data->fp, nengo_data->spikes, nengo_data->neurons_per_item, nengo_data->num_items);
  }

//  int bytesOnGPU = sizeof(float) * (8 * currentData->numEnsembles + currentData->totalInputSize + currentData->totalTransformSize + 2 * currentData->totalNumTransformRows + 2 * currentData->numTerminations + currentData->maxNumDecodedTerminations * currentData->totalEnsembleDimension + currentData->maxDimension * currentData->numNeurons + currentData->totalEnsembleDimension + currentData->numNeurons * 6 + currentData->maxNumNeurons * currentData->totalOutputSize + 2 * currentData->totalOutputSize);
 // printf("bytes on GPU: %d\n", bytesOnGPU);
}

// this function doesn't really work! because the printFloatArrayFromDevice function shouldn't be called this way
void printDynamicNengoGPUData(NengoGPUData* nengo_data)
{
  /*
    printFloatArray(currentData->input, currentData->totalInputSize, 1);

    printFloatArray(currentData->terminationOutput, currentData->totalNumTransformRows, 1);
    printFloatArray(currentData->encode_result, currentData->numNeurons, 1);
    printFloatArray(currentData->ensembleSums, currentData->totalEnsembleDimension, 1);
    printFloatArray(currentData->neuronVoltage, currentData->numNeurons, 1);
    printFloatArray(currentData->neuronReftime, currentData->numNeurons, 1);
    printFloatArray(currentData->ensembleOutput, currentData->totalOutputSize, 1);


    printFloatArray(currentData->output, currentData->totalOutputSize + currentData->numSpikesToSendBack, 1);
    printFloatArray(currentData->output_host, currentData->CPUOutputSize + currentData->numSpikesToSendBack, 1);
    //printFloatArray(currentData->spikes, currentData->numNeurons, 1);
    */
}


// m : the number of rows in the printout.
// n : the number of columns in the printout.
// Assumes the array is stored in row-major order.
void printIntArray(FILE* fp, intArray* a, int m, int n)
{
  if(!a)
    return;

  if(!a->array)
    return;


  if(a->on_device)
  {
    fp ? fprintf(fp, "On device: \n") : printf("On device:\n");
    printIntArrayFromDevice(fp, a, m, n, 0);
    return;
  }

  fp ? fprintf(fp, "On host:\n") : printf("On host:\n");

  fp ? fprintf(fp, "%s:\n", a->name) : printf("%s:\n", a->name);

  int i, j;
  for(i = 0; i < m; i++)
  {
    for(j = 0; j < n; j++)
    {
      fp ? fprintf(fp, "%d ", a->array[i * n + j]) : printf("%d ", a->array[i * n + j]);
    }
    fp ? fprintf(fp, "\n") : printf("\n");
  }

  fp ? fprintf(fp, "\n") : printf("\n");
}

// m : the number of rows in the printout.
// n : the number of columns in the printout.
// Assumes the array is stored in row-major order.
void printFloatArray(FILE* fp, floatArray* a, int m, int n)
{
  if(!a)
    return;

  if(!a->array)
    return;

  if(a->on_device)
  {
    fp ? fprintf(fp, "On device: \n") : printf("On device:\n");
    printFloatArrayFromDevice(fp, a, m, n, 0);
    return;
  }

  fp ? fprintf(fp, "On host:\n") : printf("On host:\n");

  fp ? fprintf(fp, "%s:\n", a->name) : printf("%s:\n", a->name);

  int i, j;

  for(i = 0; i < m; i++)
  {
    for(j = 0; j < n; j++)
    {
      fp ? fprintf(fp, "%f ", a->array[i * n + j]) :printf("%f ", a->array[i * n + j]);
    }

    fp ? fprintf(fp, "\n") : printf("\n");
  }

  fp ? fprintf(fp, "\n") : printf("\n");
}

void moveToDeviceIntArray(intArray* a)
{
  if(a->on_device)
    return;

  int* result;
  cudaError_t err;

  int size = a->size;

  err = cudaMalloc((void**)&result, size * sizeof(int));
  if(err)
  {
    printf("%s", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(result, a->array, size * sizeof(int), cudaMemcpyHostToDevice);
  if(err)
  {
    printf("%s", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  free(a->array);
  a->array = result;
  a->on_device = 1;
}

void moveToDeviceFloatArray(floatArray* a)
{
  if(a->on_device)
    return;

  float* result;
  cudaError_t err;

  int size = a->size;

  err = cudaMalloc((void**)&result, size * sizeof(float));
  if(err)
  {
    printf("%s", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(result, a->array, size * sizeof(float), cudaMemcpyHostToDevice);
  if(err)
  {
    printf("%s", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  free(a->array);
  a->array = result;
  a->on_device = 1;
}


void moveToHostFloatArray(floatArray* a)
{
  if(!a->on_device)
    return;

  float* result;
  int size = a->size;
  result = (float*)malloc(size * sizeof(float));
  cudaMemcpy(result, a->array, size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(a->array);
  a->array = result;
  a->on_device = 0;
}

void moveToHostIntArray(intArray* a)
{
  if(!a->on_device)
    return;

  int* result;
  int size = a->size;
  result = (int*)malloc(size * sizeof(int));
  cudaMemcpy(result, a->array, size * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(a->array);
  a->array = result;
  a->on_device = 0;
}

#ifdef __cplusplus
}
#endif

