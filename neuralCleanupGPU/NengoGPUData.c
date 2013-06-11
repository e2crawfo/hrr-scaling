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
  new->onDevice = 0;

  return new;
}

void freeIntArray(intArray* a)
{
  if(!a)
    return;

  if(a->array)
    a->onDevice ? cudaFree(a->array) : free(a->array);
  
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
  new->onDevice = 1;

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
  new->onDevice = 0;

  return new;
}

void freeFloatArray(floatArray* a)
{
  //printf("freeing %s, size: %d, onDevice: %d, address: %d\n", a->name, a->size, a->onDevice, (int)a->array);

  if(!a)
    return;

  if(a->array)
    a->onDevice ? cudaFree(a->array) : free(a->array);
  
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
  new->onDevice = 1;

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

void checkLocation(char* verb, char* name, int onDevice, int size, int index)
{
  if(onDevice)
  {
    printf("%s safe array that is not on the host, name: %s, size: %d, index:%d\n", verb, name, size, index);
    exit(EXIT_FAILURE);
  }
}

void intArraySetElement(intArray* a, int index, int value)
{
  checkBounds("Setting", a->name, a->size, index);
  checkLocation("Setting", a->name, a->onDevice, a->size, index);

  a->array[index] = value;
}

void floatArraySetElement(floatArray* a, int index, float value)
{
  checkBounds("Setting", a->name, a->size, index);
  checkLocation("Setting", a->name, a->onDevice, a->size, index);

  a->array[index] = value;
}

int intArrayGetElement(intArray* a, int index)
{
  checkBounds("Getting", a->name, a->size, index);
  checkLocation("Getting", a->name, a->onDevice, a->size, index);

  return a->array[index];
}

float floatArrayGetElement(floatArray* a, int index)
{
  checkBounds("Getting", a->name, a->size, index);
  checkLocation("Getting", a->name, a->onDevice, a->size, index);

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
  new->onDevice = 0;
  new->device = 0;
  new->maxTimeStep = 0;
  new->initialized = 0;
  new->do_print = 0;
  new->stop_early = 0;

  new->startTime = 0.0;
  new->endTime = 0.0;
   
  new->numNeuronsPerItem = 0;
  new->dimension = 0;
  new->numItems = 0;
  new->autoassociative = 0;
  new->numSpikesToReturn = 0;
  new->tau = 0;
  new->pstc = 0;
  new->tau_ref = 0;
  new->tau_rc = 0;
  new->dt = 0;
  //new->handle = NULL;
  new->handleInitialized = 0;

  new->inputHost = NULL;
  new->inputDevice = NULL;
  new->transformResult = NULL;
  new->encodeResult = NULL;
  new->decodedValues = NULL;
  new->decodedValuesHost = NULL;
  new->outputHost = NULL;
  new->outputDevice = NULL;

  new->index_vectors = NULL;
  new->encoder = NULL;
  new->decoder = NULL;
  new->result_vectors = NULL;

  new->alpha = NULL;
  new->Jbias = NULL;
  new->voltage = NULL;
  new->reftime = NULL;
  new->spikes = NULL;

  new->spikeMap = NULL;
  new->spikesHost = NULL;
  new->spikesOutput = NULL;

  return new;
}


// Should only be called once the NengoGPUData's numerical values have been set. This function allocates memory of the approprate size for each pointer.
// Memory is allocated on the host. The idea is to call this before we load the data in from the JNI structures, so we have somewhere to put that data. Later, we will move most of the data to the device.
void initializeNengoGPUData(NengoGPUData* new)
{
  if(new == NULL)
  {
     return;
  }

  char filename[50];
  int err;
  err = sprintf(filename, "../neuralCleanupGPU/gpuOutput/gpuOutput%d.txt", new->device);
  new->fp = fopen(filename, "w");

  char* name; 

  name = "inputHost";
  new->inputHost = newFloatArray(new->dimension, name);
  name = "outputHost";
  new->outputHost = newFloatArray(new->dimension, name);
  name = "decodedValuesHost";
  new->decodedValuesHost = newFloatArray(new->numItems, name);

  name = "index_vectors";
  new->index_vectors = newFloatArray(new->dimension * new->numItems, name);
  name = "result_vectors";
  new->result_vectors = newFloatArray(new->dimension * new->numItems, name);

  name = "encoder";
  new->encoder = newFloatArray(new->numNeuronsPerItem, name);
  name = "decoder";
  new->decoder = newFloatArray(new->numNeuronsPerItem, name);

  name = "alpha";
  new->alpha = newFloatArray(new->numNeuronsPerItem, name);
  name = "Jbias";
  new->Jbias = newFloatArray(new->numNeuronsPerItem, name);

  name = "spikesHost";
  new->spikesHost = newFloatArray(new->numSpikesToReturn, name);
  name = "spikeMap";
  new->spikeMap = newIntArray(new->numSpikesToReturn, name);
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
void moveToDeviceNengoGPUData(NengoGPUData* nengoData)
{
  if(!nengoData->onDevice)
  {
    // this function is in NengoGPU_CUDA.cu
    initializeDeviceInputAndOutput(nengoData);

    moveToDeviceFloatArray(nengoData->index_vectors);

    if(!nengoData->stop_early)
    {
      moveToDeviceFloatArray(nengoData->result_vectors);
    }

    moveToDeviceFloatArray(nengoData->encoder);
    moveToDeviceFloatArray(nengoData->decoder);
    moveToDeviceFloatArray(nengoData->alpha);
    moveToDeviceFloatArray(nengoData->Jbias);
    moveToDeviceIntArray(nengoData->spikeMap);

    nengoData->onDevice = 1;
  }
}

// Free the NengoGPUData. Makes certain assumptions about where each array is (device or host).
void freeNengoGPUData(NengoGPUData* nengoData)
{
  freeFloatArray(nengoData->inputHost);
  freeFloatArray(nengoData->inputDevice);
  freeFloatArray(nengoData->transformResult);
  freeFloatArray(nengoData->encodeResult);
  freeFloatArray(nengoData->decodedValues);
  freeFloatArray(nengoData->decodedValuesHost);
  freeFloatArray(nengoData->outputHost);
  freeFloatArray(nengoData->outputDevice);

  freeFloatArray(nengoData->index_vectors);
  freeFloatArray(nengoData->result_vectors);

  freeFloatArray(nengoData->encoder);
  freeFloatArray(nengoData->decoder);

  freeFloatArray(nengoData->alpha);
  freeFloatArray(nengoData->Jbias);
  freeFloatArray(nengoData->voltage);
  freeFloatArray(nengoData->reftime);
  freeFloatArray(nengoData->spikes);

  freeFloatArray(nengoData->spikesHost);
  freeFloatArray(nengoData->spikesOutput);
  freeIntArray(nengoData->spikeMap);
  
  if(nengoData->fp)
    fclose(nengoData->fp);

  if(nengoData->handleInitialized)
    cublasDestroy(nengoData->handle);
  
  free(nengoData);
};


void printVecs(NengoGPUData* nengoData)
{
  nengoData->fp ? fprintf(nengoData->fp, "printing index vectors:\n") : printf("printing index vectors:\n");
  printFloatArray(nengoData->fp, nengoData->index_vectors, nengoData->dimension, nengoData->numItems);
  nengoData->fp ? fprintf(nengoData->fp, "printing result vectors:\n") : printf("printing result vectors:\n");
  printFloatArray(nengoData->fp, nengoData->result_vectors, nengoData->dimension, nengoData->numItems);
}

// print the NengoGPUData. Should only be called once the data has been set.
void printNengoGPUData(NengoGPUData* nengoData, int printArrays)
{
  
  nengoData->fp ? fprintf(nengoData->fp, "printing NengoGPUData:\n") : printf("printing NengoGPUData:\n");

  nengoData->fp ? fprintf(nengoData->fp, "startTime; %f\n", nengoData->startTime) : printf("startTime; %f\n", nengoData->startTime);
  nengoData->fp ? fprintf(nengoData->fp, "endTime; %f\n", nengoData->endTime) : printf("endTime; %f\n", nengoData->endTime);

  nengoData->fp ? fprintf(nengoData->fp, "onDevice; %d\n", nengoData->onDevice) : printf("onDevice; %d\n", nengoData->onDevice);
  nengoData->fp ? fprintf(nengoData->fp, "initialized; %d\n", nengoData->initialized) : printf("initialized; %d\n", nengoData->initialized);
  nengoData->fp ? fprintf(nengoData->fp, "device; %d\n", nengoData->device) : printf("device; %d\n", nengoData->device);
  nengoData->fp ? fprintf(nengoData->fp, "maxTimeStep; %f\n", nengoData->maxTimeStep) : printf("maxTimeStep; %f\n", nengoData->maxTimeStep);
   
  nengoData->fp ? fprintf(nengoData->fp, "numNeuronsPerItem: %d\n", nengoData->numNeuronsPerItem) : printf("numNeuronsPerItem: %d\n", nengoData->numNeuronsPerItem);
  nengoData->fp ? fprintf(nengoData->fp, "dimension: %d\n", nengoData->dimension) : printf("dimension: %d\n", nengoData->dimension);
  nengoData->fp ? fprintf(nengoData->fp, "numItems: %d\n", nengoData->numItems) : printf("numItems: %d\n", nengoData->numItems);
  nengoData->fp ? fprintf(nengoData->fp, "autoassociative: %d\n", nengoData->autoassociative) : printf("autoassociative: %d\n", nengoData->autoassociative);
  nengoData->fp ? fprintf(nengoData->fp, "tau: %f\n", nengoData->tau) : printf("tau: %f\n", nengoData->tau);
  nengoData->fp ? fprintf(nengoData->fp, "pstc: %f\n", nengoData->pstc) : printf("pstc: %f\n", nengoData->pstc);
  nengoData->fp ? fprintf(nengoData->fp, "tau_ref: %f\n", nengoData->tau_ref) : printf("tau_ref: %f\n", nengoData->tau_ref);
  nengoData->fp ? fprintf(nengoData->fp, "tau_rc: %f\n", nengoData->tau_rc) : printf("tau_rc: %f\n", nengoData->tau_rc);
  nengoData->fp ? fprintf(nengoData->fp, "dt: %f\n", nengoData->dt) : printf("dt: %f\n", nengoData->dt);
 
  if(printArrays)
  {
    printFloatArray(nengoData->fp, nengoData->inputHost, nengoData->dimension, 1);
    
    printFloatArray(nengoData->fp, nengoData->inputDevice, nengoData->dimension, 1);
    //printFloatArray(nengoData->fp, nengoData->transformResult, nengoData->numItems, 1);
    printFloatArrayFromDevice(nengoData->fp, nengoData->transformResult, 1, nengoData->numItems, 1);
    printFloatArray(nengoData->fp, nengoData->encodeResult, nengoData->numItems * nengoData->numNeuronsPerItem, 1);
    //printFloatArray(nengoData->fp, nengoData->decodedValues, nengoData->numItems, 1);
    printFloatArrayFromDevice(nengoData->fp, nengoData->decodedValues, 1, nengoData->numItems, 1);
    printFloatArray(nengoData->fp, nengoData->outputHost, nengoData->dimension, 1);
    printFloatArray(nengoData->fp, nengoData->outputDevice, nengoData->dimension, 1);

    //printFloatArray(nengoData->fp, nengoData->index_vectors, nengoData->dimension * nengoData->numItems, 1);
    //printFloatArray(nengoData->fp, nengoData->result_vectors, nengoData->dimension * nengoData->numItems, 1);

    printFloatArray(nengoData->fp, nengoData->encoder, nengoData->numNeuronsPerItem, 1);
    printFloatArray(nengoData->fp, nengoData->decoder, nengoData->numNeuronsPerItem, 1);

    printFloatArray(nengoData->fp, nengoData->alpha, nengoData->numNeuronsPerItem, 1);
    printFloatArray(nengoData->fp, nengoData->Jbias, nengoData->numNeuronsPerItem, 1);
    printFloatArray(nengoData->fp, nengoData->voltage, nengoData->numNeuronsPerItem, nengoData->numItems);
    printFloatArray(nengoData->fp, nengoData->reftime, nengoData->numNeuronsPerItem, nengoData->numItems);
    printFloatArray(nengoData->fp, nengoData->spikes, nengoData->numNeuronsPerItem, nengoData->numItems);

    printFloatArray(nengoData->fp, nengoData->spikesHost, nengoData->numSpikesToReturn, 1);
    printIntArray(nengoData->fp, nengoData->spikeMap, nengoData->numSpikesToReturn, 1);
    printFloatArray(nengoData->fp, nengoData->spikesOutput, nengoData->numSpikesToReturn, 1);
  }

//  int bytesOnGPU = sizeof(float) * (8 * currentData->numEnsembles + currentData->totalInputSize + currentData->totalTransformSize + 2 * currentData->totalNumTransformRows + 2 * currentData->numTerminations + currentData->maxNumDecodedTerminations * currentData->totalEnsembleDimension + currentData->maxDimension * currentData->numNeurons + currentData->totalEnsembleDimension + currentData->numNeurons * 6 + currentData->maxNumNeurons * currentData->totalOutputSize + 2 * currentData->totalOutputSize);
 // printf("bytes on GPU: %d\n", bytesOnGPU);
}

// this function doesn't really work! because the printFloatArrayFromDevice function shouldn't be called this way
void printDynamicNengoGPUData(NengoGPUData* nengoData)
{
  /*
    printFloatArray(currentData->input, currentData->totalInputSize, 1);
    
    printFloatArray(currentData->terminationOutput, currentData->totalNumTransformRows, 1);
    printFloatArray(currentData->encodeResult, currentData->numNeurons, 1);
    printFloatArray(currentData->ensembleSums, currentData->totalEnsembleDimension, 1);
    printFloatArray(currentData->neuronVoltage, currentData->numNeurons, 1);
    printFloatArray(currentData->neuronReftime, currentData->numNeurons, 1);
    printFloatArray(currentData->ensembleOutput, currentData->totalOutputSize, 1);
    
    
    printFloatArray(currentData->output, currentData->totalOutputSize + currentData->numSpikesToSendBack, 1);
    printFloatArray(currentData->outputHost, currentData->CPUOutputSize + currentData->numSpikesToSendBack, 1);
    //printFloatArray(currentData->spikes, currentData->numNeurons, 1);
    */
}


void printIntArray(FILE* fp, intArray* a, int n, int m)
{
  if(!a)
    return;

  if(!a->array)
    return;


  if(a->onDevice)
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

void printFloatArray(FILE* fp, floatArray* a, int n, int m)
{
  if(!a)
    return;

  if(!a->array)
    return;

  if(a->onDevice)
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
  if(a->onDevice)
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
  a->onDevice = 1;
}
  
void moveToDeviceFloatArray(floatArray* a)
{
  if(a->onDevice)
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
  a->onDevice = 1;
}
  
  
void moveToHostFloatArray(floatArray* a)
{
  if(!a->onDevice)
    return;

  float* result;
  int size = a->size;
  result = (float*)malloc(size * sizeof(float));
  cudaMemcpy(result, a->array, size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(a->array);
  a->array = result;
  a->onDevice = 0;
}

void moveToHostIntArray(intArray* a)
{
  if(!a->onDevice)
    return;

  int* result;
  int size = a->size;
  result = (int*)malloc(size * sizeof(int));
  cudaMemcpy(result, a->array, size * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(a->array);
  a->array = result;
  a->onDevice = 0;
}

#ifdef __cplusplus
}
#endif

