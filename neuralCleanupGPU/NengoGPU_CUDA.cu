#ifdef __cplusplus
extern "C"{
#endif

#include <stdio.h>
#include <stdlib.h>
//#include <cuda_runtime.h>
//#include <cublas.h>

#include "NengoGPU.h"
#include "NengoGPU_CUDA.h"
#include "NengoGPUData.h"

# define MAX_SHARED_MEM_SIZE 16000

// print the contents of an array of integers located on the device
void printIntArrayFromDevice(FILE* fp, intArray* a, int n, int m, int labels)
{
  int* temp = (int*) malloc( m * n * sizeof(int));
  cudaMemcpy(temp, a->array, m * n * sizeof(int), cudaMemcpyDeviceToHost);

  printf("%s:\n", a->name);

  int i, j;
  for(i = 0; i < n; i++)
  {
    fp ? fprintf(fp, "line %d: ", i) : printf("line %d:", i);
    for(j = 0; j < m; j++)
    {
      if(labels)
        fp ? fprintf(fp, "(%d, %d) ", j, temp[i * n + j]) : printf("(%d, %d) ", j, temp[i * n + j]);
      else
        fp ? fprintf(fp, "%d ", temp[i * n + j]) : printf("%d ", temp[i * n + j]);
    }
    fp ? fprintf(fp, "\n") : printf("\n");
  }

  fp ? fprintf(fp, "\n") : printf("\n");

  free(temp);
}

// print the contents of an array of floats located on the device
void printFloatArrayFromDevice(FILE* fp, floatArray* a, int n, int m, int labels)
{
  cudaError_t err;
  float* temp = (float*) malloc( m * n * sizeof(float));
  err = cudaMemcpy(temp, a->array, m * n * sizeof(float), cudaMemcpyDeviceToHost);
  checkCudaError(err, "in printFloatArrayFromDevice, copying from device to host");

  printf("%s:\n", a->name);

  int i, j;
  for(i = 0; i < n; i++)
  {
    fp ? fprintf(fp, "line %d: ", i) : printf("line %d:", i);
    for(j = 0; j < m; j++)
    {
      if(labels)
        fp ? fprintf(fp, "(%d, %f) ", j, temp[i * m + j]) : printf("(%d, %f) ", j, temp[i * m + j]);
      else
        fp ? fprintf(fp, "%f ", temp[i * m + j]) : printf("%f ", temp[i * m + j]);
    }

    fp ? fprintf(fp, "\n") : printf("\n");
  }

  fp ? fprintf(fp, "\n") : printf("\n");

  free(temp);
}

void printIntColumn(FILE* fp, int* array, int m, int n, int col)
{
  int* temp = (int*) malloc( m * n * sizeof(int));
  cudaMemcpy(temp, array, m * n * sizeof(int), cudaMemcpyDeviceToHost);

  int i;
  for(i = 0; i < m; i++)
  {
    fp ? fprintf(fp, "%d ", temp[i * n + col]) : printf("%d ", temp[i * n + col]);
  }
  fp ? fprintf(fp, "\n") : printf("\n");
}

void printFloatColumn(FILE* fp, float* array, int m, int n, int col)
{
  float* temp = (float*) malloc( m * n * sizeof(float));
  cudaMemcpy(temp, array, m * n * sizeof(float), cudaMemcpyDeviceToHost);

  int i;
  for(i = 0; i < m; i++)
  {
    fp ? fprintf(fp, "%f ", temp[i * n + col]) : printf("%f ", temp[i * n + col]);
  }
  fp ? fprintf(fp, "\n") : printf("\n");
}

void printFloatRange(FILE* fp, float* array, int start, int end)
{
  float* temp = (float*) malloc((end - start + 1)  * sizeof(float));
  cudaMemcpy(temp, array + start, (end - start + 1) * sizeof(float), cudaMemcpyDeviceToHost);

  int i;
  for(i = 0; i < end - start + 1; i++)
  {
    fp ? fprintf(fp, "%f ", temp[i]) : printf("%f ", temp[i]);
  }
  fp ? fprintf(fp, "\n") : printf("\n");
}

void printIntRange(FILE* fp, int* array, int start, int end)
{
  int* temp = (int*) malloc((end - start + 1)  * sizeof(int));
  cudaMemcpy(temp, array + start, (end - start + 1) * sizeof(int), cudaMemcpyDeviceToHost);

  int i;
  for(i = 0; i < end - start + 1; i++)
  {
    fp ? fprintf(fp, "%d ", temp[i]) : printf("%d ", temp[i]);
  }
  fp ? fprintf(fp, "\n") : printf("\n");
}

// get number of devices available
int getGPUDeviceCount(){
  cudaError_t err;
  int numDevices;

  err = cudaGetDeviceCount(&numDevices);
  checkCudaError(err, "get GPU device count");

  return numDevices;
}

// Reserves device with number deviceNum for the thread that calls this function.
// No interaction with the device should take place until this has been called.
// Once the device is reserved for the thread, no other thread should try to interact
// with that device or reserve it. A thread can reserve only one device at a time
void initGPUDevice(int deviceNum)
{
  cudaError_t err = cudaSetDevice(deviceNum);
  checkCudaErrorWithDevice(err, deviceNum, "acquiring device");
}

void shutdownGPUDevice()
{
}

void checkCudaErrorWithDevice(cudaError_t err, int device, char* message)
{
  if(!err)
      return;

  printf("device: %d", device);
  checkCudaError(err, message);
}

void checkCudaError(cudaError_t err, char* message)
{
    if(!err)
        return;

    printf(" CUDA ERROR: message: %s, description: %s\n", message, cudaGetErrorString(err));

    exit(EXIT_FAILURE);
}

__global__ void integrateAfterEncode(int numNeurons, int numNeuronsPerItem, float dt, float adjusted_dt, int steps, float* encodeResult, float* voltage_array, float* reftime_array, float tau_rc, float tau_ref, float* bias, float* scale, float* spikes)
{
  int i = threadIdx.x + (blockDim.x * threadIdx.y) + (blockIdx.x + (gridDim.x * blockIdx.y)) * blockDim.x * blockDim.y;

  if( i < numNeurons)
  {
    int index = i % numNeuronsPerItem;

    float voltage = voltage_array[i];
    float refTime = reftime_array[i];
    float current = bias[index] + scale[index] * encodeResult[i];

    float dV, post_ref, spike_float, v_threshold = 1.0;
    int spike = 0.0, j;

    for(j = 0; j < steps; j++)
    {
      dV = adjusted_dt / tau_rc * (current - voltage);
      voltage = max(voltage + dV, 0.0f);

      post_ref = 1.0f - (refTime - adjusted_dt) / adjusted_dt;

      voltage = post_ref >= 1.0f ? voltage : voltage * post_ref;

      voltage = post_ref <= 0.0f ? 0.0f : voltage;

      spike = spike ? spike : voltage > v_threshold;
      spike_float = spike ? 1.0f/dt : 0.0f;
      refTime = spike ? ((adjusted_dt / dV) * (dV - voltage + v_threshold)) + tau_ref : refTime - adjusted_dt;
      voltage = spike ? 0.0 : voltage;
    }

    reftime_array[i] = refTime;
    voltage_array[i] = voltage;
    spikes[i] = spike_float;
  }
}

__global__ void moveGPUData(int size, int* map, float* to, float* from)
{
  int i = threadIdx.x + (blockDim.x * threadIdx.y) + (blockIdx.x + (gridDim.x * blockIdx.y)) * blockDim.x * blockDim.y;

  if(i < size)
  {
    to[i] = from[ map[i] ];
  }
}

// run a NengoGPUData struct for one step
void run_NEFEnsembles(NengoGPUData* nengoData, float startTime, float endTime)
{
  if(!nengoData->handleInitialized)
  {
    cublasCreate(&(nengoData->handle));
    nengoData->handleInitialized = 1;
  }

  nengoData->startTime = startTime;
  nengoData->endTime = endTime;

  float dt = endTime - startTime;

  //printf("start time: %f, end time %f, dt: %f, device: %d\n", startTime, endTime, dt, nengoData->device);

  cudaError_t err;

  dim3 dimBlock(1, 1);
  dim3 dimGrid(1, 1);

  int steps = 1;
  float adjusted_dt = dt;

  // Copy input from host to GPU

  cudaMemcpy(nengoData->inputDevice->array, nengoData->inputHost->array,
            (nengoData->dimension) * sizeof(float), cudaMemcpyHostToDevice);

  err = cudaGetLastError();
  checkCudaErrorWithDevice(err, nengoData->device, "run_NEFEnsembles: copying cpu input to device");

  //status = cublasSscal(*(nengoData->handle), nengoData->numItems, &scale, nengoData->transformResult->array, 1);

  // Multiply input vectors by corresponding termination transform
  cublasOperation_t op = CUBLAS_OP_T;
  cublasStatus_t status;

  float dt_over_tau = nengoData->dt / nengoData->tau;
  float scale = 1.0 - dt_over_tau;
  int lda = nengoData->dimension;

  status = cublasSgemv(nengoData->handle, op, nengoData->dimension, nengoData->numItems, &dt_over_tau,
                       nengoData->index_vectors->array, lda, nengoData->inputDevice->array, 1,
                       &scale, nengoData->transformResult->array, 1);

  cublasOperation_t opA = CUBLAS_OP_N;
  cublasOperation_t opB = CUBLAS_OP_N;

  // Multiply by encoders
  float one = 1.0;
  float zero = 0.0;
  lda = nengoData->numNeuronsPerItem;

  status = cublasSgemm(nengoData->handle, opA, opB, nengoData->numNeuronsPerItem, nengoData->numItems,
                       1, &one, nengoData->encoder->array, lda, nengoData->transformResult->array,
                       1, &zero, nengoData->encodeResult->array, nengoData->numNeuronsPerItem);


  dimBlock.x = 256;
  dimGrid.x = nengoData->numNeuronsPerItem  * nengoData->numItems/ dimBlock.x + 1;

  integrateAfterEncode<<<dimGrid, dimBlock>>>(nengoData->numNeuronsPerItem * nengoData->numItems, nengoData->numNeuronsPerItem,
                                              nengoData->dt, adjusted_dt, steps, nengoData->encodeResult->array,
                                              nengoData->voltage->array, nengoData->reftime->array, nengoData->tau_rc,
                                              nengoData->tau_ref, nengoData->Jbias->array, nengoData->alpha->array,
                                              nengoData->spikes->array);

  err = cudaGetLastError();
  checkCudaErrorWithDevice(err, nengoData->device, "run_NEFEnsembles: integrate after encode");

  // op has to be transposed make sure we get the decoder that corresponds to the thresholding function, not the identity decoder
  op = CUBLAS_OP_T;
  status = cublasSgemv(nengoData->handle, op, nengoData->numNeuronsPerItem, nengoData->numItems,
                       &one, nengoData->spikes->array, nengoData->numNeuronsPerItem,
                       nengoData->decoder->array, 1, &zero, nengoData->decodedValues->array, 1);

  if(nengoData->stop_early){
    // Don't do the weighting, just return values decoded from the association populations
    cudaMemcpy(nengoData->decodedValuesHost->array, nengoData->decodedValues->array, (nengoData->numItems) * sizeof(float), cudaMemcpyDeviceToHost);
  }
  else
  {
    // Multiplying the matrix whose columns are the result vectors by the vector of values
    // decoded from the association populations. The result is the decoded vector that is fed 
    // into the output population.  op should not be transposed here.
    op = CUBLAS_OP_N;
    status = cublasSgemv(nengoData->handle, op, nengoData->dimension, nengoData->numItems, &one,
                         nengoData->result_vectors->array, nengoData->dimension, nengoData->decodedValues->array,
                         1, &zero, nengoData->outputDevice->array, 1);

    // Move results to host
    cudaMemcpy(nengoData->outputHost->array, nengoData->outputDevice->array, (nengoData->dimension) * sizeof(float), cudaMemcpyDeviceToHost);
  }

  if(nengoData->numSpikesToReturn > 0)
  {
    dimBlock.x = 256;
    dimGrid.x = nengoData->numSpikesToReturn/ dimBlock.x + 1;
    moveGPUData<<<dimBlock,dimGrid>>>(nengoData->numSpikesToReturn, nengoData->spikeMap->array, nengoData->spikesOutput->array, nengoData->spikes->array);

    // move requested spikes to host
    cudaMemcpy(nengoData->spikesHost->array, nengoData->spikesOutput->array, nengoData->numSpikesToReturn * sizeof(float), cudaMemcpyDeviceToHost);
  }

  //printf("After step: %f, %f\n", startTime, endTime);
  //printNengoGPUData(nengoData, 1);
}

float* allocateCudaFloatArray(int size)
{
  float* temp;
  cudaError_t err;
  err = cudaMalloc((void**)&temp, size * sizeof(float));
  checkCudaError(err, "allocate cuda float array");
  return temp;
}

int* allocateCudaIntArray(int size)
{
  int* temp;
  cudaError_t err;
  err = cudaMalloc((void**)&temp, size * sizeof(int));
  checkCudaError(err, "allocate cuda int array");
  return temp;
}

long getDeviceCapacity(int device)
{
  cudaDeviceProp deviceProperties;
  cudaGetDeviceProperties(&deviceProperties, device);  
  return deviceProperties.totalGlobalMem;
}

// Create arrays that hold state information
void initializeDeviceInputAndOutput(NengoGPUData* nengoData)
{
  if(nengoData->do_print)
    printf("Initializing input and output: device %d\n", nengoData->device);

  char* name;

  name = "inputDevice";
  nengoData->inputDevice = newFloatArrayOnDevice(nengoData->dimension, name);
  name = "transformResult";
  nengoData->transformResult = newFloatArrayOnDevice(nengoData->numItems, name);
  name = "encodeResult";
  nengoData->encodeResult = newFloatArrayOnDevice(nengoData->numItems * nengoData->numNeuronsPerItem, name);
  name = "decodedValues";
  nengoData->decodedValues = newFloatArrayOnDevice(nengoData->numItems, name);
  name = "outputDevice";
  nengoData->outputDevice = newFloatArrayOnDevice(nengoData->dimension, name);

  name = "voltage";
  nengoData->voltage = newFloatArrayOnDevice(nengoData->numNeuronsPerItem * nengoData->numItems, name);
  name = "reftime";
  nengoData->reftime = newFloatArrayOnDevice(nengoData->numNeuronsPerItem * nengoData->numItems, name);
  name = "spikes";
  nengoData->spikes = newFloatArrayOnDevice(nengoData->numNeuronsPerItem * nengoData->numItems, name);
  name = "spikesOutput";
  nengoData->spikesOutput = newFloatArrayOnDevice(nengoData->numSpikesToReturn, name);

  reset_NEFEnsembles(nengoData);
}

void reset_NEFEnsembles(NengoGPUData* nengoData)
{
  /*
  cudaError_t err;
  printf("Resetting NEF fields: device %d\n", nengoData->device);
  err = cudaMemset(nengoData->inputDevice->array, 0, sizeof(float) * nengoData->dimension);
  checkCudaErrorWithDevice(err, nengoData->device, "cuda setup structures");
  err = cudaMemset(nengoData->transformResult->array, 0, sizeof(float) * nengoData->numItems);
  checkCudaErrorWithDevice(err, nengoData->device, "cuda setup structures");
  err = cudaMemset(nengoData->encodeResult->array, 0, sizeof(float) * nengoData->numNeuronsPerItem * nengoData->numItems);
  checkCudaErrorWithDevice(err, nengoData->device, "cuda setup structures");
  err = cudaMemset(nengoData->decodedValues, 0, sizeof(float) * nengoData->numItems);
  checkCudaErrorWithDevice(err, nengoData->device, "cuda setup structures");
  err = cudaMemset(nengoData->outputDevice, 0, sizeof(float) * nengoData->dimension);
  checkCudaErrorWithDevice(err, nengoData->device, "cuda setup structures");
  
  err = cudaMemset(nengoData->voltage, 0, sizeof(float) * nengoData->numNeuronsPerItem * nengoData->numItems);
  checkCudaErrorWithDevice(err, nengoData->device, "cuda setup structures");
  err = cudaMemset(nengoData->reftime, 0, sizeof(float) * nengoData->numNeuronsPerItem * nengoData->numItems);
  checkCudaErrorWithDevice(err, nengoData->device, "cuda setup structures");
  err = cudaMemset(nengoData->spikes, 0, sizeof(float) * nengoData->numNeuronsPerItem * nengoData->numItems);
  checkCudaErrorWithDevice(err, nengoData->device, "cuda setup structures");
  printf("Done resetting NEF fields: device %d\n", nengoData->device);
  */
  cudaError_t err;

  if(nengoData->do_print)
    printf("Resetting NEF fields: device %d\n", nengoData->device);

  err = cudaMemset(nengoData->inputDevice->array, 0, sizeof(float) * nengoData->dimension);
  checkCudaErrorWithDevice(err, nengoData->device, "cuda setup structures");
  err = cudaMemset(nengoData->transformResult->array, 0, sizeof(float) * nengoData->numItems);
  checkCudaErrorWithDevice(err, nengoData->device, "cuda setup structures");
  err = cudaMemset(nengoData->encodeResult->array, 0, sizeof(float) * nengoData->numNeuronsPerItem * nengoData->numItems);
  checkCudaErrorWithDevice(err, nengoData->device, "cuda setup structures");
  err = cudaMemset(nengoData->decodedValues->array, 0, sizeof(float) * nengoData->numItems);
  checkCudaErrorWithDevice(err, nengoData->device, "cuda setup structures");
  err = cudaMemset(nengoData->outputDevice->array, 0, sizeof(float) * nengoData->dimension);
  checkCudaErrorWithDevice(err, nengoData->device, "cuda setup structures");

  err = cudaMemset(nengoData->voltage->array, 0, sizeof(float) * nengoData->numNeuronsPerItem * nengoData->numItems);
  checkCudaErrorWithDevice(err, nengoData->device, "cuda setup structures");
  err = cudaMemset(nengoData->reftime->array, 0, sizeof(float) * nengoData->numNeuronsPerItem * nengoData->numItems);
  checkCudaErrorWithDevice(err, nengoData->device, "cuda setup structures");
  err = cudaMemset(nengoData->spikes->array, 0, sizeof(float) * nengoData->numNeuronsPerItem * nengoData->numItems);
  checkCudaErrorWithDevice(err, nengoData->device, "cuda setup structures");

  if(nengoData->do_print)
    printf("Done resetting NEF fields: device %d\n", nengoData->device);
}

#ifdef __cplusplus
}
#endif

