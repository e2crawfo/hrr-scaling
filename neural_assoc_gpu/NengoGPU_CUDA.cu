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

// Print the contents of an array of integers located on the device
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

// Print the contents of an array of floats located on the device
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

// Get number of devices available
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

__global__ void lif_math(int numNeurons, int neurons_per_item, float dt, float* encode_result,
                         float* voltage_array, float* reftime_array, float tau_rc,
                         float tau_ref, float* bias, float* scale, float* spikes)
{
  int i = threadIdx.x + (blockDim.x * threadIdx.y)
            + (blockIdx.x + (gridDim.x * blockIdx.y)) * blockDim.x * blockDim.y;

  if( i < numNeurons)
  {
    int index = i % neurons_per_item;

    float voltage = voltage_array[i];
    float reftime = reftime_array[i];
    float current = bias[i] + scale[i] * encode_result[i];
    //float current = bias[index] + scale[index] * encode_result[i];

    float dV, post_ref, spike;

    dV = dt / tau_rc * (current - voltage);
    voltage = max(voltage + dV, 0.0f);

    ref_time -= dt;

    post_ref = 1.0f - reftime / dt;

    voltage = post_ref >= 1.0f ? voltage : voltage * post_ref;
    voltage = post_ref <= 0.0f ? 0.0f : voltage;

    //spike = spike ? spike : voltage > v_threshold;
    spike = (float)(voltage > 1.0);

    if(spike){
        spike_time = dt * (1 - (voltage - 1.0) / dV);
        reftime = self.tau_ref + spike_time;
        voltage = 0.0;
    }

    reftime_array[i] = reftime;
    voltage_array[i] = voltage;
    spikes[i] = spike;
  }
}

// Assumes A and B have the same layout. Each kernel handles one row of A/B
// computing their dot product and storing the result in the appropriate place in C.
// C has to have size equal to the number of rows in A/B (ie stride).
__global__ void dot_product(int vector_length, long int stride, float* A, float* B, float* C)
{
  long int i = threadIdx.x + (blockDim.x * threadIdx.y)
                + (blockIdx.x + (gridDim.x * blockIdx.y)) * blockDim.x * blockDim.y;

  if( i < stride)
  {
      int j;
      float val = 0.0;
      for(j = i; j < vector_length; j=j+stride)
      {
          val += A[j] * B[j]
      }

      C[i] = val;
  }
}

// Run a NengoGPUData struct for one step.
void run_neural_associative_memory(NengoGPUData* nengo_data, float start_time, float end_time)
{
  if(!nengo_data->handle_initialized)
  {
    cublasCreate(&(nengo_data->handle));
    nengo_data->handle_initialized = 1;
  }

  nengo_data->start_time = start_time;
  nengo_data->end_time = end_time;

  float dt = end_time - start_time;

  //printf("start time: %f, end time %f, dt: %f, device: %d\n",
  //       start_time, end_time, dt, nengo_data->device);

  cudaError_t err;

  dim3 dimBlock(1, 1);
  dim3 dimGrid(1, 1);

  // Copy input from host to GPU
  cudaMemcpy(nengo_data->input_device->array, nengo_data->input_host->array,
            (nengo_data->dimension) * sizeof(float), cudaMemcpyHostToDevice);

  err = cudaGetLastError();
  checkCudaErrorWithDevice(err, nengo_data->device, "run_neural_associative_memory: copying cpu input to device");


  // Multiply input vectors by corresponding termination transform
  cublasOperation_t op = CUBLAS_OP_T;
  cublasStatus_t status;

  float dt_over_tau = nengo_data->dt / nengo_data->tau;
  float scale = 1.0 - dt_over_tau;
  int lda = nengo_data->dimension;

  status = cublasSgemv(nengo_data->handle, op, nengo_data->dimension, nengo_data->num_items,
                       &dt_over_tau, nengo_data->index_vectors->array, lda,
                       nengo_data->input_device->array, 1, &scale,
                       nengo_data->transformResult->array, 1);

  dimBlock.x = 256;
  dimGrid.x = nengo_data->neurons_per_item  * nengo_data->num_items / dimBlock.x + 1;

  lif_math<<<dimGrid, dimBlock>>>(nengo_data->neurons_per_item * nengo_data->num_items,
                                  nengo_data->neurons_per_item, nengo_data->dt, dt,
                                  nengo_data->encode_result->array, nengo_data->voltage->array,
                                  nengo_data->reftime->array, nengo_data->tau_rc,
                                  nengo_data->tau_ref, nengo_data->bias->array,
                                  nengo_data->gain->array, nengo_data->spikes->array);

  err = cudaGetLastError();
  checkCudaErrorWithDevice(err, nengo_data->device, "run_neural_associative_memory: lif math");

  // op has to be transposed make sure we get the decoder that
  // corresponds to the thresholding function, not the identity decoder
  float one = 1.0;
  float zero = 0.0;

  if(nengo_data->identical_ensembles)
  {
      // decoded_values(num_items, 1) =
      //    lif_output(num_items, neurons_per_item) x decoder(neurons_per_item, 1)
      op = CUBLAS_OP_T;
      status = cublasSgemv(nengo_data->handle, op, nengo_data->neurons_per_item,
                           nengo_data->num_items, &one, nengo_data->spikes->array,
                           nengo_data->neurons_per_item, nengo_data->decoder->array,
                           1, &zero, nengo_data->decoded_values->array, 1);
  }
  else
  {
      // dot 
      dot_product();
  }

  // Multiplying the matrix whose columns are the result vectors by the vector of values
  // decoded from the association populations. The result is the decoded vector that is fed
  // into the output population.  op should not be transposed here.
  //
  // output_vector(dimension, 1) =
  //    stored_vectors(dimension, num_items) x decoded_values(num_items, 1)
  op = CUBLAS_OP_N;
  status = cublasSgemv(nengo_data->handle, op, nengo_data->dimension, nengo_data->num_items,
                       &one, nengo_data->stored_vectors->array, nengo_data->dimension,
                       nengo_data->decoded_values->array, 1, &zero,
                       nengo_data->output_device->array, 1);

  // Move results to host
  cudaMemcpy(nengo_data->output_host->array, nengo_data->output_device->array,
             (nengo_data->dimension) * sizeof(float), cudaMemcpyDeviceToHost);

  //printf("After step: %f, %f\n", start_time, end_time);
  //printNengoGPUData(nengo_data, 1);
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
void initializeDeviceInputAndOutput(NengoGPUData* nengo_data)
{
  if(nengo_data->do_print)
    printf("Initializing input and output: device %d\n", nengo_data->device);

  char* name;

  name = "input_device";
  nengo_data->input_device = newFloatArrayOnDevice(nengo_data->dimension, name);
  name = "encode_result";
  nengo_data->encode_result = newFloatArrayOnDevice(nengo_data->num_items * nengo_data->neurons_per_item, name);
  name = "decoded_values";
  nengo_data->decoded_values = newFloatArrayOnDevice(nengo_data->num_items, name);
  name = "output_device";
  nengo_data->output_device = newFloatArrayOnDevice(nengo_data->dimension, name);

  name = "voltage";
  nengo_data->voltage = newFloatArrayOnDevice(nengo_data->neurons_per_item * nengo_data->num_items, name);
  name = "reftime";
  nengo_data->reftime = newFloatArrayOnDevice(nengo_data->neurons_per_item * nengo_data->num_items, name);
  name = "spikes";
  nengo_data->spikes = newFloatArrayOnDevice(nengo_data->neurons_per_item * nengo_data->num_items, name);

  reset_NEFEnsembles(nengo_data);
}

void reset_neural_associative_memory(NengoGPUData* nengo_data)
{
  /*
  cudaError_t err;
  printf("Resetting NEF fields: device %d\n", nengo_data->device);
  err = cudaMemset(nengo_data->input_device->array, 0, sizeof(float) * nengo_data->dimension);
  checkCudaErrorWithDevice(err, nengo_data->device, "cuda setup structures");
  err = cudaMemset(nengo_data->transformResult->array, 0, sizeof(float) * nengo_data->num_items);
  checkCudaErrorWithDevice(err, nengo_data->device, "cuda setup structures");
  err = cudaMemset(nengo_data->encode_result->array, 0, sizeof(float) * nengo_data->neurons_per_item * nengo_data->num_items);
  checkCudaErrorWithDevice(err, nengo_data->device, "cuda setup structures");
  err = cudaMemset(nengo_data->decoded_values, 0, sizeof(float) * nengo_data->num_items);
  checkCudaErrorWithDevice(err, nengo_data->device, "cuda setup structures");
  err = cudaMemset(nengo_data->output_device, 0, sizeof(float) * nengo_data->dimension);
  checkCudaErrorWithDevice(err, nengo_data->device, "cuda setup structures");

  err = cudaMemset(nengo_data->voltage, 0, sizeof(float) * nengo_data->neurons_per_item * nengo_data->num_items);
  checkCudaErrorWithDevice(err, nengo_data->device, "cuda setup structures");
  err = cudaMemset(nengo_data->reftime, 0, sizeof(float) * nengo_data->neurons_per_item * nengo_data->num_items);
  checkCudaErrorWithDevice(err, nengo_data->device, "cuda setup structures");
  err = cudaMemset(nengo_data->spikes, 0, sizeof(float) * nengo_data->neurons_per_item * nengo_data->num_items);
  checkCudaErrorWithDevice(err, nengo_data->device, "cuda setup structures");
  printf("Done resetting NEF fields: device %d\n", nengo_data->device);
  */
  cudaError_t err;

  if(nengo_data->do_print)
    printf("Resetting NEF fields: device %d\n", nengo_data->device);

  err = cudaMemset(nengo_data->input_device->array, 0, sizeof(float) * nengo_data->dimension);
  checkCudaErrorWithDevice(err, nengo_data->device, "cuda setup structures");
  err = cudaMemset(nengo_data->encode_result->array, 0, sizeof(float) * nengo_data->neurons_per_item * nengo_data->num_items);
  checkCudaErrorWithDevice(err, nengo_data->device, "cuda setup structures");
  err = cudaMemset(nengo_data->decoded_values->array, 0, sizeof(float) * nengo_data->num_items);
  checkCudaErrorWithDevice(err, nengo_data->device, "cuda setup structures");
  err = cudaMemset(nengo_data->output_device->array, 0, sizeof(float) * nengo_data->dimension);
  checkCudaErrorWithDevice(err, nengo_data->device, "cuda setup structures");

  err = cudaMemset(nengo_data->voltage->array, 0, sizeof(float) * nengo_data->neurons_per_item * nengo_data->num_items);
  checkCudaErrorWithDevice(err, nengo_data->device, "cuda setup structures");
  err = cudaMemset(nengo_data->reftime->array, 0, sizeof(float) * nengo_data->neurons_per_item * nengo_data->num_items);
  checkCudaErrorWithDevice(err, nengo_data->device, "cuda setup structures");
  err = cudaMemset(nengo_data->spikes->array, 0, sizeof(float) * nengo_data->neurons_per_item * nengo_data->num_items);
  checkCudaErrorWithDevice(err, nengo_data->device, "cuda setup structures");

  if(nengo_data->do_print)
    printf("Done resetting NEF fields: device %d\n", nengo_data->device);
}

#ifdef __cplusplus
}
#endif

