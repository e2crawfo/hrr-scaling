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
// m : the number of rows in the printout.
// n : the number of columns in the printout
// Assumes the array is stored in row-major order.
void printIntArrayFromDevice(FILE* fp, intArray* a, int m, int n, int labels)
{
  int* temp = (int*) malloc( m * n * sizeof(int));
  cudaMemcpy(temp, a->array, m * n * sizeof(int), cudaMemcpyDeviceToHost);

  printf("%s:\n", a->name);

  int i, j;
  for(i = 0; i < m; i++)
  {
    fp ? fprintf(fp, "line %d: ", i) : printf("line %d:", i);
    for(j = 0; j < n; j++)
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

// Print the contents of an array of ints located on the device
// m : the number of rows in the printout.
// n : the number of columns in the printout
// Assumes the array is stored in row-major order.
void printFloatArrayFromDevice(FILE* fp, floatArray* a, int m, int n, int labels)
{
  cudaError_t err;
  float* temp = (float*) malloc( m * n * sizeof(float));
  err = cudaMemcpy(temp, a->array, m * n * sizeof(float), cudaMemcpyDeviceToHost);
  checkCudaError(err, "in printFloatArrayFromDevice, copying from device to host");

  fp ? fprintf(fp, "%s:\n", a->name) : printf("%s:\n", a->name);

  int i, j;
  for(i = 0; i < m; i++)
  {
    fp ? fprintf(fp, "line %d: ", i) : printf("line %d:", i);
    for(j = 0; j < n; j++)
    {
      if(labels)
        fp ? fprintf(fp, "(%d, %f) ", j, temp[i * n + j]) : printf("(%d, %f) ", j, temp[i * n + j]);
      else
        fp ? fprintf(fp, "%f ", temp[i * n + j]) : printf("%f ", temp[i * n + j]);
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

void checkCudaErrorWithDevice(cudaError_t err, int device, const char* message)
{
  if(!err)
      return;

  printf("device: %d", device);
  checkCudaError(err, message);
}

void checkCudaError(cudaError_t err, const char* message)
{
    if(!err)
        return;

    printf(" CUDA ERROR: message: %s, description: %s\n", message, cudaGetErrorString(err));

    exit(EXIT_FAILURE);
}

__global__ void lif_math(int numNeurons, int neurons_per_item, float dt, float* encode_result,
                         float* voltage_array, float* reftime_array, float tau_rc,
                         float tau_ref, float* bias, float* gain, float* spikes)
{
  int i = threadIdx.x + (blockDim.x * threadIdx.y)
            + (blockIdx.x + (gridDim.x * blockIdx.y)) * blockDim.x * blockDim.y;

  if( i < numNeurons)
  {
    int neuron_index = i % neurons_per_item;
    int item_index = (int)(i / neurons_per_item);

    float voltage = voltage_array[i];
    float ref_time = reftime_array[i];
    float current = bias[neuron_index] + gain[neuron_index] * encode_result[item_index];
    float dV, spike, mult;

    dV = -expm1(-dt / tau_rc) * (current - voltage);
    voltage = max(voltage + dV, 0.0f);

    ref_time -= dt;

    mult = ref_time;
    mult *= -1.0 / dt;
    mult += 1.0;

    mult = mult > 1.0 ? 1.0 : mult;
    mult = mult < 0.0 ? 0.0 : mult;

    voltage *= mult;

    if(voltage > 1.0){
        spike = 1.0 / dt;
        ref_time = tau_ref + dt * (1.0 - (voltage - 1.0) / dV);
        voltage = 0.0;
    }else{
        spike = 0.0;
    }

    reftime_array[i] = ref_time;
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
          val += A[j] * B[j];
      }

      C[i] = val;
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

  printf("start time: %f, end time %f, device: %d\n",
         start_time, end_time, nengo_data->device);

  cudaError_t err;

  dim3 dimBlock(1, 1);
  dim3 dimGrid(1, 1);
  cublasOperation_t op;
  float one = 1.0;
  float zero = 0.0;
  float inv_radius = 1.0 / nengo_data->radius;
  int lda = nengo_data->dimension;

  int step;
  int input_offset = 0, output_offset = 0, probe_offset = 0, spike_offset = 0;

  // Copy input from host to GPU
  cudaMemcpy(nengo_data->input_device->array, nengo_data->input_host->array,
            (nengo_data->dimension * nengo_data->num_steps) * sizeof(float),
            cudaMemcpyHostToDevice);

  for(step = 0; step < nengo_data->num_steps; step++)
  {
      printf("NeuralAssocGPU running a step!");
      if(nengo_data->do_print && step % 10 == 0)
      {
          printf("NeuralAssocGPU: STEP %d\n", step);
      }

      // Multiply input vectors by corresponding index vector
      op = CUBLAS_OP_T;

      cublasSgemv(nengo_data->handle, op, nengo_data->dimension, nengo_data->num_items,
                  &inv_radius, nengo_data->index_vectors->array, lda,
                  nengo_data->input_device->array + input_offset, 1, &zero,
                  nengo_data->encode_result->array, 1);

      input_offset += nengo_data->dimension;

      // Run lif math
      dimBlock.x = 256;
      dimGrid.x = nengo_data->neurons_per_item  * nengo_data->num_items / dimBlock.x + 1;

      lif_math<<<dimGrid, dimBlock>>>(nengo_data->neurons_per_item * nengo_data->num_items,
                                      nengo_data->neurons_per_item, nengo_data->dt,
                                      nengo_data->encode_result->array, nengo_data->voltage->array,
                                      nengo_data->reftime->array, nengo_data->tau_rc,
                                      nengo_data->tau_ref, nengo_data->bias->array,
                                      nengo_data->gain->array, nengo_data->spikes->array);

      err = cudaGetLastError();
      checkCudaErrorWithDevice(err, nengo_data->device,
                               "run_neural_associative_memory: lif math");

      if(nengo_data->identical_ensembles)
      {
          printf("Ensembles are identical!");
          // decoded_values(num_items, 1) =
          //    lif_output(num_items, neurons_per_item) x decoder(neurons_per_item, 1)
          op = CUBLAS_OP_T;
          cublasSgemv(nengo_data->handle, op, nengo_data->neurons_per_item,
                               nengo_data->num_items, &one, nengo_data->spikes->array,
                               nengo_data->neurons_per_item, nengo_data->decoders->array,
                               1, &zero, nengo_data->decoded_values->array, 1);
      }
      else
      {
          printf("Ensembles are NOT! identical!");
          dimBlock.x = 256;
          dimGrid.x = nengo_data->num_items / dimBlock.x + 1;

          dot_product<<<dimGrid, dimBlock>>>(nengo_data->dimension, nengo_data->num_items,
                                             nengo_data->spikes->array,
                                             nengo_data->decoders->array,
                                             nengo_data->decoded_values->array);
          err = cudaGetLastError();
          checkCudaErrorWithDevice(err, nengo_data->device,
                                "run_neural_associative_memory: decoding");
      }

      if(nengo_data->num_probes > 0)
      {
          dimBlock.x = 256;
          dimGrid.x = nengo_data->num_probes / dimBlock.x + 1;
          moveGPUData<<<dimBlock, dimGrid>>>(nengo_data->num_probes,
                                             nengo_data->probe_map->array,
                                             nengo_data->probes_device->array + probe_offset,
                                             nengo_data->decoded_values->array);
          probe_offset += nengo_data->num_probes;
      }

      if(nengo_data->num_spikes > 0)
      {

          dimGrid.x = nengo_data->num_spikes / dimBlock.x + 1;
          moveGPUData<<<dimBlock, dimGrid>>>(nengo_data->num_spikes,
                                             nengo_data->spike_map->array,
                                             nengo_data->spikes_device->array + spike_offset,
                                             nengo_data->spikes->array);

          spike_offset += nengo_data->num_spikes;
      }

      // Multiplying the matrix whose columns are the stored vectors by the vector of
      // values decoded from the association populations. The result is the decoded
      // vector that is fed into the output population.  op should not be transposed
      // here.
      //
      // output_vector(dimension, 1) =
      //    stored_vectors(dimension, num_items) x decoded_values(num_items, 1)
      op = CUBLAS_OP_N;
      cublasSgemv(nengo_data->handle, op, nengo_data->dimension, nengo_data->num_items,
                           &one, nengo_data->stored_vectors->array, nengo_data->dimension,
                           nengo_data->decoded_values->array, 1, &zero,
                           nengo_data->output_device->array + output_offset, 1);

      output_offset += nengo_data->dimension;

      err = cudaGetLastError();
      checkCudaErrorWithDevice(err, nengo_data->device,
                               "run_neural_associative_memory: copying cpu input to device");
  }

  printNengoGPUData(nengo_data, 1);

  // Move output and probes to host
  cudaMemcpy(nengo_data->output_host->array, nengo_data->output_device->array,
             (nengo_data->dimension * nengo_data->num_steps) * sizeof(float),
             cudaMemcpyDeviceToHost);

  if(nengo_data->num_probes > 0)
  {
      cudaMemcpy(nengo_data->probes_host->array, nengo_data->probes_device->array,
                 (nengo_data->num_probes * nengo_data->num_steps) * sizeof(float),
                 cudaMemcpyDeviceToHost);
  }

  if(nengo_data->num_spikes > 0)
  {
      cudaMemcpy(nengo_data->spikes_host->array, nengo_data->spikes_device->array,
                 (nengo_data->num_spikes * nengo_data->num_steps) * sizeof(float),
                 cudaMemcpyDeviceToHost);
  }
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

  nengo_data->input_device = newFloatArrayOnDevice(
      nengo_data->dimension * nengo_data->num_steps, "input_device");

  nengo_data->encode_result = newFloatArrayOnDevice(nengo_data->num_items, "encode_result");

  nengo_data->decoded_values = newFloatArrayOnDevice(nengo_data->num_items, "decoded_values");

  nengo_data->output_device = newFloatArrayOnDevice(
      nengo_data->dimension * nengo_data->num_steps, "output_device");

  nengo_data->voltage = newFloatArrayOnDevice(
      nengo_data->neurons_per_item * nengo_data->num_items, "voltage");

  nengo_data->reftime = newFloatArrayOnDevice(
      nengo_data->neurons_per_item * nengo_data->num_items, "reftime");

  nengo_data->spikes = newFloatArrayOnDevice(
      nengo_data->neurons_per_item * nengo_data->num_items, "spikes");

  nengo_data->probes_device = newFloatArrayOnDevice(
      nengo_data->num_probes * nengo_data->num_steps, "probes_device");

  nengo_data->spikes_device = newFloatArrayOnDevice(
        nengo_data->num_spikes * nengo_data->num_steps, "spikes_device");

  reset_neural_associative_memory(nengo_data);
}

void reset_neural_associative_memory(NengoGPUData* nengo_data)
{
  cudaError_t err;

  if(nengo_data->do_print)
    printf("Resetting NEF fields: device %d\n", nengo_data->device);

  err = cudaMemset(nengo_data->input_device->array, 0,
      sizeof(float) * nengo_data->dimension * nengo_data->num_steps);
  checkCudaErrorWithDevice(err, nengo_data->device, "Resetting Cuda array");

  err = cudaMemset(nengo_data->encode_result->array, 0,
      sizeof(float) * nengo_data->num_items);
  checkCudaErrorWithDevice(err, nengo_data->device, "Resetting Cuda arrays");

  err = cudaMemset(nengo_data->decoded_values->array, 0,
      sizeof(float) * nengo_data->num_items);
  checkCudaErrorWithDevice(err, nengo_data->device, "Resetting Cuda arrays");

  err = cudaMemset(nengo_data->output_device->array, 0,
      sizeof(float) * nengo_data->num_steps * nengo_data->dimension);
  checkCudaErrorWithDevice(err, nengo_data->device, "Resetting Cuda arrays");

  err = cudaMemset(nengo_data->voltage->array, 0,
      sizeof(float) * nengo_data->neurons_per_item * nengo_data->num_items);
  checkCudaErrorWithDevice(err, nengo_data->device, "Resetting Cuda arrays");

  err = cudaMemset(nengo_data->reftime->array, 0,
      sizeof(float) * nengo_data->neurons_per_item * nengo_data->num_items);
  checkCudaErrorWithDevice(err, nengo_data->device, "Resetting Cuda arrays");

  err = cudaMemset(nengo_data->spikes->array, 0,
      sizeof(float) * nengo_data->neurons_per_item * nengo_data->num_items);
  checkCudaErrorWithDevice(err, nengo_data->device, "Resetting Cuda arrays");

  err = cudaMemset(nengo_data->probes_device->array, 0,
      sizeof(float) * nengo_data->num_steps * nengo_data->num_probes);
  checkCudaErrorWithDevice(err, nengo_data->device, "Resetting Cuda arrays");

  err = cudaMemset(nengo_data->spikes_device->array, 0,
      sizeof(float) * nengo_data->num_steps * nengo_data->num_spikes);
  checkCudaErrorWithDevice(err, nengo_data->device, "Resetting Cuda arrays");

  if(nengo_data->do_print)
    printf("Done resetting NEF fields: device %d\n", nengo_data->device);
}

#ifdef __cplusplus
}
#endif

