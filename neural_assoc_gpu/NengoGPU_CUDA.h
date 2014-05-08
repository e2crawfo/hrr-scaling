
#ifndef NENGO_GPU_CUDA_H
#define NENGO_GPU_CUDA_H

#ifdef __cplusplus
extern "C"{
#endif

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "NengoGPUData.h"

void printIntArrayFromDevice(FILE* fp, intArray* array, int m, int n, int labels);
void printFloatArrayFromDevice(FILE* fp, floatArray* array, int m, int n, int labels);

void printIntColumn(FILE* fp, int* array, int m, int n, int col);
void printFloatColumn(FILE* fp, float* array, int m, int n, int col);
void printIntRange(FILE* fp, int* array, int start, int end);
void printFloatRange(FILE* fp, float* array, int start, int end);

int getGPUDeviceCount();

void initGPUDevice(int device);

void shutdownGPUDevice();

void checkCudaErrorWithDevice(cudaError_t err, int device, char* message);
void checkCudaError(cudaError_t err, char* message);

__global__ void lif_math(int numNeurons, int neurons_per_item, float dt, float* encode_result,
                         float* voltage_array, float* reftime_array, float tau_rc,
                         float tau_ref, float* bias, float* scale, float* spikes);
__global__ void dot_product(int vector_length, long int stride, float* A, float* B, float* C);
__global__ void moveGPUData(int size, int* map, float* to, float* from);

void run_neural_associative_memory(NengoGPUData* nengo_data, float start_time, float end_time);

float* allocateCudaFloatArray(int size);
int* allocateCudaIntArray(int size);
long getDeviceCapacity(int device);
void initializeDeviceInputAndOutput(NengoGPUData*);

void reset_neural_associative_memory(NengoGPUData* nengo_data);

#ifdef __cplusplus
}
#endif

#endif
