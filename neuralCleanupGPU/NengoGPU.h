
#ifndef NENGO_GPU_H
#define NENGO_GPU_H

#ifdef __cplusplus
extern "C"{
#endif

#include <stdio.h>
#include "NengoGPUData.h"

extern NengoGPUData** nengoDataArray;
extern int numDevices;
extern float startTime;
extern float endTime;
extern volatile int myCVsignal;
extern pthread_mutex_t* mutex;
extern pthread_cond_t* cv_GPUThreads;
extern pthread_cond_t* cv_JNI;
extern FILE* fp;
extern int do_print;

int manipulateNumNodesProcessed(int action, int value);
int manipulateKill(int action);

void* start_GPU_thread(void* arg);
void run_start();
void run_kill();

void setup(int numDevicesRequested, float dt, int numVectors, int dimension, int autoassociative, int** index_vectors, int** result_vectors, float tau, float* encoder, float* decoder, int num_neurons, float* alpha, float* Jbias, float tau_ref, float tau_rc, int* return_spikes, int print_data);

void step(float* input, float* output, float* spikes, float start, float end);

void kill();
void reset();

#ifdef __cplusplus
}
#endif

#endif
