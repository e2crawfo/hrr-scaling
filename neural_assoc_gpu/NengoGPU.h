
#ifndef NENGO_GPU_H
#define NENGO_GPU_H

#include <stdio.h>

#include "NengoGPUData.h"

#ifdef __cplusplus
extern "C"{
#endif

extern NengoGPUData** nengo_data_array;
extern int num_devices;
extern float start_time;
extern float end_time;
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

void setup(int num_devices_requested, int* devices_to_use, float dt, int num_items,
           int dimension, int** index_vectors, int** stored_vectors, float tau,
           float* decoders, int neurons_per_item, float* gain, float* bias,
           float tau_ref, float tau_rc, float radius, int identical_ensembles,
           int print_data, int* probe_indices, int num_probes, int do_spikes,
           int num_steps);

void step(float* input, float* output, float* probes, float* spikes,
          float start, float end, int n_steps);

void kill();
void reset();

#ifdef __cplusplus
}
#endif

#endif
