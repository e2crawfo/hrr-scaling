#ifndef NENGO_GPU_DATA_H
#define NENGO_GPU_DATA_H

#ifdef __cplusplus
extern "C"{
#endif

#include <cuda_runtime.h>
#include <cublas_v2.h>


/*
  Float and int array objects that facilitate safe array accessing.
*/

struct intArray_t{
  int* array;
  int size;
  char* name;
  int onDevice;
};
typedef struct intArray_t intArray;


struct floatArray_t{
  float* array;
  int size;
  char* name;
  int onDevice;
};
typedef struct floatArray_t floatArray;

intArray* newIntArray(int size, const char* name);
void freeIntArray(intArray* a);
intArray* newIntArrayOnDevice(int size, const char* name);

floatArray* newFloatArray(int size, const char* name);
void freeFloatArray(floatArray* a);
floatArray* newFloatArrayOnDevice(int size, const char* name);

void checkBounds(char* verb, char* name, int size, int index);
void checkLocation(char* verb, char* name, int onDevice, int size, int index);

void intArraySetElement(intArray* a, int index, int value);
void floatArraySetElement(floatArray* a, int index, float value);
int intArrayGetElement(intArray* a, int index);
float floatArrayGetElement(floatArray* a, int index);

void intArraySetData(intArray* a, int* data, int dataSize);
void floatArraySetData(floatArray* a, float* data, int dataSize);


typedef struct int_list_t int_list;
typedef struct int_queue_t int_queue;

// implementation of a list
struct int_list_t{
  int first;
  int_list* next;
};

int_list* cons_int_list(int_list* list, int item);
int* first_int_list(int_list* list);
int_list* next_int_list(int_list* list);
void free_int_list(int_list* list);

// implementation of a queue
struct int_queue_t{
  int size;
  int_list* head;
  int_list* tail;
};

int_queue* new_int_queue();
int pop_int_queue(int_queue* queue);
void add_int_queue(int_queue* queue, int val);
void free_int_queue(int_queue* queue);



/* 
  The central data structure. One is created for each device in use. Holds nengo network data
  in arrays structured to be used by cuda kernels in NengoGPU_CUDA.cu.
*/
struct NengoGPUData_t{

  FILE *fp;
  int onDevice;
  int initialized;
  int device;
  int do_print;
  float maxTimeStep;

  float startTime;
  float endTime;
   
  int numNeuronsPerItem;
  int dimension;
  int numItems;
  int autoassociative;
  int numSpikesToReturn;
  float tau;
  float pstc;
  float tau_ref;
  float tau_rc;
  float dt;
  cublasHandle_t handle;
  int handleInitialized;

  floatArray* inputHost;
  floatArray* inputDevice;

  floatArray* transformResult;
  floatArray* encodeResult;
  floatArray* decodedValues;

  floatArray* outputDevice;
  floatArray* outputHost;

  floatArray* index_vectors;
  floatArray* result_vectors;

  floatArray* encoder;
  floatArray* decoder;

  floatArray* alpha;
  floatArray* Jbias;
  floatArray* voltage;
  floatArray* reftime;
  floatArray* spikes;

  // all for returning spikes
  intArray* spikeMap;
  floatArray* spikesOutput;// on the device
  floatArray* spikesHost; // on the host - has the same size as spikesOutput
};

typedef struct NengoGPUData_t NengoGPUData;

NengoGPUData* getNewNengoGPUData();
void initializeNengoGPUData(NengoGPUData*);
void checkNengoGPUData(NengoGPUData*);
void moveToDeviceNengoGPUData(NengoGPUData*);
void freeNengoGPUData(NengoGPUData*);

void printNengoGPUData(NengoGPUData*nengoData, int printArrays);
void printDynamicNengoGPUData(NengoGPUData* nengoData);
void printVecs(NengoGPUData* nengoData);
void printIntArray(FILE* fp, intArray* a, int n, int m);
void printFloatArray(FILE* fp, floatArray* a, int n, int m);
void moveToDeviceIntArray(intArray* a);
void moveToDeviceFloatArray(floatArray* a);
void moveToHostFloatArray(floatArray* a);
void moveToHostIntArray(intArray* a);

#ifdef __cplusplus
}
#endif

#endif
