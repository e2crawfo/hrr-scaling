CUDA_INC_PATH=/usr/local/cuda/include
CUDA_LIB_PATH=/usr/local/cuda/lib64

CUDA_INCLUDES= -I${CUDA_INC_PATH} 
#-I${HOME}/NVIDIA_GPU_Computing_SDK/C/common/inc
LIBPATH=-L${CUDA_LIB_PATH}

LIBS=-lcudart -lcublas -lm -lc

CC=gcc-4.4
CUDA_CC=nvcc
CFLAGS=-fPIC -c -m64 -Wall -I. ${CUDA_INCLUDES} 
CUDAFLAGS= --ptxas-options=--verbose --compiler-options '-fPIC' -c

#the order of the object files here is important, get unsatisfied link error if it gets mixed up
OBJ=NengoGPU.o NengoGPU_CUDA.o NengoGPUData.o

all: CFLAGS+= -O3
all: libNeuralCleanupGPU.so 

debug: CFLAGS+= -g -O0
debug: CUDAFLAGS+= -g -G
debug: SO_DEBUG=-g
debug: libNeuralCleanupGPU.so

libNeuralCleanupGPU.so: ${OBJ} 
	${CC} ${SO_DEBUG} -shared -Wl,-soname,libNeuralCleanupGPU.so -o libNeuralCleanupGPU.so.1.0 ${OBJ} ${LIBPATH} ${LIBS}

NengoGPUData.o: NengoGPUData.h
NengoGPU.o : NengoGPU.h NengoGPU_CUDA.h

NengoGPU_CUDA.o: NengoGPU_CUDA.cu NengoGPU_CUDA.h NengoGPU.h
	${CUDA_CC} ${CUDAFLAGS} --compiler-bindir /usr/bin/gcc-4.4 -o NengoGPU_CUDA.o NengoGPU_CUDA.cu 

clean:
	rm -rf libNengoGPU.so.1.0 *.o
