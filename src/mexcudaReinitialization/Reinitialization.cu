#include "mex.h"
#include "mexcudaReinitialization.hpp"
#include <cuda_runtime_api.h>

__global__ 
void ExploreIdx()
{
	unsigned int const block_idx = blockIdx.x;
	unsigned int const thread_idx = threadIdx.x;
	//mexPrintf("This is block (%d)", blockIdx.x);
} 

void Reinitialization(void){
	mexPrintf("hello cuda!\n");

	// 
	dim3 const dimBlock(1,1,1);
	dim3 const dimThread(2,1,1);

	mexPrintf("Block dimension (%d,%d,%d)\n", dimBlock.x, dimBlock.y, dimBlock.z);
	mexPrintf("Thread dimension (%d,%d,%d)\n", dimThread.x, dimThread.y, dimThread.z);

	ExploreIdx<<<dimBlock,dimThread>>>();
}