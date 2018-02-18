#include "mex.h"
#include "mexcudaReinitialization.hpp"
#include <cuda_runtime_api.h>

__global__ 
void ExploreIdx()
{
	unsigned int const block_idx = blockIdx.x;
	unsigned int const thread_idx = threadIdx.x;
	//printf("This is block (%d,%d,%d)", block_idx.x, block_idx.y, block_idx.z);
} 

void Reinitialization(void){
	mexPrintf("hello cuda!\n");

	// 
	dim3 const dimBlock(1,1,1);
	dim3 const dimThread(2,1,1);

	ExploreIdx<<<dimBlock,dimThread>>>();
}