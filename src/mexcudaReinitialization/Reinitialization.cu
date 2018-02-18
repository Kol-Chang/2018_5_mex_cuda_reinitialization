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

/*
 * reinitialization scheme
 * re_lsf : pointer to reinitialized level set function
 * lsf : pointer to input level set function
 */
void Reinitialization(double * re_lsf, double const * lsf, 
	int const rows, int const cols, int const pages, 
	double const dx, double const dy, double const dz)
{
	mexPrintf("hello cuda!\n");
	mexPrintf("dimension array:(%d,%d,%d)\n",rows,cols,pages);
	mexPrintf("grid spacing:(%d,%d,%d)\n",dx,dy,dz);

	// 
	dim3 const dimBlock(1,1,1);
	dim3 const dimThread(2,1,1);

	mexPrintf("Block dimension (%d,%d,%d)\n", dimBlock.x, dimBlock.y, dimBlock.z);
	mexPrintf("Thread dimension (%d,%d,%d)\n", dimThread.x, dimThread.y, dimThread.z);

	ExploreIdx<<<dimBlock,dimThread>>>();
}