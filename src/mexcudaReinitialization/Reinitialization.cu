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
void Reinitialization(double * re_lsf, double const * lsf, int const number_of_elements_lsf,
	int const rows, int const cols, int const pages, 
	double const dx, double const dy, double const dz)
{
	mexPrintf("hello cuda!\n");
	mexPrintf("number of elements : %d\n", number_of_elements_lsf);
	mexPrintf("dimension array:(%d,%d,%d)\n",rows,cols,pages);
	mexPrintf("grid spacing:(%f,%f,%f)\n",dx,dy,dz);

	// 
	int dimx = rows, dimy = 4, dimz = 1;
	dim3 const block(dimx, dimy, dimz);
	//dim3 const grid((rows + block.x - 1) / block.x, (cols + block.y - 1) / block.y, (pages + block.z - 1) / block.z);
	dim const grid(1,1,128);

	mexPrintf("block dimension (%d,%d,%d)\n", block.x, block.y, block.z);
	mexPrintf("grid dimension (%d,%d,%d)\n", grid.x, grid.y, grid.z);

	ExploreIdx<<<grid,block>>>();

	// allocate memory for input lsf and out put level set function
	double * dev_lsf, *dev_re_lsf;
	cudaMalloc((void **)&dev_lsf, sizeof(double)*number_of_elements_lsf);
	cudaMalloc((void **)&dev_re_lsf, sizeof(double)*number_of_elements_lsf);

	cudaMemcpy((void *)dev_lsf, lsf, sizeof(double)*number_of_elements_lsf, cudaMemcpyHostToDevice);
	cudaMemset((void *)dev_re_lsf, (int)0, sizeof(double)*number_of_elements_lsf);


	// copy results back 
	cudaMemcpy(re_lsf, (void *)dev_re_lsf, sizeof(double)*number_of_elements_lsf, cudaMemcpyDeviceToHost);

	cudaFree(dev_lsf);
	cudaFree(dev_re_lsf);

}












