#include "mex.h"
#include "mexcudaReinitialization.hpp"
#include <cuda_runtime_api.h>

__global__ 
void ExploreIdx(double * const dev_re_lsf, double const * const dev_lsf, int const number_of_elements_lsf,
	int const rows, int const cols, int const pages)
{
	int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int col_idx = blockIdx.y * blockDim.y + threadIdx.y;
	int pge_idx = blockIdx.z * blockDim.z + threadIdx.z;

	int idx = pge_idx * rows * cols + col_idx * rows + row_idx;

	if(idx > number_of_elements_lsf)
		return;

	dev_re_lsf[idx] = 2 * dev_lsf[idx];
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
	dim3 const thread(dimx, dimy, dimz);
	dim3 const block((rows + thread.x - 1) / thread.x, (cols + thread.y - 1) / thread.y, (pages + thread.z - 1) / thread.z);

	mexPrintf("thread dimension (%d,%d,%d)\n", thread.x, thread.y, thread.z);
	mexPrintf("block dimension (%d,%d,%d)\n", block.x, block.y, block.z);

	// allocate memory for input lsf and out put level set function
	double * dev_lsf, * dev_re_lsf;
	cudaMalloc((void **)&dev_lsf, sizeof(double)*number_of_elements_lsf);
	cudaMalloc((void **)&dev_re_lsf, sizeof(double)*number_of_elements_lsf);

	// record information 
	//bool * dev_mask, * dev_mxr, * dev_mxl, * dev_myf, * dev_myb, * dev_mzu, * dev_mzd;

	cudaMemcpy((void *)dev_lsf, lsf, sizeof(double)*number_of_elements_lsf, cudaMemcpyHostToDevice);
	//cudaMemset((void *)dev_re_lsf, (int)0, sizeof(double)*number_of_elements_lsf);

	ExploreIdx<<<block, thread>>>(dev_re_lsf, dev_lsf, number_of_elements_lsf, rows, cols, pages);

	// copy results back 
	cudaMemcpy(re_lsf, (void *)dev_re_lsf, sizeof(double)*number_of_elements_lsf, cudaMemcpyDeviceToHost);

	cudaFree(dev_lsf);
	cudaFree(dev_re_lsf);

}












