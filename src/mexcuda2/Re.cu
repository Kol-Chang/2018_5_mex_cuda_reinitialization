#include <algorithm>
#include <cuda_runtime_api.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"

#include "Re.hpp"

/**	
 * convert subindices to global indices 
 * for a 3d array stored in colum major
 */
__device__ inline 
int sub2ind(int const row_idx, int const col_idx, int const pge_idx, int const rows, int const cols, int const pages)
{
	return (pge_idx * rows * cols + col_idx * rows + row_idx);
}


__global__ 
void explore(double * const dev_re_lsf, double const * const dev_lsf,
	int number_of_elements_lsf, int rows, int cols, int pages)
{
	int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int col_idx = blockIdx.y * blockDim.y + threadIdx.y;
	int pge_idx = blockIdx.z * blockDim.z + threadIdx.z;

	int idx = sub2ind(row_idx, col_idx, pge_idx, rows, cols, pages);

	if(idx > number_of_elements_lsf-1)
		return;

	dev_re_lsf[idx] = 2*  dev_lsf[idx];
}

void Reinitialization(double * const dev_re_lsf, double const * const dev_lsf, 
	double * const dev_xpr, double * const dev_ypf, double * const dev_zpu, 
	double * dev_new_lsf, double * dev_intermediate_lsf, double * dev_cur_lsf,
	int const number_of_elements_lsf, int const rows, int const cols, int const pages,
	double const dx, double const dy, double const dz)
{
	mexPrintf("reintializing ...\n");

	int dimx = rows, dimy = 4, dimz = 1;
	dim3 const thread(dimx, dimy, dimz);
	dim3 const block(	(rows + thread.x - 1) / thread.x, 
						(cols + thread.y - 1) / thread.y, 
						(pages + thread.z - 1) / thread.z);

	
	explore<<<block, thread>>>(dev_re_lsf, dev_lsf, number_of_elements_lsf, rows, cols, pages);

}