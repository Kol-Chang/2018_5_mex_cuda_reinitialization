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

__device__ inline 
double min_mod(double x, double y)
{
	return ( (x*y<0) ? 0 : (fabs(x)<fabs(y) ? x : y ));
}

__device__ inline
double min2(double x, double y)
{
	return ( (x<y) ? x : y );
}

__device__ inline
double max2(double x, double y)
{
	return ( (x<y) ? y : x );
}

__device__ inline
double sign(double x)
{
	return ( x>0 ? 1. : -1. );
}

__device__ inline
double discriminant(double p2, double v0, double v2)
{
	return ( pow((0.5*p2-v0-v2),2) - 4.*v0*v2 );
}

__device__ inline
double dist(double disc, double ds, double p2, double v0, double v2)
{
	return ( ds * (0.5 + (v0 - v2 - sign(v0-v2)*sqrt(disc)) / p2 ) );
}

__device__ inline
double dist_turn(double ds, double v0, double v2)
{
	return ( ds * v0 / (v0 - v2) );
}

__global__
void boundary_correction(double * const dev_xpr, double * const dev_ypf, double * const dev_zpu,
	double const * const dev_lsf, int number_of_elements_lsf, int rows, int cols, int pages,
	double dx, double dy, double dz)
{	
	double epislon = 10e-10;

	int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int col_idx = blockIdx.y * blockDim.y + threadIdx.y;
	int pge_idx = blockIdx.z * blockDim.z + threadIdx.z;

	int idx = sub2ind(row_idx, col_idx, pge_idx, rows, cols, pages);

	if(idx > number_of_elements_lsf-1)
		return;

	double f0 = dev_lsf[idx]; // grab the current/left/back/lower node
	// fill in dev_xpr and make correction near boundary

	dev_xpr[idx] = dx;
	//int idx_right = sub2ind(row_idx, (col_idx < (cols-1)) ? col_idx+1 : col_idx+1-cols, pge_idx, rows, cols, pages );	
	int idx_right = sub2ind(row_idx, (col_idx < (cols-1)) ? col_idx+1 : cols, pge_idx, rows, cols, pages );	
	double f2 = dev_lsf[idx_right]; // grab the right node

	double p2;

	if(f0*f2 < 0) // if there is a boundary to the right
	{
		//int idx_left = sub2ind(row_idx, (col_idx > 0) ? col_idx-1 : col_idx-1+cols, pge_idx, rows, cols, pages);
		int idx_left = sub2ind(row_idx, (col_idx > 0) ? col_idx-1 : 0, pge_idx, rows, cols, pages);
		//int idx_2right = sub2ind(row_idx, (col_idx < (cols-2) ? col_idx+2 : col_idx+2-cols), pge_idx, rows, cols, pages);
		int idx_2right = sub2ind(row_idx, (col_idx < (cols-2) ? col_idx+2 : cols), pge_idx, rows, cols, pages);

		double p2xl = dev_lsf[idx_left] - 2.0 * f0 + f2; // 2nd difference on the left node
		double p2xr = f0 - 2.0 * f2 + dev_lsf[idx_2right]; // 2nd difference on the right node
		p2 = min_mod(p2xl, p2xr);
		if(p2>epislon){
			dev_xpr[idx] = dist(discriminant(p2,f0,f2),dx,p2,f0,f2);
		} else{
			dev_xpr[idx] = dist_turn(dx,f0,f2);
		}
	}

	// fill in dev_ypf
	dev_ypf[idx] = dy;
	//int idx_front = sub2ind( (row_idx < (rows-1)) ? row_idx+1 : row_idx+1-rows, col_idx, pge_idx, rows, cols, pages);
	int idx_front = sub2ind( (row_idx < (rows-1)) ? row_idx+1 : rows, col_idx, pge_idx, rows, cols, pages);
	f2 = dev_lsf[idx_front]; // grab the front node value

	if(f0*f2 < 0) // if there is a boundary to the front
	{
		//int idx_back = sub2ind( (row_idx > 0) ? row_idx-1 : row_idx-1+rows, col_idx, pge_idx, rows, cols, pages );
		int idx_back = sub2ind( (row_idx > 0) ? row_idx-1 : 0, col_idx, pge_idx, rows, cols, pages );
		//int idx_2front = sub2ind( (row_idx < (rows-2)) ? row_idx+2 : row_idx+2-rows, col_idx, pge_idx, rows, cols, pages );
		int idx_2front = sub2ind( (row_idx < (rows-2)) ? row_idx+2 : rows, col_idx, pge_idx, rows, cols, pages );

		double p2yb = dev_lsf[idx_back] - 2.0 * f0 + f2;
		double p2yf = f0 - 2.0 * f2 + dev_lsf[idx_2front];
		p2 = min_mod(p2yb, p2yf);
		if(p2>epislon){
			dev_ypf[idx] = dist(discriminant(p2,f0,f2),dy,p2,f0,f2);
		}else{
			dev_ypf[idx] = dist_turn(dy,f0,f2);
		}
	}

	// fill in dev_zpu
	dev_zpu[idx] = dz;
	//int idx_upper = sub2ind( row_idx, col_idx, (pge_idx < (pages-1)) ? pge_idx+1 : pge_idx+1-pages, rows, cols, pages );
	int idx_upper = sub2ind( row_idx, col_idx, (pge_idx < (pages-1)) ? pge_idx+1 : pages, rows, cols, pages );
	f2 = dev_lsf[idx_upper]; // grab the upper node value

	if(f0*f2 < 0) // if there is a boundary to the upper side
	{
		//int idx_lower = sub2ind(row_idx, col_idx, (pge_idx > 0) ? pge_idx-1: pge_idx-1+pages, rows, cols, pages);
		int idx_lower = sub2ind(row_idx, col_idx, (pge_idx > 0) ? pge_idx-1: pages, rows, cols, pages);
		//int idx_2upper = sub2ind( row_idx, col_idx, (pge_idx < (pages-2)) ? pge_idx+2 : pge_idx+2-pages, rows, cols, pages );
		int idx_2upper = sub2ind( row_idx, col_idx, (pge_idx < (pages-2)) ? pge_idx+2 : pages, rows, cols, pages );

		double p2zd = dev_lsf[idx_lower] - 2.0 * f0 + f2;
		double p2zu = f0 - 2.0 * f2 + dev_lsf[idx_2upper];
		p2 = min_mod(p2zd, p2zu);
		if(p2>epislon){
			dev_zpu[idx] = dist(discriminant(p2,f0,f2),dz,p2,f0,f2);
		}else{
			dev_ypf[idx] = dist_turn(dz,f0,f2);
		}
	}
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

	dev_re_lsf[idx] = 3. * dev_lsf[idx];
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


	// fill in dev_xpr,ypf,zpu
	boundary_correction<<<block, thread>>>(dev_xpr, dev_ypf, dev_zpu, 
		dev_lsf, number_of_elements_lsf, rows, cols, pages, dx, dy, dz);

}