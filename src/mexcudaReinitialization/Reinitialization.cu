#include "mex.h"
#include "math.h"
#include "mexcudaReinitialization.hpp"
#include <cuda_runtime_api.h>
#include <algorithm> 
#include <utility>

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
void explore(double * out, double * in, int number_of_elements_lsf, int rows, int cols, int pages)
{
	int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int col_idx = blockIdx.y * blockDim.y + threadIdx.y;
	int pge_idx = blockIdx.z * blockDim.z + threadIdx.z;

	int idx = sub2ind(row_idx, col_idx, pge_idx, rows, cols, pages);

	if(idx > number_of_elements_lsf-1)
		return;

	out[idx] = 2*  in[idx];
}

__global__ 
void time_step_lsf(double * dev_new_lsf, double * dev_intermediate_lsf, double * dev_cur_lsf, double * dev_lsf,
	double const * const dev_xpr, double const * const dev_ypf, double const * const dev_zpu,
	int number_of_elements_lsf, int rows, int cols, int pages, double dx, double dy, double dz, bool const flag)
{
	int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int col_idx = blockIdx.y * blockDim.y + threadIdx.y;
	int pge_idx = blockIdx.z * blockDim.z + threadIdx.z;

	int idx = sub2ind(row_idx, col_idx, pge_idx, rows, cols, pages);

	if(idx > number_of_elements_lsf-1)
		return;

	double f0 = dev_cur_lsf[idx];
	double p2m, p2, p2r, p2l;

	// compute xR & xL
	//int idx_right = sub2ind(row_idx, (col_idx < (cols-1)) ? col_idx+1 : col_idx+1-cols, pge_idx, rows, cols, pages );
	//int idx_2right = sub2ind(row_idx, (col_idx < (cols-2)) ? col_idx+2 : col_idx+2-cols, pge_idx, rows, cols, pages );
	//int idx_left = sub2ind(row_idx, (col_idx > 0) ? col_idx-1 : col_idx-1+cols, pge_idx, rows, cols, pages);
	//int idx_2left = sub2ind(row_idx, (col_idx > 1) ? col_idx-2 : col_idx-2+cols, pge_idx, rows, cols, pages);
	int idx_right = sub2ind(row_idx, (col_idx < (cols-1)) ? col_idx+1 : cols, pge_idx, rows, cols, pages );
	int idx_2right = sub2ind(row_idx, (col_idx < (cols-2)) ? col_idx+2 : cols, pge_idx, rows, cols, pages );
	int idx_left = sub2ind(row_idx, (col_idx > 0) ? col_idx-1 : 0, pge_idx, rows, cols, pages);
	int idx_2left = sub2ind(row_idx, (col_idx > 1) ? col_idx-2 : 0, pge_idx, rows, cols, pages);
	double fr = dev_cur_lsf[idx_right];
	double f2r = dev_cur_lsf[idx_2right];
	double fl = dev_cur_lsf[idx_left];
	double f2l = dev_cur_lsf[idx_2left];

	p2 = fr - 2.0 * f0 + fl;

	p2r = f0 - 2.0 * fr + f2r;
	p2l = f2l - 2.0 * fl + f0;

	p2m = 0.5 * min_mod(p2, p2r) / pow(dx, 2);
	double xpr = dev_xpr[idx];
	fr = (xpr<dx) ? 0 : fr;
	double xR = (fr-f0)/xpr - xpr * p2m;

	p2m = 0.5 * min_mod(p2, p2l) / pow(dx, 2);
	double xpl = (dev_xpr[idx_left]<dx) ? (dx-dev_xpr[idx_left]) : dx;
	fl = (xpl<dx) ? 0 : fl;
	double xL = (f0-fl) / xpl + xpl * p2m;

	// compute yF & yB
	//int idx_front = sub2ind( (row_idx < (rows-1)) ? row_idx+1 : row_idx+1-rows, col_idx, pge_idx, rows, cols, pages);
	//int idx_2front = sub2ind( (row_idx < (rows-2)) ? row_idx+2 : row_idx+2-rows, col_idx, pge_idx, rows, cols, pages);
	//int idx_back = sub2ind( (row_idx > 0) ? row_idx-1 : row_idx-1+rows, col_idx, pge_idx, rows, cols, pages );
	//int idx_2back = sub2ind( (row_idx > 1) ? row_idx-2 : row_idx-2+rows, col_idx, pge_idx, rows, cols, pages );

	int idx_front = sub2ind( (row_idx < (rows-1)) ? row_idx+1 : rows, col_idx, pge_idx, rows, cols, pages);
	int idx_2front = sub2ind( (row_idx < (rows-2)) ? row_idx+2 : rows, col_idx, pge_idx, rows, cols, pages);
	int idx_back = sub2ind( (row_idx > 0) ? row_idx-1 : 0, col_idx, pge_idx, rows, cols, pages );
	int idx_2back = sub2ind( (row_idx > 1) ? row_idx-2 : 0, col_idx, pge_idx, rows, cols, pages );

	fr = dev_cur_lsf[idx_front];
	f2r = dev_cur_lsf[idx_2front];
	fl = dev_cur_lsf[idx_back];
	f2l = dev_cur_lsf[idx_2back];

	p2 = fr - 2.0 * f0 + fl;

	p2r = f0 - 2.0 * fr + f2r;
	p2l = f2l - 2.0 * fl + f0;

	p2m = 0.5 * min_mod(p2, p2r) / pow(dy, 2);
	double ypf = dev_ypf[idx];
	fr = (ypf<dy) ? 0 : fr;
	double yF = (fr-f0)/ypf - ypf * p2m;

	p2m = 0.5 * min_mod(p2, p2l) / pow(dy, 2);
	double ypb = (dev_ypf[idx_back]<dy) ? (dy-dev_ypf[idx_back]) : dy;
	fl = (ypb<dy) ? 0 : fl;
	double yB = (f0-fl) / ypb + ypb * p2m;

	// compute zU & zD
	//int idx_upper = sub2ind( row_idx, col_idx, (pge_idx < (pages-1)) ? pge_idx+1 : pge_idx+1-pages, rows, cols, pages );
	//int idx_2upper = sub2ind( row_idx, col_idx, (pge_idx < (pages-2)) ? pge_idx+2 : pge_idx+2-pages, rows, cols, pages );
	//int idx_lower = sub2ind(row_idx, col_idx, (pge_idx > 0) ? pge_idx-1: pge_idx-1+pages, rows, cols, pages);
	//int idx_2lower = sub2ind(row_idx, col_idx, (pge_idx > 1) ? pge_idx-2: pge_idx-2+pages, rows, cols, pages);

	int idx_upper = sub2ind( row_idx, col_idx, (pge_idx < (pages-1)) ? pge_idx+1 : pages, rows, cols, pages );
	int idx_2upper = sub2ind( row_idx, col_idx, (pge_idx < (pages-2)) ? pge_idx+2 : pages, rows, cols, pages );
	int idx_lower = sub2ind(row_idx, col_idx, (pge_idx > 0) ? pge_idx-1: 0, rows, cols, pages);
	int idx_2lower = sub2ind(row_idx, col_idx, (pge_idx > 1) ? pge_idx-2: 0, rows, cols, pages);

	fr = dev_cur_lsf[idx_upper];
	f2r = dev_cur_lsf[idx_2upper];
	fl = dev_cur_lsf[idx_lower];
	f2l = dev_cur_lsf[idx_2lower];

	p2 = fr - 2.0 * f0 + fl;

	p2r = f0 - 2.0 * fr + f2r;
	p2l = f2l - 2.0 * fl + f0;

	p2m = 0.5 * min_mod(p2, p2r) / pow(dz, 2);
	double zpu = dev_zpu[idx];
	fr = (zpu<dz) ? 0 : fr;
	double zU = (fr-f0)/zpu - zpu * p2m;

	p2m = 0.5 * min_mod(p2, p2l) / pow(dz, 2);
	double zpl = (dev_zpu[idx_lower]<dz) ? (dz-dev_zpu[idx_lower]) : dz;
	fl = (zpl<dz) ? 0 : fl;
	double zL = (f0-fl)/zpl + zpl * p2m;

	// calculate time step
	double step;
	double deltat = min2(min2(xpr,xpl),min2(ypb,ypf));
	deltat = min2(deltat, min2(zpl,zpu));
	deltat = 0.3 * deltat;
	if(dev_lsf[idx] < 0) // if inside
	{
		step = (sqrt( 	max2(pow(min2(0,xL),2), pow(max2(0,xR),2)) + 
						max2(pow(min2(0,yB),2), pow(max2(0,yF),2)) +
						max2(pow(min2(0,zL),2), pow(max2(0,zU),2)) ) - 1)
				* deltat * (-1.0);
	}else{
		step = (sqrt( 	max2(pow(max2(0,xL),2), pow(min2(0,xR),2)) + 
						max2(pow(max2(0,yB),2), pow(min2(0,yF),2)) +
						max2(pow(max2(0,zL),2), pow(min2(0,zU),2)) ) - 1)
				* deltat * (1.0);
	}

	// flag is true : calculate itermediate lsf
	// flag is false : calculate a new lsf
	if(flag){
		dev_intermediate_lsf[idx] = dev_cur_lsf[idx] - step;
	}else{
		dev_new_lsf[idx] = (dev_cur_lsf[idx] + dev_intermediate_lsf[idx] - step) / 2;
	}

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
	mexPrintf("reitializing ...\n");
	//mexPrintf("number of elements : %d\n", number_of_elements_lsf);
	//mexPrintf("dimension array:(%d,%d,%d)\n",rows,cols,pages);
	//mexPrintf("grid spacing:(%f,%f,%f)\n",dx,dy,dz);

	// 
	int dimx = rows, dimy = 4, dimz = 1;
	dim3 const thread(dimx, dimy, dimz);
	dim3 const block((rows + thread.x - 1) / thread.x, (cols + thread.y - 1) / thread.y, (pages + thread.z - 1) / thread.z);

	//mexPrintf("thread dimension (%d,%d,%d)\n", thread.x, thread.y, thread.z);
	//mexPrintf("block dimension (%d,%d,%d)\n", block.x, block.y, block.z);

	// allocate memory for input lsf and out put level set function
	double * dev_lsf;
	cudaMalloc((void **)&dev_lsf, sizeof(double)*number_of_elements_lsf);
	cudaMemcpy((void *)dev_lsf, lsf, sizeof(double)*number_of_elements_lsf, cudaMemcpyHostToDevice);

	cudaMemcpy(re_lsf, (void *)dev_lsf, sizeof(double)*number_of_elements_lsf, cudaMemcpyDeviceToHost);
	for(int i = 0;i < 10; i++){
		mexPrintf("dev_lsf[%d] : %f \n", i, re_lsf[i] );
	}
	mexPrintf("\n");

	// allocate memory for boundary corrections
	double * dev_xpr, * dev_ypf, * dev_zpu;
	cudaMalloc((void **)&dev_xpr, sizeof(double)*number_of_elements_lsf);
	cudaMalloc((void **)&dev_ypf, sizeof(double)*number_of_elements_lsf);
	cudaMalloc((void **)&dev_zpu, sizeof(double)*number_of_elements_lsf);

	boundary_correction<<<block, thread>>>(dev_xpr, dev_ypf, dev_zpu, dev_lsf, number_of_elements_lsf, rows, cols, pages, dx, dy, dz);

	double * dev_new_lsf, * dev_intermediate_lsf, * dev_cur_lsf;
	cudaMalloc((void **)&dev_new_lsf, sizeof(double)*number_of_elements_lsf);
	cudaMalloc((void **)&dev_intermediate_lsf, sizeof(double)*number_of_elements_lsf);
	cudaMalloc((void **)&dev_cur_lsf, sizeof(double)*number_of_elements_lsf);

	cudaMemcpy((void *)dev_cur_lsf, lsf, sizeof(double)*number_of_elements_lsf, cudaMemcpyHostToDevice);

for(int i = 0; i < 100; ++i){
		mexPrintf("iteration %d \n", i);
		// fill in dev_intermediate_lsf
		time_step_lsf<<<block, thread>>>(dev_new_lsf, dev_intermediate_lsf, dev_cur_lsf, dev_lsf, dev_xpr, dev_ypf, dev_zpu, 
			number_of_elements_lsf, rows, cols, pages, dx, dy, dz, true); 

		// fill in dev_new_lsf
		time_step_lsf<<<block, thread>>>(dev_new_lsf, dev_cur_lsf, dev_intermediate_lsf, dev_lsf, dev_xpr, dev_ypf, dev_zpu, 
			number_of_elements_lsf, rows, cols, pages, dx, dy, dz, false); 

		std::swap(dev_new_lsf,dev_cur_lsf);
		//dev_cur_lsf = dev_new_lsf;

		//double * tmp;
		//tmp = dev_new_lsf;
		//dev_new_lsf = dev_cur_lsf;
		//dev_cur_lsf = tmp;
	}


	// copy results back 
	//cudaMemcpy((void *)re_lsf, (const void *)dev_cur_lsf, sizeof(double)*number_of_elements_lsf, cudaMemcpyDeviceToHost);

	

	//cudaMemcpy((void *)re_lsf, dev_cur_lsf, sizeof(double)*number_of_elements_lsf, cudaMemcpyDeviceToHost);
	//cudaMemcpy(re_lsf, (void *)dev_intermediate_lsf, sizeof(double)*number_of_elements_lsf, cudaMemcpyDeviceToHost);
	//cudaMemcpy(re_lsf, (void *)dev_cur_lsf, sizeof(double)*number_of_elements_lsf, cudaMemcpyDeviceToHost);
	
	//cudaMemcpy(re_lsf, (void *)dev_lsf, sizeof(double)*number_of_elements_lsf, cudaMemcpyDeviceToHost);
	//for(int i = 0;i < 10; i++){
	//	mexPrintf("dev_lsf[%d] : %f \n", i, re_lsf[i] );
	//}

	//explore<<<block, thread>>>(dev_cur_lsf, dev_lsf, number_of_elements_lsf, rows, cols, pages);
	//cudaMemcpy(re_lsf, (void *)dev_intermediate_lsf, sizeof(double)*number_of_elements_lsf, cudaMemcpyDeviceToHost);
	//for(int i = 0;i < 10; i++){
	//	mexPrintf("dev_intermediate_lsf[%d] : %f \n", i, re_lsf[i] );
	//}

	//cudaMemcpy(re_lsf, (void *)dev_cur_lsf, sizeof(double)*number_of_elements_lsf, cudaMemcpyDeviceToHost);
	//for(int i = 0;i < 10; i++){
	//	mexPrintf("dev_cur_lsf[%d] : %f \n", i, re_lsf[i] );
	//}

	//cudaMemcpy(re_lsf, (void *)dev_new_lsf, sizeof(double)*number_of_elements_lsf, cudaMemcpyDeviceToHost);
	//for(int i = 0;i < 10; i++){
	//	mexPrintf("dev_new_lsf[%d] : %f \n", i, re_lsf[i] );
	//}

	cudaMemcpy(re_lsf, (void *)dev_cur_lsf, sizeof(double)*number_of_elements_lsf, cudaMemcpyDeviceToHost);
	

	cudaFree(dev_lsf);
	cudaFree(dev_xpr);
	cudaFree(dev_ypf);
	cudaFree(dev_zpu);
	cudaFree(dev_new_lsf);
	cudaFree(dev_intermediate_lsf);
	cudaFree(dev_cur_lsf);

}












