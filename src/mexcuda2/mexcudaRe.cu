#include <algorithm>
#include <cuda_runtime_api.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"

#include "Re.hpp"

// enumerate input index
enum in_put{
	level_set_function = 0,
	grid_spacing = 1
};

// enumerate output index
enum out_put{
	reinitialized_lsf = 0
};

/**
 * MEX gateway
 */
void mexFunction(int nlhs , mxArray *plhs[], int nrhs, mxArray const * prhs[])
{
	// Initialize the MathWorks GPU API.
    mxInitGPU();

    mxClassID category;

    if(nrhs != 2){
		mexErrMsgIdAndTxt("mexReinitialization:wrong_number_of_inputs",
			"expecting 2 inputs: a 3d array representing level set function and a 1x3 array representing grid spacing");
	}

	// assign level set function
	mxGPUArray const * lsf = mxGPUCreateFromMxArray(prhs[level_set_function]);
	double const *dev_lsf = (double const *)(mxGPUGetDataReadOnly(lsf)); // pointer to input data on device

	mwSize number_of_dimensions;
	const mwSize *dimension_array;
	size_t number_of_elements_lsf;

	category = mxGPUGetClassID(lsf);
	number_of_dimensions = mxGPUGetNumberOfDimensions(lsf);
	dimension_array = mxGPUGetDimensions(lsf);
	number_of_elements_lsf = mxGPUGetNumberOfElements(lsf);

	if (category != mxDOUBLE_CLASS || number_of_dimensions != (mwSize)3 || !mxIsGPUArray(prhs[level_set_function])){
		mexErrMsgIdAndTxt("mexReinitialization:Invalid_Input",
			"Argument %d must be a 3 dimension GPUarray of double precision!",
			level_set_function);
	}
	// finish assigning level set function

	// assign grid spacing array
	double *ds;
	size_t rows, cols;

	category = mxGetClassID(prhs[grid_spacing]);
	rows = mxGetM(prhs[grid_spacing]);
	cols = mxGetN(prhs[grid_spacing]);
	if (category != mxDOUBLE_CLASS || rows != (size_t)1 || cols != (size_t)3){
		mexErrMsgIdAndTxt("mexReinitialization:Invalid_Input",
			"Argument %d must be a 1X3 double array of the grid spacing",
			grid_spacing);
	}
	ds = (double *)mxGetData(prhs[grid_spacing]);
	// finish assigning spacing array


	/* Create a GPUArray to hold the result and get its underlying pointer. */
 	mxGPUArray *re_lsf = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(lsf),
                            				 mxGPUGetDimensions(lsf),
                            				 mxGPUGetClassID(lsf),
                            				 mxGPUGetComplexity(lsf),
                            				 MX_GPU_DO_NOT_INITIALIZE);

 	double * dev_re_lsf = (double *)(mxGPUGetData(re_lsf)); // pointer to data on device


    /*
     * workspace gpuArrays
     */
    mxGPUArray *xpr = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(lsf),
                            				 mxGPUGetDimensions(lsf),
                            				 mxGPUGetClassID(lsf),
                            				 mxGPUGetComplexity(lsf),
                            				 MX_GPU_DO_NOT_INITIALIZE);
 	double * dev_xpr = (double *)(mxGPUGetData(xpr)); // pointer to data on device

 	mxGPUArray *ypf = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(lsf),
                            				 mxGPUGetDimensions(lsf),
                            				 mxGPUGetClassID(lsf),
                            				 mxGPUGetComplexity(lsf),
                            				 MX_GPU_DO_NOT_INITIALIZE);
 	double * dev_ypf = (double *)(mxGPUGetData(ypf)); // pointer to data on device

 	mxGPUArray *zpu = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(lsf),
                            				 mxGPUGetDimensions(lsf),
                            				 mxGPUGetClassID(lsf),
                            				 mxGPUGetComplexity(lsf),
                            				 MX_GPU_DO_NOT_INITIALIZE);
 	double * dev_zpu = (double *)(mxGPUGetData(zpu)); // pointer to data on device

 	mxGPUArray *new_lsf = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(lsf),
                            				 mxGPUGetDimensions(lsf),
                            				 mxGPUGetClassID(lsf),
                            				 mxGPUGetComplexity(lsf),
                            				 MX_GPU_DO_NOT_INITIALIZE);
 	double * dev_new_lsf = (double *)(mxGPUGetData(new_lsf)); // pointer to data on device

 	mxGPUArray *intermediate_lsf = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(lsf),
                            				 mxGPUGetDimensions(lsf),
                            				 mxGPUGetClassID(lsf),
                            				 mxGPUGetComplexity(lsf),
                            				 MX_GPU_DO_NOT_INITIALIZE);
 	double * dev_intermediate_lsf = (double *)(mxGPUGetData(intermediate_lsf)); // pointer to data on device

 	mxGPUArray *cur_lsf = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(lsf),
                            				 mxGPUGetDimensions(lsf),
                            				 mxGPUGetClassID(lsf),
                            				 mxGPUGetComplexity(lsf),
                            				 MX_GPU_DO_NOT_INITIALIZE);
 	double * dev_cur_lsf = (double *)(mxGPUGetData(cur_lsf)); // pointer to data on device

 	// call the computation routine
 	Reinitialization(dev_re_lsf, dev_lsf, 
 		dev_xpr, dev_ypf, dev_zpu,
 		dev_new_lsf, dev_intermediate_lsf, dev_cur_lsf,
 		number_of_elements_lsf, dimension_array[0], dimension_array[1], dimension_array[2],
 		ds[0], ds[1], ds[2]);



 	/* Wrap the result up as a MATLAB gpuArray for return. */
    //plhs[reinitialized_lsf] = mxGPUCreateMxArrayOnGPU(re_lsf);

	/* Wrap the result up as a MATLAB cpuArray for return. */
	mexPrintf("trying to return a cpu array \n");
    //plhs[reinitialized_lsf] = mxGPUCreateMxArrayOnCPU(cur_lsf);
    plhs[reinitialized_lsf] = mxGPUCreateMxArrayOnCPU(re_lsf);

    mexPrintf("returned a cpu array \n");

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
	mxGPUDestroyGPUArray(lsf);
	mxGPUDestroyGPUArray(re_lsf);
	mxGPUDestroyGPUArray(xpr);
	mxGPUDestroyGPUArray(ypf);
	mxGPUDestroyGPUArray(zpu);
	mxGPUDestroyGPUArray(new_lsf);
	mxGPUDestroyGPUArray(intermediate_lsf);
	mxGPUDestroyGPUArray(cur_lsf);
}



























































