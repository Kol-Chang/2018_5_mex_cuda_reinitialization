#include "mex.h"
#include "gpu/mxGPUArray.h"

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

    mexPrintf("2nd try ...\n");

    if(nrhs != 2){
		mexErrMsgIdAndTxt("mexReinitialization:wrong_number_of_inputs",
			"expecting 2 inputs");
	}

	// assign level set function
	if(mxIsGPUArray(prhs[level_set_function]))
		mexPrintf("Get a GPU array \n");

	mxGPUArray const * lsf = mxGPUCreateFromMxArray(prhs[level_set_function]);
	double const *dev_lsf = (double const *)(mxGPUGetDataReadOnly(lsf)); // pointer to input data on device

	//
	mxGPUArray const * ds = mxGPUCreateFromMxArray(prhs[grid_spacing]);
	double const *dev_ds = (double const *)(mxGPUGetDataReadOnly(ds)); 

	/* Create a GPUArray to hold the result and get its underlying pointer. */
 	mxGPUArray *re_lsf = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(lsf),
                            				 mxGPUGetDimensions(lsf),
                            				 mxGPUGetClassID(lsf),
                            				 mxGPUGetComplexity(lsf),
                            				 MX_GPU_DO_NOT_INITIALIZE);

 	double * dev_re_lsf = (double *)(mxGPUGetData(re_lsf)); // pointer to data on device

	/* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[reinitialized_lsf] = mxGPUCreateMxArrayOnGPU(re_lsf);


    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
	mxGPUDestroyGPUArray(lsf);
	mxGPUDestroyGPUArray(ds);
}



























































