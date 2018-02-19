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



	mxGPUDestroyGPUArray(lsf);
}



























































