#include "mex.h"
#include "mexcudaReinitialization.hpp"

// enumerate input index
enum In_put{
	level_set_function = 0,
	grid_spacing = 1
};

// enumerate output index
enum out_put{
	reinitialized_lsf = 0
};

// the gateway function
void mexFunction(int nlhs, mxArray * plhs[], int nrhs, mxArray const * prhs[])
{
	
	mxClassID category;

	if(nrhs != 2){
		mexErrMsgIdAndTxt("mexReinitialization:wrong_number_of_inputs",
			"expecting 2 inputs");
	}

	// assign level set function to LSF
	double *lsf;

	mwSize number_of_dimensions;
	const mwSize *dimension_array;
	size_t number_of_elements_lsf;

	category = mxGetClassID(prhs[level_set_function]);
	number_of_dimensions = mxGetNumberOfDimensions(prhs[level_set_function]);
	dimension_array = mxGetDimensions(prhs[level_set_function]);
	number_of_elements_lsf = mxGetNumberOfElements(prhs[level_set_function]);
	if (category != mxDOUBLE_CLASS || number_of_dimensions != (mwSize)3){
		mexErrMsgIdAndTxt("mexReinitialization:Invalid_Input",
			"Argument %d must be a 3 dimension array of double precision!",
			level_set_function);
	}
	lsf = (double *)mxGetData(prhs[level_set_function]);
	// finish assigning LSF

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

	// create an output array
	double *re_lsf; // pointer to reinitialized level set function
	plhs[reinitialized_lsf] = mxCreateNumericArray(number_of_dimensions, dimension_array, mxDOUBLE_CLASS, mxREAL);
	re_lsf = (double *)mxGetData(plhs[reinitialized_lsf]);
	//  finish creating output array

	// calling computation routine
	Reinitialization(re_lsf, lsf, dimension_array[0], dimension_array[1], dimension_array[2], ds[0], ds[1], ds[2]);
}










