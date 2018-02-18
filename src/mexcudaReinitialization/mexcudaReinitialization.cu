#include "mex.h"
#include "mexcudaReinitialization.hpp"

// the gateway function
void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[]){
	
	// calling computation routine
	Reinitialization();
}