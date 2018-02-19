#include <algorithm>
#include <cuda_runtime_api.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"

#include "Re.hpp"

void Reinitialization(double * const dev_re_lsf, double const * const dev_lsf, 
	double * const dev_xpr, double * const dev_ypf, double * const dev_zpu, 
	double * dev_new_lsf, double * dev_intermediate_lsf, double * dev_cur_lsf,
	int const number_of_elements_lsf, int const rows, int const cols, int const pages,
	double const dx, double const dy, double const dz);

{
	mexPrintf("try again ...\n");
}