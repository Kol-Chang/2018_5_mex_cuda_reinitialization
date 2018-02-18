#ifndef MEXCUDAREINITIALIZATION
#define MEXCUDAREINITIALIZATION

void Reinitialization(double * re_lsf, double const * lsf, 
	int const rows, int const cols, int const pages, 
	double const dx, double const dy, double const dz);

#endif