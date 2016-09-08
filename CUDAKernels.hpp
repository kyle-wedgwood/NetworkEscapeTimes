#ifndef CUDAKERNELSHEADERDEF
#define CUDAKERNELSHEADERDEF

#include <curand.h>
#include <curand_kernel.h>

__global__ void InitalisePRNGKernel( const unsigned int noReal,
                                     curandState* pGlobalState);

__global__ void SimulateNetworkKernel( const unsigned int noReal,
                                       const unsigned int noSims,
                                       const unsigned int noBeta,
                                       curandState* pGlobalState,
                                       unsigned int* pNoFinished,
                                       float* pEscapeTimes,
                                       int2* pCouplingList,
                                       float* pCouplingStrength);

#endif
