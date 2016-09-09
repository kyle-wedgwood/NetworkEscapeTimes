#include <curand_kernel.h>
#include "CUDAKernels.hpp"
#include "HeunSolver.hpp"
#include "Benjamin.hpp"
#include "NonlinearProblem.hpp"
#include "parameters.hpp"

__global__ void InitalisePRNGKernel( const unsigned int noReal,
                                     curandState* pGlobalState)
{
  int index = blockIdx.z * blockDim.x + threadIdx.x;
  if (index<noReal)
  {
    curand_init(1337, index, 0, &pGlobalState[index]);
  }
}

__global__ void DebugKernel( const unsigned int noReal,
                             const unsigned int noSims,
                             const unsigned int noBeta,
                             curandState* pGlobalState,
                             unsigned int* pNoFinished,
                             float* pEscapeTimes,
                             int2* pCouplingList,
                             float* pCouplingStrength)
{
  float coupling_strength = pCouplingStrength[blockIdx.x];
  int2 p_coupling_list[noNeurons*noNeurons];
  for (int i=0;i<noNeurons*noNeurons;++i)
  {
    p_coupling_list[i].x = pCouplingList[i+blockIdx.y*noNeurons*noNeurons].x;
    p_coupling_list[i].y = pCouplingList[i+blockIdx.y*noNeurons*noNeurons].y;
    printf("Network no: %d, Output neuron: %d, Input neuron: %d.\n" \
        ,blockIdx.y,p_coupling_list[i].x,p_coupling_list[i].y);
  }
  printf("Network no: %d, Strength index: %d, Coupling strength: %f.\n",blockIdx.y,blockIdx.x,coupling_strength);
}

__global__ void SimulateNetworkKernel( const unsigned int noReal,
                                       const unsigned int noSims,
                                       const unsigned int noBeta,
                                       curandState* pGlobalState,
                                       unsigned int* pNoFinished,
                                       float* pEscapeTimes,
                                       int2* pCouplingList,
                                       float* pCouplingStrength)
{
  int index = blockIdx.z * blockDim.x + threadIdx.x;
  if (index<noReal)
  {
    __shared__ int2 p_coupling_list[noNeurons*noNeurons];
    curandState local_state = pGlobalState[index];

    // Load coupling list
    if (threadIdx.x<noNeurons*noNeurons)
    {
      p_coupling_list[threadIdx.x].x =
        pCouplingList[threadIdx.x+blockIdx.y*noNeurons*noNeurons].x;
      p_coupling_list[threadIdx.x].y =
        pCouplingList[threadIdx.x+blockIdx.y*noNeurons*noNeurons].y;
    }
    Benjamin* p_problem = new Benjamin( p_coupling_list,
                                        pCouplingStrength[blockIdx.x]);
    HeunSolver* p_solver = new HeunSolver( timestep, local_state, p_problem);

    // Initialise system
    float time = 0.0f;
    float2 u[noNeurons];

    # pragma unroll
    for (int i=0;i<noNeurons;++i)
    {
      u[i].x = 0.0f;
      u[i].y = 0.0f;
    }

    int noCrossed = 0;
    int crossings = 0;

    do
    {
      p_solver->HeunStep( time, u);

      # pragma unroll
      for (int i=0;i<noNeurons;++i)
      {
        if (!((crossings & (1<<i)) || (u[i].x*u[i].x+u[i].y*u[i].y<unstable_radius_squared)))
        {
          atomicAdd( pEscapeTimes+(blockIdx.x+blockIdx.y*noBeta)*noNeurons+noCrossed, time);
          noCrossed++;
          crossings += (1<<i);
        }
      }
      time += timestep;
    } while (noCrossed<noNeurons);

    printf("Finished simulation %d of %d.", atomicAdd( pNoFinished,1)+1,noSims);
    delete( p_solver);
    delete( p_problem);
  }
}
