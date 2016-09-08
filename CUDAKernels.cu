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
    curandState local_state = pGlobalState[index];
    Benjamin* p_problem = new Benjamin(
        pCouplingList+blockIdx.y*noNeurons*noNeurons, pCouplingStrength[blockIdx.x]);
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
