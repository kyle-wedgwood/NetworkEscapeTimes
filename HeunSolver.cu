#include <curand_kernel.h>
#include <cstdlib>
#include "HeunSolver.hpp"
#include "NonlinearProblem.hpp"
#include "parameters.hpp"

__device__ HeunSolver::HeunSolver( float dt, curandState global_state, NonlinearProblem*
    pProblem)
{
  mDt = dt;
  mState = global_state;
  mpProblem = pProblem;
}

__device__ void HeunSolver::HeunStep( float t, float2* u)
{
  float2 f0[noNeurons], f1[noNeurons], k1[noNeurons];

  // Take Euler step
  mpProblem->ComputeF( t, u, f0);
  mpProblem->Coupling( u, f0);

  # pragma unroll
  for (int i=0;i<noNeurons;++i)
  {
    k1[i].x = u[i].x + f0[i].x*mDt + alpha*sqrt(mDt)*make_rand();
    k1[i].y = u[i].y + f0[i].y*mDt + alpha*sqrt(mDt)*make_rand();;
  }

  // Make prediction step
  mpProblem->ComputeF( t+mDt, k1, f1);
  mpProblem->Coupling( k1, f1);

  # pragma unroll
  for (int i=0;i<noNeurons;++i)
  {
    u[i].x += mDt/2.0f*(f0[i].x+f1[i].x) + alpha*sqrt(mDt)*make_rand();
    u[i].y += mDt/2.0f*(f0[i].y+f1[i].y) + alpha*sqrt(mDt)*make_rand();
  }
}

__device__ float HeunSolver::make_rand()
{
  return curand_normal( &mState);
}
