#ifndef HEUNKERNELSHEADERDER
#define HEUNKERNELSHEADERDER

#include <iostream>
#include "NonlinearProblem.hpp"

class HeunSolver
{

  public:

    __device__ HeunSolver( float dt, curandState global_state, NonlinearProblem* pProblem);

    __device__ void HeunStep( float t, float2* u);

  private:

    __device__ float make_rand();

    float mDt;
    curandState mState;
    NonlinearProblem* mpProblem;

};

#endif
