#ifndef BENJAMINPROBLEMHEADERDEF
#define BENJAMINPROBLEMHEADERDEF

class Benjamin:
  public NonlinearProblem
{

  public:

    __device__ Benjamin( int2* pCouplingList,
                         const float couplingStrength);

    __device__ void ComputeF( float t, float2 u, float2* f);

    __device__ void ComputeF( float t, float2* u, float2* f);

    __device__ void Coupling( const float2* u, float2* f);

  private:

    int2*  mpCouplingList;
    float  mCouplingStrength;
};

#endif
