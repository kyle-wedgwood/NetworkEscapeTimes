#ifndef NONLINEARPROBLEMHEADERDEF
#define NONLINEARPROBLEMHEADERDEF

class NonlinearProblem
{

  public:

    __device__ virtual void ComputeF( float t, float2* u, float2* f) = 0;

    __device__ virtual void Coupling( const float2* u, float2* f) = 0;

};

#endif
