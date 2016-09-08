#include "Benjamin.hpp"
#include "parameters.hpp"

__device__ Benjamin::Benjamin( int2* pCouplingList, const float couplingStrength)
{
  mpCouplingList   = pCouplingList;
  mCouplingStrength = couplingStrength;
}

__device__ void Benjamin::ComputeF( float t, float2 u, float2* f)
{
  (*f).x =
    (lambda-1)*u.x-omega*u.y+2.0f*u.x*(u.x*u.x+u.y*u.y)-u.x*powf(u.x*u.x+u.y*u.y,2);
  (*f).y =
    (lambda-1)*u.y+omega*u.x+2.0f*u.y*(u.x*u.x+u.y*u.y)-u.y*powf(u.x*u.x+u.y*u.y,2);
}

__device__ void Benjamin::ComputeF( float t, float2* u, float2* f)
{
  # pragma unroll
  for (int i=0;i<noNeurons;++i)
  {
    f[i].x =
      (lambda-1)*u[i].x-omega*u[i].y+2.0f*u[i].x*(u[i].x*u[i].x+u[i].y*u[i].y)-u[i].x*powf(u[i].x*u[i].x+u[i].y*u[i].y,2);
    f[i].y =
      (lambda-1)*u[i].y+omega*u[i].x+2.0f*u[i].y*(u[i].x*u[i].x+u[i].y*u[i].y)-u[i].y*powf(u[i].x*u[i].x+u[i].y*u[i].y,2);
  }
}

__device__ void Benjamin::Coupling( const float2* u,
                                    float2* f)
{
  # pragma unroll
  for (int i=0;i<noNeurons*noNeurons;++i)
  {
    if (mpCouplingList[i].x==-1)
      break;
    f[mpCouplingList[i].y].x +=
      mCouplingStrength*(u[mpCouplingList[i].x].x-u[mpCouplingList[i].y].x);
    f[mpCouplingList[i].y].y +=
      mCouplingStrength*(u[mpCouplingList[i].x].y-u[mpCouplingList[i].y].y);
  }
}
