#include <iostream>
#include <cstdlib>
#include <curand_kernel.h>

__global__ void init_stuff(curandState *state)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   curand_init(1337, idx, 0, &state[idx]);
}

__global__ void make_rand(curandState *state, float
*randArray)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   randArray[idx] = curand_normal(&state[idx]);
}

void SaveData( int npts, float *x, char *filename) {
  FILE *fp = fopen(filename,"w");
  for (int i=0;i<npts;i++) {
    fprintf(fp,"%f\n",x[i]);
  }
  fclose(fp);
}

int main( int argc, char* argv[])
{
   curandState *d_state;
   float* randArray;
   float* host_randArray;
   int nThreads = 512;
   int nBlocks  = 1000;

   host_randArray = (float*) malloc( nThreads*nBlocks*sizeof(float));
   cudaMalloc (&d_state, nThreads*nBlocks*sizeof(curandState));
   cudaMalloc( &randArray, nThreads*nBlocks*sizeof(float));

   init_stuff<<<nBlocks, nThreads>>>(d_state);
   make_rand<<<nBlocks, nThreads>>>(d_state, randArray);

   cudaMemcpy( host_randArray, randArray, nThreads*nBlocks*sizeof(float),
       cudaMemcpyDeviceToHost);

   char filename[] = "testPRNG.dat";
   SaveData(nThreads*nBlocks,host_randArray,filename);

   free(host_randArray);
   cudaFree(randArray);
   cudaFree(d_state);

   return 0;
}
