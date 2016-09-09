#include <iostream>
#include <cstdlib>
#include <cmath>
#include <curand_kernel.h>
#include <assert.h>
#include <fstream>
#include <sstream>
#include "cu_error_functions.hpp"
#include "parameters.hpp"
#include "NetworkProblem.hpp"
#include "CUDAKernels.hpp"

using namespace std;

NetworkProblem::NetworkProblem( unsigned int noReal,
                    unsigned int noNetworks,
                    unsigned int noCouplingStrengths)
{
  mNoReal     = noReal;
  mNoThreads  = 1024;
  mNoNetworks = noNetworks;
  mNoBeta     = noCouplingStrengths;
  mNoSims     = mNoNetworks*mNoBeta;
  mNoBlocks.x = mNoBeta;
  mNoBlocks.y = mNoNetworks;
  mNoBlocks.z = (mNoReal+mNoThreads-1)/mNoThreads;

  // Allocate memory
  mpHost_couplingList = (int2*) malloc( mNoNetworks*noNeurons*noNeurons*sizeof(int2));
  mpHost_couplingStrength = (float*) malloc( mNoBeta*sizeof(float));
  mpHost_escapeTimes = (float*) malloc( mNoSims*noNeurons*sizeof(float));
  mpCurrentIndex = (int*) malloc( mNoNetworks*sizeof(int));
  memset( mpCurrentIndex, 0, mNoNetworks*sizeof(int));

  CUDA_CALL( cudaMalloc( &mpGlobalState, mNoReal*sizeof(curandState)));
  CUDA_CALL( cudaMalloc( &mpNoFinished, sizeof(int)));
  CUDA_CALL( cudaMalloc( &mpEscapeTimes, noNeurons*mNoSims*sizeof(float)));
  CUDA_CALL( cudaMalloc( &mpCouplingList, mNoNetworks*noNeurons*noNeurons*sizeof(int2)));
  CUDA_CALL( cudaMalloc( &mpCouplingStrength, mNoBeta*sizeof(float)));

  cout << "Created network object with "
       << mNoReal
       << " Realisations"
       << endl;
}

NetworkProblem::~NetworkProblem()
{
  cudaFree( mpGlobalState);
  cudaFree( mpNoFinished);
  cudaFree( mpEscapeTimes);
  cudaFree( mpCouplingList);
  cudaFree( mpCouplingStrength);

  free( mpHost_couplingList);
  free( mpHost_couplingStrength);
  free( mpHost_escapeTimes);
  free( mpCurrentIndex);
}

void NetworkProblem::SimulateNetwork()
{
  // Check if networks are configured
  if (!mNetworksCreatedFlag)
  {
    cout << "Networks not initialised. Aborting." << endl;
    return;
  }
  if (!mCouplingFlag)
  {
    cout << "Coupling strengths not set. Aborting." << endl;
    return;
  }

  // Reset memory
  CUDA_CALL( cudaMemset( mpNoFinished, 0, sizeof(int)));
  CUDA_CALL( cudaMemset( mpEscapeTimes, 0.0f, noNeurons*mNoSims*sizeof(float)));

  // Initialise the PRNG
  InitalisePRNGKernel<<<mNoBlocks,mNoThreads>>>( mNoReal, mpGlobalState);
  CUDA_CHECK_ERROR();

  cout << "Random number generator configured." << endl;

  // Debug
  if (mDebug)
  {
    DebugKernel<<<mNoBlocks,1>>>( mNoReal,
                                  mNoSims,
                                  mNoBeta,
                                  mpGlobalState,
                                  mpNoFinished,
                                  mpEscapeTimes,
                                  mpCouplingList,
                                  mpCouplingStrength);
    CUDA_CALL( cudaDeviceSynchronize());
    cout << "Debug information displayed." << endl;
    cout << "Press return to continue." << endl;
    getchar();
  }

  // Actually run the network
  cout << "Starting simulation..." << endl;
  SimulateNetworkKernel<<<mNoBlocks,mNoThreads>>>( mNoReal,
                                                   mNoSims,
                                                   mNoBeta,
                                                   mpGlobalState,
                                                   mpNoFinished,
                                                   mpEscapeTimes,
                                                   mpCouplingList,
                                                   mpCouplingStrength);
  CUDA_CHECK_ERROR();
  CUDA_CALL( cudaDeviceSynchronize());

  cout << "Simulation completed successfully" << endl;

  // Copy data back
  CUDA_CALL( cudaMemcpy( mpHost_escapeTimes, mpEscapeTimes,
        mNoSims*noNeurons*sizeof(float), cudaMemcpyDeviceToHost));

  // Collect averages
  CalculateAverages();

  // Save and print the result
  PrintResult();

}

void NetworkProblem::SetNoRealisations( unsigned int noReal)
{
  mNoReal = noReal;
  cudaFree( mpGlobalState);
  CUDA_CALL( cudaMalloc( &mpGlobalState, mNoReal*sizeof(curandState)));
  cout << "Number of realisation set to " << mNoReal << endl;
}

void NetworkProblem::SetCouplingStrength( float min, float max, int npts)
{
  float dbeta = (max-min)/(npts-1);
  float beta = min;
  cout << beta << dbeta << endl;

  // Populate beta vector
  for (int i=0;i<npts;++i)
  {
    mpHost_couplingStrength[i] = pow(10,beta);
    beta += dbeta;
  }

  CUDA_CALL( cudaMemcpy( mpCouplingStrength, mpHost_couplingStrength,
        mNoBeta*sizeof(float), cudaMemcpyHostToDevice));

  mCouplingFlag = 1;
  if (mDebug)
  {
    cout << "Coupling strengths:";
    for (int i=0;i<npts;++i)
    {
      cout << "\t" << mpHost_couplingStrength[i];
    }
    cout << endl;
  }
  cout << "Coupling strengths set." << endl;
}

void NetworkProblem::CalculateAverages()
{
  for (int i=0;i<mNoSims*noNeurons;++i)
  {
    mpHost_escapeTimes[i] /= mNoReal;
  }
}

void NetworkProblem::PrintResult()
{
  int networkNo, betaNo;
  for (int i=0;i<mNoSims;++i)
  {
    networkNo = i/mNoBeta;
    betaNo = i%mNoBeta;
    cout << "Mean first passages times for network "
         << networkNo << " with strength "
         << mpHost_couplingStrength[betaNo]
         << ":";
    for (int j=0;j<1;++j)
    {
      cout << mpHost_escapeTimes[i*noNeurons+j] << endl;
    }
  }
}

void NetworkProblem::SaveData( char* filename)
{
  ofstream file;
  file.open( filename);
  for (int i=0;i<mNoSims;++i)
  {
    file << i/mNoBeta << mpHost_couplingStrength[i%mNoBeta];
    for (int j=0;j<noNeurons;++j)
    {
      file << mpHost_escapeTimes[i*noNeurons+j];
    }
    file << "\n";
  }
  file.close();
  cout << "Escape times saved" << endl;
}

void NetworkProblem::LoadNetworks( char* filename)
{
  unsigned int networkNo;
  unsigned int outputNeuron;
  unsigned int inputNeuron;
  ifstream file( filename, ios::in);
  string str;

  if (file.good())
  {
    while (getline(file,str))
    {
      istringstream ss(str);
      ss >> networkNo >> outputNeuron >> inputNeuron;
      AddLink( networkNo, outputNeuron, inputNeuron);
    }
  }

  FinishNetworks();

  file.close();
  cout << "Network file loaded successfully." << endl;
  if (mDebug)
  {
    for (int i=0;i<mNoNetworks*noNeurons*noNeurons;++i)
    {
      cout << "Network: " << i/(noNeurons*noNeurons)
           << ", Output neuron: " << mpHost_couplingList[i].x
           << ", Input neuron: " << mpHost_couplingList[i].y
           << endl;
    }
    cout << "Debug information displayed." << endl;
    cout << "Press return to continue." << endl;
    getchar();
  }
}

void NetworkProblem::AddLink( unsigned int networkNo,
                        unsigned int outputNeuron,
                        unsigned int inputNeuron)
{
  assert( networkNo<mNoNetworks);
  assert( outputNeuron<noNeurons);
  assert( inputNeuron<noNeurons);
  assert( mpCurrentIndex[networkNo]<noNeurons*noNeurons);

  unsigned int local_index =
    networkNo*noNeurons*noNeurons+mpCurrentIndex[networkNo];
  mpHost_couplingList[local_index].x = outputNeuron;
  mpHost_couplingList[local_index].y = inputNeuron;

  mpCurrentIndex[networkNo]++;

  cout << "Added link from neuron " << outputNeuron
       << " to neuron " << inputNeuron
       << " in network " << networkNo << endl;
}

void NetworkProblem::RemoveLink( unsigned int networkNo,
                           unsigned int outputNeuron,
                           unsigned int inputNeuron)
{
  assert( networkNo<mNoNetworks);
  assert( outputNeuron<noNeurons);
  assert( inputNeuron<noNeurons);
  assert( mpCurrentIndex[networkNo]>0);

  int i;
  unsigned int local_index = networkNo*noNeurons*noNeurons;

  for (i=0;i<noNeurons*noNeurons;++i)
  {
    if ((mpHost_couplingList[local_index+i].x = outputNeuron)
      && (mpHost_couplingList[local_index+i].y = inputNeuron))
    {
      break;
    }
  }
  for (i=i;i<noNeurons*noNeurons;++i)
  {
    mpHost_couplingList[local_index+i].x =
      mpHost_couplingList[local_index+i+1].x;
    mpHost_couplingList[local_index+i].y =
      mpHost_couplingList[local_index+i+1].y;
  }

  mpCurrentIndex[networkNo]--;

  cout << "Removed link from neuron " << outputNeuron
       << " to neuron " << inputNeuron
       << " in network " << networkNo << endl;
}

void NetworkProblem::FinishNetworks()
{
  for (int i=0;i<mNoNetworks;++i)
  {
    mpHost_couplingList[i*noNeurons*noNeurons+mpCurrentIndex[i]].x = -1;
  }

  CUDA_CALL( cudaMemcpy( mpCouplingList, mpHost_couplingList,
        mNoNetworks*noNeurons*noNeurons*sizeof(int2), cudaMemcpyHostToDevice));
  mNetworksCreatedFlag = 1;

  cout << "Networks set." << endl;
}

void NetworkProblem::SetDebugFlag( bool val)
{
  mDebug = val;
}
