#ifndef NETWORKPROBLEMHEADERDEF
#define NETWORKPROBLEMHEADERDEF

#include <iostream>
#include <curand_kernel.h>

class NetworkProblem
{
  public:

    NetworkProblem( unsigned int, unsigned int, unsigned int);

    ~NetworkProblem();

    void SimulateNetwork();

    void SetNoRealisations( unsigned int);

    void CalculateAverages();

    void PrintResult();

    void SaveData( char*);

    // For setting beta
    void SetCouplingStrength( float, float, int);

    // For making networks
    void LoadNetworks( char*);

    void AddLink( unsigned int, unsigned int, unsigned int);

    void RemoveLink( unsigned int, unsigned int, unsigned int);

    void FinishNetworks();

  private:

    unsigned int mNoReal;

    // CUDA stuff
    unsigned int mNoThreads;
    dim3 mNoBlocks;

    // For PRNG
    curandState* mpGlobalState;

    // Coupling strength
    float* mpCouplingStrength;
    bool mCouplingFlag;

    float* mpEscapeTimes;

    // For bookkeeping
    unsigned int* mpNoFinished;

    // For coupling
    int2* mpHost_couplingList;
    int2* mpCouplingList;
    float* mpHost_couplingStrength;
    float* mpCouplingStrength;
    int* mpCurrentIndex;
    bool mNetworksCreatedFlag;

    // For doing multiple networks simultaneously
    unsigned int mNoNetworks;
    unsigned int mNoBeta;

    // Total number of simulations to perform
    unsigned int mNoSims;

    // For copying back
    float* mpHost_escapeTimes;

    // Hide default constructor
    NetworkProblem();

};

#endif
