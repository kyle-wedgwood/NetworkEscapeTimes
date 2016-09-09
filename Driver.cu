/* Driver for one simulation of the Benjamin model */
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "NetworkProblem.hpp"
#include "parameters.hpp"

int main( int argc, char* argv[])
{

  int noReal     = 1024;
  int noNetworks = 13;
  int noCouplingStrengths = 10;

  float beta_min = -2.0f;
  float beta_max = 3.0f;
  char network_filename[] = "networks3.dat";
  char escape_time_filename[] = "netork3EscapeTimes.dat";

  NetworkProblem* p_network = new NetworkProblem( noReal, noNetworks, noCouplingStrengths);

  p_network->SetDebugFlag(1);

  p_network->SetCouplingStrength( beta_min, beta_max, noCouplingStrengths);

  p_network->LoadNetworks( network_filename);

  p_network->SimulateNetwork();

  p_network->SaveData( escape_time_filename);

  delete( p_network);

  return 0;
}
