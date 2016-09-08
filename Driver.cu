/* Driver for one simulation of the Benjamin model */
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "NetworkProblem.hpp"
#include "parameters.hpp"

int main( int argc, char* argv[])
{

  int noReal     = 1000;
  int noNetworks = 6;
  int noCouplingStrengths = 10;

  float beta_min = log10(-2);
  float beta_max = log10(5);
  char filename[] = "networks3.dat";

  Benjamin* p_network = new Benjamin( noReal, noNetworks, noCouplingStrengths);

  p_network->SetCouplingStrength( beta_min, beta_max, noCouplingStrengths);

  p_network->LoadNetworks( filename);
  p_network->SimulateNetwork();

  delete( p_network);

  return 0;
}
