#ifndef PARAMETERSHEADERDEF
#define PARAMETERSHEADERDEF

#define lambda 0.8f
#define omega 0.0f
#define alpha 0.05f

// TIME STEPPER
#define timestep 0.001f
#define unstable_radius_squared (1-powf(lambda,0.5))

// NETWORK SIZE
#define noNeurons 3

#endif
