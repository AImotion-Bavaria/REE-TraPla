// ispc_parameters.h
#ifndef ISPC_PARAMETERS_H
#define ISPC_PARAMETERS_H

#ifdef __ISPC__
// ISPC-specific definitions
#define UNIFORM UNIFORM
#else
// C++ definitions
#define UNIFORM
#endif

#define M_PI_ISPC 3.14159265358979323846 // #include <math.h>

// For c++ code only
UNIFORM const double L_Max_Double = 64.0d;
UNIFORM const double  V_Max_Double = 64.0d;
UNIFORM const double  C_Max_Double = 4096.0d;

#endif // ISPC_PARAMETERS_H