#ifndef PARAMETERS_H
#define PARAMETERS_H

enum STATES {x, y, v, psi, s, l};
enum INPUTS {d, F};

struct Parameters
{
    const double tau = 2.0;
    const double k = 2.0;
    const double r_safe = 2.5;
    const double r_lane = 3.5;
    // #if REAL_BITS == 32
    // const double eps = 3.3e-4;
    // #elif REAL_BITS == 64
    // const double eps = 1.5e-8;
    // #elif REAL_BITS == 16
    // const double eps = 3.2e-2;
    // #endif 
    const double eps = 3.3e-4;
    //const double eps = 2.9e-4;
    const double length = 5.0;
    const double cg_ratio = 0.5;
    const double pi = M_PI;
    const double pi_2 = 2.0*M_PI;
    const double v_max = 10.0;

    constexpr static const int N = 20;                                  /** number of integration nodes */
    constexpr static const int nX = 6;                                  /** <X, Y, V, PSI, S, L> */
    constexpr static const int nU = 2;                                  /** <d, F> */
    constexpr static const int N_interpolation = 60;
    constexpr static const double dt_interpolation = 0.1;

    const double dt = 0.25;                                                    /** integration time step */
    const double d_up = 0.7;                                                  /** upper bound yaw rate */
    const double d_low = -0.7;                                                /** lower bound yaw rate */
    const double F_up = 3.0;                                                  /** upper bound force */
    const double F_low = -3.0;                                                /** lower bound force */

    // Parameters:
    const double weight_target_speed = 1e0;                                      /** weight for the maximum speed in the lagrangian */
    const double weight_center_lane = 1e-1;                                   /** weight for the center lane in the lagrangian */
    const double weight_heading = 1e2;                                        /** weight for the heading in the lagrangian */
    const double weight_input = 0.0;                                          /** weight for the input in the lagrangian */
};

#endif // PARAMETERS_H
