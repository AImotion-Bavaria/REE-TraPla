#include "ispc_parameter.h"


// Integration parameters
#define N 20                                  // number of integration nodes
#define nX 6                                  // <X, Y, V, PSI, S, L>
#define nU 2                                  // <d, F>
#define MAX_OBSTACLES 20
#define NUM_State 21

#if REAL_BITS == 16
typedef float16 myType;
// Parameters for normalization
UNIFORM const myType L_Max = (float16)64.0;
UNIFORM const myType  V_Max = (float16)64.0;
UNIFORM const myType  M_normalized = (float16)(1.0/16.0);
UNIFORM const myType  C_Max = (float16)4096.0;
// Numerical constants
UNIFORM const myType tau = (float16)2.0;
UNIFORM const myType tau_normalized = (float16) (tau * V_Max / L_Max);
UNIFORM const myType k = (float16)2.0;
UNIFORM const myType k_normalized = (float16) (k * M_normalized);
UNIFORM const myType length = (float16)5.0;
UNIFORM const myType length_normalized = (float16)(length / L_Max);
UNIFORM const myType cg_ratio = (float16)0.5;
UNIFORM const myType dt = (float16)0.25;
UNIFORM const myType dt_normalized = dt  * V_Max / L_Max;
// Weights for cost function
UNIFORM const myType weight_target_speed = (float16)1e0;            // weight for max speed in lagrangian
UNIFORM const myType weight_target_speed_normalized = (float16)(weight_target_speed * V_Max * L_Max / C_Max);
UNIFORM const myType weight_center_lane = (float16)1e-1;            // weight for center lane
UNIFORM const myType weight_heading = (float16)1e2;                 // weight for heading
UNIFORM const myType weight_input = (float16)0.0;                   // weight for input
UNIFORM const myType pi_2 = (float16)(2.0*M_PI_ISPC);
UNIFORM const myType zero = (float16)0.0;
#elif REAL_BITS == 32
typedef float myType;
// Parameters for normalization
UNIFORM const myType L_Max = 64.0f;
UNIFORM const myType  V_Max = 64.0f;
UNIFORM const myType  M_normalized = 1.0f/16.0f;
UNIFORM const myType  C_Max = 4096.0f;
// Numerical constants
UNIFORM const myType tau = 2.0f;
UNIFORM const myType tau_normalized = tau * V_Max / L_Max;
UNIFORM const myType k = 2.0f;
UNIFORM const myType k_normalized = k * M_normalized;
UNIFORM const myType length = 5.0f;
UNIFORM const myType length_normalized = length / L_Max;
UNIFORM const myType cg_ratio = 0.5f;
UNIFORM const myType dt = 0.25f;
UNIFORM const myType dt_normalized = dt  * V_Max / L_Max;
// Weights for cost function
UNIFORM const myType weight_target_speed = 1e0f;
UNIFORM const myType weight_target_speed_normalized = (weight_target_speed * V_Max * L_Max / C_Max);            // weight for max speed in lagrangian
UNIFORM const myType weight_center_lane = 1e-1f;            // weight for center lane
UNIFORM const myType weight_heading = 1e2f;                 // weight for heading
UNIFORM const myType weight_input = 0.0f;                   // weight for input
UNIFORM const myType pi_2 = 2.0f*(float)(M_PI_ISPC);
UNIFORM const myType zero = 0.0f;
#else
typedef double myType;
// Parameters for normalization
UNIFORM const myType L_Max = 64.0d;
UNIFORM const myType  V_Max = 64.0d;
UNIFORM const myType  M_normalized = 1.0d/16.0d;
UNIFORM const myType  C_Max = 4096.0d;
// Numerical constants
UNIFORM const myType tau = 2.0d;
UNIFORM const myType tau_normalized = tau * V_Max / L_Max;
UNIFORM const myType k = 2.0d;
UNIFORM const myType k_normalized = k * M_normalized;
UNIFORM const myType length = 5.0d;
UNIFORM const myType length_normalized = length / L_Max;
UNIFORM const myType cg_ratio = 0.5d;
UNIFORM const myType dt = 0.25d;
UNIFORM const myType dt_normalized = dt  * V_Max / L_Max;
// Weights for cost function
UNIFORM const myType weight_target_speed = 1e0d;
UNIFORM const myType weight_target_speed_normalized = (weight_target_speed * V_Max * L_Max / C_Max);           // weight for max speed in lagrangian
UNIFORM const myType weight_center_lane = 1e-1d;            // weight for center lane
UNIFORM const myType weight_heading = 1e2d;                 // weight for heading
UNIFORM const myType weight_input = 0.0d;                   // weight for input
UNIFORM const myType pi_2 = 2.0d*(double)M_PI_ISPC;
UNIFORM const myType zero = 0.0d;
#endif

struct SoA_X_Double {
    double x[NUM_State*MAX_OBSTACLES];
    double y[NUM_State*MAX_OBSTACLES];
    double psi[NUM_State*MAX_OBSTACLES];
    double v[NUM_State*MAX_OBSTACLES];
    double s[NUM_State*MAX_OBSTACLES];
    double l[NUM_State*MAX_OBSTACLES];
};
struct SoA_U_Double {
    double d[NUM_State*MAX_OBSTACLES];
    double F[NUM_State*MAX_OBSTACLES];
};

struct State_ISPC{
    myType x[MAX_OBSTACLES];
    myType y[MAX_OBSTACLES];
    myType v[MAX_OBSTACLES];
    myType psi[MAX_OBSTACLES];
    myType v_target[MAX_OBSTACLES];
};

struct VehicleState_SIMD
{
    myType x;
    myType y;
    myType v;
    myType psi;
    myType s;
    myType l;
};

struct PhysicState_SIMD
{
    myType d;
    myType F;
};

struct SoA_U {
    myType d[NUM_State*MAX_OBSTACLES];
    myType F[NUM_State*MAX_OBSTACLES];
};

struct SoA_X {
    myType x[NUM_State*MAX_OBSTACLES];
    myType y[NUM_State*MAX_OBSTACLES];
    myType psi[NUM_State*MAX_OBSTACLES];
    myType v[NUM_State*MAX_OBSTACLES];
    myType s[NUM_State*MAX_OBSTACLES];
    myType l[NUM_State*MAX_OBSTACLES];
};
/** integrates the input U to get the state X */
export void integrate_ispc(uniform SoA_X X_[], uniform const SoA_U U_[],
                            uniform const State_ISPC& states, uniform const int traffic_num)
{
    varying int index;
    VehicleState_SIMD s_t0;
    varying PhysicState_SIMD u_t0;
    VehicleState_SIMD ds_t0;
    myType target_speed;
    foreach (i = 0 ... traffic_num){
        s_t0.x = states.x[i] / L_Max;
        s_t0.y = states.y[i] / L_Max;
        s_t0.v = states.v[i] / V_Max;
        s_t0.psi = states.psi[i] / pi_2;
        s_t0.s = zero;
        s_t0.l = zero;

        target_speed = states.v_target[i] / V_Max;

        for (uniform int j = 0; j < N + 1; j++){
            // Reference point on the center lane:
            for (uniform int count = 0; count < programCount; ++count) {
                uniform myType s_ref = extract(s_t0.s, count);
                uniform myType lane_num = extract(i, count);
                if(lane_num >= traffic_num) {
                    break;
                }
            }

            // Input control:
            uniform int adress = MAX_OBSTACLES*j;
            u_t0.d = U_->d[adress+i] / pi_2;
            u_t0.F = U_->F[adress+i] * L_Max / (V_Max * V_Max * M_normalized); // a = F/m
            
            // Derivatives:
            myType angle_1 = cg_ratio * u_t0.d;
            myType angle_2 = s_t0.psi + angle_1;
            myType sin_angle_2;
            myType cos_angle_2;
            sincos(angle_2 * pi_2, &sin_angle_2, &cos_angle_2);
            ds_t0.x = s_t0.v * cos_angle_2;
            ds_t0.y = s_t0.v * sin_angle_2;
            ds_t0.v =  - s_t0.v / tau_normalized + (k_normalized) * u_t0.F;
            ds_t0.psi = s_t0.v * tan(u_t0.d * pi_2) * cos(angle_1*pi_2)/ (length_normalized*pi_2);
            ds_t0.l = weight_target_speed_normalized * (s_t0.v - target_speed) * (s_t0.v - target_speed);
            ds_t0.s = s_t0.v;
            
            // Integration to compute the new state: 
            s_t0.x += dt_normalized * ds_t0.x;
            s_t0.y += dt_normalized * ds_t0.y;
            s_t0.v += dt_normalized * ds_t0.v;
            s_t0.psi += dt_normalized * ds_t0.psi;
            s_t0.s += dt_normalized * ds_t0.s;
            s_t0.l += dt_normalized * ds_t0.l;

            // if (s_t0.v < 0.0){s_t0.v = 0.0;}
            myType min_speed = zero;
            s_t0.v = max(min_speed, s_t0.v);            // for (uniform int count = 0; count < programCount; ++count) {
            //     uniform myType s_ref = extract(s_t0.s, count);
            //     uniform myType lane_num = extract(i, count);
            //     if(lane_num >= traffic_num) {
            //         break;
            //     }
            // }

            X_->x[adress+i] = s_t0.x;
            X_->y[adress+i] = s_t0.y;
            X_->v[adress+i] = s_t0.v;
            X_->psi[adress+i] = s_t0.psi;
            X_->s[adress+i] = s_t0.s;
            X_->l[adress+i] = s_t0.l;  
        }
    }
}

export void keep4DataTypeInHeader(uniform SoA_U_Double U_Double[], uniform SoA_X_Double U[], uniform int num_elements) {
}
