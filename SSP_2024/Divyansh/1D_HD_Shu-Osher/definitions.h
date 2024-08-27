#define  PHYSICS                        HD
#define  DIMENSIONS                     1
#define  GEOMETRY                       CARTESIAN
#define  BODY_FORCE                     NO
#define  COOLING                        NO
#define  RECONSTRUCTION                 WENOZ_FD
#define  TIME_STEPPING                  RK3
#define  NTRACER                        0
#define  PARTICLES                      NO
#define  USER_DEF_PARAMETERS            2

/* -- physics dependent declarations -- */

#define  DUST_FLUID                     NO
#define  EOS                            IDEAL
#define  ENTROPY_SWITCH                 NO
#define  THERMAL_CONDUCTION             NO
#define  VISCOSITY                      NO
#define  ROTATING_FRAME                 NO

/* -- user-defined parameters (labels) -- */

#define  GAMMA_EOS                      0
#define  X_CUT                          1

/* [Beg] user-defined constants (do not change this line) */

#define  CHAR_LIMITING                  YES
#define  LIMITER                        MC_LIM

/* [End] user-defined constants (do not change this line) */
