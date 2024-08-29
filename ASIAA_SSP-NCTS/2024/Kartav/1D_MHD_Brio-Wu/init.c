/*
* @Author: Somdeb Bandopadhyay
* @Date:   2024-08-03
* @Last Modified by:   somdeb
* @Last Modified time: 2024-08-03
*/
#include "pluto.h"

/* ********************************************************************* */
void Init (double *us, double x1, double x2, double x3)
/*
 *
 *
 *
 *********************************************************************** */
{
  #if EOS == IDEAL
   g_gamma = g_inputParam[GAMMA_EOS];
  #endif

  if (x1 < 0.5){

    us[RHO] = g_inputParam[RHO_LEFT];
    us[VX1] = g_inputParam[VX_LEFT];
    us[VX2] = g_inputParam[VY_LEFT];
    us[VX3] = g_inputParam[VZ_LEFT];
    us[BX1] = g_inputParam[BX_CONST];
    us[BX2] = g_inputParam[BY_LEFT];
    us[BX3] = g_inputParam[BZ_LEFT];
    #if EOS == IDEAL
     us[PRS] = g_inputParam[PR_LEFT];
    #endif

  }else{

    us[RHO] = g_inputParam[RHO_RIGHT];
    us[VX1] = g_inputParam[VX_RIGHT];
    us[VX2] = g_inputParam[VY_RIGHT];
    us[VX3] = g_inputParam[VZ_RIGHT];
    us[BX1] = g_inputParam[BX_CONST];
    us[BX2] = g_inputParam[BY_RIGHT];
    us[BX3] = g_inputParam[BZ_RIGHT];
    #if EOS == IDEAL
     us[PRS] = g_inputParam[PR_RIGHT];
    #endif 
  }


  if (g_inputParam[DIVIDE_BY_4PI] > 0.5){
    us[BX1] /= sqrt(4.0*CONST_PI);
    us[BX2] /= sqrt(4.0*CONST_PI);
    us[BX3] /= sqrt(4.0*CONST_PI);
  }

}

/* ********************************************************************* */
void InitDomain (Data *d, Grid *grid)
/*! 
 * Assign initial condition by looping over the computational domain.
 * Called after the usual Init() function to assign initial conditions
 * on primitive variables.
 * Value assigned here will overwrite those prescribed during Init().
 *
 *
 *********************************************************************** */
{
}

/* ********************************************************************* */
void Analysis (const Data *d, Grid *grid)
/* 
 *
 *
 *********************************************************************** */
{
}

/* ********************************************************************* */
void UserDefBoundary (const Data *d, RBox *box, int side, Grid *grid)
{
int  i, j, k, nv;
double *x1, *x2, *x3;
x1 = grid->x[IDIR];
x2 = grid->x[JDIR];
x3 = grid->x[KDIR];
/* -- array pointer to x1 coordinate -- */
/* -- array pointer to x2 coordinate -- */
/* -- array pointer to x3 coordinate -- */
if (side == X1_BEG){
/* -- select the boundary side -- */
  BOX_LOOP(box,k,j,i){
  /* -- Loop over boundary zones -- */
  /* -- set jet values for r <= 1 -- */
      d->Vc[RHO][k][j][i] = g_inputParam[RHO_LEFT]; 
      d->Vc[VX1][k][j][i] = g_inputParam[VX_LEFT];  
      d->Vc[VX2][k][j][i] = g_inputParam[VY_LEFT];  
      d->Vc[VX3][k][j][i] = g_inputParam[VZ_LEFT];  
      d->Vc[BX1][k][j][i] = g_inputParam[BX_CONST]; 
      d->Vc[BX2][k][j][i] = g_inputParam[BY_LEFT];  
      d->Vc[BX3][k][j][i] = g_inputParam[BZ_LEFT];  
      d->Vc[PRS][k][j][i] = g_inputParam[PR_LEFT];  
  }
}

if (side == X1_END){
/* -- select the boundary side -- */
  BOX_LOOP(box,k,j,i){
  /* -- Loop over boundary zones -- */
  /* -- set jet values for r <= 1 -- */
      d->Vc[RHO][k][j][i] = g_inputParam[RHO_RIGHT]; 
      d->Vc[VX1][k][j][i] = g_inputParam[VX_RIGHT];  
      d->Vc[VX2][k][j][i] = g_inputParam[VY_RIGHT];  
      d->Vc[VX3][k][j][i] = g_inputParam[VZ_RIGHT];  
      d->Vc[BX1][k][j][i] = g_inputParam[BX_CONST]; 
      d->Vc[BX2][k][j][i] = g_inputParam[BY_RIGHT];  
      d->Vc[BX3][k][j][i] = g_inputParam[BZ_RIGHT];  
      d->Vc[PRS][k][j][i] = g_inputParam[PR_RIGHT];  
  }
}

}





/*! 
 *  Assign user-defined boundary conditions.
 *
 * \param [in/out] d  pointer to the PLUTO data structure containing
 *                    cell-centered primitive quantities (d->Vc) and 
 *                    staggered magnetic fields (d->Vs, when used) to 
 *                    be filled.
 * \param [in] box    pointer to a RBox structure containing the lower
 *                    and upper indices of the ghost zone-centers/nodes
 *                    or edges at which data values should be assigned.
 * \param [in] side   specifies on which side boundary conditions need 
 *                    to be assigned. side can assume the following 
 *                    pre-definite values: X1_BEG, X1_END,
 *                                         X2_BEG, X2_END, 
 *                                         X3_BEG, X3_END.
 *                    The special value side == 0 is used to control
 *                    a region inside the computational domain.
 * \param [in] grid  pointer to an array of Grid structures.
 *
 *********************************************************************** */

/* ************************************************************** */
void USERDEF_BOUNDARY (const Data *d, int side, Grid *grid) 
/* 
 *
 * 
 **************************************************************** */
{
}
