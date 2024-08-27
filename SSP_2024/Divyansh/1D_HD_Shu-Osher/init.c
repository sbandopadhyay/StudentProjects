/*
* @Author: Somdeb Bandopadhyay
* @Date:   2024-08-03
* @Last Modified by:   somdeb
* @Last Modified time: 2024-08-27
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
   g_gamma = g_inputParam[GAMMA_EOS];
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
int i, j, k;
int id;
double *x1 = grid->x[IDIR], *x1r = grid->xr[IDIR], *x1l = grid->xl[IDIR];
double *x2 = grid->x[JDIR], *x2r = grid->xr[JDIR], *x2l = grid->xl[JDIR];
double *x3 = grid->x[KDIR], *x3r = grid->xr[KDIR], *x3l = grid->xl[KDIR];
double *dx;
dx = grid->dx[IDIR];

double xcut = g_inputParam[X_CUT];

TOT_LOOP(k,j,i){

   double dn1 =  3.8571;
   double vx1 =  2.6293;
   double pr1 = 10.3333;
   double dn2 = 1.0+0.200*sin(5.00*x1[i]);
   double vx2 = 0.0;
   double pr2 = 1.0;



   if(x1r[i] <= xcut ){
          d->Vc[RHO][k][j][i] = dn1;
          d->Vc[VX1][k][j][i] = vx1;
          d->Vc[PRS][k][j][i] = pr1;
   }else if(x1l[i] >= xcut ){
          d->Vc[RHO][k][j][i] = dn2;
          d->Vc[VX1][k][j][i] = vx2;
          d->Vc[PRS][k][j][i] = pr2;
 }else{
          d->Vc[RHO][k][j][i] =  
              ((x1r[i]-xcut) - (0.2/5.0)*(cos(5.0*x1r[i])-cos(5.0*xcut)) + (xcut-x1l[i]) * dn1) / dx[i];
          d->Vc[VX1][k][j][i] =  ((x1r[i]-xcut) * vx2 + (xcut-x1l[i]) * vx1) / dx[i];
          d->Vc[PRS][k][j][i] =  ((x1r[i]-xcut) * pr2 + (xcut-x1l[i]) * pr1) / dx[i];
   }

d->Vc[VX2][k][j][i] = 0.0;
d->Vc[VX3][k][j][i] = 0.0;
}


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
      d->Vc[RHO][k][j][i] = 3.8571; 
      d->Vc[VX1][k][j][i] = 2.6293;  
      d->Vc[VX2][k][j][i] = 0.0;  
      d->Vc[VX3][k][j][i] = 0.0;
      d->Vc[PRS][k][j][i] = 10.3333;  
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
