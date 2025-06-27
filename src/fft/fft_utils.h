/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef FFT_UTILS_H
#define FFT_UTILS_H

#include <complex.h>

/*******************************************************************************
 * \brief Local transposition.
 * \author Frederick Stein
 ******************************************************************************/
void transpose_local_complex(double complex *grid,
                             double complex *grid_transposed,
                             const int number_of_columns_grid,
                             const int number_of_rows_grid);

/*******************************************************************************
 * \brief Local transposition.
 * \author Frederick Stein
 ******************************************************************************/
void transpose_local_double(double *grid, double *grid_transposed,
                            const int number_of_columns_grid,
                            const int number_of_rows_grid);

/*******************************************************************************
 * \brief Local transposition of blocks.
 * \author Frederick Stein
 ******************************************************************************/
void transpose_local_complex_block(double complex *grid,
                                   double complex *grid_transposed,
                                   const int number_of_columns_grid,
                                   const int number_of_rows_grid,
                                   const int block_size);

/*******************************************************************************
 * \brief Local transposition of blocks.
 * \author Frederick Stein
 ******************************************************************************/
void transpose_local_double_block(double *grid, double *grid_transposed,
                                  const int number_of_columns_grid,
                                  const int number_of_rows_grid,
                                  const int block_size);

                                  

/*******************************************************************************
 * \brief Returns the smaller of two given integer (missing from the C standard)
 * \author Ole Schuett
 ******************************************************************************/
static inline int imin(int x, int y) {
  return (x < y ? x : y);
}

/*******************************************************************************
 * \brief Returns the larger of two given integer (missing from the C standard)
 * \author Ole Schuett
 ******************************************************************************/
static inline int imax(int x, int y) {
  return (x > y ? x : y);
}

/*******************************************************************************
 * \brief Returns the smaller of two given integer (missing from the C standard)
 * \author Frederick Stein
 ******************************************************************************/
static inline int dmin(double x, double y) {
  return (x < y ? x : y);
}

/*******************************************************************************
 * \brief Returns the larger of two given integer (missing from the C standard)
 * \author Frederick Stein
 ******************************************************************************/
static inline int dmax(double x, double y) {
  return (x > y ? x : y);
}

/*******************************************************************************
 * \brief Equivalent of Fortran's MODULO which always returns a positive number.
 *        https://gcc.gnu.org/onlinedocs/gfortran/MODULO.html
 * \author Ole Schuett
 ******************************************************************************/
static inline int modulo(int a, int m) {
  return ((a % m + m) % m);
}

/*******************************************************************************
 * \brief Calculates the product of three numbers.
 * \author Frederick Stein
 ******************************************************************************/
static inline int product3(const int array3[3]) {
  return array3[0] * array3[1] * array3[2];
}

#endif /* FFT_UTILS_H */

// EOF
