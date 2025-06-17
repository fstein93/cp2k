/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef GRID_FFT_UTILS_H
#define GRID_FFT_UTILS_H

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

#endif /* GRID_FFT_UTILS_H */

// EOF
