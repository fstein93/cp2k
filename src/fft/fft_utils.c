/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "fft_utils.h"

#include <complex.h>
#include <string.h>

/*******************************************************************************
 * \brief Local transposition.
 * \author Frederick Stein
 ******************************************************************************/
void transpose_local_complex(double complex *grid,
                             double complex *grid_transposed,
                             const int number_of_columns_grid,
                             const int number_of_rows_grid) {
  for (int column_index = 0; column_index < number_of_columns_grid;
       column_index++) {
    for (int row_index = 0; row_index < number_of_rows_grid; row_index++) {
      grid_transposed[column_index * number_of_rows_grid + row_index] =
          grid[row_index * number_of_columns_grid + column_index];
    }
  }
}

/*******************************************************************************
 * \brief Local transposition.
 * \author Frederick Stein
 ******************************************************************************/
void transpose_local_double(double *grid, double *grid_transposed,
                            const int number_of_columns_grid,
                            const int number_of_rows_grid) {
  for (int column_index = 0; column_index < number_of_columns_grid;
       column_index++) {
    for (int row_index = 0; row_index < number_of_rows_grid; row_index++) {
      grid_transposed[column_index * number_of_rows_grid + row_index] =
          grid[row_index * number_of_columns_grid + column_index];
    }
  }
}

/*******************************************************************************
 * \brief Local transposition of blocks.
 * \author Frederick Stein
 ******************************************************************************/
void transpose_local_complex_block(double complex *grid,
                                   double complex *grid_transposed,
                                   const int number_of_columns_grid,
                                   const int number_of_rows_grid,
                                   const int block_size) {
  for (int column_index = 0; column_index < number_of_columns_grid;
       column_index++) {
    for (int row_index = 0; row_index < number_of_rows_grid; row_index++) {
      memcpy(&grid_transposed[(column_index * number_of_rows_grid + row_index) *
                              block_size],
             &grid[(row_index * number_of_columns_grid + column_index) *
                   block_size],
             block_size * sizeof(double complex));
    }
  }
}

/*******************************************************************************
 * \brief Local transposition of blocks.
 * \author Frederick Stein
 ******************************************************************************/
void transpose_local_double_block(double *grid, double *grid_transposed,
                                  const int number_of_columns_grid,
                                  const int number_of_rows_grid,
                                  const int block_size) {
  for (int column_index = 0; column_index < number_of_columns_grid;
       column_index++) {
    for (int row_index = 0; row_index < number_of_rows_grid; row_index++) {
      memcpy(&grid_transposed[(column_index * number_of_rows_grid + row_index) *
                              block_size],
             &grid[(row_index * number_of_columns_grid + column_index) *
                   block_size],
             block_size * sizeof(double));
    }
  }
}

// EOF
