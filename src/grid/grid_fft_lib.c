/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "grid_fft_lib.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(__FFTW3)
#include <fftw3.h>
#endif

/*******************************************************************************
 * \brief Initialize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_init_lib() {
#if defined(__FFTW3)
  fftw_init_threads();
#endif
}

/*******************************************************************************
 * \brief Finalize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_finalize_lib() {
#if defined(__FFTW3)
  fftw_cleanup();
#endif
}

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_allocate_double(const int length, double **buffer) {
  assert(buffer != NULL);
  assert(*buffer == NULL);
#if defined(__FFTW3)
  *buffer = fftw_alloc_real(length);
#else
  *buffer = (double *)malloc(length * sizeof(double));
#endif
}

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_allocate_complex(const int length, double complex **buffer) {
  assert(buffer != NULL);
  assert(*buffer == NULL);
#if defined(__FFTW3)
  *buffer = fftw_alloc_complex(length);
#else
  *buffer = (double complex *)malloc(length * sizeof(double complex));
#endif
}

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_free_double(double *buffer) {
#if defined(__FFTW3)
  fftw_free(buffer);
#else
  free(buffer);
#endif
}

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_free_complex(double complex *buffer) {
#if defined(__FFTW3)
  fftw_free(buffer);
#else
  free(buffer);
#endif
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_1d_fw_local(const double complex *grid_rs, double complex *grid_gs,
                     const int fft_size, const int number_of_ffts) {
  const double pi = acos(-1.0);
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_rs, grid_gs, fft_size, number_of_ffts, pi)
  for (int fft = 0; fft < number_of_ffts; fft++) {
    for (int index_out = 0; index_out < fft_size; index_out++) {
      double complex tmp = 0.0;
      for (int index_in = 0; index_in < fft_size; index_in++) {
        tmp += grid_rs[index_in * number_of_ffts + fft] *
               cexp(-2.0 * I * pi * index_out * index_in / fft_size);
      }
      grid_gs[fft * fft_size + index_out] = tmp;
    }
  }
}

/*******************************************************************************
 * \brief Naive implementation of backwards FFT to transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_1d_bw_local(const double complex *grid_gs, double complex *grid_rs,
                     const int fft_size, const int number_of_ffts) {
  const double pi = acos(-1.0);
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_rs, grid_gs, fft_size, number_of_ffts, pi)
  for (int fft = 0; fft < number_of_ffts; fft++) {
    for (int index_out = 0; index_out < fft_size; index_out++) {
      double complex tmp = 0.0;
      for (int index_in = 0; index_in < fft_size; index_in++) {
        tmp += grid_gs[fft * fft_size + index_in] *
               cexp(2.0 * I * pi * index_out * index_in / fft_size);
      }
      grid_rs[index_out * number_of_ffts + fft] = tmp;
    }
  }
}

/*******************************************************************************
 * \brief Local transposition.
 * \author Frederick Stein
 ******************************************************************************/
void transpose_local(double complex *grid, double complex *grid_transposed,
                     const int number_of_columns_grid,
                     const int number_of_rows_grid) {
#pragma omp parallel for collapse(2) default(none)                             \
    shared(grid, grid_transposed, number_of_columns_grid, number_of_rows_grid)
  for (int column_index = 0; column_index < number_of_columns_grid;
       column_index++) {
    for (int row_index = 0; row_index < number_of_rows_grid; row_index++) {
      grid_transposed[column_index * number_of_rows_grid + row_index] =
          grid[row_index * number_of_columns_grid + column_index];
    }
  }
}

/*******************************************************************************
 * \brief Naive implementation of 2D FFT (transposed format, no normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_fw_local(double complex *grid_rs, double complex *grid_gs,
                     const int size_of_first_fft, const int size_of_second_fft,
                     const int number_of_ffts) {

  // Perform the first FFT along z
  fft_1d_fw_local(grid_rs, grid_gs, size_of_first_fft,
                  size_of_second_fft * number_of_ffts);

  // Perform the second FFT along y
  fft_1d_fw_local(grid_gs, grid_rs, size_of_second_fft,
                  size_of_first_fft * number_of_ffts);

  memcpy(grid_gs, grid_rs,
         size_of_first_fft * size_of_second_fft * number_of_ffts *
             sizeof(double complex));
}

/*******************************************************************************
 * \brief Performs local 2D FFT (reverse to fw routine, no normalization).
 * \note fft_2d_bw_local(grid_gs, grid_rs, n1, n2, m) is the reverse to
 * fft_2d_rw_local(grid_rs, grid_gs, n1, n2, m) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_bw_local(double complex *grid_gs, double complex *grid_rs,
                     const int size_of_first_fft, const int size_of_second_fft,
                     const int number_of_ffts) {

  // Perform the second FFT along y
  fft_1d_bw_local(grid_gs, grid_rs, size_of_second_fft,
                  size_of_first_fft * number_of_ffts);

  // Perform the third FFT along z
  fft_1d_bw_local(grid_rs, grid_gs, size_of_first_fft,
                  size_of_second_fft * number_of_ffts);

  memcpy(grid_rs, grid_gs,
         size_of_first_fft * size_of_second_fft * number_of_ffts *
             sizeof(double complex));
}

/*******************************************************************************
 * \brief Performs local 3D FFT (no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_local(double complex *grid_rs, double complex *grid_gs,
                     const int fft_size[3]) {

  // Perform the first FFT along z
  fft_1d_fw_local(grid_rs, grid_gs, fft_size[2], fft_size[0] * fft_size[1]);

  // Perform the second FFT along y
  fft_1d_fw_local(grid_gs, grid_rs, fft_size[1], fft_size[0] * fft_size[2]);

  // Perform the third FFT along x
  fft_1d_fw_local(grid_rs, grid_gs, fft_size[0], fft_size[1] * fft_size[2]);
}

/*******************************************************************************
 * \brief Performs local 3D FFT (reverse to fw routine, no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_local(double complex *grid_gs, double complex *grid_rs,
                     const int fft_size[3]) {

  // Perform the first FFT along x
  fft_1d_bw_local(grid_gs, grid_rs, fft_size[0], fft_size[1] * fft_size[2]);

  // Perform the second FFT along y
  fft_1d_bw_local(grid_rs, grid_gs, fft_size[1], fft_size[0] * fft_size[2]);

  // Perform the third FFT along z
  fft_1d_bw_local(grid_gs, grid_rs, fft_size[2], fft_size[0] * fft_size[1]);
}

// EOF
