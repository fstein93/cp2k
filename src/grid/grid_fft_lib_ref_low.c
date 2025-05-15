/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "grid_fft_lib_ref.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PROFILE_CODE 1

// Assume a cache line size of 64 bytes and 8 bytes per double
#define DOUBLES_PER_CACHE_LINE 8

#if PROFILE_CODE
#include <time.h>

static double time_reorder_2 = 0.0;
static double time_reorder_3 = 0.0;
static double time_reorder_4 = 0.0;
static double time_2 = 0.0;
static double time_3 = 0.0;
static double time_4 = 0.0;
static double time_naive = 0.0;
static double time_twiddle = 0.0;
#endif

// We need these definitions for the recursion within this module
void fft_ref_1d_fw_local_internal(double *restrict grid_in_real,
                                  double *restrict grid_in_imag,
                                  double *restrict grid_out_real,
                                  double *restrict grid_out_imag,
                                  const int fft_size, const int number_of_ffts);
void fft_ref_1d_bw_local_internal(double *grid_in_real, double *grid_in_imag,
                                  double *grid_out_real, double *grid_out_imag,
                                  const int fft_size, const int number_of_ffts);

void reorder_indices_2(const double *grid, double *grid_reordered,
                       const int sizes[2], const int strides_in[2],
                       const int strides_out[2]) {
#if PROFILE_CODE
  time_t begin = clock();
#endif
  for (int index_0 = 0; index_0 < sizes[0]; index_0++) {
    for (int index_1 = 0; index_1 < sizes[1]; index_1++) {
      grid_reordered[index_0 * strides_out[0] + index_1 * strides_out[1]] =
          grid[index_0 * strides_in[0] + index_1 * strides_in[1]];
    }
  }
#if PROFILE_CODE
  time_reorder_2 += (double)(clock() - begin) / CLOCKS_PER_SEC;
#endif
}

/*******************************************************************************
 * \brief Local transposition.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_transpose_local_complex_low(double complex *grid,
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
void fft_ref_transpose_local_double_low(double *grid, double *grid_transposed,
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
void fft_ref_transpose_local_double_block(double *grid, double *grid_transposed,
                                          const int number_of_columns_grid,
                                          const int number_of_rows_grid,
                                          const int block_size) {
  for (int column_index = 0; column_index < number_of_columns_grid;
       column_index++) {
    for (int row_index = 0; row_index < number_of_rows_grid; row_index++) {
      memcpy(grid_transposed +
                 (column_index * number_of_rows_grid + row_index) * block_size,
             grid + (row_index * number_of_columns_grid + column_index) *
                        block_size,
             block_size * sizeof(double));
    }
  }
}

void reorder_input(const double complex *restrict grid_in,
                   double *restrict grid_out_real,
                   double *restrict grid_out_imag, const int fft_size,
                   const int number_of_ffts, const int stride_in,
                   const int distance_in) {
  if (distance_in == 1 && stride_in == number_of_ffts) {
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in, grid_out_real, grid_out_imag, number_of_ffts, fft_size)
    for (int index = 0; index < fft_size; index++) {
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out_real[fft + index * number_of_ffts] =
            creal(grid_in[index * number_of_ffts + fft]);
        grid_out_imag[fft + index * number_of_ffts] =
            cimag(grid_in[index * number_of_ffts + fft]);
      }
    }
  } else if (distance_in == fft_size && stride_in == 1) {
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in, grid_out_real, grid_out_imag, number_of_ffts, stride_in,   \
               distance_in, fft_size)
    for (int fft = 0; fft < number_of_ffts; fft++) {
      for (int index = 0; index < fft_size; index++) {
        grid_out_real[fft + index * number_of_ffts] =
            creal(grid_in[index + fft * fft_size]);
        grid_out_imag[fft + index * number_of_ffts] =
            cimag(grid_in[index + fft * fft_size]);
      }
    }
  } else {
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in, grid_out_real, grid_out_imag, number_of_ffts, stride_in,   \
               distance_in, fft_size)
    for (int index = 0; index < fft_size; index++) {
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out_real[fft + index * number_of_ffts] =
            creal(grid_in[index * stride_in + fft * distance_in]);
        grid_out_imag[fft + index * number_of_ffts] =
            cimag(grid_in[index * stride_in + fft * distance_in]);
      }
    }
  }
}

void reorder_input_r2c(const double *restrict grid_in,
                       double *restrict grid_out, const int fft_size,
                       const int number_of_ffts, const int stride_in,
                       const int distance_in) {
  if (distance_in == 1 && stride_in == number_of_ffts) {
    memcpy(grid_out, grid_in, number_of_ffts * fft_size * sizeof(double));
  } else if (distance_in == fft_size && stride_in == 1) {
#pragma omp parallel for default(none) collapse(2) shared(                     \
        grid_in, grid_out, number_of_ffts, stride_in, distance_in, fft_size)
    for (int fft = 0; fft < number_of_ffts; fft++) {
      for (int index = 0; index < fft_size; index++) {
        grid_out[fft + index * number_of_ffts] =
            grid_in[index + fft * fft_size];
      }
    }
  } else {
#pragma omp parallel for default(none) collapse(2) shared(                     \
        grid_in, grid_out, number_of_ffts, stride_in, distance_in, fft_size)
    for (int index = 0; index < fft_size; index++) {
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out[fft + index * number_of_ffts] =
            grid_in[index * stride_in + fft * distance_in];
      }
    }
  }
}

void reorder_input_r2c_for_c2c(const double *restrict grid_in,
                               double *restrict grid_out_real,
                               double *restrict grid_out_imag,
                               const int fft_size[2], const int number_of_ffts,
                               const int stride_in, const int distance_in) {
#pragma omp parallel for default(none) collapse(3)                             \
    shared(grid_in, grid_out_real, grid_out_imag, number_of_ffts, stride_in,   \
               distance_in, fft_size)
  for (int index_large = 0; index_large < fft_size[0] / 2; index_large++) {
    for (int index_2 = 0; index_2 < fft_size[1]; index_2++) {
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out_real[(index_large * fft_size[1] + index_2) * number_of_ffts +
                      fft] =
            grid_in[(2 * index_large * fft_size[1] + index_2) * stride_in +
                    fft * distance_in];
        grid_out_imag[(index_large * fft_size[1] + index_2) * number_of_ffts +
                      fft] =
            grid_in[((2 * index_large + 1) * fft_size[1] + index_2) *
                        stride_in +
                    fft * distance_in];
      }
    }
  }
}

void reorder_input_c2r(const double complex *restrict grid_in,
                       double *restrict grid_out_real,
                       double *restrict grid_out_imag, const int fft_size,
                       const int number_of_ffts, const int stride_in,
                       const int distance_in) {
  // If the distance is 1, we can use a simple copy
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in, grid_out_real, grid_out_imag, number_of_ffts, stride_in,   \
               distance_in, fft_size)
  for (int index = 0; index < fft_size / 2 + 1; index++) {
    for (int fft = 0; fft < number_of_ffts; fft++) {
      grid_out_real[fft + index * number_of_ffts] =
          creal(grid_in[fft * distance_in + index * stride_in]);
      grid_out_imag[fft + index * number_of_ffts] =
          cimag(grid_in[fft * distance_in + index * stride_in]);
    }
  }
}

void reorder_input_c2r_2d(const double complex *restrict grid_in,
                          double *restrict grid_out_real,
                          double *restrict grid_out_imag, const int fft_size[2],
                          const int number_of_ffts, const int stride_in,
                          const int distance_in) {
  // If the distance is 1, we can use a simple copy
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in, grid_out_real, grid_out_imag, number_of_ffts, stride_in,   \
               distance_in, fft_size)
  for (int index_0 = 0; index_0 < fft_size[0] / 2 + 1; index_0++) {
    for (int index_1 = 0; index_1 < fft_size[1]; index_1++) {
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out_real[fft + (index_1 * (fft_size[0] / 2 + 1) + index_0) *
                                number_of_ffts] =
            creal(grid_in[fft * distance_in +
                          (index_0 * fft_size[1] + index_1) * stride_in]);
        grid_out_imag[fft + (index_1 * (fft_size[0] / 2 + 1) + index_0) *
                                number_of_ffts] =
            cimag(grid_in[fft * distance_in +
                          (index_0 * fft_size[1] + index_1) * stride_in]);
      }
    }
  }
}

void reorder_input_c2r_for_c2c(const double complex *restrict grid_in,
                               double *restrict grid_out_real,
                               double *restrict grid_out_imag,
                               const int large_factor, const int number_of_ffts,
                               const int stride_in, const int distance_in) {
  const double pi = acos(-1);
#pragma omp parallel for default(none)                                         \
    shared(grid_in, grid_out_real, grid_out_imag, number_of_ffts, stride_in,   \
               distance_in, large_factor, pi)
  for (int index_large = 0; index_large < large_factor; index_large++) {
    const double complex phase_factor =
        cexp(pi * I * index_large / large_factor);
    // This refers to Re(1+i*phase_factor) and Re(1-i*phase_factor)
    const double factor_plus_real = 1.0 - cimag(phase_factor);
    const double factor_minus_real = 1.0 + cimag(phase_factor);
    const double half_factor_real = creal(phase_factor);
    for (int fft = 0; fft < number_of_ffts; fft++) {
      grid_out_real[index_large * number_of_ffts + fft] =
          factor_plus_real *
              creal(grid_in[index_large * stride_in + fft * distance_in]) -
          half_factor_real *
              cimag(grid_in[index_large * stride_in + fft * distance_in]) +
          factor_minus_real *
              creal(grid_in[(large_factor - index_large) * stride_in +
                            fft * distance_in]) -
          half_factor_real *
              cimag(grid_in[(large_factor - index_large) * stride_in +
                            fft * distance_in]);
      grid_out_imag[index_large * number_of_ffts + fft] =
          half_factor_real *
              creal(grid_in[index_large * stride_in + fft * distance_in]) +
          factor_plus_real *
              cimag(grid_in[index_large * stride_in + fft * distance_in]) -
          half_factor_real *
              creal(grid_in[(large_factor - index_large) * stride_in +
                            fft * distance_in]) -
          factor_minus_real *
              cimag(grid_in[(large_factor - index_large) * stride_in +
                            fft * distance_in]);
    }
  }
}

void reorder_input_c2r_for_c2c_2d(const double *restrict grid_in_real,
                                  const double *restrict grid_in_imag,
                                  double *restrict grid_out_real,
                                  double *restrict grid_out_imag,
                                  const int fft_size[2],
                                  const int number_of_ffts) {
  assert(fft_size[0] % 2 == 0);
  const double pi = acos(-1);
  const int large_factor = fft_size[0] / 2;
#pragma omp parallel for default(none)                                         \
    shared(grid_in_real, grid_in_imag, grid_out_real, grid_out_imag,           \
               number_of_ffts, large_factor, fft_size, pi)
  for (int index_large = 0; index_large < large_factor; index_large++) {
    const double complex phase_factor =
        cexp(pi * I * index_large / large_factor);
    // This refers to Re(1+i*phase_factor) and Re(1-i*phase_factor)
    const double factor_plus_real = 1.0 - cimag(phase_factor);
    const double factor_minus_real = 1.0 + cimag(phase_factor);
    const double half_factor_real = creal(phase_factor);
    for (int index_1 = 0; index_1 < fft_size[1]; index_1++) {
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out_real[(index_1 + index_large * fft_size[1]) * number_of_ffts +
                      fft] =
            factor_plus_real *
                grid_in_real[(index_large + index_1 * (large_factor + 1)) *
                                 number_of_ffts +
                             fft] -
            half_factor_real *
                grid_in_imag[(index_large + index_1 * (large_factor + 1)) *
                                 number_of_ffts +
                             fft] +
            factor_minus_real * grid_in_real[((large_factor - index_large) +
                                              index_1 * (large_factor + 1)) *
                                                 number_of_ffts +
                                             fft] -
            half_factor_real * grid_in_imag[((large_factor - index_large) +
                                             index_1 * (large_factor + 1)) *
                                                number_of_ffts +
                                            fft];
        grid_out_imag[(index_1 + index_large * fft_size[1]) * number_of_ffts +
                      fft] =
            half_factor_real *
                grid_in_real[(index_large + index_1 * (large_factor + 1)) *
                                 number_of_ffts +
                             fft] +
            factor_plus_real *
                grid_in_imag[(index_large + index_1 * (large_factor + 1)) *
                                 number_of_ffts +
                             fft] -
            half_factor_real * grid_in_real[((large_factor - index_large) +
                                             index_1 * (large_factor + 1)) *
                                                number_of_ffts +
                                            fft] -
            factor_minus_real * grid_in_imag[((large_factor - index_large) +
                                              index_1 * (large_factor + 1)) *
                                                 number_of_ffts +
                                             fft];
      }
    }
  }
}

void reorder_output(const double *grid_in_real, const double *grid_in_imag,
                    double complex *grid_out, const int fft_size,
                    const int number_of_ffts, const int stride_out,
                    const int distance_out) {
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in_real, grid_in_imag, grid_out, number_of_ffts, stride_out,   \
               distance_out, fft_size)
  for (int index = 0; index < fft_size; index++) {
    for (int fft = 0; fft < number_of_ffts; fft++) {
      grid_out[fft * distance_out + index * stride_out] =
          CMPLX(grid_in_real[fft + index * number_of_ffts],
                grid_in_imag[fft + index * number_of_ffts]);
    }
  }
}

void reorder_output_2d(const double *grid_in_real, const double *grid_in_imag,
                       double complex *grid_out, const int fft_size[2],
                       const int number_of_ffts, const int stride_out,
                       const int distance_out) {
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in_real, grid_in_imag, grid_out, number_of_ffts, stride_out,   \
               distance_out, fft_size)
  for (int index_1 = 0; index_1 < fft_size[1]; index_1++) {
    for (int index_0 = 0; index_0 < fft_size[0]; index_0++) {
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out[fft * distance_out +
                 (index_0 * fft_size[1] + index_1) * stride_out] =
            CMPLX(grid_in_real[fft + (index_1 * fft_size[0] + index_0) *
                                         number_of_ffts],
                  grid_in_imag[fft + (index_1 * fft_size[0] + index_0) *
                                         number_of_ffts]);
      }
    }
  }
}

void reorder_output_r2c(const double *grid_in_real, const double *grid_in_imag,
                        double *grid_out, const int fft_size,
                        const int number_of_ffts, const int stride_out,
                        const int distance_out, const int imag_shift) {
// The first element is just given by the real part
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in_real, grid_in_imag, grid_out, number_of_ffts, stride_out,   \
               distance_out, fft_size, imag_shift)
  for (int index = 0; index < fft_size / 2 + 1; index++) {
    for (int fft = 0; fft < number_of_ffts; fft++) {
      grid_out[2 * fft * distance_out + 2 * index * stride_out] =
          grid_in_real[fft + index * number_of_ffts];
      grid_out[2 * fft * distance_out + 2 * index * stride_out + imag_shift] =
          grid_in_imag[fft + index * number_of_ffts];
    }
  }
}

void reorder_output_c2r(const double *grid_in, double *grid_out,
                        const int fft_size, const int number_of_ffts,
                        const int stride_out, const int distance_out) {
#pragma omp parallel for default(none) collapse(2) shared(                     \
        grid_in, grid_out, number_of_ffts, stride_out, distance_out, fft_size)
  for (int index = 0; index < fft_size; index++) {
    for (int fft = 0; fft < number_of_ffts; fft++) {
      grid_out[fft * distance_out + index * stride_out] =
          grid_in[fft + index * number_of_ffts];
    }
  }
}

void reorder_output_c2r_from_c2c(const double *grid_in_real,
                                 const double *grid_in_imag, double *grid_out,
                                 const int large_factor,
                                 const int number_of_ffts, const int stride_out,
                                 const int distance_out) {
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in_real, grid_in_imag, grid_out, number_of_ffts, stride_out,   \
               distance_out, large_factor)
  for (int index_large = 0; index_large < large_factor; index_large++) {
    for (int fft = 0; fft < number_of_ffts; fft++) {
      grid_out[2 * index_large * stride_out + fft * distance_out] =
          grid_in_real[index_large * number_of_ffts + fft];
      grid_out[(2 * index_large + 1) * stride_out + fft * distance_out] =
          grid_in_imag[index_large * number_of_ffts + fft];
    }
  }
}

void reorder_output_c2r_from_c2c_2d(
    const double *grid_in_real, const double *grid_in_imag, double *grid_out,
    const int large_factor, const int fft_size_2, const int number_of_ffts,
    const int stride_out, const int distance_out) {
#pragma omp parallel for default(none) collapse(3)                             \
    shared(grid_in_real, grid_in_imag, grid_out, number_of_ffts, stride_out,   \
               distance_out, large_factor, fft_size_2)
  for (int index_large = 0; index_large < large_factor; index_large++) {
    for (int index_2 = 0; index_2 < fft_size_2; index_2++) {
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out[(2 * index_large * fft_size_2 + index_2) * stride_out +
                 fft * distance_out] =
            grid_in_real[(index_large * fft_size_2 + index_2) * number_of_ffts +
                         fft];
        grid_out[((2 * index_large + 1) * fft_size_2 + index_2) * stride_out +
                 fft * distance_out] =
            grid_in_imag[(index_large * fft_size_2 + index_2) * number_of_ffts +
                         fft];
      }
    }
  }
}

/*******************************************************************************
 * \brief Naive implementation of FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_fw_local_naive(const double *restrict grid_in_real,
                               const double *restrict grid_in_imag,
                               double *restrict grid_out_real,
                               double *restrict grid_out_imag,
                               const int fft_size, const int number_of_ffts) {
#if PROFILE_CODE
  clock_t begin = clock();
#endif
  // Perform FFTs along the first dimension
  if (fft_size == 1) {
    // If the FFT size is 1, we can use a simple copy
    memcpy(grid_out_real, grid_in_real,
           fft_size * number_of_ffts * sizeof(double));
    memcpy(grid_out_imag, grid_in_imag,
           fft_size * number_of_ffts * sizeof(double));
  } else if (fft_size == 2) {
    for (int fft = 0; fft < number_of_ffts; fft++) {
      grid_out_real[fft] =
          grid_in_real[fft] + grid_in_real[number_of_ffts + fft];
      grid_out_imag[fft] =
          grid_in_imag[fft] + grid_in_imag[number_of_ffts + fft];
      grid_out_real[number_of_ffts + fft] =
          grid_in_real[fft] - grid_in_real[number_of_ffts + fft];
      grid_out_imag[number_of_ffts + fft] =
          grid_in_imag[fft] - grid_in_imag[number_of_ffts + fft];
    }
#if PROFILE_CODE
    time_2 += (double)(clock() - begin) / CLOCKS_PER_SEC;
#endif
  } else if (fft_size == 4) {
#pragma omp parallel for default(none)                                         \
    shared(grid_in_real, grid_in_imag, grid_out_real, grid_out_imag,           \
               number_of_ffts)
    for (int fft = 0; fft < number_of_ffts; fft++) {
      const double real_sum1 =
          grid_in_real[fft] + grid_in_real[fft + 2 * number_of_ffts];
      const double real_sum2 = grid_in_real[fft + number_of_ffts] +
                               grid_in_real[fft + 3 * number_of_ffts];
      const double real_diff1 =
          grid_in_real[fft] - grid_in_real[fft + 2 * number_of_ffts];
      const double real_diff2 = grid_in_real[fft + number_of_ffts] -
                                grid_in_real[fft + 3 * number_of_ffts];
      const double imag_sum1 =
          grid_in_imag[fft] + grid_in_imag[fft + 2 * number_of_ffts];
      const double imag_sum2 = grid_in_imag[fft + number_of_ffts] +
                               grid_in_imag[fft + 3 * number_of_ffts];
      const double imag_diff1 =
          grid_in_imag[fft] - grid_in_imag[fft + 2 * number_of_ffts];
      const double imag_diff2 = grid_in_imag[fft + number_of_ffts] -
                                grid_in_imag[fft + 3 * number_of_ffts];
      grid_out_real[fft] = real_sum1 + real_sum2;
      grid_out_real[fft + number_of_ffts] = real_diff1 + imag_diff2;
      grid_out_real[fft + 2 * number_of_ffts] = real_sum1 - real_sum2;
      grid_out_real[fft + 3 * number_of_ffts] = real_diff1 - imag_diff2;
      grid_out_imag[fft] = imag_sum1 + imag_sum2;
      grid_out_imag[fft + number_of_ffts] = imag_diff1 - real_diff2;
      grid_out_imag[fft + 2 * number_of_ffts] = imag_sum1 - imag_sum2;
      grid_out_imag[fft + 3 * number_of_ffts] = imag_diff1 + real_diff2;
    }
#if PROFILE_CODE
    time_4 += (double)(clock() - begin) / CLOCKS_PER_SEC;
#endif
  } else if (fft_size == 3) {
    const double half_sqrt3 = 0.5 * sqrt(3.0);
#pragma omp parallel for default(none)                                         \
    shared(grid_in_real, grid_in_imag, grid_out_real, grid_out_imag,           \
               number_of_ffts, half_sqrt3)
    for (int fft = 0; fft < number_of_ffts; fft++) {
      const double real_sum = grid_in_real[fft + number_of_ffts] +
                              grid_in_real[fft + 2 * number_of_ffts];
      const double imag_sum = grid_in_imag[fft + number_of_ffts] +
                              grid_in_imag[fft + 2 * number_of_ffts];
      const double real_diff = grid_in_real[fft + number_of_ffts] -
                               grid_in_real[fft + 2 * number_of_ffts];
      const double imag_diff = grid_in_imag[fft + number_of_ffts] -
                               grid_in_imag[fft + 2 * number_of_ffts];
      grid_out_real[fft] = grid_in_real[fft] + real_sum;
      grid_out_imag[fft] = grid_in_imag[fft] + imag_sum;
      grid_out_real[fft + number_of_ffts] =
          grid_in_real[fft] - 0.5 * real_sum + half_sqrt3 * imag_diff;
      grid_out_imag[fft + number_of_ffts] =
          grid_in_imag[fft] - half_sqrt3 * real_diff - 0.5 * imag_sum;
      grid_out_real[fft + 2 * number_of_ffts] =
          grid_in_real[fft] - 0.5 * real_sum - half_sqrt3 * imag_diff;
      grid_out_imag[fft + 2 * number_of_ffts] =
          grid_in_imag[fft] + half_sqrt3 * real_diff - 0.5 * imag_sum;
    }
#if PROFILE_CODE
    time_3 += (double)(clock() - begin) / CLOCKS_PER_SEC;
#endif
  } else {
    const double pi = acos(-1.0);
    memset(grid_out_real, 0, fft_size * number_of_ffts * sizeof(double));
    memset(grid_out_imag, 0, fft_size * number_of_ffts * sizeof(double));
#pragma omp parallel for default(none)                                         \
    shared(grid_in_real, grid_in_imag, grid_out_real, grid_out_imag, fft_size, \
               number_of_ffts, pi)
    for (int index_out = 0; index_out < fft_size; index_out++) {
      for (int index_in = 0; index_in < fft_size; index_in++) {
        const double complex phase_factor =
            cexp(-2.0 * I * pi * index_out * index_in / fft_size);
        for (int fft = 0; fft < number_of_ffts; fft++) {
          grid_out_real[fft + index_out * number_of_ffts] +=
              grid_in_real[fft + index_in * number_of_ffts] *
                  creal(phase_factor) -
              grid_in_imag[fft + index_in * number_of_ffts] *
                  cimag(phase_factor);
          grid_out_imag[fft + index_out * number_of_ffts] +=
              grid_in_real[fft + index_in * number_of_ffts] *
                  cimag(phase_factor) +
              grid_in_imag[fft + index_in * number_of_ffts] *
                  creal(phase_factor);
        }
      }
    }
#if PROFILE_CODE
    time_naive += (double)(clock() - begin) / CLOCKS_PER_SEC;
#endif
  }
}

/*******************************************************************************
 * \brief Naive implementation of FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_fw_local_r2c_naive(const double *restrict grid_in,
                                   double *restrict grid_out,
                                   const int fft_size,
                                   const int number_of_ffts) {
#if PROFILE_CODE
  clock_t begin = clock();
#endif
  double *grid_out_real = grid_out;
  double *grid_out_imag = grid_out_real + (fft_size / 2 + 1) * number_of_ffts;
  // Perform FFTs along the first dimension
  const double pi = acos(-1.0);
  memset(grid_out_real, 0,
         (fft_size / 2 + 1) * number_of_ffts * sizeof(double));
  memset(grid_out_imag, 0,
         (fft_size / 2 + 1) * number_of_ffts * sizeof(double));
#pragma omp parallel for default(none) shared(                                 \
        grid_in, grid_out_real, grid_out_imag, fft_size, number_of_ffts, pi)
  for (int index_out = 0; index_out < fft_size / 2 + 1; index_out++) {
    for (int index_in = 0; index_in < fft_size; index_in++) {
      const double complex phase_factor =
          cexp(-2.0 * I * pi * index_out * index_in / fft_size);
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out_real[fft + index_out * number_of_ffts] +=
            grid_in[fft + index_in * number_of_ffts] * creal(phase_factor);
        grid_out_imag[fft + index_out * number_of_ffts] +=
            grid_in[fft + index_in * number_of_ffts] * cimag(phase_factor);
      }
    }
  }
#if PROFILE_CODE
  time_naive += 0.0 * (double)(clock() - begin) / CLOCKS_PER_SEC;
#endif
}

/*******************************************************************************
 * \brief Naive implementation of FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_bw_local_naive(const double *restrict grid_in_real,
                               const double *restrict grid_in_imag,
                               double *restrict grid_out_real,
                               double *restrict grid_out_imag,
                               const int fft_size, const int number_of_ffts) {
#if PROFILE_CODE
  clock_t begin = clock();
#endif
  if (fft_size == 1) {
    // If the FFT size is 1, we can use a simple copy
    memcpy(grid_out_real, grid_in_real, number_of_ffts * sizeof(double));
    memcpy(grid_out_imag, grid_in_imag, number_of_ffts * sizeof(double));
  } else if (fft_size == 2) {
    for (int fft = 0; fft < number_of_ffts; fft++) {
      grid_out_real[fft] =
          grid_in_real[fft] + grid_in_real[fft + number_of_ffts];
      grid_out_imag[fft] =
          grid_in_imag[fft] + grid_in_imag[fft + number_of_ffts];
      grid_out_real[fft + number_of_ffts] =
          grid_in_real[fft] - grid_in_real[fft + number_of_ffts];
      grid_out_imag[fft + number_of_ffts] =
          grid_in_imag[fft] - grid_in_imag[fft + number_of_ffts];
    }
#if PROFILE_CODE
    time_2 += (double)(clock() - begin) / CLOCKS_PER_SEC;
#endif
  } else if (fft_size == 4) {
#pragma omp parallel for default(none)                                         \
    shared(grid_in_real, grid_in_imag, grid_out_real, grid_out_imag,           \
               number_of_ffts)
    for (int fft = 0; fft < number_of_ffts; fft++) {
      const double real_sum1 =
          grid_in_real[fft] + grid_in_real[fft + 2 * number_of_ffts];
      const double real_sum2 = grid_in_real[fft + number_of_ffts] +
                               grid_in_real[fft + 3 * number_of_ffts];
      const double real_diff1 =
          grid_in_real[fft] - grid_in_real[fft + 2 * number_of_ffts];
      const double real_diff2 = grid_in_real[fft + number_of_ffts] -
                                grid_in_real[fft + 3 * number_of_ffts];
      const double imag_sum1 =
          grid_in_imag[fft] + grid_in_imag[fft + 2 * number_of_ffts];
      const double imag_sum2 = grid_in_imag[fft + number_of_ffts] +
                               grid_in_imag[fft + 3 * number_of_ffts];
      const double imag_diff1 =
          grid_in_imag[fft] - grid_in_imag[fft + 2 * number_of_ffts];
      const double imag_diff2 = grid_in_imag[fft + number_of_ffts] -
                                grid_in_imag[fft + 3 * number_of_ffts];
      grid_out_real[fft] = real_sum1 + real_sum2;
      grid_out_real[fft + number_of_ffts] = real_diff1 - imag_diff2;
      grid_out_real[fft + 2 * number_of_ffts] = real_sum1 - real_sum2;
      grid_out_real[fft + 3 * number_of_ffts] = real_diff1 + imag_diff2;
      grid_out_imag[fft] = imag_sum1 + imag_sum2;
      grid_out_imag[fft + number_of_ffts] = imag_diff1 + real_diff2;
      grid_out_imag[fft + 2 * number_of_ffts] = imag_sum1 - imag_sum2;
      grid_out_imag[fft + 3 * number_of_ffts] = imag_diff1 - real_diff2;
    }
#if PROFILE_CODE
    time_4 += (double)(clock() - begin) / CLOCKS_PER_SEC;
#endif
  } else if (fft_size == 3) {
    const double half_sqrt3 = 0.5 * sqrt(3.0);
#pragma omp parallel for default(none)                                         \
    shared(grid_in_real, grid_in_imag, grid_out_real, grid_out_imag,           \
               number_of_ffts, half_sqrt3)
    for (int fft = 0; fft < number_of_ffts; fft++) {
      const double real_sum = grid_in_real[fft + number_of_ffts] +
                              grid_in_real[fft + 2 * number_of_ffts];
      const double imag_sum = grid_in_imag[fft + number_of_ffts] +
                              grid_in_imag[fft + 2 * number_of_ffts];
      const double real_diff = grid_in_real[fft + number_of_ffts] -
                               grid_in_real[fft + 2 * number_of_ffts];
      const double imag_diff = grid_in_imag[fft + number_of_ffts] -
                               grid_in_imag[fft + 2 * number_of_ffts];
      grid_out_real[fft] = grid_in_real[fft] + real_sum;
      grid_out_imag[fft] = grid_in_imag[fft] + imag_sum;
      grid_out_real[fft + number_of_ffts] =
          grid_in_real[fft] - 0.5 * real_sum - half_sqrt3 * imag_diff;
      grid_out_imag[fft + number_of_ffts] =
          grid_in_imag[fft] + half_sqrt3 * real_diff - 0.5 * imag_sum;
      grid_out_real[fft + 2 * number_of_ffts] =
          grid_in_real[fft] - 0.5 * real_sum + half_sqrt3 * imag_diff;
      grid_out_imag[fft + 2 * number_of_ffts] =
          grid_in_imag[fft] - half_sqrt3 * real_diff - 0.5 * imag_sum;
    }
#if PROFILE_CODE
    time_3 += (double)(clock() - begin) / CLOCKS_PER_SEC;
#endif
  } else {
    const double pi = acos(-1.0);
    memset(grid_out_real, 0, fft_size * number_of_ffts * sizeof(double));
    memset(grid_out_imag, 0, fft_size * number_of_ffts * sizeof(double));
#pragma omp parallel for default(none)                                         \
    shared(grid_in_real, grid_in_imag, grid_out_real, grid_out_imag, fft_size, \
               number_of_ffts, pi)
    for (int index_out = 0; index_out < fft_size; index_out++) {
      for (int index_in = 0; index_in < fft_size; index_in++) {
        const double complex phase_factor =
            cexp(2.0 * I * pi * index_out * index_in / fft_size);
        for (int fft = 0; fft < number_of_ffts; fft++) {
          grid_out_real[fft + index_out * number_of_ffts] +=
              grid_in_real[fft + index_in * number_of_ffts] *
                  creal(phase_factor) -
              grid_in_imag[fft + index_in * number_of_ffts] *
                  cimag(phase_factor);
          grid_out_imag[fft + index_out * number_of_ffts] +=
              grid_in_real[fft + index_in * number_of_ffts] *
                  cimag(phase_factor) +
              grid_in_imag[fft + index_in * number_of_ffts] *
                  creal(phase_factor);
        }
      }
    }
#if PROFILE_CODE
    time_naive += (double)(clock() - begin) / CLOCKS_PER_SEC;
#endif
  }
}

/*******************************************************************************
 * \brief Naive implementation of FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_bw_local_c2r_naive(double *restrict grid_in,
                                   double *restrict grid_out,
                                   const int fft_size,
                                   const int number_of_ffts) {
#if PROFILE_CODE
  clock_t begin = clock();
#endif

  double *grid_in_real = grid_in;
  double *grid_in_imag = grid_in + (fft_size / 2 + 1) * number_of_ffts;

  for (int index_out = 0; index_out < fft_size; index_out++) {
    for (int fft = 0; fft < number_of_ffts; fft++) {
      grid_out[fft + index_out * number_of_ffts] = 0.0;
    }
    for (int index_in = 0; index_in < fft_size / 2 + 1; index_in++) {
      double complex phase_factor =
          cexp(2.0 * I * acos(-1) * index_in * index_out / fft_size);
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out[fft + index_out * number_of_ffts] +=
            grid_in_real[fft + index_in * number_of_ffts] *
                creal(phase_factor) -
            grid_in_imag[fft + index_in * number_of_ffts] * cimag(phase_factor);
      }
    }
    for (int index_in = fft_size / 2 + 1; index_in < fft_size; index_in++) {
      double complex phase_factor2 =
          cexp(2.0 * I * acos(-1) * index_in * index_out / fft_size);
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out[fft + index_out * number_of_ffts] +=
            grid_in_real[fft + (fft_size - index_in) * number_of_ffts] *
                creal(phase_factor2) +
            grid_in_imag[fft + (fft_size - index_in) * number_of_ffts] *
                cimag(phase_factor2);
      }
    }
  }
#if PROFILE_CODE
  time_naive += (double)(clock() - begin) / CLOCKS_PER_SEC;
#endif
}

void fft_twiddle_r2c_fw_for_c2c(const double *grid_in_real,
                                const double *grid_in_imag, double *grid_out,
                                const int fft_size, const int number_of_ffts,
                                const int stride_out, const int distance_out,
                                const int imag_shift) {
#if PROFILE_CODE
  clock_t begin = clock();
#endif
  const int large_factor = fft_size / 2;
  for (int index_large = 0; index_large < large_factor + 1; index_large++) {
    const double complex phase_factor =
        cexp(-acos(-1) * I * index_large / large_factor);
    // This refers to Re(1+i*phase_factor) and Re(1-i*phase_factor)
    const double factor_plus_real = 0.5 - 0.5 * cimag(phase_factor);
    const double factor_minus_real = 0.5 + 0.5 * cimag(phase_factor);
    const double half_factor_real = 0.5 * creal(phase_factor);
    for (int fft = 0; fft < number_of_ffts; fft++) {
      grid_out[2 * index_large * stride_out + 2 * fft * distance_out] =
          factor_minus_real *
              grid_in_real[index_large % large_factor * number_of_ffts + fft] +
          half_factor_real *
              grid_in_imag[index_large % large_factor * number_of_ffts + fft] +
          factor_plus_real * grid_in_real[(large_factor - index_large) %
                                              large_factor * number_of_ffts +
                                          fft] +
          half_factor_real * grid_in_imag[(large_factor - index_large) %
                                              large_factor * number_of_ffts +
                                          fft];
      grid_out[2 * index_large * stride_out + 2 * fft * distance_out +
               imag_shift] =
          -half_factor_real *
              grid_in_real[index_large % large_factor * number_of_ffts + fft] +
          factor_minus_real *
              grid_in_imag[index_large % large_factor * number_of_ffts + fft] +
          half_factor_real * grid_in_real[(large_factor - index_large) %
                                              large_factor * number_of_ffts +
                                          fft] -
          factor_plus_real * grid_in_imag[(large_factor - index_large) %
                                              large_factor * number_of_ffts +
                                          fft];
    }
  }
#if PROFILE_CODE
  time_twiddle += (double)(clock() - begin) / CLOCKS_PER_SEC;
#endif
}

void fft_twiddle_r2c_fw_for_c2c_2d(const double *grid_in_real,
                                   const double *grid_in_imag, double *grid_out,
                                   const int fft_size[2],
                                   const int number_of_ffts) {
#if PROFILE_CODE
  clock_t begin = clock();
#endif
  const int large_factor = fft_size[0] / 2;
  for (int index_large = 0; index_large < large_factor + 1; index_large++) {
    const double complex phase_factor =
        cexp(-acos(-1) * I * index_large / large_factor);
    // This refers to Re(1+i*phase_factor) and Re(1-i*phase_factor)
    const double factor_plus_real = 0.5 - 0.5 * cimag(phase_factor);
    const double factor_minus_real = 0.5 + 0.5 * cimag(phase_factor);
    const double half_factor_real = 0.5 * creal(phase_factor);
    for (int index_1 = 0; index_1 < fft_size[1]; index_1++) {
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out[(index_1 * (large_factor + 1) + index_large) * number_of_ffts +
                 fft] =
            factor_minus_real *
                grid_in_real[(index_large % large_factor * fft_size[1] +
                              index_1) *
                                 number_of_ffts +
                             fft] +
            half_factor_real *
                grid_in_imag[(index_large % large_factor * fft_size[1] +
                              index_1) *
                                 number_of_ffts +
                             fft] +
            factor_plus_real * grid_in_real[((large_factor - index_large) %
                                                 large_factor * fft_size[1] +
                                             index_1) *
                                                number_of_ffts +
                                            fft] +
            half_factor_real * grid_in_imag[((large_factor - index_large) %
                                                 large_factor * fft_size[1] +
                                             index_1) *
                                                number_of_ffts +
                                            fft];
        grid_out[(large_factor + 1) * number_of_ffts * fft_size[1] +
                 (index_1 * (large_factor + 1) + index_large) * number_of_ffts +
                 fft] =
            -half_factor_real *
                grid_in_real[(index_large % large_factor * fft_size[1] +
                              index_1) *
                                 number_of_ffts +
                             fft] +
            factor_minus_real *
                grid_in_imag[(index_large % large_factor * fft_size[1] +
                              index_1) *
                                 number_of_ffts +
                             fft] +
            half_factor_real * grid_in_real[((large_factor - index_large) %
                                                 large_factor * fft_size[1] +
                                             index_1) *
                                                number_of_ffts +
                                            fft] -
            factor_plus_real * grid_in_imag[((large_factor - index_large) %
                                                 large_factor * fft_size[1] +
                                             index_1) *
                                                number_of_ffts +
                                            fft];
      }
    }
  }
#if PROFILE_CODE
  time_twiddle += (double)(clock() - begin) / CLOCKS_PER_SEC;
#endif
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_fw_local_internal(double *restrict grid_in_real,
                                  double *restrict grid_in_imag,
                                  double *restrict grid_out_real,
                                  double *restrict grid_out_imag,
                                  const int fft_size,
                                  const int number_of_ffts) {

  // Determine a factorization of the FFT size
  int small_factor = 1;
  for (int i = 2; i * i <= fft_size; i++) {
    if (fft_size % i == 0) {
      small_factor = i;
      break;
    }
  }
  if (small_factor == 2 && fft_size % 4 == 0)
    small_factor = 4;
  const int large_factor = fft_size / small_factor;
  assert(small_factor * large_factor == fft_size);

  const double pi = acos(-1.0);
  if (small_factor == 1 || large_factor == 1) {
    // If the FFT size is prime, we can use the naive implementation
    fft_ref_1d_fw_local_naive(grid_in_real, grid_in_imag, grid_out_real,
                              grid_out_imag, fft_size, number_of_ffts);
  } else {
    // Perform FFTs along the shorter sub-dimension
    fft_ref_1d_fw_local_naive(grid_in_real, grid_in_imag, grid_out_real,
                              grid_out_imag, small_factor,
                              number_of_ffts * large_factor);
// Transpose and multiply with twiddle factors
#if PROFILE_CODE
    clock_t begin = clock();
#endif
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in_real, grid_in_imag, grid_out_real, grid_out_imag, fft_size, \
               number_of_ffts, pi, small_factor, large_factor)
    for (int index_small = 0; index_small < small_factor; index_small++) {
      for (int index_large = 0; index_large < large_factor; index_large++) {
        const double complex phase_factor =
            cexp(-2.0 * I * pi * index_small * index_large / fft_size);
        for (int fft = 0; fft < number_of_ffts; fft++) {
          grid_in_real[fft + (index_large * small_factor + index_small) *
                                 number_of_ffts] =
              grid_out_real[fft + (index_small * large_factor + index_large) *
                                      number_of_ffts] *
                  creal(phase_factor) -
              grid_out_imag[fft + (index_small * large_factor + index_large) *
                                      number_of_ffts] *
                  cimag(phase_factor);
          grid_in_imag[fft + (index_large * small_factor + index_small) *
                                 number_of_ffts] =
              grid_out_real[fft + (index_small * large_factor + index_large) *
                                      number_of_ffts] *
                  cimag(phase_factor) +
              grid_out_imag[fft + (index_small * large_factor + index_large) *
                                      number_of_ffts] *
                  creal(phase_factor);
        }
      }
    }
#if PROFILE_CODE
    time_twiddle += (double)(clock() - begin) / CLOCKS_PER_SEC;
#endif
    fft_ref_1d_fw_local_internal(grid_in_real, grid_in_imag, grid_out_real,
                                 grid_out_imag, large_factor,
                                 number_of_ffts * small_factor);
  }
}

/*******************************************************************************
 * \brief Cooley-Tukey step (For internal use only, special format).
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_bw_local_internal(double *restrict grid_in_real,
                                  double *restrict grid_in_imag,
                                  double *restrict grid_out_real,
                                  double *restrict grid_out_imag,
                                  const int fft_size,
                                  const int number_of_ffts) {

  // Determine a factorization of the FFT size
  int small_factor = 1;
  for (int i = 2; i * i <= fft_size; i++) {
    if (fft_size % i == 0) {
      small_factor = i;
      break;
    }
  }
  if (small_factor == 2 && fft_size % 4 == 0)
    small_factor = 4;
  const int large_factor = fft_size / small_factor;
  assert(small_factor * large_factor == fft_size);

  const double pi = acos(-1.0);
  if (small_factor == 1 || large_factor == 1) {
    // If the FFT size is prime, we can use the naive implementation
    fft_ref_1d_bw_local_naive(grid_in_real, grid_in_imag, grid_out_real,
                              grid_out_imag, fft_size, number_of_ffts);
  } else {
    // Perform FFTs along the shorter sub-dimension
    fft_ref_1d_bw_local_naive(grid_in_real, grid_in_imag, grid_out_real,
                              grid_out_imag, small_factor,
                              number_of_ffts * large_factor);
// Transpose and multiply with twiddle factors
#if PROFILE_CODE
    clock_t begin = clock();
#endif
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in_real, grid_in_imag, grid_out_real, grid_out_imag, fft_size, \
               number_of_ffts, pi, small_factor, large_factor)
    for (int index_small = 0; index_small < small_factor; index_small++) {
      for (int index_large = 0; index_large < large_factor; index_large++) {
        const double complex phase_factor =
            cexp(2.0 * I * pi * index_small * index_large / fft_size);
        for (int fft = 0; fft < number_of_ffts; fft++) {
          grid_in_real[fft + (index_large * small_factor + index_small) *
                                 number_of_ffts] =
              grid_out_real[fft + (index_small * large_factor + index_large) *
                                      number_of_ffts] *
                  creal(phase_factor) -
              grid_out_imag[fft + (index_small * large_factor + index_large) *
                                      number_of_ffts] *
                  cimag(phase_factor);
          grid_in_imag[fft + (index_large * small_factor + index_small) *
                                 number_of_ffts] =
              grid_out_real[fft + (index_small * large_factor + index_large) *
                                      number_of_ffts] *
                  cimag(phase_factor) +
              grid_out_imag[fft + (index_small * large_factor + index_large) *
                                      number_of_ffts] *
                  creal(phase_factor);
        }
      }
    }
#if PROFILE_CODE
    time_twiddle += (double)(clock() - begin) / CLOCKS_PER_SEC;
#endif
    fft_ref_1d_bw_local_internal(grid_in_real, grid_in_imag, grid_out_real,
                                 grid_out_imag, large_factor,
                                 number_of_ffts * small_factor);
  }
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_fw_local_low(double complex *restrict grid_in,
                             double complex *restrict grid_out,
                             const int fft_size, const int number_of_ffts,
                             const int stride_in, const int stride_out,
                             const int distance_in, const int distance_out) {

#if PROFILE_CODE
  time_reorder_2 = 0.0;
  time_reorder_3 = 0.0;
  time_reorder_4 = 0.0;
  time_2 = 0.0;
  time_3 = 0.0;
  time_4 = 0.0;
  time_naive = 0.0;
  time_twiddle = 0.0;

  clock_t begin = clock();
#endif

  // We reorder the data to a format more suitable for vectorization
  double *grid_in_real = (double *)grid_in;
  double *grid_in_imag = ((double *)grid_in) + fft_size * number_of_ffts;
  double *grid_out_real = (double *)grid_out;
  double *grid_out_imag = ((double *)grid_out) + fft_size * number_of_ffts;

  reorder_input(grid_in, grid_out_real, grid_out_imag, fft_size, number_of_ffts,
                stride_in, distance_in);
  fft_ref_1d_fw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size, number_of_ffts);
  reorder_output(grid_in_real, grid_in_imag, grid_out, fft_size, number_of_ffts,
                 stride_out, distance_out);

#if PROFILE_CODE
  clock_t end = clock();
  double time_total = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Time Reorder 2: %f\n", time_reorder_2);
  printf("Time Reorder 3: %f\n", time_reorder_3);
  printf("Time Reorder 4: %f\n", time_reorder_4);
  printf("Time 2: %f\n", time_2);
  printf("Time 3: %f\n", time_3);
  printf("Time 4: %f\n", time_4);
  printf("Time naive: %f\n", time_naive);
  printf("Time twiddle: %f\n", time_twiddle);
  printf("Time rest: %f\n", time_total - time_naive - time_twiddle -
                                time_reorder_2 - time_reorder_3 -
                                time_reorder_4 - time_2 - time_3 - time_4);
  printf("Total Time 1D FW %i %i: %f\n", fft_size, number_of_ffts, time_total);
  fflush(stdout);
#endif
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_fw_local_r2c_low(double *restrict grid_in,
                                 double complex *restrict grid_out,
                                 const int fft_size, const int number_of_ffts,
                                 const int stride_in, const int stride_out,
                                 const int distance_in,
                                 const int distance_out) {

#if PROFILE_CODE
  time_reorder_2 = 0.0;
  time_reorder_3 = 0.0;
  time_reorder_4 = 0.0;
  time_2 = 0.0;
  time_3 = 0.0;
  time_4 = 0.0;
  time_naive = 0.0;
  time_twiddle = 0.0;

  clock_t begin = clock();
#endif

  double *grid_in_real = grid_in;
  double *grid_out_real = (double *)grid_out;
  double *grid_in_imag = grid_in_real + (fft_size / 2 + 1) * number_of_ffts;
  double *grid_out_imag = grid_out_real + (fft_size / 2 + 1) * number_of_ffts;

  if (fft_size % 2 == 0) {
    const int large_factor = fft_size / 2;
    reorder_input_r2c_for_c2c(grid_in, grid_out_real, grid_out_imag,
                              (const int[2]){fft_size, 1}, number_of_ffts,
                              stride_in, distance_in);
    fft_ref_1d_fw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                                 grid_in_imag, large_factor, number_of_ffts);
    fft_twiddle_r2c_fw_for_c2c(grid_in_real, grid_in_imag, grid_out_real,
                               fft_size, number_of_ffts, stride_out,
                               distance_out, 1);
  } else {
    reorder_input_r2c(grid_in, grid_out_real, fft_size, number_of_ffts,
                      stride_in, distance_in);
    fft_ref_1d_fw_local_r2c_naive(grid_out_real, grid_in_real, fft_size,
                                  number_of_ffts);
    double *grid_in_imag = grid_in_real + (fft_size / 2 + 1) * number_of_ffts;
    reorder_output_r2c(grid_in_real, grid_in_imag, grid_out_real, fft_size,
                       number_of_ffts, stride_out, distance_out, 1);
  }

#if PROFILE_CODE
  clock_t end = clock();
  double time_total = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Time Reorder 2: %f\n", time_reorder_2);
  printf("Time Reorder 3: %f\n", time_reorder_3);
  printf("Time Reorder 4: %f\n", time_reorder_4);
  printf("Time 2: %f\n", time_2);
  printf("Time 3: %f\n", time_3);
  printf("Time 4: %f\n", time_4);
  printf("Time naive: %f\n", time_naive);
  printf("Time twiddle: %f\n", time_twiddle);
  printf("Time rest: %f\n", time_total - time_reorder_2 - time_reorder_3 -
                                time_reorder_4 - time_2 - time_3 - time_4);
  printf("Total Time 1D R2C FW %i %i: %f\n", fft_size, number_of_ffts,
         time_total);
  fflush(stdout);
#endif
}

/*******************************************************************************
 * \brief Naive implementation of backwards FFT to transposed format (for
 *easier transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_bw_local_low(double complex *restrict grid_in,
                             double complex *restrict grid_out,
                             const int fft_size, const int number_of_ffts,
                             const int stride_in, const int stride_out,
                             const int distance_in, const int distance_out) {

#if PROFILE_CODE
  time_reorder_2 = 0.0;
  time_reorder_3 = 0.0;
  time_reorder_4 = 0.0;
  time_2 = 0.0;
  time_3 = 0.0;
  time_4 = 0.0;
  time_naive = 0.0;
  time_twiddle = 0.0;

  clock_t begin = clock();
#endif

  // We reorder the data to a format more suitable for vectorization
  double *grid_in_real = (double *)grid_in;
  double *grid_in_imag = ((double *)grid_in) + fft_size * number_of_ffts;
  double *grid_out_real = (double *)grid_out;
  double *grid_out_imag = ((double *)grid_out) + fft_size * number_of_ffts;

  reorder_input(grid_in, grid_out_real, grid_out_imag, fft_size, number_of_ffts,
                stride_in, distance_in);
  fft_ref_1d_bw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size, number_of_ffts);
  reorder_output(grid_in_real, grid_in_imag, grid_out, fft_size, number_of_ffts,
                 stride_out, distance_out);

#if PROFILE_CODE
  clock_t end = clock();
  double time_total = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Time Reorder 2: %f\n", time_reorder_2);
  printf("Time Reorder 3: %f\n", time_reorder_3);
  printf("Time Reorder 4: %f\n", time_reorder_4);
  printf("Time 2: %f\n", time_2);
  printf("Time 3: %f\n", time_3);
  printf("Time 4: %f\n", time_4);
  printf("Time naive: %f\n", time_naive);
  printf("Time twiddle: %f\n", time_twiddle);
  printf("Time rest: %f\n", time_total - time_reorder_2 - time_reorder_3 -
                                time_reorder_4 - time_2 - time_3 - time_4);
  printf("Total Time 1D BW %i %i: %f\n", fft_size, number_of_ffts, time_total);
  fflush(stdout);
#endif
}

/*******************************************************************************
 * \brief Naive implementation of backwards FFT to transposed format (for
 *easier transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_bw_local_c2r_low(double complex *restrict grid_in,
                                 double *restrict grid_out, const int fft_size,
                                 const int number_of_ffts, const int stride_in,
                                 const int stride_out, const int distance_in,
                                 const int distance_out) {

#if PROFILE_CODE
  time_reorder_2 = 0.0;
  time_reorder_3 = 0.0;
  time_reorder_4 = 0.0;
  time_2 = 0.0;
  time_3 = 0.0;
  time_4 = 0.0;
  time_naive = 0.0;
  time_twiddle = 0.0;

  clock_t begin = clock();
#endif

  if (fft_size % 2 == 0) {
    const int large_factor = fft_size / 2;
    double *grid_in_real = (double *)grid_in;
    double *grid_in_imag = grid_in_real + large_factor * number_of_ffts;
    double *grid_out_real = grid_out;
    double *grid_out_imag = grid_out + large_factor * number_of_ffts;

    reorder_input_c2r_for_c2c(grid_in, grid_out_real, grid_out_imag,
                              large_factor, number_of_ffts, stride_in,
                              distance_in);
    fft_ref_1d_bw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                                 grid_in_imag, large_factor, number_of_ffts);
    reorder_output_c2r_from_c2c(grid_in_real, grid_in_imag, grid_out,
                                large_factor, number_of_ffts, stride_out,
                                distance_out);
  } else {
    double *grid_in_real = (double *)grid_in;
    double *grid_out_real = grid_out;
    double *grid_out_imag = grid_out + (fft_size / 2 + 1) * number_of_ffts;
    reorder_input_c2r(grid_in, grid_out_real, grid_out_imag, fft_size,
                      number_of_ffts, stride_in, distance_in);
    fft_ref_1d_bw_local_c2r_naive(grid_out_real, grid_in_real, fft_size,
                                  number_of_ffts);
    reorder_output_c2r(grid_in_real, grid_out, fft_size, number_of_ffts,
                       stride_out, distance_out);
  }

#if PROFILE_CODE
  clock_t end = clock();
  double time_total = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Time Reorder 2: %f\n", time_reorder_2);
  printf("Time Reorder 3: %f\n", time_reorder_3);
  printf("Time Reorder 4: %f\n", time_reorder_4);
  printf("Time 2: %f\n", time_2);
  printf("Time 3: %f\n", time_3);
  printf("Time 4: %f\n", time_4);
  printf("Time naive: %f\n", time_naive);
  printf("Time twiddle: %f\n", time_twiddle);
  printf("Time rest: %f\n", time_total - time_reorder_2 - time_reorder_3 -
                                time_reorder_4 - time_2 - time_3 - time_4);
  printf("Total Time 1D C2R BW %i %i: %f\n", fft_size, number_of_ffts,
         time_total);
  fflush(stdout);
#endif
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_2d_fw_local_low(double complex *restrict grid_in,
                             double complex *restrict grid_out,
                             const int fft_size[2], const int number_of_ffts,
                             const int stride_in, const int stride_out,
                             const int distance_in, const int distance_out) {

#if PROFILE_CODE
  time_reorder_2 = 0.0;
  time_reorder_3 = 0.0;
  time_reorder_4 = 0.0;
  time_2 = 0.0;
  time_3 = 0.0;
  time_4 = 0.0;
  time_naive = 0.0;
  time_twiddle = 0.0;

  clock_t begin = clock();
#endif

  // We reorder the data to a format more suitable for vectorization
  double *grid_in_real = (double *)grid_in;
  double *grid_in_imag =
      ((double *)grid_in) + fft_size[0] * fft_size[1] * number_of_ffts;
  double *grid_out_real = (double *)grid_out;
  double *grid_out_imag =
      ((double *)grid_out) + fft_size[0] * fft_size[1] * number_of_ffts;

  reorder_input(grid_in, grid_out_real, grid_out_imag,
                fft_size[0] * fft_size[1], number_of_ffts, stride_in,
                distance_in);
  fft_ref_1d_fw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size[0],
                               number_of_ffts * fft_size[1]);
  // Transpose the data
  fft_ref_transpose_local_double_block(grid_in_real, grid_out_real, fft_size[1],
                                       fft_size[0], number_of_ffts);
  fft_ref_transpose_local_double_block(grid_in_imag, grid_out_imag, fft_size[1],
                                       fft_size[0], number_of_ffts);
  fft_ref_1d_fw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size[1],
                               number_of_ffts * fft_size[0]);
  reorder_output_2d(grid_in_real, grid_in_imag, grid_out, fft_size,
                    number_of_ffts, stride_out, distance_out);

#if PROFILE_CODE
  clock_t end = clock();
  double time_total = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Time Reorder 2: %f\n", time_reorder_2);
  printf("Time Reorder 3: %f\n", time_reorder_3);
  printf("Time Reorder 4: %f\n", time_reorder_4);
  printf("Time 2: %f\n", time_2);
  printf("Time 3: %f\n", time_3);
  printf("Time 4: %f\n", time_4);
  printf("Time naive: %f\n", time_naive);
  printf("Time twiddle: %f\n", time_twiddle);
  printf("Time rest: %f\n", time_total - time_reorder_2 - time_reorder_3 -
                                time_reorder_4 - time_2 - time_3 - time_4);
  printf("Total Time 2D FW %i %i %i: %f\n", fft_size[0], fft_size[1],
         number_of_ffts, time_total);
  fflush(stdout);
#endif
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_2d_fw_local_r2c_low(double *restrict grid_in,
                                 double complex *restrict grid_out,
                                 const int fft_size[2],
                                 const int number_of_ffts, const int stride_in,
                                 const int stride_out, const int distance_in,
                                 const int distance_out) {

#if PROFILE_CODE
  time_reorder_2 = 0.0;
  time_reorder_3 = 0.0;
  time_reorder_4 = 0.0;
  time_2 = 0.0;
  time_3 = 0.0;
  time_4 = 0.0;
  time_naive = 0.0;
  time_twiddle = 0.0;

  clock_t begin = clock();
#endif

  // We reorder the data to a format more suitable for vectorization
  double *grid_in_real = (double *)grid_in;
  double *grid_out_real = (double *)grid_out;
  double *grid_out_imag =
      ((double *)grid_out) + fft_size[0] * fft_size[1] * number_of_ffts;
  double *grid_in_imag =
      grid_in_real + fft_size[0] * fft_size[1] * number_of_ffts;

  // Perform the first FFT (R2C along the first dimension) and prepare for the
  // second FFT
  if (fft_size[0] % 2 == 0) {
    // Perform two FFTs of half length
    grid_in_imag =
        grid_in_real + fft_size[0] / 2 * fft_size[1] * number_of_ffts;
    grid_out_imag =
        grid_out_real + fft_size[0] / 2 * fft_size[1] * number_of_ffts;
    reorder_input_r2c_for_c2c(grid_in, grid_out_real, grid_out_imag, fft_size,
                              number_of_ffts, stride_in, distance_in);
    fft_ref_1d_fw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                                 grid_in_imag, fft_size[0] / 2,
                                 number_of_ffts * fft_size[1]);
    fft_twiddle_r2c_fw_for_c2c_2d(grid_in_real, grid_in_imag, grid_out_real,
                                  fft_size, number_of_ffts);
  } else {
    reorder_input_r2c(grid_in, grid_out_real, fft_size[0] * fft_size[1],
                      number_of_ffts, stride_in, distance_in);
    fft_ref_1d_fw_local_r2c_naive(grid_out_real, grid_in_real, fft_size[0],
                                  number_of_ffts * fft_size[1]);
    grid_in_imag =
        grid_in_real + (fft_size[0] / 2 + 1) * fft_size[1] * number_of_ffts;
    grid_out_imag =
        grid_out_real + (fft_size[0] / 2 + 1) * fft_size[1] * number_of_ffts;
    // Transpose the data
    fft_ref_transpose_local_double_block(grid_in_real, grid_out_real,
                                         fft_size[1], fft_size[0] / 2 + 1,
                                         number_of_ffts);
    fft_ref_transpose_local_double_block(grid_in_imag, grid_out_imag,
                                         fft_size[1], fft_size[0] / 2 + 1,
                                         number_of_ffts);
  }
  grid_out_imag =
      grid_out_real + (fft_size[0] / 2 + 1) * fft_size[1] * number_of_ffts;
  grid_in_imag =
      grid_in_real + (fft_size[0] / 2 + 1) * fft_size[1] * number_of_ffts;

  fft_ref_1d_fw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size[1],
                               number_of_ffts * (fft_size[0] / 2 + 1));
  reorder_output_2d(grid_in_real, grid_in_imag, grid_out,
                    (const int[2]){fft_size[0] / 2 + 1, fft_size[1]},
                    number_of_ffts, stride_out, distance_out);

#if PROFILE_CODE
  clock_t end = clock();
  double time_total = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Time Reorder 2: %f\n", time_reorder_2);
  printf("Time Reorder 3: %f\n", time_reorder_3);
  printf("Time Reorder 4: %f\n", time_reorder_4);
  printf("Time 2: %f\n", time_2);
  printf("Time 3: %f\n", time_3);
  printf("Time 4: %f\n", time_4);
  printf("Time naive: %f\n", time_naive);
  printf("Time twiddle: %f\n", time_twiddle);
  printf("Time rest: %f\n", time_total - time_reorder_2 - time_reorder_3 -
                                time_reorder_4 - time_2 - time_3 - time_4);
  printf("Total Time 2D R2C FW %i %i %i: %f\n", fft_size[0], fft_size[1],
         number_of_ffts, time_total);
  fflush(stdout);
#endif
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_2d_bw_local_low(double complex *restrict grid_in,
                             double complex *restrict grid_out,
                             const int fft_size[2], const int number_of_ffts,
                             const int stride_in, const int stride_out,
                             const int distance_in, const int distance_out) {

#if PROFILE_CODE
  time_reorder_2 = 0.0;
  time_reorder_3 = 0.0;
  time_reorder_4 = 0.0;
  time_2 = 0.0;
  time_3 = 0.0;
  time_4 = 0.0;
  time_naive = 0.0;
  time_twiddle = 0.0;

  clock_t begin = clock();
#endif

  // We reorder the data to a format more suitable for vectorization
  double *grid_in_real = (double *)grid_in;
  double *grid_in_imag =
      ((double *)grid_in) + fft_size[0] * fft_size[1] * number_of_ffts;
  double *grid_out_real = (double *)grid_out;
  double *grid_out_imag =
      ((double *)grid_out) + fft_size[0] * fft_size[1] * number_of_ffts;

  reorder_input(grid_in, grid_out_real, grid_out_imag,
                fft_size[0] * fft_size[1], number_of_ffts, stride_in,
                distance_in);
  fft_ref_1d_bw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size[0],
                               number_of_ffts * fft_size[1]);
  // Transpose the data
  fft_ref_transpose_local_double_block(grid_in_real, grid_out_real, fft_size[1],
                                       fft_size[0], number_of_ffts);
  fft_ref_transpose_local_double_block(grid_in_imag, grid_out_imag, fft_size[1],
                                       fft_size[0], number_of_ffts);
  fft_ref_1d_bw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size[1],
                               number_of_ffts * fft_size[0]);
  reorder_output_2d(grid_in_real, grid_in_imag, grid_out, fft_size,
                    number_of_ffts, stride_out, distance_out);

#if PROFILE_CODE
  clock_t end = clock();
  double time_total = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Time Reorder 2: %f\n", time_reorder_2);
  printf("Time Reorder 3: %f\n", time_reorder_3);
  printf("Time Reorder 4: %f\n", time_reorder_4);
  printf("Time 2: %f\n", time_2);
  printf("Time 3: %f\n", time_3);
  printf("Time 4: %f\n", time_4);
  printf("Time naive: %f\n", time_naive);
  printf("Time twiddle: %f\n", time_twiddle);
  printf("Time rest: %f\n", time_total - time_reorder_2 - time_reorder_3 -
                                time_reorder_4 - time_2 - time_3 - time_4);
  printf("Total Time 2D BW %i %i %i: %f\n", fft_size[0], fft_size[1],
         number_of_ffts, time_total);
  fflush(stdout);
#endif
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_2d_bw_local_c2r_low(double complex *restrict grid_in,
                                 double *restrict grid_out,
                                 const int fft_size[2],
                                 const int number_of_ffts, const int stride_in,
                                 const int stride_out, const int distance_in,
                                 const int distance_out) {

#if PROFILE_CODE
  time_reorder_2 = 0.0;
  time_reorder_3 = 0.0;
  time_reorder_4 = 0.0;
  time_2 = 0.0;
  time_3 = 0.0;
  time_4 = 0.0;
  time_naive = 0.0;
  time_twiddle = 0.0;

  clock_t begin = clock();
#endif

  // We reorder the data to a format more suitable for vectorization
  double *grid_in_real = (double *)grid_in;
  double *grid_in_imag = ((double *)grid_in) +
                         (fft_size[0] / 2 + 1) * fft_size[1] * number_of_ffts;
  double *grid_out_real = (double *)grid_out;
  double *grid_out_imag = ((double *)grid_out) +
                          (fft_size[0] / 2 + 1) * fft_size[1] * number_of_ffts;

  reorder_input_c2r_2d(grid_in, grid_out_real, grid_out_imag, fft_size,
                       number_of_ffts, stride_in, distance_in);
  fft_ref_1d_bw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size[1],
                               number_of_ffts * (fft_size[0] / 2 + 1));
  // Transpose the data
  if (fft_size[0] % 2 == 0) {
    // Reorder the data and apply the twiddle factors
    const int large_factor = fft_size[0] / 2;
    double *grid_out_imag =
        grid_out + large_factor * fft_size[1] * number_of_ffts;

    reorder_input_c2r_for_c2c_2d(grid_in_real, grid_in_imag, grid_out_real,
                                 grid_out_imag, fft_size, number_of_ffts);
    // Now, we have FFTs of half size
    grid_in_imag = grid_in_real + large_factor * fft_size[1] * number_of_ffts;
    fft_ref_1d_bw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                                 grid_in_imag, large_factor,
                                 number_of_ffts * fft_size[1]);
    reorder_output_c2r_from_c2c_2d(grid_in_real, grid_in_imag, grid_out,
                                   large_factor, fft_size[1], number_of_ffts,
                                   stride_out, distance_out);
  } else {
    fft_ref_transpose_local_double_block(grid_in_real, grid_out_real,
                                         fft_size[0] / 2 + 1, fft_size[1],
                                         number_of_ffts);
    fft_ref_transpose_local_double_block(grid_in_imag, grid_out_imag,
                                         fft_size[0] / 2 + 1, fft_size[1],
                                         number_of_ffts);
    fft_ref_1d_bw_local_c2r_naive(grid_out_real, grid_in_real, fft_size[0],
                                  number_of_ffts * fft_size[1]);
    reorder_output_c2r(grid_in_real, grid_out, fft_size[0] * fft_size[1],
                       number_of_ffts, stride_out, distance_out);
  }

#if PROFILE_CODE
  clock_t end = clock();
  double time_total = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Time Reorder 2: %f\n", time_reorder_2);
  printf("Time Reorder 3: %f\n", time_reorder_3);
  printf("Time Reorder 4: %f\n", time_reorder_4);
  printf("Time 2: %f\n", time_2);
  printf("Time 3: %f\n", time_3);
  printf("Time 4: %f\n", time_4);
  printf("Time naive: %f\n", time_naive);
  printf("Time twiddle: %f\n", time_twiddle);
  printf("Time rest: %f\n", time_total - time_reorder_2 - time_reorder_3 -
                                time_reorder_4 - time_2 - time_3 - time_4);
  printf("Total Time 2D C2R BW %i %i %i: %f\n", fft_size[0], fft_size[1],
         number_of_ffts, time_total);
  fflush(stdout);
#endif
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_3d_fw_local_low(double complex *restrict grid_in,
                             double complex *restrict grid_out,
                             const int fft_size[3]) {

#if PROFILE_CODE
  time_reorder_2 = 0.0;
  time_reorder_3 = 0.0;
  time_reorder_4 = 0.0;
  time_2 = 0.0;
  time_3 = 0.0;
  time_4 = 0.0;
  time_naive = 0.0;
  time_twiddle = 0.0;

  clock_t begin = clock();
#endif

  // We reorder the data to a format more suitable for vectorization
  double *grid_in_real = (double *)grid_in;
  double *grid_in_imag =
      ((double *)grid_in) + fft_size[0] * fft_size[1] * fft_size[2];
  double *grid_out_real = (double *)grid_out;
  double *grid_out_imag =
      ((double *)grid_out) + fft_size[0] * fft_size[1] * fft_size[2];

  reorder_input(grid_in, grid_out_real, grid_out_imag,
                fft_size[0] * fft_size[1] * fft_size[2], 1, 1,
                fft_size[0] * fft_size[1] * fft_size[2]);
  fft_ref_1d_fw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size[2],
                               fft_size[0] * fft_size[1]);
  // Transpose the data
  fft_ref_transpose_local_double_low(grid_in_real, grid_out_real,
                                     fft_size[0] * fft_size[1], fft_size[2]);
  fft_ref_transpose_local_double_low(grid_in_imag, grid_out_imag,
                                     fft_size[0] * fft_size[1], fft_size[2]);
  fft_ref_1d_fw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size[1],
                               fft_size[0] * fft_size[2]);
  // Transpose the data
  fft_ref_transpose_local_double_low(grid_in_real, grid_out_real,
                                     fft_size[0] * fft_size[2], fft_size[1]);
  fft_ref_transpose_local_double_low(grid_in_imag, grid_out_imag,
                                     fft_size[0] * fft_size[2], fft_size[1]);
  fft_ref_1d_fw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size[0],
                               fft_size[1] * fft_size[2]);
  // Transpose the data
  for (int index_0 = 0; index_0 < fft_size[0]; index_0++) {
    for (int index_1 = 0; index_1 < fft_size[1] * fft_size[2]; index_1++) {
      grid_out[index_1 * fft_size[0] + index_0] =
          CMPLX(grid_in_real[index_0 * fft_size[1] * fft_size[2] + index_1],
                grid_in_imag[index_0 * fft_size[1] * fft_size[2] + index_1]);
    }
  }

#if PROFILE_CODE
  clock_t end = clock();
  double time_total = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Time Reorder 2: %f\n", time_reorder_2);
  printf("Time Reorder 3: %f\n", time_reorder_3);
  printf("Time Reorder 4: %f\n", time_reorder_4);
  printf("Time 2: %f\n", time_2);
  printf("Time 3: %f\n", time_3);
  printf("Time 4: %f\n", time_4);
  printf("Time naive: %f\n", time_naive);
  printf("Time twiddle: %f\n", time_twiddle);
  printf("Time rest: %f\n", time_total - time_reorder_2 - time_reorder_3 -
                                time_reorder_4 - time_2 - time_3 - time_4);
  printf("Total Time 3D FW %i %i %i: %f\n", fft_size[0], fft_size[1],
         fft_size[2], time_total);
  fflush(stdout);
#endif
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_3d_fw_local_r2c_low(double *restrict grid_in,
                                 double complex *restrict grid_out,
                                 const int fft_size[3]) {

#if PROFILE_CODE
  time_reorder_2 = 0.0;
  time_reorder_3 = 0.0;
  time_reorder_4 = 0.0;
  time_2 = 0.0;
  time_3 = 0.0;
  time_4 = 0.0;
  time_naive = 0.0;
  time_twiddle = 0.0;

  clock_t begin = clock();
#endif

  // We reorder the data to a format more suitable for vectorization
  double *grid_in_real = (double *)grid_in;
  double *grid_in_imag =
      ((double *)grid_in) + fft_size[0] * fft_size[1] * fft_size[2];
  double *grid_out_real = (double *)grid_out;
  double *grid_out_imag =
      ((double *)grid_out) + fft_size[0] * fft_size[1] * fft_size[2];

  if (false && fft_size[2] % 2 == 0) {
    // Perform two FFTs of half length
    grid_in_imag = grid_in_real + fft_size[2] / 2 * fft_size[0] * fft_size[1];
    grid_out_imag = grid_out_real + fft_size[2] / 2 * fft_size[0] * fft_size[1];
    reorder_input_r2c_for_c2c(
        grid_in, grid_out_real, grid_out_imag,
        (const int[2]){fft_size[2], fft_size[0] * fft_size[1]}, 1, 1, 1);
    fft_ref_1d_fw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                                 grid_in_imag, fft_size[2] / 2,
                                 fft_size[0] * fft_size[1]);
    fft_twiddle_r2c_fw_for_c2c_2d(
        grid_in_real, grid_in_imag, grid_out_real,
        (const int[2]){fft_size[2], fft_size[0] * fft_size[1]}, 1);
    // Now, we switch to the new format
    grid_in_imag =
        grid_in_real + (fft_size[2] / 2 + 1) * fft_size[0] * fft_size[1];
    grid_out_imag =
        grid_out_real + (fft_size[2] / 2 + 1) * fft_size[0] * fft_size[1];
  } else {
    memcpy(grid_out_real, grid_in,
           fft_size[0] * fft_size[1] * fft_size[2] * sizeof(double));
    fft_ref_1d_fw_local_r2c_naive(grid_out_real, grid_in_real, fft_size[2],
                                  fft_size[0] * fft_size[1]);
    grid_in_imag =
        grid_in_real + (fft_size[2] / 2 + 1) * fft_size[0] * fft_size[1];
    grid_out_imag =
        grid_out_real + (fft_size[2] / 2 + 1) * fft_size[0] * fft_size[1];
    // Transpose the data
    fft_ref_transpose_local_double_low(grid_in_real, grid_out_real,
                                       fft_size[0] * fft_size[1],
                                       fft_size[2] / 2 + 1);
    fft_ref_transpose_local_double_low(grid_in_imag, grid_out_imag,
                                       fft_size[0] * fft_size[1],
                                       fft_size[2] / 2 + 1);
  }
  fft_ref_1d_fw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size[1],
                               fft_size[0] * (fft_size[2] / 2 + 1));
  // Transpose the data
  fft_ref_transpose_local_double_low(grid_in_real, grid_out_real,
                                     fft_size[0] * (fft_size[2] / 2 + 1),
                                     fft_size[1]);
  fft_ref_transpose_local_double_low(grid_in_imag, grid_out_imag,
                                     fft_size[0] * (fft_size[2] / 2 + 1),
                                     fft_size[1]);
  fft_ref_1d_fw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size[0],
                               fft_size[1] * (fft_size[2] / 2 + 1));
  // Transpose the data
  for (int index_0 = 0; index_0 < fft_size[0]; index_0++) {
    for (int index_1 = 0; index_1 < fft_size[1] * (fft_size[2] / 2 + 1);
         index_1++) {
      grid_out[index_1 * fft_size[0] + index_0] = CMPLX(
          grid_in_real[index_0 * fft_size[1] * (fft_size[2] / 2 + 1) + index_1],
          grid_in_imag[index_0 * fft_size[1] * (fft_size[2] / 2 + 1) +
                       index_1]);
    }
  }

#if PROFILE_CODE
  clock_t end = clock();
  double time_total = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Time Reorder 2: %f\n", time_reorder_2);
  printf("Time Reorder 3: %f\n", time_reorder_3);
  printf("Time Reorder 4: %f\n", time_reorder_4);
  printf("Time 2: %f\n", time_2);
  printf("Time 3: %f\n", time_3);
  printf("Time 4: %f\n", time_4);
  printf("Time naive: %f\n", time_naive);
  printf("Time twiddle: %f\n", time_twiddle);
  printf("Time rest: %f\n", time_total - time_reorder_2 - time_reorder_3 -
                                time_reorder_4 - time_2 - time_3 - time_4);
  printf("Total Time 3D R2C FW %i %i %i: %f\n", fft_size[0], fft_size[1],
         fft_size[2], time_total);
  fflush(stdout);
#endif
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_3d_bw_local_low(double complex *restrict grid_in,
                             double complex *restrict grid_out,
                             const int fft_size[3]) {

#if PROFILE_CODE
  time_reorder_2 = 0.0;
  time_reorder_3 = 0.0;
  time_reorder_4 = 0.0;
  time_2 = 0.0;
  time_3 = 0.0;
  time_4 = 0.0;
  time_naive = 0.0;
  time_twiddle = 0.0;

  clock_t begin = clock();
#endif

  // We reorder the data to a format more suitable for vectorization
  double *grid_in_real = (double *)grid_in;
  double *grid_in_imag =
      ((double *)grid_in) + fft_size[0] * fft_size[1] * fft_size[2];
  double *grid_out_real = (double *)grid_out;
  double *grid_out_imag =
      ((double *)grid_out) + fft_size[0] * fft_size[1] * fft_size[2];

  reorder_input(grid_in, grid_out_real, grid_out_imag,
                fft_size[0] * fft_size[1] * fft_size[2], 1, 1, 1);
  fft_ref_1d_bw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size[2],
                               fft_size[0] * fft_size[1]);
  // Transpose the data
  fft_ref_transpose_local_double_low(grid_in_real, grid_out_real,
                                     fft_size[0] * fft_size[1], fft_size[2]);
  fft_ref_transpose_local_double_low(grid_in_imag, grid_out_imag,
                                     fft_size[0] * fft_size[1], fft_size[2]);
  fft_ref_1d_bw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size[1],
                               fft_size[0] * fft_size[2]);
  // Transpose the data
  fft_ref_transpose_local_double_low(grid_in_real, grid_out_real,
                                     fft_size[0] * fft_size[2], fft_size[1]);
  fft_ref_transpose_local_double_low(grid_in_imag, grid_out_imag,
                                     fft_size[0] * fft_size[2], fft_size[1]);
  fft_ref_1d_bw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size[0],
                               fft_size[1] * fft_size[2]);
  // Transpose the data
  for (int index_0 = 0; index_0 < fft_size[0]; index_0++) {
    for (int index_1 = 0; index_1 < fft_size[1] * fft_size[2]; index_1++) {
      grid_out[index_1 * fft_size[0] + index_0] =
          CMPLX(grid_in_real[index_0 * fft_size[1] * fft_size[2] + index_1],
                grid_in_imag[index_0 * fft_size[1] * fft_size[2] + index_1]);
    }
  }

#if PROFILE_CODE
  clock_t end = clock();
  double time_total = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Time Reorder 2: %f\n", time_reorder_2);
  printf("Time Reorder 3: %f\n", time_reorder_3);
  printf("Time Reorder 4: %f\n", time_reorder_4);
  printf("Time 2: %f\n", time_2);
  printf("Time 3: %f\n", time_3);
  printf("Time 4: %f\n", time_4);
  printf("Time naive: %f\n", time_naive);
  printf("Time twiddle: %f\n", time_twiddle);
  printf("Time rest: %f\n", time_total - time_reorder_2 - time_reorder_3 -
                                time_reorder_4 - time_2 - time_3 - time_4);
  printf("Total Time 3D BW %i %i %i: %f\n", fft_size[0], fft_size[1],
         fft_size[2], time_total);
  fflush(stdout);
#endif
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_3d_bw_local_c2r_low(double complex *restrict grid_in,
                                 double *restrict grid_out,
                                 const int fft_size[3]) {

#if PROFILE_CODE
  time_reorder_2 = 0.0;
  time_reorder_3 = 0.0;
  time_reorder_4 = 0.0;
  time_2 = 0.0;
  time_3 = 0.0;
  time_4 = 0.0;
  time_naive = 0.0;
  time_twiddle = 0.0;

  clock_t begin = clock();
#endif

  // We reorder the data to a format more suitable for vectorization
  double *grid_in_real = (double *)grid_in;
  double *grid_in_imag =
      ((double *)grid_in) + fft_size[0] * fft_size[1] * (fft_size[2] / 2 + 1);
  double *grid_out_real = (double *)grid_out;
  double *grid_out_imag =
      ((double *)grid_out) + fft_size[0] * fft_size[1] * (fft_size[2] / 2 + 1);

  reorder_input(grid_in, grid_out_real, grid_out_imag, fft_size[0],
                fft_size[1] * (fft_size[2] / 2 + 1), 1, fft_size[0]);
  fft_ref_1d_bw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size[0],
                               fft_size[1] * (fft_size[2] / 2 + 1));
  // Transpose the data
  fft_ref_transpose_local_double_low(grid_in_real, grid_out_real, fft_size[1],
                                     fft_size[0] * (fft_size[2] / 2 + 1));
  fft_ref_transpose_local_double_low(grid_in_imag, grid_out_imag, fft_size[1],
                                     fft_size[0] * (fft_size[2] / 2 + 1));
  fft_ref_1d_bw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size[1],
                               fft_size[0] * (fft_size[2] / 2 + 1));
  // Transpose the data
  if (fft_size[2] % 2 == 0) {
    // Reorder the data and apply the twiddle factors
    const int large_factor = fft_size[2] / 2;
    double *grid_out_imag = grid_out + large_factor * fft_size[0] * fft_size[1];

    reorder_input_c2r_for_c2c_2d(
        grid_in_real, grid_in_imag, grid_out_real, grid_out_imag,
        (const int[2]){fft_size[2], fft_size[0] * fft_size[1]}, 1);
    // Now, we have FFTs of half size
    grid_in_imag = grid_in_real + large_factor * fft_size[0] * fft_size[1];
    fft_ref_1d_bw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                                 grid_in_imag, large_factor,
                                 fft_size[0] * fft_size[1]);
    reorder_output_c2r_from_c2c_2d(grid_in_real, grid_in_imag, grid_out,
                                   large_factor, fft_size[0] * fft_size[1], 1,
                                   1, 1);
  } else {
    fft_ref_transpose_local_double_low(grid_in_real, grid_out_real,
                                       fft_size[2] / 2 + 1,
                                       fft_size[0] * fft_size[1]);
    fft_ref_transpose_local_double_low(grid_in_imag, grid_out_imag,
                                       fft_size[2] / 2 + 1,
                                       fft_size[0] * fft_size[1]);
    fft_ref_1d_bw_local_c2r_naive(grid_out_real, grid_in_real, fft_size[2],
                                  fft_size[0] * fft_size[1]);
    memcpy(grid_out, grid_in_real,
           fft_size[0] * fft_size[1] * fft_size[2] * sizeof(double));
  }

#if PROFILE_CODE
  clock_t end = clock();
  double time_total = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("Time Reorder 2: %f\n", time_reorder_2);
  printf("Time Reorder 3: %f\n", time_reorder_3);
  printf("Time Reorder 4: %f\n", time_reorder_4);
  printf("Time 2: %f\n", time_2);
  printf("Time 3: %f\n", time_3);
  printf("Time 4: %f\n", time_4);
  printf("Time naive: %f\n", time_naive);
  printf("Time twiddle: %f\n", time_twiddle);
  printf("Time rest: %f\n", time_total - time_reorder_2 - time_reorder_3 -
                                time_reorder_4 - time_2 - time_3 - time_4);
  printf("Total Time 3D C2R BW %i %i %i: %f\n", fft_size[0], fft_size[1],
         fft_size[2], time_total);
  fflush(stdout);
#endif
}

// EOF
