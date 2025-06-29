/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "../mpiwrap/mp_mpi.h"
#include "fft_lib_ref.h"
#include "fft_utils.h"

#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void determine_factorization(const int n, int *m1, int *m2) {
  *m1 = 1;
  for (int candidate = 2; candidate * candidate <= n; candidate++) {
    if (n % candidate == 0) {
      *m1 = candidate;
      break;
    }
  }
  *m2 = n / *m1;
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_fw_local_naive(double complex *restrict grid_in,
                               double complex *restrict grid_out,
                               const int fft_size, const int number_of_ffts,
                               const int stride_in, const int stride_out,
                               const int distance_in, const int distance_out) {

  const double pi = acos(-1);

#pragma omp parallel for default(none)                                         \
    shared(number_of_ffts, grid_in, grid_out, fft_size, stride_in, stride_out, \
               distance_in, distance_out, pi)                                  \
    collapse(2) if (!omp_in_parallel())
  for (int fft = 0; fft < number_of_ffts; fft++) {
    for (int index_out = 0; index_out < fft_size; index_out++) {
      grid_out[index_out * stride_out + fft * distance_out] = 0.0;
      for (int index_in = 0; index_in < fft_size; index_in++) {
        grid_out[index_out * stride_out + fft * distance_out] +=
            cexp(-2.0 * pi * I * index_in * index_out / fft_size) *
            grid_in[index_in * stride_in + fft * distance_in];
      }
    }
  }
}

/*******************************************************************************
 * \brief Naive implementation of backwards FFT to transposed format (for
 *easier transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_bw_local_naive(double complex *restrict grid_in,
                               double complex *restrict grid_out,
                               const int fft_size, const int number_of_ffts,
                               const int stride_in, const int stride_out,
                               const int distance_in, const int distance_out) {

  const double pi = acos(-1);

#pragma omp parallel for default(none)                                         \
    shared(number_of_ffts, grid_in, grid_out, fft_size, stride_in, stride_out, \
               distance_in, distance_out, pi)                                  \
    collapse(2) if (!omp_in_parallel())
  for (int fft = 0; fft < number_of_ffts; fft++) {
    for (int index_out = 0; index_out < fft_size; index_out++) {
      grid_out[index_out * stride_out + fft * distance_out] = 0.0;
      for (int index_in = 0; index_in < fft_size; index_in++) {
        grid_out[index_out * stride_out + fft * distance_out] +=
            cexp(2.0 * pi * I * index_in * index_out / fft_size) *
            grid_in[index_in * stride_in + fft * distance_in];
      }
    }
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

  int small_divisor, large_divisor;
  determine_factorization(fft_size, &small_divisor, &large_divisor);

  if (small_divisor == 1) {
    fft_ref_1d_fw_local_naive(grid_in, grid_out, fft_size, number_of_ffts,
                              stride_in, stride_out, distance_in, distance_out);
  } else {
    const double pi = acos(-1);

#pragma omp parallel for default(none)                                         \
    shared(number_of_ffts, small_divisor, large_divisor, stride_in,            \
               stride_out, distance_in, distance_out, fft_size, grid_in,       \
               grid_out) if (number_of_ffts >= omp_get_max_threads() &&        \
                                 !omp_in_parallel())
    for (int fft = 0; fft < number_of_ffts; fft++) {
      fft_ref_1d_fw_local_low(
          grid_in + fft * distance_in, grid_out + fft * distance_out,
          small_divisor, large_divisor, stride_in * large_divisor, stride_out,
          stride_in, stride_out * small_divisor);
      for (int index_small = 0; index_small < small_divisor; index_small++) {
        for (int index_large = 0; index_large < large_divisor; index_large++) {
          grid_in[(index_large * small_divisor + index_small) * stride_in +
                  fft * distance_in] =
              cexp(-I * (2.0 * pi * index_small * index_large / fft_size)) *
              grid_out[(index_large * small_divisor + index_small) *
                           stride_out +
                       fft * distance_out];
        }
      }
      fft_ref_1d_fw_local_low(
          grid_in + fft * distance_in, grid_out + fft * distance_out,
          large_divisor, small_divisor, stride_in * small_divisor,
          stride_out * small_divisor, stride_in, stride_out);
    }
  }
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

  const double pi = acos(-1);

#pragma omp parallel for default(none)                                         \
    shared(number_of_ffts, grid_in, grid_out, fft_size, stride_in, stride_out, \
               distance_in, distance_out, pi)                                  \
    collapse(2) if (!omp_in_parallel())
  for (int fft = 0; fft < number_of_ffts; fft++) {
    for (int index_out = 0; index_out < fft_size / 2 + 1; index_out++) {
      grid_out[index_out * stride_out + fft * distance_out] = 0.0;
      for (int index_in = 0; index_in < fft_size; index_in++) {
        grid_out[index_out * stride_out + fft * distance_out] +=
            cexp(-2.0 * pi * I * index_in * index_out / fft_size) *
            grid_in[index_in * stride_in + fft * distance_in];
      }
    }
  }
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

  int small_divisor, large_divisor;
  determine_factorization(fft_size, &small_divisor, &large_divisor);

  if (small_divisor == 1) {
    fft_ref_1d_bw_local_naive(grid_in, grid_out, fft_size, number_of_ffts,
                              stride_in, stride_out, distance_in, distance_out);
  } else {
    const double pi = acos(-1);

#pragma omp parallel for default(none)                                         \
    shared(number_of_ffts, small_divisor, large_divisor, stride_in,            \
               stride_out, distance_in, distance_out, fft_size, grid_in,       \
               grid_out) if (number_of_ffts >= omp_get_max_threads() &&        \
                                 !omp_in_parallel())
    for (int fft = 0; fft < number_of_ffts; fft++) {
      fft_ref_1d_bw_local_low(
          grid_in + fft * distance_in, grid_out + fft * distance_out,
          small_divisor, large_divisor, stride_in * large_divisor, stride_out,
          stride_in, stride_out * small_divisor);
      for (int index_small = 0; index_small < small_divisor; index_small++) {
        for (int index_large = 0; index_large < large_divisor; index_large++) {
          grid_in[(index_large * small_divisor + index_small) * stride_in +
                  fft * distance_in] =
              cexp(I * (2.0 * pi * index_small * index_large / fft_size)) *
              grid_out[(index_large * small_divisor + index_small) *
                           stride_out +
                       fft * distance_out];
        }
      }
      fft_ref_1d_bw_local_low(
          grid_in + fft * distance_in, grid_out + fft * distance_out,
          large_divisor, small_divisor, stride_in * small_divisor,
          stride_out * small_divisor, stride_in, stride_out);
    }
  }
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

  const double pi = acos(-1);

#pragma omp parallel for default(none)                                         \
    shared(number_of_ffts, grid_in, grid_out, fft_size, stride_in, stride_out, \
               distance_in, distance_out, pi)                                  \
    collapse(2) if (!omp_in_parallel())
  for (int fft = 0; fft < number_of_ffts; fft++) {
    for (int index_out = 0; index_out < fft_size; index_out++) {
      grid_out[index_out * stride_out + fft * distance_out] = 0.0;
      for (int index_in = 0; index_in < fft_size / 2 + 1; index_in++) {
        grid_out[index_out * stride_out + fft * distance_out] +=
            creal(cexp(2.0 * pi * I * index_in * index_out / fft_size) *
                  grid_in[index_in * stride_in + fft * distance_in]);
      }
      for (int index_in = fft_size / 2 + 1; index_in < fft_size; index_in++) {
        grid_out[index_out * stride_out + fft * distance_out] +=
            creal(cexp(2.0 * pi * I * index_in * index_out / fft_size) *
                  conj(grid_in[(fft_size - index_in) * stride_in +
                               fft * distance_in]));
      }
    }
  }
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

  double complex *buffer = malloc(fft_size[0] * fft_size[1] * number_of_ffts *
                                  sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(grid_in, grid_out, fft_size, number_of_ffts, stride_in, stride_out, \
               distance_in, distance_out,                                      \
               buffer) if (number_of_ffts > omp_get_max_threads())
  for (int fft = 0; fft < number_of_ffts; fft++) {
    fft_ref_1d_fw_local_low(grid_in + distance_in * fft, buffer + fft,
                            fft_size[0], fft_size[1], fft_size[1] * stride_in,
                            fft_size[1] * number_of_ffts, stride_in,
                            number_of_ffts);
  }
#pragma omp parallel for default(none)                                         \
    shared(grid_in, grid_out, fft_size, number_of_ffts, stride_in, stride_out, \
               distance_in, distance_out,                                      \
               buffer) if (number_of_ffts > omp_get_max_threads())
  for (int fft = 0; fft < number_of_ffts; fft++) {
    fft_ref_1d_fw_local_low(buffer + fft, grid_out + distance_out * fft,
                            fft_size[1], fft_size[0], number_of_ffts,
                            stride_out, number_of_ffts * fft_size[1],
                            stride_out * fft_size[1]);
  }
  free(buffer);
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

  double complex *buffer = malloc(fft_size[0] * (fft_size[1] / 2 + 1) *
                                  number_of_ffts * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(grid_in, grid_out, fft_size, number_of_ffts, stride_in, stride_out, \
               distance_in, distance_out,                                      \
               buffer) if (number_of_ffts > omp_get_max_threads())
  for (int fft = 0; fft < number_of_ffts; fft++) {
    fft_ref_1d_fw_local_r2c_low(grid_in + distance_in * fft, buffer + fft,
                                fft_size[1], fft_size[0], stride_in,
                                number_of_ffts, stride_in * fft_size[1],
                                number_of_ffts * (fft_size[1] / 2 + 1));
  }
#pragma omp parallel for default(none)                                         \
    shared(grid_in, grid_out, fft_size, number_of_ffts, stride_in, stride_out, \
               distance_in, distance_out,                                      \
               buffer) if (number_of_ffts > omp_get_max_threads())
  for (int fft = 0; fft < number_of_ffts; fft++) {
    fft_ref_1d_fw_local_low(
        buffer + fft, grid_out + distance_out * fft, fft_size[0],
        fft_size[1] / 2 + 1, number_of_ffts * (fft_size[1] / 2 + 1),
        stride_out * (fft_size[1] / 2 + 1), number_of_ffts, stride_out);
  }
  free(buffer);
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

  double complex *buffer = malloc(fft_size[0] * fft_size[1] * number_of_ffts *
                                  sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(grid_in, grid_out, fft_size, number_of_ffts, stride_in, stride_out, \
               distance_in, distance_out,                                      \
               buffer) if (number_of_ffts > omp_get_max_threads())
  for (int fft = 0; fft < number_of_ffts; fft++) {
    fft_ref_1d_bw_local_low(grid_in + distance_in * fft, buffer + fft,
                            fft_size[0], fft_size[1], fft_size[1] * stride_in,
                            fft_size[1] * number_of_ffts, stride_in,
                            number_of_ffts);
  }
#pragma omp parallel for default(none)                                         \
    shared(grid_in, grid_out, fft_size, number_of_ffts, stride_in, stride_out, \
               distance_in, distance_out,                                      \
               buffer) if (number_of_ffts > omp_get_max_threads())
  for (int fft = 0; fft < number_of_ffts; fft++) {
    fft_ref_1d_bw_local_low(buffer + fft, grid_out + distance_out * fft,
                            fft_size[1], fft_size[0], number_of_ffts,
                            stride_out, number_of_ffts * fft_size[1],
                            stride_out * fft_size[1]);
  }
  free(buffer);
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

  double complex *buffer = malloc(fft_size[0] * (fft_size[1] / 2 + 1) *
                                  number_of_ffts * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(grid_in, grid_out, fft_size, number_of_ffts, stride_in, stride_out, \
               distance_in, distance_out,                                      \
               buffer) if (number_of_ffts > omp_get_max_threads())
  for (int fft = 0; fft < number_of_ffts; fft++) {
    fft_ref_1d_bw_local_low(
        grid_in + distance_in * fft, buffer + fft, fft_size[0],
        (fft_size[1] / 2 + 1), (fft_size[1] / 2 + 1) * stride_in,
        (fft_size[1] / 2 + 1) * number_of_ffts, stride_in, number_of_ffts);
  }
#pragma omp parallel for default(none)                                         \
    shared(grid_in, grid_out, fft_size, number_of_ffts, stride_in, stride_out, \
               distance_in, distance_out,                                      \
               buffer) if (number_of_ffts > omp_get_max_threads())
  for (int fft = 0; fft < number_of_ffts; fft++) {
    fft_ref_1d_bw_local_c2r_low(
        buffer + fft, grid_out + distance_out * fft, fft_size[1], fft_size[0],
        number_of_ffts, stride_out, number_of_ffts * (fft_size[1] / 2 + 1),
        stride_out * fft_size[1]);
  }
  free(buffer);
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_3d_fw_local_low(double complex *restrict grid_in,
                             double complex *restrict grid_out,
                             const int fft_size[3]) {

  fft_ref_1d_fw_local_low(grid_in, grid_out, fft_size[2],
                          fft_size[0] * fft_size[1], 1,
                          fft_size[0] * fft_size[1], fft_size[2], 1);
  fft_ref_1d_fw_local_low(grid_out, grid_in, fft_size[1],
                          fft_size[0] * fft_size[2], 1,
                          fft_size[0] * fft_size[2], fft_size[1], 1);
  fft_ref_1d_fw_local_low(grid_in, grid_out, fft_size[0],
                          fft_size[1] * fft_size[2], 1,
                          fft_size[1] * fft_size[2], fft_size[0], 1);
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_3d_fw_local_r2c_low(double *restrict grid_in,
                                 double complex *restrict grid_out,
                                 const int fft_size[3]) {

  fft_ref_1d_fw_local_r2c_low(grid_in, grid_out, fft_size[2],
                              fft_size[0] * fft_size[1], 1,
                              fft_size[0] * fft_size[1], fft_size[2], 1);
  fft_ref_1d_fw_local_low(grid_out, (double complex *)grid_in, fft_size[1],
                          fft_size[0] * (fft_size[2] / 2 + 1), 1,
                          fft_size[0] * (fft_size[2] / 2 + 1), fft_size[1], 1);
  fft_ref_1d_fw_local_low((double complex *)grid_in, grid_out, fft_size[0],
                          fft_size[1] * (fft_size[2] / 2 + 1), 1,
                          fft_size[1] * (fft_size[2] / 2 + 1), fft_size[0], 1);
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_3d_bw_local_low(double complex *restrict grid_in,
                             double complex *restrict grid_out,
                             const int fft_size[3]) {

  fft_ref_1d_bw_local_low(grid_in, grid_out, fft_size[0],
                          fft_size[1] * fft_size[2], fft_size[1] * fft_size[2],
                          1, 1, fft_size[0]);
  fft_ref_1d_bw_local_low(grid_out, grid_in, fft_size[1],
                          fft_size[0] * fft_size[2], fft_size[0] * fft_size[2],
                          1, 1, fft_size[1]);
  fft_ref_1d_bw_local_low(grid_in, grid_out, fft_size[2],
                          fft_size[0] * fft_size[1], fft_size[0] * fft_size[1],
                          1, 1, fft_size[2]);
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_3d_bw_local_c2r_low(double complex *restrict grid_in,
                                 double *restrict grid_out,
                                 const int fft_size[3]) {

  fft_ref_1d_bw_local_low(grid_in, (double complex *)grid_out, fft_size[0],
                          fft_size[1] * (fft_size[2] / 2 + 1),
                          fft_size[1] * (fft_size[2] / 2 + 1), 1, 1,
                          fft_size[0]);
  fft_ref_1d_bw_local_low((double complex *)grid_out, grid_in, fft_size[1],
                          fft_size[0] * (fft_size[2] / 2 + 1),
                          fft_size[0] * (fft_size[2] / 2 + 1), 1, 1,
                          fft_size[1]);
  fft_ref_1d_bw_local_c2r_low(grid_in, grid_out, fft_size[2],
                              fft_size[0] * fft_size[1],
                              fft_size[0] * fft_size[1], 1, 1, fft_size[2]);
}

// EOF
