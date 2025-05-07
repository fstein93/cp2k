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
#include <stdlib.h>
#include <string.h>

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_fw_local_low(const double complex *grid_in,
                             double complex *grid_out, const int fft_size,
                             const int number_of_ffts, const int stride_in,
                             const int stride_out, const int distance_in,
                             const int distance_out) {
  const double pi = acos(-1.0);
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in, grid_out, fft_size, number_of_ffts, pi, stride_in,         \
               stride_out, distance_in, distance_out)
  for (int fft = 0; fft < number_of_ffts; fft++) {
    for (int index_out = 0; index_out < fft_size; index_out++) {
      double complex tmp = 0.0;
      for (int index_in = 0; index_in < fft_size; index_in++) {
        tmp += grid_in[index_in * stride_in + fft * distance_in] *
               cexp(-2.0 * I * pi * index_out * index_in / fft_size);
      }
      grid_out[fft * distance_out + index_out * stride_out] = tmp;
    }
  }
}

/*******************************************************************************
 * \brief Naive implementation of backwards FFT to transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_bw_local_low(const double complex *grid_in,
                             double complex *grid_out, const int fft_size,
                             const int number_of_ffts, const int stride_in,
                             const int stride_out, const int distance_in,
                             const int distance_out) {
  const double pi = acos(-1.0);
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in, grid_out, fft_size, number_of_ffts, pi, stride_in,         \
               stride_out, distance_in, distance_out)
  for (int fft = 0; fft < number_of_ffts; fft++) {
    for (int index_out = 0; index_out < fft_size; index_out++) {
      double complex tmp = 0.0;
      for (int index_in = 0; index_in < fft_size; index_in++) {
        tmp += grid_in[index_in * stride_in + fft * distance_in] *
               cexp(2.0 * I * pi * index_out * index_in / fft_size);
      }
      grid_out[fft * distance_out + index_out * stride_out] = tmp;
    }
  }
}

// EOF
