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

  // Determine a factorization of the FFT size
  int small_factor = 1;
  for (int i = 2; i * i <= fft_size; i++) {
    if (fft_size % i == 0) {
      small_factor = i;
      break;
    }
  }
  const int large_factor = fft_size / small_factor;

  const double pi = acos(-1.0);
#pragma omp parallel for default(none) collapse(2) shared(                     \
        grid_in, grid_out, fft_size, number_of_ffts, pi, stride_in,            \
            stride_out, distance_in, distance_out, small_factor, large_factor)
  for (int fft = 0; fft < number_of_ffts; fft++) {
    for (int index_out_small = 0; index_out_small < small_factor;
         index_out_small++) {
      for (int index_out_large = 0; index_out_large < large_factor;
           index_out_large++) {
        double complex tmp = 0.0;
        for (int index_in_large = 0; index_in_large < large_factor;
             index_in_large++) {
          for (int index_in_small = 0; index_in_small < small_factor;
               index_in_small++) {
            tmp += grid_in[(index_in_small * large_factor + index_in_large) *
                               stride_in +
                           fft * distance_in] *
                   cexp(-2.0 * I * pi *
                        (index_out_large * small_factor + index_out_small) *
                        (index_in_small * large_factor + index_in_large) /
                        fft_size);
          }
        }
        grid_out[fft * distance_out +
                 (index_out_large * small_factor + index_out_small) *
                     stride_out] = tmp;
      }
    }
  }
}

/*******************************************************************************
 * \brief Naive implementation of backwards FFT to transposed format (for
 *easier transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_bw_local_low(const double complex *grid_in,
                             double complex *grid_out, const int fft_size,
                             const int number_of_ffts, const int stride_in,
                             const int stride_out, const int distance_in,
                             const int distance_out) {

  // Determine a factorization of the FFT size
  int small_factor = 1;
  for (int i = 2; i * i <= fft_size; i++) {
    if (fft_size % i == 0) {
      small_factor = i;
      break;
    }
  }
  const int large_factor = fft_size / small_factor;

  const double pi = acos(-1.0);
#pragma omp parallel for default(none) collapse(2) shared(                     \
        grid_in, grid_out, fft_size, number_of_ffts, pi, stride_in,            \
            stride_out, distance_in, distance_out, small_factor, large_factor)
  for (int fft = 0; fft < number_of_ffts; fft++) {
    for (int index_out_small = 0; index_out_small < small_factor;
         index_out_small++) {
      for (int index_out_large = 0; index_out_large < large_factor;
           index_out_large++) {
        double complex tmp = 0.0;
        for (int index_in_large = 0; index_in_large < large_factor;
             index_in_large++) {
          for (int index_in_small = 0; index_in_small < small_factor;
               index_in_small++) {
            tmp += grid_in[(index_in_small * large_factor + index_in_large) *
                               stride_in +
                           fft * distance_in] *
                   cexp(2.0 * I * pi *
                        (index_out_large * small_factor + index_out_small) *
                        (index_in_small * large_factor + index_in_large) /
                        fft_size);
          }
        }
        grid_out[fft * distance_out +
                 (index_out_large * small_factor + index_out_small) *
                     stride_out] = tmp;
      }
    }
  }
}

// EOF
