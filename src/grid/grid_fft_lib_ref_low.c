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

// We need these definitions for the recursion within this module
void fft_ref_1d_fw_local_internal(double complex *grid_in,
                                  double complex *grid_out, const int fft_size,
                                  const int number_of_ffts);
void fft_ref_1d_bw_local_internal(double complex *grid_in,
                                  double complex *grid_out, const int fft_size,
                                  const int number_of_ffts);

void reorder_input(double complex *grid_in, double complex *grid_out,
                   const int fft_size, const int number_of_ffts,
                   const int stride_in, const int distance_in) {
  if (distance_in == 1 && stride_in == number_of_ffts) {
    memcpy(grid_out, grid_in,
           fft_size * number_of_ffts * sizeof(double complex));
  } else {
    // If the distance is 1, we can use a simple copy
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

void reorder_output(double complex *grid_in, double complex *grid_out,
                    const int fft_size, const int number_of_ffts,
                    const int stride_out, const int distance_out) {
  if (distance_out == 1 && stride_out == number_of_ffts) {
    memcpy(grid_out, grid_in,
           fft_size * number_of_ffts * sizeof(double complex));
  } else {
#pragma omp parallel for default(none) collapse(2) shared(                     \
        grid_in, grid_out, number_of_ffts, stride_out, distance_out, fft_size)
    for (int index = 0; index < fft_size; index++) {
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out[fft * distance_out + index * stride_out] =
            grid_in[fft + index * number_of_ffts];
      }
    }
  }
}

/*******************************************************************************
 * \brief Naive implementation of FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_fw_local_naive(double complex *grid_in,
                               double complex *grid_out, const int fft_size,
                               const int number_of_ffts) {
  memset(grid_out, 0, fft_size * number_of_ffts * sizeof(double complex));
  // Perform FFTs along the first dimension
  const double pi = acos(-1.0);
#pragma omp parallel for default(none)                                         \
    shared(grid_in, grid_out, fft_size, number_of_ffts, pi)
  for (int index_out = 0; index_out < fft_size; index_out++) {
    for (int index_in = 0; index_in < fft_size; index_in++) {
      const double complex phase_factor =
          cexp(-2.0 * I * pi * index_out * index_in / fft_size);
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out[fft + index_out * number_of_ffts] +=
            grid_in[fft + index_in * number_of_ffts] * phase_factor;
      }
    }
  }
}

/*******************************************************************************
 * \brief Naive implementation of FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_bw_local_naive(const double complex *grid_in,
                               double complex *grid_out, const int fft_size,
                               const int number_of_ffts) {
  memset(grid_out, 0, fft_size * number_of_ffts * sizeof(double complex));
  const double pi = acos(-1.0);
#pragma omp parallel for default(none)                                         \
    shared(grid_in, grid_out, fft_size, number_of_ffts, pi)
  for (int index_out = 0; index_out < fft_size; index_out++) {
    for (int index_in = 0; index_in < fft_size; index_in++) {
      const double complex phase_factor =
          cexp(2.0 * I * pi * index_out * index_in / fft_size);
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out[fft + index_out * number_of_ffts] +=
            grid_in[fft + index_in * number_of_ffts] * phase_factor;
      }
    }
  }
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_fw_local_internal(double complex *grid_in,
                                  double complex *grid_out, const int fft_size,
                                  const int number_of_ffts) {

  // Determine a factorization of the FFT size
  int small_factor = 1;
  for (int i = 2; i * i <= fft_size; i++) {
    if (fft_size % i == 0) {
      small_factor = i;
      break;
    }
  }
  const int large_factor = fft_size / small_factor;
  assert(small_factor * large_factor == fft_size);
  assert(small_factor <= large_factor);

  const double pi = acos(-1.0);
  if (small_factor == 1 || large_factor == 1) {
    // If the FFT size is prime, we can use the naive implementation
    fft_ref_1d_fw_local_naive(grid_in, grid_out, fft_size, number_of_ffts);
  } else {
    // Perform FFTs along the shorter sub-dimension
    fft_ref_1d_fw_local_naive(grid_in, grid_out, small_factor,
                              number_of_ffts * large_factor);
// Transpose and multiply with twiddle factors
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in, grid_out, fft_size, number_of_ffts, pi, small_factor,      \
               large_factor)
    for (int index_small = 0; index_small < small_factor; index_small++) {
      for (int index_large = 0; index_large < large_factor; index_large++) {
        const double complex phase_factor =
            cexp(-2.0 * I * pi * index_small * index_large / fft_size);
        for (int fft = 0; fft < number_of_ffts; fft++) {
          grid_in[fft +
                  (index_large * small_factor + index_small) * number_of_ffts] =
              grid_out[fft + (index_small * large_factor + index_large) *
                                 number_of_ffts] *
              phase_factor;
        }
      }
    }
    fft_ref_1d_fw_local_internal(grid_in, grid_out, large_factor,
                                 number_of_ffts * small_factor);
  }
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_fw_local_low(double complex *grid_in, double complex *grid_out,
                             const int fft_size, const int number_of_ffts,
                             const int stride_in, const int stride_out,
                             const int distance_in, const int distance_out) {

  // Determine a factorization of the FFT size
  int small_factor = 1;
  for (int i = 2; i * i <= fft_size; i++) {
    if (fft_size % i == 0) {
      small_factor = i;
      break;
    }
  }
  const int large_factor = fft_size / small_factor;
  assert(small_factor * large_factor == fft_size);
  assert(small_factor <= large_factor);

  const double pi = acos(-1.0);
  reorder_input(grid_in, grid_out, fft_size, number_of_ffts, stride_in,
                distance_in);
  if (small_factor == 1 || large_factor == 1) {
    // If the FFT size is prime, we can use the naive implementation
    fft_ref_1d_fw_local_naive(grid_out, grid_in, fft_size, number_of_ffts);
  } else {
    // Perform FFTs along the shorter sub-dimension
    fft_ref_1d_fw_local_naive(grid_out, grid_in, small_factor,
                              number_of_ffts * large_factor);
// Transpose and multiply with twiddle factors
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in, grid_out, fft_size, number_of_ffts, pi, small_factor,      \
               large_factor)
    for (int index_small = 0; index_small < small_factor; index_small++) {
      for (int index_large = 0; index_large < large_factor; index_large++) {
        const double complex phase_factor =
            cexp(-2.0 * I * pi * index_small * index_large / fft_size);
        for (int fft = 0; fft < number_of_ffts; fft++) {
          grid_out[fft + (index_large * small_factor + index_small) *
                             number_of_ffts] =
              grid_in[fft + (index_small * large_factor + index_large) *
                                number_of_ffts] *
              phase_factor;
        }
      }
    }
    fft_ref_1d_fw_local_internal(grid_out, grid_in, large_factor,
                                 number_of_ffts * small_factor);
  }
  reorder_output(grid_in, grid_out, fft_size, number_of_ffts, stride_out,
                 distance_out);
}

/*******************************************************************************
 * \brief Cooley-Tukey step (For internal use only, special format).
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_bw_local_internal(double complex *grid_in,
                                  double complex *grid_out, const int fft_size,
                                  const int number_of_ffts) {

  // Determine a factorization of the FFT size
  int small_factor = 1;
  for (int i = 2; i * i <= fft_size; i++) {
    if (fft_size % i == 0) {
      small_factor = i;
      break;
    }
  }
  const int large_factor = fft_size / small_factor;
  assert(small_factor * large_factor == fft_size);
  assert(small_factor <= large_factor);

  const double pi = acos(-1.0);
  if (small_factor == 1 || large_factor == 1) {
    // If the FFT size is prime, we can use the naive implementation
    fft_ref_1d_bw_local_naive(grid_in, grid_out, fft_size, number_of_ffts);
  } else {
    // Perform FFTs along the shorter sub-dimension
    fft_ref_1d_bw_local_naive(grid_in, grid_out, small_factor,
                              number_of_ffts * large_factor);
// Transpose and multiply with twiddle factors
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in, grid_out, fft_size, number_of_ffts, pi, small_factor,      \
               large_factor)
    for (int index_small = 0; index_small < small_factor; index_small++) {
      for (int index_large = 0; index_large < large_factor; index_large++) {
        const double complex phase_factor =
            cexp(2.0 * I * pi * index_small * index_large / fft_size);
        for (int fft = 0; fft < number_of_ffts; fft++) {
          grid_in[fft +
                  (index_large * small_factor + index_small) * number_of_ffts] =
              grid_out[fft + (index_small * large_factor + index_large) *
                                 number_of_ffts] *
              phase_factor;
        }
      }
    }
    fft_ref_1d_bw_local_internal(grid_in, grid_out, large_factor,
                                 number_of_ffts * small_factor);
  }
}

/*******************************************************************************
 * \brief Naive implementation of backwards FFT to transposed format (for
 *easier transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_bw_local_low(double complex *grid_in, double complex *grid_out,
                             const int fft_size, const int number_of_ffts,
                             const int stride_in, const int stride_out,
                             const int distance_in, const int distance_out) {

  // Determine a factorization of the FFT size
  int small_factor = 1;
  for (int i = 2; i * i <= fft_size; i++) {
    if (fft_size % i == 0) {
      small_factor = i;
      break;
    }
  }
  const int large_factor = fft_size / small_factor;
  assert(small_factor * large_factor == fft_size);
  assert(small_factor <= large_factor);

  const double pi = acos(-1.0);
  reorder_input(grid_in, grid_out, fft_size, number_of_ffts, stride_in,
                distance_in);
  if (small_factor == 1 || large_factor == 1) {
    // If the FFT size is prime, we can use the naive implementation
    fft_ref_1d_bw_local_naive(grid_out, grid_in, fft_size, number_of_ffts);
  } else {
    // Perform FFTs along the shorter sub-dimension
    fft_ref_1d_bw_local_naive(grid_out, grid_in, small_factor,
                              number_of_ffts * large_factor);
// Transpose and multiply with twiddle factors
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in, grid_out, fft_size, number_of_ffts, pi, small_factor,      \
               large_factor)
    for (int index_small = 0; index_small < small_factor; index_small++) {
      for (int index_large = 0; index_large < large_factor; index_large++) {
        const double complex phase_factor =
            cexp(2.0 * I * pi * index_small * index_large / fft_size);
        for (int fft = 0; fft < number_of_ffts; fft++) {
          grid_out[fft + (index_large * small_factor + index_small) *
                             number_of_ffts] =
              grid_in[fft + (index_small * large_factor + index_large) *
                                number_of_ffts] *
              phase_factor;
        }
      }
    }
    fft_ref_1d_bw_local_internal(grid_out, grid_in, large_factor,
                                 number_of_ffts * small_factor);
  }
  reorder_output(grid_in, grid_out, fft_size, number_of_ffts, stride_out,
                 distance_out);
}

// EOF
