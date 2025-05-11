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
#include <time.h>

// Assume a cache line size of 64 bytes and 8 bytes per double
#define DOUBLES_PER_CACHE_LINE 8

static double time_reorder_input = 0.0;
static double time_reorder_output = 0.0;
static double time_1 = 0.0;
static double time_2 = 0.0;
static double time_4 = 0.0;
static double time_naive = 0.0;

// We need these definitions for the recursion within this module
void fft_ref_1d_fw_local_internal(double *restrict grid_in_real,
                                  double *restrict grid_in_imag,
                                  double *restrict grid_out_real,
                                  double *restrict grid_out_imag,
                                  const int fft_size, const int number_of_ffts,
                                  const int stride_size);
void fft_ref_1d_bw_local_internal(double *grid_in_real, double *grid_in_imag,
                                  double *grid_out_real, double *grid_out_imag,
                                  const int fft_size, const int number_of_ffts,
                                  const int stride_size);

void reorder_input(const double complex *restrict grid_in,
                   double *restrict grid_out_real,
                   double *restrict grid_out_imag, const int fft_size,
                   const int number_of_ffts, const int stride_in,
                   const int distance_in) {
  clock_t begin = clock();

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
  time_reorder_input += (double)(clock() - begin) / CLOCKS_PER_SEC;
}

void reorder_input_r2c(const double *restrict grid_in,
                       double *restrict grid_out, const int fft_size,
                       const int number_of_ffts, const int stride_in,
                       const int distance_in) {
  clock_t begin = clock();
  if (distance_in == 1 && stride_in == number_of_ffts) {
    memcpy(grid_out, grid_in, number_of_ffts * fft_size * sizeof(double));
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
  time_reorder_input += (double)(clock() - begin) / CLOCKS_PER_SEC;
}

void reorder_input_c2r(const double complex *restrict grid_in,
                       double *restrict grid_out_real,
                       double *restrict grid_out_imag, const int fft_size,
                       const int number_of_ffts, const int stride_in,
                       const int distance_in) {
  clock_t begin = clock();
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
  time_reorder_input += (double)(clock() - begin) / CLOCKS_PER_SEC;
}

void reorder_output(const double *grid_in_real, const double *grid_in_imag,
                    double complex *grid_out, const int fft_size,
                    const int number_of_ffts, const int stride_out,
                    const int distance_out) {
  clock_t begin = clock();
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
  time_reorder_output += (double)(clock() - begin) / CLOCKS_PER_SEC;
}

void reorder_output_2d(const double *grid_in_real, const double *grid_in_imag,
                       double complex *grid_out, const int fft_size[2],
                       const int number_of_ffts, const int stride_out,
                       const int distance_out) {
  clock_t begin = clock();
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
  time_reorder_output += (double)(clock() - begin) / CLOCKS_PER_SEC;
}

void reorder_output_r2c(const double *grid_in_real, const double *grid_in_imag,
                        double complex *grid_out, const int fft_size,
                        const int number_of_ffts, const int stride_out,
                        const int distance_out) {
  clock_t begin = clock();
// The first element is just given by the real part
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in_real, grid_in_imag, grid_out, number_of_ffts, stride_out,   \
               distance_out, fft_size)
  for (int index = 0; index < fft_size / 2 + 1; index++) {
    for (int fft = 0; fft < number_of_ffts; fft++) {
      grid_out[fft * distance_out + index * stride_out] =
          CMPLX(grid_in_real[fft + index * number_of_ffts],
                grid_in_imag[fft + index * number_of_ffts]);
    }
  }
  time_reorder_output += (double)(clock() - begin) / CLOCKS_PER_SEC;
}

void reorder_output_c2r(const double *grid_in, double *grid_out,
                        const int fft_size, const int number_of_ffts,
                        const int stride_out, const int distance_out) {
  clock_t begin = clock();
#pragma omp parallel for default(none) collapse(2) shared(                     \
        grid_in, grid_out, number_of_ffts, stride_out, distance_out, fft_size)
  for (int index = 0; index < fft_size; index++) {
    for (int fft = 0; fft < number_of_ffts; fft++) {
      grid_out[fft * distance_out + index * stride_out] =
          grid_in[fft + index * number_of_ffts];
    }
  }
  time_reorder_output += (double)(clock() - begin) / CLOCKS_PER_SEC;
}

/*******************************************************************************
 * \brief Naive implementation of FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_fw_local_naive(const double *restrict grid_in_real,
                               const double *restrict grid_in_imag,
                               double *restrict grid_out_real,
                               double *restrict grid_out_imag,
                               const int fft_size, const int number_of_ffts,
                               const int stride_size) {
  clock_t begin = clock();
  // Perform FFTs along the first dimension
  if (fft_size == 1) {
    // If the FFT size is 1, we can use a simple copy
    memcpy(grid_out_real, grid_in_real,
           fft_size * number_of_ffts * sizeof(double));
    memcpy(grid_out_imag, grid_in_imag,
           fft_size * number_of_ffts * sizeof(double));
    time_1 += (double)(clock() - begin) / CLOCKS_PER_SEC;
  } else if (fft_size == 2) {
    for (int fft = 0; fft < number_of_ffts; fft++) {
      grid_out_real[fft] = grid_in_real[fft] + grid_in_real[stride_size + fft];
      grid_out_imag[fft] = grid_in_imag[fft] + grid_in_imag[stride_size + fft];
      grid_out_real[stride_size + fft] =
          grid_in_real[fft] - grid_in_real[stride_size + fft];
      grid_out_imag[stride_size + fft] =
          grid_in_imag[fft] - grid_in_imag[stride_size + fft];
    }
    time_2 += (double)(clock() - begin) / CLOCKS_PER_SEC;
  } else if (fft_size == 4) {
    for (int fft = 0; fft < number_of_ffts; fft++) {
      grid_out_real[fft] = grid_in_real[fft];
      grid_out_imag[fft] = grid_in_imag[fft];
      grid_out_real[fft] += grid_in_real[fft + stride_size];
      grid_out_imag[fft] += grid_in_imag[fft + stride_size];
      grid_out_real[fft] += grid_in_real[fft + 2 * stride_size];
      grid_out_imag[fft] += grid_in_imag[fft + 2 * stride_size];
      grid_out_real[fft] += grid_in_real[fft + 3 * stride_size];
      grid_out_imag[fft] += grid_in_imag[fft + 3 * stride_size];
      grid_out_real[fft + stride_size] = grid_in_real[fft];
      grid_out_imag[fft + stride_size] = grid_in_imag[fft];
      grid_out_real[fft + stride_size] += grid_in_imag[fft + stride_size];
      grid_out_imag[fft + stride_size] -= grid_in_real[fft + stride_size];
      grid_out_real[fft + stride_size] -= grid_in_real[fft + 2 * stride_size];
      grid_out_imag[fft + stride_size] -= grid_in_imag[fft + 2 * stride_size];
      grid_out_real[fft + stride_size] -= grid_in_imag[fft + 3 * stride_size];
      grid_out_imag[fft + stride_size] += grid_in_real[fft + 3 * stride_size];
      grid_out_real[fft + 2 * stride_size] = grid_in_real[fft];
      grid_out_imag[fft + 2 * stride_size] = grid_in_imag[fft];
      grid_out_real[fft + 2 * stride_size] -= grid_in_real[fft + stride_size];
      grid_out_imag[fft + 2 * stride_size] -= grid_in_imag[fft + stride_size];
      grid_out_real[fft + 2 * stride_size] +=
          grid_in_real[fft + 2 * stride_size];
      grid_out_imag[fft + 2 * stride_size] +=
          grid_in_imag[fft + 2 * stride_size];
      grid_out_real[fft + 2 * stride_size] -=
          grid_in_real[fft + 3 * stride_size];
      grid_out_imag[fft + 2 * stride_size] -=
          grid_in_imag[fft + 3 * stride_size];
      grid_out_real[fft + 3 * stride_size] = grid_in_real[fft];
      grid_out_imag[fft + 3 * stride_size] = grid_in_imag[fft];
      grid_out_real[fft + 3 * stride_size] -= grid_in_imag[fft + stride_size];
      grid_out_imag[fft + 3 * stride_size] += grid_in_real[fft + stride_size];
      grid_out_real[fft + 3 * stride_size] -=
          grid_in_real[fft + 2 * stride_size];
      grid_out_imag[fft + 3 * stride_size] -=
          grid_in_imag[fft + 2 * stride_size];
      grid_out_real[fft + 3 * stride_size] +=
          grid_in_imag[fft + 3 * stride_size];
      grid_out_imag[fft + 3 * stride_size] -=
          grid_in_real[fft + 3 * stride_size];
    }
    time_4 += (double)(clock() - begin) / CLOCKS_PER_SEC;
  } else {
    const double pi = acos(-1.0);
    memset(grid_out_real, 0, fft_size * number_of_ffts * sizeof(double));
    memset(grid_out_imag, 0, fft_size * number_of_ffts * sizeof(double));
#pragma omp parallel for default(none)                                         \
    shared(grid_in_real, grid_in_imag, grid_out_real, grid_out_imag, fft_size, \
               number_of_ffts, stride_size, pi)
    for (int index_out = 0; index_out < fft_size; index_out++) {
      for (int index_in = 0; index_in < fft_size; index_in++) {
        const double complex phase_factor =
            cexp(-2.0 * I * pi * index_out * index_in / fft_size);
        for (int fft = 0; fft < number_of_ffts; fft++) {
          grid_out_real[fft + index_out * stride_size] +=
              grid_in_real[fft + index_in * stride_size] * creal(phase_factor) -
              grid_in_imag[fft + index_in * stride_size] * cimag(phase_factor);
          grid_out_imag[fft + index_out * stride_size] +=
              grid_in_real[fft + index_in * stride_size] * cimag(phase_factor) +
              grid_in_imag[fft + index_in * stride_size] * creal(phase_factor);
        }
      }
    }
    time_naive += (double)(clock() - begin) / CLOCKS_PER_SEC;
  }
}

/*******************************************************************************
 * \brief Naive implementation of FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_fw_local_r2c_naive(const double *restrict grid_in,
                                   double *restrict grid_out,
                                   const int fft_size, const int number_of_ffts,
                                   int *offset_imaginary) {
  clock_t begin = clock();
  *offset_imaginary = (fft_size / 2 + 1) * number_of_ffts;
  double *grid_out_real = grid_out;
  double *grid_out_imag = grid_out_real + *offset_imaginary;
  // Perform FFTs along the first dimension
  const double pi = acos(-1.0);
  memset(grid_out_real, 0,
         (fft_size / 2 + 1) * number_of_ffts * sizeof(double));
  memset(grid_out_imag, 0,
         (fft_size / 2 + 1) * number_of_ffts * sizeof(double));

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

  double *buffer_real = calloc(number_of_ffts * fft_size, sizeof(double));
  double *buffer_imag = calloc(number_of_ffts * fft_size, sizeof(double));
  for (int index_out_small = 0; index_out_small < small_factor;
       index_out_small++) {
    for (int index_in_small = 0; index_in_small < small_factor;
         index_in_small++) {
      const double complex phase_factor =
          cexp(-2.0 * I * pi * index_out_small * index_in_small / small_factor);
      for (int index_large = 0; index_large < large_factor; index_large++) {
        for (int fft = 0; fft < number_of_ffts; fft++) {
          buffer_real[fft + (index_out_small * large_factor + index_large) *
                                number_of_ffts] +=
              creal(
                  grid_in[fft + (index_in_small * large_factor + index_large) *
                                    number_of_ffts] *
                  phase_factor);
          buffer_imag[fft + (index_out_small * large_factor + index_large) *
                                number_of_ffts] +=
              cimag(
                  grid_in[fft + (index_in_small * large_factor + index_large) *
                                    number_of_ffts] *
                  phase_factor);
        }
      }
    }
  }
  double *buffer_real2 = calloc(number_of_ffts * fft_size, sizeof(double));
  double *buffer_imag2 = calloc(number_of_ffts * fft_size, sizeof(double));
  for (int index_small = 0; index_small < small_factor; index_small++) {
    for (int index_large = 0; index_large < large_factor; index_large++) {
      const double complex phase_factor =
          cexp(-2.0 * I * pi * index_small * index_large / fft_size);
      for (int fft = 0; fft < number_of_ffts; fft++) {
        buffer_real2[fft + (index_large * small_factor + index_small) *
                               number_of_ffts] =
            buffer_real[fft + (index_small * large_factor + index_large) *
                                  number_of_ffts] *
                creal(phase_factor) -
            buffer_imag[fft + (index_small * large_factor + index_large) *
                                  number_of_ffts] *
                cimag(phase_factor);
        buffer_imag2[fft + (index_large * small_factor + index_small) *
                               number_of_ffts] =
            buffer_real[fft + (index_small * large_factor + index_large) *
                                  number_of_ffts] *
                cimag(phase_factor) +
            buffer_imag[fft + (index_small * large_factor + index_large) *
                                  number_of_ffts] *
                creal(phase_factor);
      }
    }
  }

  // l is even: 0=(0, 0), ... , l*s/2=(l/2, 0)
  // l is odd: 0, ... , l'*s+s/2=(l/2, s/2)
  for (int index_out_large = 0; index_out_large < large_factor / 2;
       index_out_large++) {
    for (int index_in_large = 0; index_in_large < large_factor;
         index_in_large++) {
      const double complex phase_factor =
          cexp(-2.0 * I * pi * index_out_large * index_in_large / large_factor);
      for (int index_small = 0; index_small < small_factor; index_small++) {
        for (int fft = 0; fft < number_of_ffts; fft++) {
          grid_out_real[fft + (index_out_large * small_factor + index_small) *
                                  number_of_ffts] +=
              buffer_real2[fft + (index_in_large * small_factor + index_small) *
                                     number_of_ffts] *
                  creal(phase_factor) -
              buffer_imag2[fft + (index_in_large * small_factor + index_small) *
                                     number_of_ffts] *
                  cimag(phase_factor);
          grid_out_imag[fft + (index_out_large * small_factor + index_small) *
                                  number_of_ffts] +=
              buffer_real2[fft + (index_in_large * small_factor + index_small) *
                                     number_of_ffts] *
                  cimag(phase_factor) +
              buffer_imag2[fft + (index_in_large * small_factor + index_small) *
                                     number_of_ffts] *
                  creal(phase_factor);
        }
      }
    }
  }
  if (large_factor % 2 == 0) {
    for (int index_in_large = 0; index_in_large < large_factor;
         index_in_large++) {
      const double phase_factor = index_in_large % 2 == 0 ? 1.0 : -1.0;
      for (int fft = 0; fft < number_of_ffts; fft++) {
        grid_out_real[fft + fft_size / 2 * number_of_ffts] +=
            buffer_real2[fft + index_in_large * small_factor * number_of_ffts] *
            phase_factor;
      }
    }
  } else {
    for (int index_in_large = 0; index_in_large < large_factor;
         index_in_large++) {
      const double complex phase_factor =
          cexp(-2.0 * I * pi * (large_factor - 1) / large_factor / 2 *
               index_in_large);
      for (int index_small = 0; index_small < small_factor / 2 + 1;
           index_small++) {
        for (int fft = 0; fft < number_of_ffts; fft++) {
          grid_out_real[fft + (large_factor / 2 * small_factor + index_small) *
                                  number_of_ffts] +=
              buffer_real2[fft + (index_in_large * small_factor + index_small) *
                                     number_of_ffts] *
                  creal(phase_factor) -
              buffer_imag2[fft + (index_in_large * small_factor + index_small) *
                                     number_of_ffts] *
                  cimag(phase_factor);
          grid_out_imag[fft + (large_factor / 2 * small_factor + index_small) *
                                  number_of_ffts] +=
              buffer_real2[fft + (index_in_large * small_factor + index_small) *
                                     number_of_ffts] *
                  cimag(phase_factor) +
              buffer_imag2[fft + (index_in_large * small_factor + index_small) *
                                     number_of_ffts] *
                  creal(phase_factor);
        }
      }
    }
  }
  free(buffer_real2);
  free(buffer_imag2);
  free(buffer_real);
  free(buffer_imag);
  time_naive += 0.0 * (double)(clock() - begin) / CLOCKS_PER_SEC;
}

/*******************************************************************************
 * \brief Naive implementation of FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_bw_local_naive(const double *restrict grid_in_real,
                               const double *restrict grid_in_imag,
                               double *restrict grid_out_real,
                               double *restrict grid_out_imag,
                               const int fft_size, const int number_of_ffts,
                               const int stride_size) {
  clock_t begin = clock();
  if (fft_size == 1) {
    // If the FFT size is 1, we can use a simple copy
    memcpy(grid_out_real, grid_in_real, number_of_ffts * sizeof(double));
    memcpy(grid_out_imag, grid_in_imag, number_of_ffts * sizeof(double));
    time_1 += (double)(clock() - begin) / CLOCKS_PER_SEC;
  } else if (fft_size == 2) {
    for (int fft = 0; fft < number_of_ffts; fft++) {
      grid_out_real[fft] = grid_in_real[fft] + grid_in_real[fft + stride_size];
      grid_out_imag[fft] = grid_in_imag[fft] + grid_in_imag[fft + stride_size];
      grid_out_real[fft + stride_size] =
          grid_in_real[fft] - grid_in_real[fft + stride_size];
      grid_out_imag[fft + stride_size] =
          grid_in_imag[fft] - grid_in_imag[fft + stride_size];
    }
    time_2 += (double)(clock() - begin) / CLOCKS_PER_SEC;
  } else if (fft_size == 4) {
    for (int fft = 0; fft < number_of_ffts; fft++) {
      grid_out_real[fft] = grid_in_real[fft];
      grid_out_real[fft] += grid_in_real[fft + stride_size];
      grid_out_real[fft] += grid_in_real[fft + 2 * stride_size];
      grid_out_real[fft] += grid_in_real[fft + 3 * stride_size];
      grid_out_real[fft + stride_size] = grid_in_real[fft];
      grid_out_real[fft + stride_size] -= grid_in_imag[fft + stride_size];
      grid_out_real[fft + stride_size] -= grid_in_real[fft + 2 * stride_size];
      grid_out_real[fft + stride_size] += grid_in_imag[fft + 3 * stride_size];
      grid_out_real[fft + 2 * stride_size] = grid_in_real[fft];
      grid_out_real[fft + 2 * stride_size] -= grid_in_real[fft + stride_size];
      grid_out_real[fft + 2 * stride_size] +=
          grid_in_real[fft + 2 * stride_size];
      grid_out_real[fft + 2 * stride_size] -=
          grid_in_real[fft + 3 * stride_size];
      grid_out_real[fft + 3 * stride_size] = grid_in_real[fft];
      grid_out_real[fft + 3 * stride_size] += grid_in_imag[fft + stride_size];
      grid_out_real[fft + 3 * stride_size] -=
          grid_in_real[fft + 2 * stride_size];
      grid_out_real[fft + 3 * stride_size] -=
          grid_in_imag[fft + 3 * stride_size];
      grid_out_imag[fft] = grid_in_imag[fft];
      grid_out_imag[fft] += grid_in_imag[fft + stride_size];
      grid_out_imag[fft] += grid_in_imag[fft + 2 * stride_size];
      grid_out_imag[fft] += grid_in_imag[fft + 3 * stride_size];
      grid_out_imag[fft + stride_size] = grid_in_imag[fft];
      grid_out_imag[fft + stride_size] += grid_in_real[fft + stride_size];
      grid_out_imag[fft + stride_size] -= grid_in_imag[fft + 2 * stride_size];
      grid_out_imag[fft + stride_size] -= grid_in_real[fft + 3 * stride_size];
      grid_out_imag[fft + 2 * stride_size] = grid_in_imag[fft];
      grid_out_imag[fft + 2 * stride_size] -= grid_in_imag[fft + stride_size];
      grid_out_imag[fft + 2 * stride_size] +=
          grid_in_imag[fft + 2 * stride_size];
      grid_out_imag[fft + 2 * stride_size] -=
          grid_in_imag[fft + 3 * stride_size];
      grid_out_imag[fft + 3 * stride_size] = grid_in_imag[fft];
      grid_out_imag[fft + 3 * stride_size] -= grid_in_real[fft + stride_size];
      grid_out_imag[fft + 3 * stride_size] -=
          grid_in_imag[fft + 2 * stride_size];
      grid_out_imag[fft + 3 * stride_size] -=
          -grid_in_real[fft + 3 * stride_size];
    }
    time_4 += (double)(clock() - begin) / CLOCKS_PER_SEC;
  } else {
    const double pi = acos(-1.0);
    memset(grid_out_real, 0, fft_size * number_of_ffts * sizeof(double));
    memset(grid_out_imag, 0, fft_size * number_of_ffts * sizeof(double));
#pragma omp parallel for default(none)                                         \
    shared(grid_in_real, grid_in_imag, grid_out_real, grid_out_imag, fft_size, \
               number_of_ffts, stride_size, pi)
    for (int index_out = 0; index_out < fft_size; index_out++) {
      for (int index_in = 0; index_in < fft_size; index_in++) {
        const double complex phase_factor =
            cexp(2.0 * I * pi * index_out * index_in / fft_size);
        for (int fft = 0; fft < number_of_ffts; fft++) {
          grid_out_real[fft + index_out * stride_size] +=
              grid_in_real[fft + index_in * stride_size] * creal(phase_factor) -
              grid_in_imag[fft + index_in * stride_size] * cimag(phase_factor);
          grid_out_imag[fft + index_out * stride_size] +=
              grid_in_real[fft + index_in * stride_size] * cimag(phase_factor) +
              grid_in_imag[fft + index_in * stride_size] * creal(phase_factor);
        }
      }
    }
    time_naive += (double)(clock() - begin) / CLOCKS_PER_SEC;
  }
}

/*******************************************************************************
 * \brief Naive implementation of FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_bw_local_c2r_naive(double *restrict grid_in,
                                   double *restrict grid_out,
                                   const int fft_size, const int number_of_ffts,
                                   const int position_imag) {
  clock_t begin = clock();

  double *grid_in_real = grid_in;
  double *grid_in_imag = grid_in + position_imag;

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
  time_naive += (double)(clock() - begin) / CLOCKS_PER_SEC;
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_fw_local_internal(double *restrict grid_in_real,
                                  double *restrict grid_in_imag,
                                  double *restrict grid_out_real,
                                  double *restrict grid_out_imag,
                                  const int fft_size, const int number_of_ffts,
                                  const int stride_size) {

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
                              grid_out_imag, fft_size, number_of_ffts,
                              stride_size);
  } else {
    // Perform FFTs along the shorter sub-dimension
    fft_ref_1d_fw_local_naive(
        grid_in_real, grid_in_imag, grid_out_real, grid_out_imag, small_factor,
        number_of_ffts * large_factor, stride_size * large_factor);
// Transpose and multiply with twiddle factors
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in_real, grid_in_imag, grid_out_real, grid_out_imag, fft_size, \
               number_of_ffts, stride_size, pi, small_factor, large_factor)
    for (int index_small = 0; index_small < small_factor; index_small++) {
      for (int index_large = 0; index_large < large_factor; index_large++) {
        const double complex phase_factor =
            cexp(-2.0 * I * pi * index_small * index_large / fft_size);
        for (int fft = 0; fft < number_of_ffts; fft++) {
          grid_in_real[fft + (index_large * small_factor + index_small) *
                                 stride_size] =
              grid_out_real[fft + (index_small * large_factor + index_large) *
                                      stride_size] *
                  creal(phase_factor) -
              grid_out_imag[fft + (index_small * large_factor + index_large) *
                                      stride_size] *
                  cimag(phase_factor);
          grid_in_imag[fft + (index_large * small_factor + index_small) *
                                 stride_size] =
              grid_out_real[fft + (index_small * large_factor + index_large) *
                                      stride_size] *
                  cimag(phase_factor) +
              grid_out_imag[fft + (index_small * large_factor + index_large) *
                                      stride_size] *
                  creal(phase_factor);
        }
      }
    }
    fft_ref_1d_fw_local_internal(
        grid_in_real, grid_in_imag, grid_out_real, grid_out_imag, large_factor,
        number_of_ffts * small_factor, stride_size * small_factor);
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
                                  const int fft_size, const int number_of_ffts,
                                  const int stride_size) {

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
                              grid_out_imag, fft_size, number_of_ffts,
                              stride_size);
  } else {
    // Perform FFTs along the shorter sub-dimension
    fft_ref_1d_bw_local_naive(
        grid_in_real, grid_in_imag, grid_out_real, grid_out_imag, small_factor,
        number_of_ffts * large_factor, stride_size * large_factor);
// Transpose and multiply with twiddle factors
#pragma omp parallel for default(none) collapse(2)                             \
    shared(grid_in_real, grid_in_imag, grid_out_real, grid_out_imag, fft_size, \
               number_of_ffts, stride_size, pi, small_factor, large_factor)
    for (int index_small = 0; index_small < small_factor; index_small++) {
      for (int index_large = 0; index_large < large_factor; index_large++) {
        const double complex phase_factor =
            cexp(2.0 * I * pi * index_small * index_large / fft_size);
        for (int fft = 0; fft < number_of_ffts; fft++) {
          grid_in_real[fft + (index_large * small_factor + index_small) *
                                 stride_size] =
              grid_out_real[fft + (index_small * large_factor + index_large) *
                                      stride_size] *
                  creal(phase_factor) -
              grid_out_imag[fft + (index_small * large_factor + index_large) *
                                      stride_size] *
                  cimag(phase_factor);
          grid_in_imag[fft + (index_large * small_factor + index_small) *
                                 stride_size] =
              grid_out_real[fft + (index_small * large_factor + index_large) *
                                      stride_size] *
                  cimag(phase_factor) +
              grid_out_imag[fft + (index_small * large_factor + index_large) *
                                      stride_size] *
                  creal(phase_factor);
        }
      }
    }
    fft_ref_1d_bw_local_internal(
        grid_in_real, grid_in_imag, grid_out_real, grid_out_imag, large_factor,
        number_of_ffts * small_factor, stride_size * small_factor);
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

  time_reorder_input = 0.0;
  time_reorder_output = 0.0;
  time_1 = 0.0;
  time_2 = 0.0;
  time_4 = 0.0;
  time_naive = 0.0;

  clock_t begin = clock();

  // We reorder the data to a format more suitable for vectorization
  double *grid_in_real = (double *)grid_in;
  double *grid_in_imag = ((double *)grid_in) + fft_size * number_of_ffts;
  double *grid_out_real = (double *)grid_out;
  double *grid_out_imag = ((double *)grid_out) + fft_size * number_of_ffts;

  reorder_input(grid_in, grid_out_real, grid_out_imag, fft_size, number_of_ffts,
                stride_in, distance_in);
  fft_ref_1d_fw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size, number_of_ffts,
                               number_of_ffts);
  reorder_output(grid_in_real, grid_in_imag, grid_out, fft_size, number_of_ffts,
                 stride_out, distance_out);

  clock_t end = clock();
  printf("Time Reorder input: %f\n", time_reorder_input);
  printf("Time Reorder output: %f\n", time_reorder_output);
  printf("Time 1: %f\n", time_1);
  printf("Time 2: %f\n", time_2);
  printf("Time 4: %f\n", time_4);
  printf("Time naive: %f\n", time_naive);
  printf("Total Time 1D FW %i %i: %f\n", fft_size, number_of_ffts,
         (double)(end - begin) / CLOCKS_PER_SEC);
  fflush(stdout);
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

#if 0
  time_reorder_input = 0.0;
  time_reorder_output = 0.0;
  time_1 = 0.0;
  time_2 = 0.0;
  time_4 = 0.0;
  time_naive = 0.0;

  clock_t begin = clock();
#endif

  int offset_imaginary;
  double *grid_in_real = grid_in;
  double *grid_out_real = (double *)grid_out;

  reorder_input_r2c(grid_in, grid_out_real, fft_size, number_of_ffts, stride_in,
                    distance_in);
  fft_ref_1d_fw_local_r2c_naive(grid_out_real, grid_in_real, fft_size,
                                number_of_ffts, &offset_imaginary);
  double *grid_in_imag = grid_in_real + offset_imaginary;
  reorder_output_r2c(grid_in_real, grid_in_imag, grid_out, fft_size,
                     number_of_ffts, stride_out, distance_out);

#if 0
  clock_t end = clock();
  printf("Time Reorder input: %f\n", time_reorder_input);
  printf("Time Reorder output: %f\n", time_reorder_output);
  printf("Time 1: %f\n", time_1);
  printf("Time 2: %f\n", time_2);
  printf("Time 4: %f\n", time_4);
  printf("Time naive: %f\n", time_naive);
  printf("Total Time R2C FW %i %i: %f\n", fft_size, number_of_ffts,
         (double)(end - begin) / CLOCKS_PER_SEC);
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

  time_reorder_input = 0.0;
  time_reorder_output = 0.0;
  time_1 = 0.0;
  time_2 = 0.0;
  time_4 = 0.0;
  time_naive = 0.0;

  clock_t begin = clock();

  // We reorder the data to a format more suitable for vectorization
  double *grid_in_real = (double *)grid_in;
  double *grid_in_imag = ((double *)grid_in) + fft_size * number_of_ffts;
  double *grid_out_real = (double *)grid_out;
  double *grid_out_imag = ((double *)grid_out) + fft_size * number_of_ffts;

  reorder_input(grid_in, grid_out_real, grid_out_imag, fft_size, number_of_ffts,
                stride_in, distance_in);
  fft_ref_1d_bw_local_internal(grid_out_real, grid_out_imag, grid_in_real,
                               grid_in_imag, fft_size, number_of_ffts,
                               number_of_ffts);
  reorder_output(grid_in_real, grid_in_imag, grid_out, fft_size, number_of_ffts,
                 stride_out, distance_out);

  clock_t end = clock();
  printf("Time Reorder input: %f\n", time_reorder_input);
  printf("Time Reorder output: %f\n", time_reorder_output);
  printf("Time 1: %f\n", time_1);
  printf("Time 2: %f\n", time_2);
  printf("Time 4: %f\n", time_4);
  printf("Time naive: %f\n", time_naive);
  printf("Total Time 1D BW %i %i: %f\n", fft_size, number_of_ffts,
         (double)(end - begin) / CLOCKS_PER_SEC);
  fflush(stdout);
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

#if 0
  time_reorder_input = 0.0;
  time_reorder_output = 0.0;
  time_1 = 0.0;
  time_2 = 0.0;
  time_4 = 0.0;
  time_naive = 0.0;

  clock_t begin = clock();
#endif

  const int offset_imaginary = (fft_size / 2 + 1) * number_of_ffts;
  double *grid_in_real = (double *)grid_in;
  double *grid_out_real = grid_out;
  double *grid_out_imag = grid_out + offset_imaginary;

  reorder_input_c2r(grid_in, grid_out_real, grid_out_imag, fft_size,
                    number_of_ffts, stride_in, distance_in);
  fft_ref_1d_bw_local_c2r_naive(grid_out_real, grid_in_real, fft_size,
                                number_of_ffts, offset_imaginary);
  reorder_output_c2r(grid_in_real, grid_out, fft_size, number_of_ffts,
                     stride_out, distance_out);

#if 0
  clock_t end = clock();
  printf("Time Reorder input: %f\n", time_reorder_input);
  printf("Time Reorder output: %f\n", time_reorder_output);
  printf("Time 1: %f\n", time_1);
  printf("Time 2: %f\n", time_2);
  printf("Time 4: %f\n", time_4);
  printf("Time naive: %f\n", time_naive);
  printf("Total Time 1D C2R BW %i %i: %f\n", fft_size, number_of_ffts,
         (double)(end - begin) / CLOCKS_PER_SEC);
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

  time_reorder_input = 0.0;
  time_reorder_output = 0.0;
  time_1 = 0.0;
  time_2 = 0.0;
  time_4 = 0.0;
  time_naive = 0.0;

  clock_t begin = clock();

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
  fft_ref_1d_fw_local_internal(
      grid_out_real, grid_out_imag, grid_in_real, grid_in_imag, fft_size[0],
      number_of_ffts * fft_size[1], number_of_ffts * fft_size[1]);
// Transpose the data
#pragma omp parallel for collapse(2) default(none)                             \
    shared(fft_size, grid_out_real, grid_out_imag, grid_in_real, grid_in_imag, \
               number_of_ffts)
  for (int index_0 = 0; index_0 < fft_size[0]; index_0++) {
    for (int index_1 = 0; index_1 < fft_size[1]; index_1++) {
      memcpy(grid_out_real + index_1 * fft_size[0] * number_of_ffts +
                 index_0 * number_of_ffts,
             grid_in_real + index_0 * fft_size[1] * number_of_ffts +
                 index_1 * number_of_ffts,
             number_of_ffts * sizeof(double));
      memcpy(grid_out_imag + index_1 * fft_size[0] * number_of_ffts +
                 index_0 * number_of_ffts,
             grid_in_imag + index_0 * fft_size[1] * number_of_ffts +
                 index_1 * number_of_ffts,
             number_of_ffts * sizeof(double));
    }
  }
  fft_ref_1d_fw_local_internal(
      grid_out_real, grid_out_imag, grid_in_real, grid_in_imag, fft_size[1],
      number_of_ffts * fft_size[0], number_of_ffts * fft_size[0]);
  reorder_output_2d(grid_in_real, grid_in_imag, grid_out, fft_size,
                    number_of_ffts, stride_out, distance_out);

  clock_t end = clock();
  printf("Time Reorder input: %f\n", time_reorder_input);
  printf("Time Reorder output: %f\n", time_reorder_output);
  printf("Time 1: %f\n", time_1);
  printf("Time 2: %f\n", time_2);
  printf("Time 4: %f\n", time_4);
  printf("Time naive: %f\n", time_naive);
  printf("Total Time 2D FW %i %i %i: %f\n", fft_size[0], fft_size[1],
         number_of_ffts, (double)(end - begin) / CLOCKS_PER_SEC);
  fflush(stdout);
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

  time_reorder_input = 0.0;
  time_reorder_output = 0.0;
  time_1 = 0.0;
  time_2 = 0.0;
  time_4 = 0.0;
  time_naive = 0.0;

  clock_t begin = clock();

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
  fft_ref_1d_bw_local_internal(
      grid_out_real, grid_out_imag, grid_in_real, grid_in_imag, fft_size[0],
      number_of_ffts * fft_size[1], number_of_ffts * fft_size[1]);
// Transpose the data
#pragma omp parallel for collapse(2) default(none)                             \
    shared(fft_size, grid_out_real, grid_out_imag, grid_in_real, grid_in_imag, \
               number_of_ffts)
  for (int index_0 = 0; index_0 < fft_size[0]; index_0++) {
    for (int index_1 = 0; index_1 < fft_size[1]; index_1++) {
      memcpy(grid_out_real + index_1 * fft_size[0] * number_of_ffts +
                 index_0 * number_of_ffts,
             grid_in_real + index_0 * fft_size[1] * number_of_ffts +
                 index_1 * number_of_ffts,
             number_of_ffts * sizeof(double));
      memcpy(grid_out_imag + index_1 * fft_size[0] * number_of_ffts +
                 index_0 * number_of_ffts,
             grid_in_imag + index_0 * fft_size[1] * number_of_ffts +
                 index_1 * number_of_ffts,
             number_of_ffts * sizeof(double));
    }
  }
  fft_ref_1d_bw_local_internal(
      grid_out_real, grid_out_imag, grid_in_real, grid_in_imag, fft_size[1],
      number_of_ffts * fft_size[0], number_of_ffts * fft_size[0]);
  reorder_output_2d(grid_in_real, grid_in_imag, grid_out, fft_size,
                    number_of_ffts, stride_out, distance_out);

  clock_t end = clock();
  printf("Time Reorder input: %f\n", time_reorder_input);
  printf("Time Reorder output: %f\n", time_reorder_output);
  printf("Time 1: %f\n", time_1);
  printf("Time 2: %f\n", time_2);
  printf("Time 4: %f\n", time_4);
  printf("Time naive: %f\n", time_naive);
  printf("Total Time 2D BW %i %i %i: %f\n", fft_size[0], fft_size[1],
         number_of_ffts, (double)(end - begin) / CLOCKS_PER_SEC);
  fflush(stdout);
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_3d_fw_local_low(double complex *restrict grid_in,
                             double complex *restrict grid_out,
                             const int fft_size[3]) {

  time_reorder_input = 0.0;
  time_reorder_output = 0.0;
  time_1 = 0.0;
  time_2 = 0.0;
  time_4 = 0.0;
  time_naive = 0.0;

  clock_t begin = clock();

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
  fft_ref_1d_fw_local_internal(
      grid_out_real, grid_out_imag, grid_in_real, grid_in_imag, fft_size[2],
      fft_size[0] * fft_size[1], fft_size[0] * fft_size[1]);
  // Transpose the data
  for (int index_0 = 0; index_0 < fft_size[2]; index_0++) {
    for (int index_1 = 0; index_1 < fft_size[0] * fft_size[1]; index_1++) {
      grid_out_real[index_1 * fft_size[2] + index_0] =
          grid_in_real[index_0 * fft_size[0] * fft_size[1] + index_1];
      grid_out_imag[index_1 * fft_size[2] + index_0] =
          grid_in_imag[index_0 * fft_size[0] * fft_size[1] + index_1];
    }
  }
  fft_ref_1d_fw_local_internal(
      grid_out_real, grid_out_imag, grid_in_real, grid_in_imag, fft_size[1],
      fft_size[0] * fft_size[2], fft_size[0] * fft_size[2]);
  // Transpose the data
  for (int index_0 = 0; index_0 < fft_size[1]; index_0++) {
    for (int index_1 = 0; index_1 < fft_size[0] * fft_size[2]; index_1++) {
      grid_out_real[index_1 * fft_size[1] + index_0] =
          grid_in_real[index_0 * fft_size[0] * fft_size[2] + index_1];
      grid_out_imag[index_1 * fft_size[1] + index_0] =
          grid_in_imag[index_0 * fft_size[0] * fft_size[2] + index_1];
    }
  }
  fft_ref_1d_fw_local_internal(
      grid_out_real, grid_out_imag, grid_in_real, grid_in_imag, fft_size[0],
      fft_size[1] * fft_size[2], fft_size[1] * fft_size[2]);
  // Transpose the data
  for (int index_0 = 0; index_0 < fft_size[0]; index_0++) {
    for (int index_1 = 0; index_1 < fft_size[1] * fft_size[2]; index_1++) {
      grid_out[index_1 * fft_size[0] + index_0] =
          CMPLX(grid_in_real[index_0 * fft_size[1] * fft_size[2] + index_1],
                grid_in_imag[index_0 * fft_size[1] * fft_size[2] + index_1]);
    }
  }

  clock_t end = clock();
  printf("Time Reorder input: %f\n", time_reorder_input);
  printf("Time Reorder output: %f\n", time_reorder_output);
  printf("Time 1: %f\n", time_1);
  printf("Time 2: %f\n", time_2);
  printf("Time 4: %f\n", time_4);
  printf("Time naive: %f\n", time_naive);
  printf("Total Time 3D FW %i %i %i: %f\n", fft_size[0], fft_size[1],
         fft_size[2], (double)(end - begin) / CLOCKS_PER_SEC);
  fflush(stdout);
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_ref_3d_bw_local_low(double complex *restrict grid_in,
                             double complex *restrict grid_out,
                             const int fft_size[3]) {

  time_reorder_input = 0.0;
  time_reorder_output = 0.0;
  time_1 = 0.0;
  time_2 = 0.0;
  time_4 = 0.0;
  time_naive = 0.0;

  clock_t begin = clock();

  // We reorder the data to a format more suitable for vectorization
  double *grid_in_real = (double *)grid_in;
  double *grid_in_imag =
      ((double *)grid_in) + fft_size[0] * fft_size[1] * fft_size[2];
  double *grid_out_real = (double *)grid_out;
  double *grid_out_imag =
      ((double *)grid_out) + fft_size[0] * fft_size[1] * fft_size[2];

  reorder_input(grid_in, grid_out_real, grid_out_imag,
                fft_size[0] * fft_size[1] * fft_size[2], 1, 1, 1);
  fft_ref_1d_bw_local_internal(
      grid_out_real, grid_out_imag, grid_in_real, grid_in_imag, fft_size[2],
      fft_size[0] * fft_size[1], fft_size[0] * fft_size[1]);
  // Transpose the data
  for (int index_0 = 0; index_0 < fft_size[2]; index_0++) {
    for (int index_1 = 0; index_1 < fft_size[0] * fft_size[1]; index_1++) {
      grid_out_real[index_1 * fft_size[2] + index_0] =
          grid_in_real[index_0 * fft_size[0] * fft_size[1] + index_1];
      grid_out_imag[index_1 * fft_size[2] + index_0] =
          grid_in_imag[index_0 * fft_size[0] * fft_size[1] + index_1];
    }
  }
  fft_ref_1d_bw_local_internal(
      grid_out_real, grid_out_imag, grid_in_real, grid_in_imag, fft_size[1],
      fft_size[0] * fft_size[2], fft_size[0] * fft_size[2]);
  // Transpose the data
  for (int index_0 = 0; index_0 < fft_size[1]; index_0++) {
    for (int index_1 = 0; index_1 < fft_size[0] * fft_size[2]; index_1++) {
      grid_out_real[index_1 * fft_size[1] + index_0] =
          grid_in_real[index_0 * fft_size[0] * fft_size[2] + index_1];
      grid_out_imag[index_1 * fft_size[1] + index_0] =
          grid_in_imag[index_0 * fft_size[0] * fft_size[2] + index_1];
    }
  }
  fft_ref_1d_bw_local_internal(
      grid_out_real, grid_out_imag, grid_in_real, grid_in_imag, fft_size[0],
      fft_size[1] * fft_size[2], fft_size[1] * fft_size[2]);
  // Transpose the data
  for (int index_0 = 0; index_0 < fft_size[0]; index_0++) {
    for (int index_1 = 0; index_1 < fft_size[1] * fft_size[2]; index_1++) {
      grid_out[index_1 * fft_size[0] + index_0] =
          CMPLX(grid_in_real[index_0 * fft_size[1] * fft_size[2] + index_1],
                grid_in_imag[index_0 * fft_size[1] * fft_size[2] + index_1]);
    }
  }

  clock_t end = clock();
  printf("Time Reorder input: %f\n", time_reorder_input);
  printf("Time Reorder output: %f\n", time_reorder_output);
  printf("Time 1: %f\n", time_1);
  printf("Time 2: %f\n", time_2);
  printf("Time 4: %f\n", time_4);
  printf("Time naive: %f\n", time_naive);
  printf("Total Time 3D BW %i %i %i: %f\n", fft_size[0], fft_size[1],
         fft_size[2], (double)(end - begin) / CLOCKS_PER_SEC);
  fflush(stdout);
}

// EOF
