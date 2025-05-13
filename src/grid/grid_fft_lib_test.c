/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "grid_fft_lib_test.h"

#include "common/grid_common.h"
#include "common/grid_mpi.h"
#include "grid_fft_grid.h"
#include "grid_fft_grid_layout.h"
#include "grid_fft_lib.h"
#include "grid_fft_reorder.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*******************************************************************************
 * \brief Function to test the local FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_1d_local_low(const int fft_size, const int number_of_ffts,
                          const int transpose_rs, const int transpose_gs) {
  const int my_process = grid_mpi_comm_rank(grid_mpi_comm_world);

  int errors = 0;

  const double pi = acos(-1);

  double complex *input_array = NULL, *output_array = NULL;
  fft_allocate_complex(fft_size * number_of_ffts, &input_array);
  fft_allocate_complex(fft_size * number_of_ffts, &output_array);

  // Check the forward FFT
  if (transpose_rs) {
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      input_array[(number_of_fft % fft_size) * number_of_ffts + number_of_fft] =
          1.0;
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      input_array[(number_of_fft % fft_size) + number_of_fft * fft_size] = 1.0;
    }
  }

  fft_1d_fw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                  input_array, output_array);

  double max_error = 0.0;
  if (transpose_gs) {
#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts, pi)                         \
    reduction(max : max_error) collapse(2)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index = 0; index < fft_size; index++) {
        max_error =
            fmax(max_error,
                 cabs(output_array[number_of_fft + index * number_of_ffts] -
                      cexp(-2.0 * I * pi * (number_of_fft % fft_size) * index /
                           fft_size)));
      }
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts, pi)                         \
    reduction(max : max_error) collapse(2)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index = 0; index < fft_size; index++) {
        max_error = fmax(max_error,
                         cabs(output_array[number_of_fft * fft_size + index] -
                              cexp(-2.0 * I * pi * (number_of_fft % fft_size) *
                                   index / fft_size)));
      }
    }
  }
  fflush(stdout);

  if (max_error > 1.0e-12) {
    if (my_process == 0) {
      printf("The fw 1D-FFT does not work correctly (%i %i): %f!\n", fft_size,
             number_of_ffts, max_error);
      fflush(stdout);
    }
    errors++;
  }

  // Check the backward FFT
  memset(output_array, 0, fft_size * number_of_ffts * sizeof(double complex));

  if (transpose_gs) {
#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      output_array[number_of_fft + number_of_fft % fft_size * number_of_ffts] =
          1.0;
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      output_array[number_of_fft * fft_size + number_of_fft % fft_size] = 1.0;
    }
  }

  fft_1d_bw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                  output_array, input_array);

  max_error = 0.0;
  if (transpose_rs) {
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts, pi)                          \
    reduction(max : max_error) collapse(2)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index = 0; index < fft_size; index++) {
        max_error =
            fmax(max_error,
                 cabs(input_array[index * number_of_ffts + number_of_fft] -
                      cexp(2.0 * I * pi * (number_of_fft % fft_size) * index /
                           fft_size)));
      }
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts, pi)                          \
    reduction(max : max_error) collapse(2)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index = 0; index < fft_size; index++) {
        max_error = fmax(max_error,
                         cabs(input_array[index + number_of_fft * fft_size] -
                              cexp(2.0 * I * pi * (number_of_fft % fft_size) *
                                   index / fft_size)));
      }
    }
  }
  fflush(stdout);

  fft_free_complex(input_array);
  fft_free_complex(output_array);

  if (max_error > 1e-12) {
    if (my_process == 0) {
      printf("The bw 1D FFT does not work correctly (%i %i): %f!\n", fft_size,
             number_of_ffts, max_error);
      fflush(stdout);
    }
    errors++;
  }

  if (errors == 0 && my_process == 0)
    printf("The 1D FFT does work correctly (%i %i)!\n", fft_size,
           number_of_ffts);
  return errors;
}

/*******************************************************************************
 * \brief Function to test the local FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_1d_local_r2c_low(const int fft_size, const int number_of_ffts,
                              const int transpose_rs, const int transpose_gs) {
  const int my_process = grid_mpi_comm_rank(grid_mpi_comm_world);

  int errors = 0;

  const double pi = acos(-1);

  double *input_array = NULL;
  double complex *output_array = NULL;
  fft_allocate_double(2 * (fft_size / 2 + 1) * number_of_ffts, &input_array);
  fft_allocate_complex((fft_size / 2 + 1) * number_of_ffts + 4, &output_array);

  memset(input_array, 0, fft_size * number_of_ffts * sizeof(double));
  // Check the forward FFT
  if (transpose_rs) {
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      input_array[(number_of_fft % fft_size) * number_of_ffts + number_of_fft] =
          1.0;
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      input_array[(number_of_fft % fft_size) + number_of_fft * fft_size] = 1.0;
    }
  }

  fft_1d_fw_local_r2c(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                      input_array, output_array);

  double max_error = 0.0;
  if (transpose_gs) {
#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts, pi, my_process)             \
    reduction(max : max_error) collapse(2)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index = 0; index < fft_size / 2 + 1; index++) {
        const double complex my_value =
            output_array[number_of_fft + index * number_of_ffts];
        const double complex ref_value =
            cexp(-2.0 * I * pi * (number_of_fft % fft_size) * index / fft_size);
        const double current_error = cabs(my_value - ref_value);
        if (my_process == 0 && current_error > 1e-12)
          printf("Error %i %i / %i %i: (%f %f) (%f %f)\n", index, number_of_fft,
                 fft_size, number_of_ffts, creal(my_value), cimag(my_value),
                 creal(ref_value), cimag(ref_value));
        max_error = fmax(max_error, current_error);
      }
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts, pi, my_process)             \
    reduction(max : max_error) collapse(2)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index = 0; index < fft_size / 2 + 1; index++) {
        const double complex my_value =
            output_array[number_of_fft * (fft_size / 2 + 1) + index];
        const double complex ref_value =
            cexp(-2.0 * I * pi * (number_of_fft % fft_size) * index / fft_size);
        const double current_error = cabs(my_value - ref_value);
        if (my_process == 0 && current_error > 1e-12)
          printf("Error %i %i / %i %i: (%f %f) (%f %f)\n", index, number_of_fft,
                 fft_size, number_of_ffts, creal(my_value), cimag(my_value),
                 creal(ref_value), cimag(ref_value));
        max_error = fmax(max_error, current_error);
      }
    }
  }
  fflush(stdout);

  if (max_error > 1.0e-8) {
    if (my_process == 0) {
      printf("The fw R2C 1D-FFT does not work correctly (%i %i): %f!\n",
             fft_size, number_of_ffts, max_error);
      fflush(stdout);
    }
    errors++;
  }

  // Check the backward FFT
  memset(output_array, 0,
         (fft_size / 2 + 1) * number_of_ffts * sizeof(double complex));

  if (transpose_gs) {
#pragma omp parallel for default(none) collapse(2)                             \
    shared(output_array, fft_size, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index = 0; index < fft_size / 2 + 1; index++) {
        output_array[number_of_fft + index * number_of_ffts] =
            cexp(-2.0 * I * acos(-1) * index * number_of_fft / fft_size);
      }
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index = 0; index < fft_size / 2 + 1; index++) {
        output_array[number_of_fft * (fft_size / 2 + 1) + index] =
            cexp(-2.0 * I * acos(-1) * index * number_of_fft / fft_size);
      }
    }
  }

  fft_1d_bw_local_c2r(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                      output_array, input_array);

  max_error = 0.0;
  if (transpose_rs) {
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts, my_process)                  \
    reduction(max : max_error) collapse(2)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index = 0; index < fft_size; index++) {
        const double my_value =
            input_array[index * number_of_ffts + number_of_fft];
        const double ref_value =
            (number_of_fft % fft_size == index ? (double)fft_size : 0.0);
        const double current_error = fabs(my_value - ref_value);
        if (my_process == 0 && current_error > 1e-4)
          printf("ERROR %i %i / %i %i : %f %f\n", number_of_fft, index,
                 number_of_ffts, fft_size, my_value, ref_value);
        max_error = fmax(max_error, current_error);
      }
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts, my_process)                  \
    reduction(max : max_error) collapse(2)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index = 0; index < fft_size; index++) {
        const double my_value = input_array[index + number_of_fft * fft_size];
        const double ref_value =
            (number_of_fft % fft_size == index ? (double)fft_size : 0.0);
        const double current_error = fabs(my_value - ref_value);
        if (my_process == 0 && current_error > 1e-4)
          printf("ERROR %i %i / %i %i : %f %f\n", number_of_fft, index,
                 number_of_ffts, fft_size, my_value, ref_value);
        max_error = fmax(max_error, current_error);
      }
    }
  }
  fflush(stdout);

  if (max_error > 1e-8) {
    if (my_process == 0) {
      printf("The bw C2R-1D FFT does not work correctly (%i %i): %f!\n",
             fft_size, number_of_ffts, max_error);
      fflush(stdout);
    }
    errors++;
  }

  fft_free_double(input_array);
  fft_free_complex(output_array);

  if (errors == 0 && my_process == 0)
    printf("The 1D R/C FFT does work correctly (%i %i)!\n", fft_size,
           number_of_ffts);
  return errors;
}

/*******************************************************************************
 * \brief Function to test the local FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_2d_local_low(const int fft_size[2], const int number_of_ffts,
                          const int transpose_rs, const int transpose_gs) {
  const int my_process = grid_mpi_comm_rank(grid_mpi_comm_world);

  int errors = 0;

  const double pi = acos(-1);

  double complex *input_array = NULL, *output_array = NULL;
  fft_allocate_complex(fft_size[0] * fft_size[1] * number_of_ffts,
                       &input_array);
  fft_allocate_complex(fft_size[0] * fft_size[1] * number_of_ffts,
                       &output_array);

  double max_error = 0.0;
  // Check the forward FFT
  if (transpose_rs) {
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      input_array[number_of_fft % (fft_size[0] * fft_size[1]) * number_of_ffts +
                  number_of_fft] = 1.0;
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      input_array[number_of_fft % (fft_size[0] * fft_size[1]) +
                  number_of_fft * (fft_size[0] * fft_size[1])] = 1.0;
    }
  }

  fft_2d_fw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                  input_array, output_array);

  if (transpose_gs) {
#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts, pi)                         \
    reduction(max : max_error) collapse(3)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index_1 = 0; index_1 < fft_size[0]; index_1++) {
        for (int index_2 = 0; index_2 < fft_size[1]; index_2++) {
          const double complex my_value =
              output_array[number_of_fft +
                           (index_1 * fft_size[1] + index_2) * number_of_ffts];
          const double complex ref_value = cexp(
              -2.0 * I * pi *
              ((double)(number_of_fft / fft_size[1]) * index_1 / fft_size[0] +
               (double)(number_of_fft % fft_size[1]) * index_2 / fft_size[1]));
          double current_error = cabs(my_value - ref_value);
          max_error = fmax(max_error, current_error);
        }
      }
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts, pi)                         \
    reduction(max : max_error) collapse(3)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index_1 = 0; index_1 < fft_size[0]; index_1++) {
        for (int index_2 = 0; index_2 < fft_size[1]; index_2++) {
          const double complex my_value =
              output_array[number_of_fft * fft_size[0] * fft_size[1] +
                           index_1 * fft_size[1] + index_2];
          const double complex ref_value = cexp(
              -2.0 * I * pi *
              ((double)(number_of_fft / fft_size[1]) * index_1 / fft_size[0] +
               (double)(number_of_fft % fft_size[1]) * index_2 / fft_size[1]));
          double current_error = cabs(my_value - ref_value);
          max_error = fmax(max_error, current_error);
        }
      }
    }
  }
  fflush(stdout);

  if (max_error > 1.0e-12) {
    if (my_process == 0) {
      printf("The fw 2D-FFT does not work correctly (%i %i/%i): %f!\n",
             fft_size[0], fft_size[1], number_of_ffts, max_error);
      fflush(stdout);
    }
    errors++;
  }

  // Check the backward FFT
  memset(output_array, 0,
         fft_size[0] * fft_size[1] * number_of_ffts * sizeof(double complex));

  if (transpose_gs) {
#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      output_array[number_of_fft +
                   (number_of_fft % (fft_size[0] * fft_size[1])) *
                       number_of_ffts] = 1.0;
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      output_array[number_of_fft * fft_size[0] * fft_size[1] +
                   (number_of_fft % (fft_size[0] * fft_size[1]))] = 1.0;
    }
  }

  fft_2d_bw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                  output_array, input_array);

  max_error = 0.0;
  if (transpose_rs) {
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts, pi, my_process)              \
    reduction(max : max_error) collapse(3)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index_1 = 0; index_1 < fft_size[0]; index_1++) {
        for (int index_2 = 0; index_2 < fft_size[1]; index_2++) {
          const double complex my_value =
              input_array[(index_1 * fft_size[1] + index_2) * number_of_ffts +
                          number_of_fft];
          const double complex ref_value = cexp(
              2.0 * I * pi *
              ((double)(number_of_fft / fft_size[1]) * index_1 / fft_size[0] +
               (double)(number_of_fft % fft_size[1]) * index_2 / fft_size[1]));
          double current_error = cabs(my_value - ref_value);
          if (my_process == 0 && current_error > 1e-12)
            printf("Error %i %i %i: (%f %f) (%f %f)\n", index_1, index_2,
                   number_of_fft, creal(my_value), cimag(my_value),
                   creal(ref_value), cimag(ref_value));
          max_error = fmax(max_error, current_error);
        }
      }
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts, pi, my_process)              \
    reduction(max : max_error) collapse(3)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index_1 = 0; index_1 < fft_size[0]; index_1++) {
        for (int index_2 = 0; index_2 < fft_size[1]; index_2++) {
          const double complex my_value =
              input_array[index_1 * fft_size[1] + index_2 +
                          number_of_fft * fft_size[0] * fft_size[1]];
          const double complex ref_value = cexp(
              2.0 * I * pi *
              ((double)(number_of_fft / fft_size[1]) * index_1 / fft_size[0] +
               (double)(number_of_fft % fft_size[1]) * index_2 / fft_size[1]));
          double current_error = cabs(my_value - ref_value);
          if (my_process == 0 && current_error > 1e-12)
            printf("Error %i %i %i: (%f %f) (%f %f)\n", index_1, index_2,
                   number_of_fft, creal(my_value), cimag(my_value),
                   creal(ref_value), cimag(ref_value));
          max_error = fmax(max_error, current_error);
        }
      }
    }
  }
  fflush(stdout);

  fft_free_complex(input_array);
  fft_free_complex(output_array);

  if (max_error > 1e-12) {
    if (my_process == 0) {
      printf("The bw 2D-FFT does not work correctly (%i %i/%i): %f!\n",
             fft_size[0], fft_size[1], number_of_ffts, max_error);
      fflush(stdout);
    }
    errors++;
  }

  if (errors == 0 && my_process == 0)
    printf("The 2D FFT does work correctly (%i %i/%i)!\n", fft_size[0],
           fft_size[1], number_of_ffts);
  return errors;
}

/*******************************************************************************
 * \brief Function to test the local FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_2d_local_r2c_low(const int fft_size[2], const int number_of_ffts,
                              const int transpose_rs, const int transpose_gs) {
  const int my_process = grid_mpi_comm_rank(grid_mpi_comm_world);

  int errors = 0;

  const double pi = acos(-1);

  double *input_array = NULL;
  double complex *output_array = NULL;
  fft_allocate_double(2 * (fft_size[0] / 2 + 1) * fft_size[1] * number_of_ffts,
                      &input_array);
  fft_allocate_complex((fft_size[0] / 2 + 1) * fft_size[1] * number_of_ffts,
                       &output_array);

  double max_error = 0.0;
  // Check the forward FFT
  if (transpose_rs) {
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      input_array[number_of_fft % (fft_size[0] * fft_size[1]) * number_of_ffts +
                  number_of_fft] = 1.0;
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      input_array[number_of_fft % (fft_size[0] * fft_size[1]) +
                  number_of_fft * (fft_size[0] * fft_size[1])] = 1.0;
    }
  }

  fft_2d_fw_local_r2c(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                      input_array, output_array);

  if (transpose_gs) {
#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts, pi)                         \
    reduction(max : max_error) collapse(3)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index_1 = 0; index_1 < fft_size[0] / 2 + 1; index_1++) {
        for (int index_2 = 0; index_2 < fft_size[1]; index_2++) {
          const double complex my_value =
              output_array[number_of_fft +
                           (index_1 * fft_size[1] + index_2) * number_of_ffts];
          const double complex ref_value = cexp(
              -2.0 * I * pi *
              ((double)(number_of_fft / fft_size[1]) * index_1 / fft_size[0] +
               (double)(number_of_fft % fft_size[1]) * index_2 / fft_size[1]));
          double current_error = cabs(my_value - ref_value);
          max_error = fmax(max_error, current_error);
        }
      }
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts, pi)                         \
    reduction(max : max_error) collapse(3)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index_1 = 0; index_1 < fft_size[0] / 2 + 1; index_1++) {
        for (int index_2 = 0; index_2 < fft_size[1]; index_2++) {
          const double complex my_value =
              output_array[number_of_fft * (fft_size[0] / 2 + 1) * fft_size[1] +
                           index_1 * fft_size[1] + index_2];
          const double complex ref_value = cexp(
              -2.0 * I * pi *
              ((double)(number_of_fft / fft_size[1]) * index_1 / fft_size[0] +
               (double)(number_of_fft % fft_size[1]) * index_2 / fft_size[1]));
          double current_error = cabs(my_value - ref_value);
          max_error = fmax(max_error, current_error);
        }
      }
    }
  }
  fflush(stdout);

  if (max_error > 1.0e-12) {
    if (my_process == 0) {
      printf("The fw R2C 2D-FFT does not work correctly (%i %i/%i): %f!\n",
             fft_size[0], fft_size[1], number_of_ffts, max_error);
      fflush(stdout);
    }
    errors++;
  }

  // Check the backward FFT
  memset(output_array, 0,
         (fft_size[0] / 2 + 1) * fft_size[1] * number_of_ffts *
             sizeof(double complex));

  if (transpose_gs) {
#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts, pi)                         \
    reduction(max : max_error) collapse(3)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index_1 = 0; index_1 < fft_size[0] / 2 + 1; index_1++) {
        for (int index_2 = 0; index_2 < fft_size[1]; index_2++) {
          output_array[number_of_fft +
                       (index_1 * fft_size[1] + index_2) * number_of_ffts] =
              cexp(-2.0 * I * pi *
                   ((double)(number_of_fft / fft_size[1]) * index_1 /
                        fft_size[0] +
                    (double)(number_of_fft % fft_size[1]) * index_2 /
                        fft_size[1]));
        }
      }
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts, pi)                         \
    reduction(max : max_error) collapse(3)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index_1 = 0; index_1 < fft_size[0] / 2 + 1; index_1++) {
        for (int index_2 = 0; index_2 < fft_size[1]; index_2++) {
          output_array[number_of_fft * (fft_size[0] / 2 + 1) * fft_size[1] +
                       index_1 * fft_size[1] + index_2] =
              cexp(-2.0 * I * pi *
                   ((double)(number_of_fft / fft_size[1]) * index_1 /
                        fft_size[0] +
                    (double)(number_of_fft % fft_size[1]) * index_2 /
                        fft_size[1]));
        }
      }
    }
  }

  fft_2d_bw_local_c2r(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                      output_array, input_array);

  max_error = 0.0;
  if (transpose_rs) {
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts, pi, my_process)              \
    reduction(max : max_error) collapse(3)
    for (int index_1 = 0; index_1 < fft_size[0]; index_1++) {
      for (int index_2 = 0; index_2 < fft_size[1]; index_2++) {
        for (int number_of_fft = 0; number_of_fft < number_of_ffts;
             number_of_fft++) {
          const double my_value =
              input_array[(index_1 * fft_size[1] + index_2) * number_of_ffts +
                          number_of_fft];
          const double ref_value =
              index_1 == number_of_fft / fft_size[1] &&
                      index_2 == number_of_fft % fft_size[1]
                  ? (double)(fft_size[0] * fft_size[1])
                  : 0.0;
          const double current_error = fabs(my_value - ref_value);
          if (my_process == 0 && current_error > 1e-6)
            printf("Error %i %i %i: %f %f\n", index_1, index_2, number_of_fft,
                   my_value, ref_value);
          max_error = fmax(max_error, current_error);
        }
      }
    }
  } else {
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts, pi, my_process)              \
    reduction(max : max_error) collapse(3)
    for (int number_of_fft = 0; number_of_fft < number_of_ffts;
         number_of_fft++) {
      for (int index_1 = 0; index_1 < fft_size[0]; index_1++) {
        for (int index_2 = 0; index_2 < fft_size[1]; index_2++) {
          const double my_value =
              input_array[index_1 * fft_size[1] + index_2 +
                          number_of_fft * fft_size[0] * fft_size[1]];
          const double ref_value =
              index_1 == number_of_fft / fft_size[1] &&
                      index_2 == number_of_fft % fft_size[1]
                  ? (double)(fft_size[0] * fft_size[1])
                  : 0.0;
          double current_error = fabs(my_value - ref_value);
          if (my_process == 0 && current_error > 1e-8)
            printf("Error %i %i %i: %f %f\n", index_1, index_2, number_of_fft,
                   my_value, ref_value);
          max_error = fmax(max_error, current_error);
        }
      }
    }
  }
  fflush(stdout);

  fft_free_double(input_array);
  fft_free_complex(output_array);

  if (max_error > 1e-8) {
    if (my_process == 0) {
      printf("The bw C2R 2D-FFT does not work correctly (%i %i/%i): %f!\n",
             fft_size[0], fft_size[1], number_of_ffts, max_error);
      fflush(stdout);
    }
    errors++;
  }

  if (errors == 0 && my_process == 0)
    printf("The 2D R2C/C2R FFT does work correctly (%i %i/%i)!\n", fft_size[0],
           fft_size[1], number_of_ffts);
  return errors;
}

/*******************************************************************************
 * \brief Function to test the local FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_3d_local_low(const int fft_size[3], const int test_every) {
  const int my_process = grid_mpi_comm_rank(grid_mpi_comm_world);

  int errors = 0;

  const double pi = acos(-1);

  double complex *input_array = NULL, *output_array = NULL;
  fft_allocate_complex(fft_size[0] * fft_size[1] * fft_size[2], &input_array);
  fft_allocate_complex(fft_size[0] * fft_size[1] * fft_size[2], &output_array);

  double max_error = 0.0;
  int number_of_tests = 0;
  for (int mx = 0; mx < fft_size[0]; mx++) {
    for (int my = 0; my < fft_size[1]; my++) {
      for (int mz = 0; mz < fft_size[2]; mz++) {
        if (test_every > 0 && number_of_tests % test_every != 0) {
          number_of_tests++;
          continue;
        }
        number_of_tests++;
        memset(input_array, 0,
               fft_size[0] * fft_size[1] * fft_size[2] *
                   sizeof(double complex));
        input_array[mz * fft_size[0] * fft_size[1] + my * fft_size[0] + mx] =
            1.0;
        fft_3d_fw_local(fft_size, input_array, output_array);

#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, pi, mx, my, mz) reduction(max : max_error)  \
    collapse(3)
        for (int nx = 0; nx < fft_size[0]; nx++) {
          for (int ny = 0; ny < fft_size[1]; ny++) {
            for (int nz = 0; nz < fft_size[2]; nz++) {
              const double complex my_value =
                  output_array[nz * fft_size[0] * fft_size[1] +
                               ny * fft_size[0] + nx];
              const double complex ref_value =
                  cexp(-2.0 * I * pi *
                       (((double)mx) * nx / fft_size[0] +
                        ((double)my) * ny / fft_size[1] +
                        ((double)mz) * nz / fft_size[2]));
              double current_error = cabs(my_value - ref_value);
              max_error = fmax(max_error, current_error);
            }
          }
        }
      }
    }
  }
  fflush(stdout);

  if (max_error > 1.0e-12) {
    if (my_process == 0) {
      printf("The fw 3D-FFT does not work correctly (%i %i %i): %f!\n",
             fft_size[0], fft_size[1], fft_size[2], max_error);
      fflush(stdout);
    }
    errors++;
  }

  max_error = 0.0;
  number_of_tests = 0;
  for (int mx = 0; mx < fft_size[0]; mx++) {
    for (int my = 0; my < fft_size[1]; my++) {
      for (int mz = 0; mz < fft_size[2]; mz++) {
        if (test_every > 0 && number_of_tests % test_every != 0) {
          number_of_tests++;
          continue;
        }
        number_of_tests++;
        memset(output_array, 0,
               fft_size[0] * fft_size[1] * fft_size[2] *
                   sizeof(double complex));
        output_array[mz * fft_size[0] * fft_size[1] + my * fft_size[0] + mx] =
            1.0;
        fft_3d_bw_local(fft_size, output_array, input_array);

#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, pi, mx, my, mz) reduction(max : max_error)   \
    collapse(3)
        for (int nx = 0; nx < fft_size[0]; nx++) {
          for (int ny = 0; ny < fft_size[1]; ny++) {
            for (int nz = 0; nz < fft_size[2]; nz++) {
              const double complex my_value =
                  input_array[nz * fft_size[0] * fft_size[1] +
                              ny * fft_size[0] + nx];
              const double complex ref_value =
                  cexp(2.0 * I * pi *
                       (((double)mx) * nx / fft_size[0] +
                        ((double)my) * ny / fft_size[1] +
                        ((double)mz) * nz / fft_size[2]));
              double current_error = cabs(my_value - ref_value);
              max_error = fmax(max_error, current_error);
            }
          }
        }
      }
    }
  }
  fflush(stdout);

  fft_free_complex(input_array);
  fft_free_complex(output_array);

  if (max_error > 1e-12) {
    if (my_process == 0) {
      printf("The bw 3D-FFT does not work correctly (%i %i %i): %f!\n",
             fft_size[0], fft_size[1], fft_size[2], max_error);
      fflush(stdout);
    }
    errors++;
  }

  if (errors == 0 && my_process == 0)
    printf("The 3D FFT does work correctly (%i %i %i)!\n", fft_size[0],
           fft_size[1], fft_size[2]);
  return errors;
}

/*******************************************************************************
 * \brief Function to test the local FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_3d_local_r2c_low(const int fft_size[3], const int test_every) {
  const int my_process = grid_mpi_comm_rank(grid_mpi_comm_world);

  int errors = 0;

  const double pi = acos(-1);

  double *input_array = NULL;
  double complex *output_array = NULL;
  fft_allocate_double(2 * fft_size[0] * fft_size[1] * (fft_size[2] / 2 + 1),
                      &input_array);
  fft_allocate_complex(fft_size[0] * fft_size[1] * (fft_size[2] / 2 + 1),
                       &output_array);

  double max_error = 0.0;
  int number_of_tests = 0;
  for (int mx = 0; mx < fft_size[0]; mx++) {
    for (int my = 0; my < fft_size[1]; my++) {
      for (int mz = 0; mz < fft_size[2]; mz++) {
        if (test_every > 0 && number_of_tests % test_every != 0) {
          number_of_tests++;
          continue;
        }
        number_of_tests++;
        memset(input_array, 0,
               fft_size[0] * fft_size[1] * fft_size[2] *
                   sizeof(double complex));
        input_array[mz * fft_size[0] * fft_size[1] + my * fft_size[0] + mx] =
            1.0;
        fft_3d_fw_local_r2c(fft_size, input_array, output_array);

#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, pi, mx, my, mz, my_process)                 \
    reduction(max : max_error) collapse(3)
        for (int nz = 0; nz < fft_size[2] / 2 + 1; nz++) {
          for (int ny = 0; ny < fft_size[1]; ny++) {
            for (int nx = 0; nx < fft_size[0]; nx++) {
              const double complex my_value =
                  output_array[nz * fft_size[0] * fft_size[1] +
                               ny * fft_size[0] + nx];
              const double complex ref_value =
                  cexp(-2.0 * I * pi *
                       (((double)mx) * nx / fft_size[0] +
                        ((double)my) * ny / fft_size[1] +
                        ((double)mz) * nz / fft_size[2]));
              double current_error = cabs(my_value - ref_value);
              if (my_process == 0 && current_error > 1e-6) {
                printf("ERROR %i %i %i/%i %i %i: (%f %f) (%f %f)\n", nx, ny, nz,
                       mx, my, mz, creal(my_value), cimag(my_value),
                       creal(ref_value), cimag(ref_value));
              }
              max_error = fmax(max_error, current_error);
            }
          }
        }
      }
    }
  }
  fflush(stdout);

  if (max_error > 1.0e-6) {
    if (my_process == 0) {
      printf("The fw R2C 3D-FFT does not work correctly (%i %i %i): %f!\n",
             fft_size[0], fft_size[1], fft_size[2], max_error);
      fflush(stdout);
    }
    errors++;
  }

  max_error = 0.0;
  number_of_tests = 0;
  for (int mx = 0; mx < fft_size[0]; mx++) {
    for (int my = 0; my < fft_size[1]; my++) {
      for (int mz = 0; mz < fft_size[2]; mz++) {
        if (test_every > 0 && number_of_tests % test_every != 0) {
          number_of_tests++;
          continue;
        }
        number_of_tests++;

#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, pi, mx, my, mz, my_process)                 \
    reduction(max : max_error) collapse(3)
        for (int nz = 0; nz < fft_size[2] / 2 + 1; nz++) {
          for (int ny = 0; ny < fft_size[1]; ny++) {
            for (int nx = 0; nx < fft_size[0]; nx++) {
              output_array[nz * fft_size[0] * fft_size[1] + ny * fft_size[0] +
                           nx] = cexp(-2.0 * I * pi *
                                      (((double)mx) * nx / fft_size[0] +
                                       ((double)my) * ny / fft_size[1] +
                                       ((double)mz) * nz / fft_size[2]));
            }
          }
        }
        fft_3d_bw_local_c2r(fft_size, output_array, input_array);

#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, pi, mx, my, mz, my_process)                  \
    reduction(max : max_error) collapse(3)
        for (int nx = 0; nx < fft_size[0]; nx++) {
          for (int ny = 0; ny < fft_size[1]; ny++) {
            for (int nz = 0; nz < fft_size[2]; nz++) {
              const double my_value =
                  input_array[nz * fft_size[0] * fft_size[1] +
                              ny * fft_size[0] + nx];
              const double ref_value = (nx == mx && ny == my && nz == mz)
                                           ? (double)product3(fft_size)
                                           : 0.0;
              double current_error = fabs(my_value - ref_value);
              if (my_process == 0 && current_error > 1e-6) {
                printf("ERROR %i %i %i/%i %i %i: %f %f\n", nx, ny, nz, mx, my,
                       mz, my_value, ref_value);
              }
              max_error = fmax(max_error, current_error);
            }
          }
        }
      }
    }
  }
  fflush(stdout);

  fft_free_double(input_array);
  fft_free_complex(output_array);

  if (max_error > 1e-6) {
    if (my_process == 0) {
      printf("The bw 3D C2R FFT does not work correctly (%i %i %i): %f!\n",
             fft_size[0], fft_size[1], fft_size[2], max_error);
      fflush(stdout);
    }
    errors++;
  }

  if (errors == 0 && my_process == 0)
    printf("The 3D R2C/C2R FFT does work correctly (%i %i %i)!\n", fft_size[0],
           fft_size[1], fft_size[2]);
  return errors;
}

/*******************************************************************************
 * \brief Function to test the local FFT backend (1-3D).
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_local() {
  int errors = 0;

  clock_t begin = clock();
  errors += fft_test_1d_local_low(15, 26, true, true);
  errors += fft_test_1d_local_low(18, 22, true, false);
  errors += fft_test_1d_local_low(20, 28, false, true);
  errors += fft_test_1d_local_low(12, 14, false, false);
  // A larger test
  errors += fft_test_1d_local_low(144, 14, true, false);

  errors += fft_test_1d_local_r2c_low(4, 4, false, false);
  errors += fft_test_1d_local_r2c_low(15, 22, true, false);
  errors += fft_test_1d_local_r2c_low(12, 14, false, false);
  errors += fft_test_1d_local_r2c_low(20, 28, false, true);
  errors += fft_test_1d_local_r2c_low(16, 26, true, true);
  // A larger test
  errors += fft_test_1d_local_r2c_low(280, 12, true, false);

  errors += fft_test_2d_local_low((const int[2]){10, 10}, 20, true, true);
  errors += fft_test_2d_local_low((const int[2]){15, 9}, 90, true, false);
  errors += fft_test_2d_local_low((const int[2]){7, 20}, 70, false, true);
  errors += fft_test_2d_local_low((const int[2]){12, 14}, 50, false, false);
  // A larger test
  errors += fft_test_2d_local_low((const int[2]){96, 96}, 10, false, false);

  errors += fft_test_2d_local_r2c_low((const int[2]){10, 10}, 20, true, true);
  errors += fft_test_2d_local_r2c_low((const int[2]){15, 9}, 90, true, false);
  errors += fft_test_2d_local_r2c_low((const int[2]){7, 20}, 70, false, true);
  errors += fft_test_2d_local_r2c_low((const int[2]){12, 14}, 50, false, false);
  // A larger test
  errors += fft_test_2d_local_r2c_low((const int[2]){96, 96}, 10, false, false);

  // Reduce tests to ca 10 per set
  errors += fft_test_3d_local_low((const int[3]){8, 8, 8}, 23);
  errors += fft_test_3d_local_low((const int[3]){3, 4, 5}, 13);
  errors += fft_test_3d_local_low((const int[3]){4, 8, 2}, 11);
  errors += fft_test_3d_local_low((const int[3]){7, 5, 3}, 17);
  // A larger test
  errors += fft_test_3d_local_low((const int[3]){72, 72, 84}, 54321);

  // Reduce tests to ca 10 per set
  // errors += fft_test_3d_local_r2c_low((const int[3]){8, 8, 8}, 23);
  // errors += fft_test_3d_local_r2c_low((const int[3]){3, 4, 5}, 13);
  // errors += fft_test_3d_local_r2c_low((const int[3]){4, 8, 2}, 11);
  // errors += fft_test_3d_local_r2c_low((const int[3]){7, 5, 3}, 17);
  // A larger test
  // errors += fft_test_3d_local_r2c_low((const int[3]){72, 72, 84}, 54321);
  clock_t end = clock();
  printf("Time to test local FFTs with planning: %f\n",
         (double)(end - begin) / CLOCKS_PER_SEC);

  begin = clock();
  errors += fft_test_1d_local_low(15, 26, true, true);
  errors += fft_test_1d_local_low(18, 22, true, false);
  errors += fft_test_1d_local_low(20, 28, false, true);
  errors += fft_test_1d_local_low(12, 14, false, false);
  // A larger test
  errors += fft_test_1d_local_low(144, 14, true, false);

  errors += fft_test_1d_local_r2c_low(4, 4, false, false);
  errors += fft_test_1d_local_r2c_low(15, 22, true, false);
  errors += fft_test_1d_local_r2c_low(12, 14, false, false);
  errors += fft_test_1d_local_r2c_low(18, 28, false, true);
  errors += fft_test_1d_local_r2c_low(16, 26, true, true);
  // A larger test
  errors += fft_test_1d_local_r2c_low(280, 12, true, false);

  errors += fft_test_2d_local_low((const int[2]){10, 10}, 20, true, true);
  errors += fft_test_2d_local_low((const int[2]){15, 9}, 90, true, false);
  errors += fft_test_2d_local_low((const int[2]){7, 20}, 70, false, true);
  errors += fft_test_2d_local_low((const int[2]){12, 14}, 50, false, false);
  // A larger test
  errors += fft_test_2d_local_low((const int[2]){96, 96}, 10, false, false);

  errors += fft_test_2d_local_r2c_low((const int[2]){10, 10}, 20, true, true);
  errors += fft_test_2d_local_r2c_low((const int[2]){15, 9}, 90, true, false);
  errors += fft_test_2d_local_r2c_low((const int[2]){7, 20}, 70, false, true);
  errors += fft_test_2d_local_r2c_low((const int[2]){12, 14}, 50, false, false);
  // A larger test
  errors += fft_test_2d_local_r2c_low((const int[2]){96, 96}, 10, false, false);

  // Reduce tests to ca 10 per set
  errors += fft_test_3d_local_low((const int[3]){8, 8, 8}, 23);
  errors += fft_test_3d_local_low((const int[3]){3, 4, 5}, 13);
  errors += fft_test_3d_local_low((const int[3]){4, 8, 2}, 11);
  errors += fft_test_3d_local_low((const int[3]){7, 5, 3}, 17);
  // A larger test
  errors += fft_test_3d_local_low((const int[3]){72, 72, 84}, 54321);

  // Reduce tests to ca 10 per set
  // errors += fft_test_3d_local_r2c_low((const int[3]){8, 8, 8}, 23);
  // errors += fft_test_3d_local_r2c_low((const int[3]){3, 4, 5}, 13);
  // errors += fft_test_3d_local_r2c_low((const int[3]){4, 8, 2}, 11);
  // errors += fft_test_3d_local_r2c_low((const int[3]){7, 5, 3}, 17);
  // A larger test
  // errors += fft_test_3d_local_r2c_low((const int[3]){72, 72, 84}, 54321);
  end = clock();
  printf("Time to test local FFTs without planning: %f\n",
         (double)(end - begin) / CLOCKS_PER_SEC);

  return errors;
}

/*******************************************************************************
 * \brief Function to test the local FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_2d_distributed_low(const int fft_size[2],
                                const int number_of_ffts) {
  const grid_mpi_comm comm = grid_mpi_comm_world;
  const int my_process = grid_mpi_comm_rank(comm);

  int errors = 0;

  const double pi = acos(-1);

  int local_n0, local_n0_start;
  int local_n1, local_n1_start;
  const int buffer_size =
      fft_2d_distributed_sizes(fft_size, number_of_ffts, comm, &local_n0,
                               &local_n0_start, &local_n1, &local_n1_start);

  double complex *input_array = NULL, *output_array = NULL;
  fft_allocate_complex(buffer_size, &input_array);
  fft_allocate_complex(buffer_size, &output_array);

  double max_error = 0.0;
  // Check the forward FFT
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts, local_n0, local_n0_start)
  for (int number_of_fft = 0; number_of_fft < number_of_ffts; number_of_fft++) {
    const int index_0 = number_of_fft / fft_size[1];
    const int index_1 = number_of_fft % fft_size[1];
    if (index_0 >= local_n0_start && index_0 < local_n0_start + local_n0) {
      input_array[number_of_fft +
                  ((index_0 - local_n0_start) * fft_size[1] + index_1) *
                      number_of_ffts] = 1.0;
    }
  }

  fft_2d_fw_distributed(fft_size, number_of_ffts, comm, input_array,
                        output_array);

#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts, pi, local_n1,               \
               local_n1_start) reduction(max : max_error) collapse(3)
  for (int number_of_fft = 0; number_of_fft < number_of_ffts; number_of_fft++) {
    for (int index_0 = 0; index_0 < fft_size[0]; index_0++) {
      for (int index_1 = 0; index_1 < local_n1; index_1++) {
        const double complex my_value =
            output_array[number_of_fft +
                         (index_1 * fft_size[0] + index_0) * number_of_ffts];
        const double complex ref_value = cexp(
            -2.0 * I * pi *
            ((double)(number_of_fft / fft_size[1]) * index_0 / fft_size[0] +
             (double)(number_of_fft % fft_size[1]) *
                 (index_1 + local_n1_start) / fft_size[1]));
        double current_error = cabs(my_value - ref_value);
        if (current_error > 1e-12)
          printf("Error %i %i %i: %f %f\n", index_0, index_1, number_of_fft,
                 cabs(my_value), cabs(ref_value));
        max_error = fmax(max_error, current_error);
      }
    }
  }
  fflush(stdout);
  grid_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1.0e-12) {
    if (my_process == 0)
      printf("The distributed fw 2D-FFT does not work correctly (%i %i/%i): "
             "%f!\n",
             fft_size[0], fft_size[1], number_of_ffts, max_error);
    errors++;
  }

  // Check the backward FFT
  memset(input_array, 0, buffer_size * sizeof(double complex));
  memset(output_array, 0, buffer_size * sizeof(double complex));
  // Check the forward FFT
#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, number_of_ffts, local_n1, local_n1_start)
  for (int number_of_fft = 0; number_of_fft < number_of_ffts; number_of_fft++) {
    const int index_0 = number_of_fft / fft_size[1];
    const int index_1 = number_of_fft % fft_size[1];
    if (index_1 >= local_n1_start && index_1 < local_n1_start + local_n1) {
      input_array[number_of_fft +
                  ((index_1 - local_n1_start) * fft_size[0] + index_0) *
                      number_of_ffts] = 1.0;
    }
  }

  fft_2d_bw_distributed(fft_size, number_of_ffts, comm, input_array,
                        output_array);

#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, number_of_ffts, pi, local_n0,               \
               local_n0_start) reduction(max : max_error) collapse(3)
  for (int number_of_fft = 0; number_of_fft < number_of_ffts; number_of_fft++) {
    for (int index_0 = 0; index_0 < local_n0; index_0++) {
      for (int index_1 = 0; index_1 < fft_size[1]; index_1++) {
        const double complex my_value =
            output_array[number_of_fft +
                         (index_0 * fft_size[1] + index_1) * number_of_ffts];
        const double complex ref_value = cexp(
            2.0 * I * pi *
            ((double)(number_of_fft / fft_size[1]) *
                 (index_0 + local_n0_start) / fft_size[0] +
             (double)(number_of_fft % fft_size[1]) * index_1 / fft_size[1]));
        double current_error = cabs(my_value - ref_value);
        if (current_error > 1e-12)
          printf("Error %i %i %i: %f %f\n", index_0, index_1, number_of_fft,
                 cabs(my_value), cabs(ref_value));
        max_error = fmax(max_error, current_error);
      }
    }
  }
  fflush(stdout);
  grid_mpi_max_double(&max_error, 1, comm);
  fft_free_complex(input_array);
  fft_free_complex(output_array);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The bw 2D-FFT does not work correctly (%i %i/%i): %f!\n",
             fft_size[0], fft_size[1], number_of_ffts, max_error);
    errors++;
  }

  if (errors == 0 && my_process == 0)
    printf("The 2D FFT does work correctly (%i %i/%i)!\n", fft_size[0],
           fft_size[1], number_of_ffts);
  return errors;
}

/*******************************************************************************
 * \brief Function to test the local FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_3d_distributed_low(const int fft_size[3], const int test_every) {
  const grid_mpi_comm comm = grid_mpi_comm_world;
  const int my_process = grid_mpi_comm_rank(comm);

  int errors = 0;

  const double pi = acos(-1);

  int local_n2, local_n2_start;
  int local_n1, local_n1_start;
  const int buffer_size = fft_3d_distributed_sizes(
      fft_size, comm, &local_n2, &local_n2_start, &local_n1, &local_n1_start);

  double complex *input_array = NULL, *output_array = NULL;
  fft_allocate_complex(buffer_size, &input_array);
  fft_allocate_complex(buffer_size, &output_array);

  double max_error = 0.0;
  int number_of_tests = 0;
  for (int mx = 0; mx < fft_size[0]; mx++) {
    for (int my = 0; my < fft_size[1]; my++) {
      for (int mz = 0; mz < fft_size[2]; mz++) {
        if (test_every > 0 && number_of_tests % test_every != 0) {
          number_of_tests++;
          continue;
        }
        number_of_tests++;
        memset(input_array, 0, buffer_size * sizeof(double complex));
        if (mz >= local_n2_start && mz < local_n2_start + local_n2)
          input_array[(mz - local_n2_start) * fft_size[0] * fft_size[1] +
                      my * fft_size[0] + mx] = 1.0;
        fft_3d_fw_distributed(fft_size, comm, input_array, output_array);

#pragma omp parallel for default(none)                                         \
    shared(output_array, fft_size, pi, mx, my, mz, local_n1, local_n1_start)   \
    reduction(max : max_error) collapse(3)
        for (int nx = 0; nx < fft_size[0]; nx++) {
          for (int ny = 0; ny < local_n1; ny++) {
            for (int nz = 0; nz < fft_size[2]; nz++) {
              const double complex my_value =
                  output_array[ny * fft_size[0] * fft_size[2] +
                               nz * fft_size[0] + nx];
              const double complex ref_value =
                  cexp(-2.0 * I * pi *
                       (((double)mx) * nx / fft_size[0] +
                        ((double)my) * (ny + local_n1_start) / fft_size[1] +
                        ((double)mz) * nz / fft_size[2]));
              double current_error = cabs(my_value - ref_value);
              if (current_error > 1e-12) {
                printf("Error %i %i %i/ %i %i %i: (%f %f) (%f %f)\n", nx, ny,
                       nz, mx, my, mz, creal(my_value), cimag(my_value),
                       creal(ref_value), cimag(ref_value));
              }
              max_error = fmax(max_error, current_error);
            }
          }
        }
      }
    }
  }
  fflush(stdout);
  grid_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1.0e-12) {
    if (my_process == 0)
      printf("The distributed fw 3D-FFT does not work correctly (%i %i %i): "
             "%f!\n",
             fft_size[0], fft_size[1], fft_size[2], max_error);
    errors++;
  }

  max_error = 0.0;
  number_of_tests = 0;
  for (int mx = 0; mx < fft_size[0]; mx++) {
    for (int my = 0; my < fft_size[1]; my++) {
      for (int mz = 0; mz < fft_size[2]; mz++) {
        if (test_every > 0 && number_of_tests % test_every != 0) {
          number_of_tests++;
          continue;
        }
        number_of_tests++;
        memset(output_array, 0, buffer_size * sizeof(double complex));
        if (my >= local_n1_start && my < local_n1_start + local_n1)
          output_array[(my - local_n1_start) * fft_size[0] * fft_size[2] +
                       mz * fft_size[0] + mx] = 1.0;

        fft_3d_bw_distributed(fft_size, comm, output_array, input_array);

#pragma omp parallel for default(none)                                         \
    shared(input_array, fft_size, pi, mx, my, mz, local_n2, local_n2_start)    \
    reduction(max : max_error) collapse(3)
        for (int nx = 0; nx < fft_size[0]; nx++) {
          for (int ny = 0; ny < fft_size[1]; ny++) {
            for (int nz = 0; nz < local_n2; nz++) {
              const double complex my_value =
                  input_array[nz * fft_size[0] * fft_size[1] +
                              ny * fft_size[0] + nx];
              const double complex ref_value =
                  cexp(2.0 * I * pi *
                       (((double)mx) * nx / fft_size[0] +
                        ((double)my) * ny / fft_size[1] +
                        ((double)mz) * (nz + local_n2_start) / fft_size[2]));
              double current_error = cabs(my_value - ref_value);
              if (current_error > 1e-12) {
                printf("Error %i %i %i/ %i %i %i: (%f %f) (%f %f)\n", nx, ny,
                       nz, mx, my, mz, creal(my_value), cimag(my_value),
                       creal(ref_value), cimag(ref_value));
              }
              max_error = fmax(max_error, current_error);
            }
          }
        }
      }
    }
  }
  fflush(stdout);
  grid_mpi_max_double(&max_error, 1, comm);

  fft_free_complex(input_array);
  fft_free_complex(output_array);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The distributed bw 3D-FFT does not work correctly (%i %i %i): "
             "%f!\n",
             fft_size[0], fft_size[1], fft_size[2], max_error);
    errors++;
  }

  if (errors == 0 && my_process == 0)
    printf("The distributed 3D FFT does work correctly (%i %i %i)!\n",
           fft_size[0], fft_size[1], fft_size[2]);
  return errors;
}

/*******************************************************************************
 * \brief Function to test the distributed FFT backend (2-3D, 1D not used).
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_distributed() {
  int errors = 0;

  if (!fft_lib_use_mpi()) {
    printf("Skipped testing the distributed FFT backend!\n");
    return 0;
  }

  errors += fft_test_2d_distributed_low((const int[2]){10, 10}, 19);
  errors += fft_test_2d_distributed_low((const int[2]){16, 9}, 51);
  errors += fft_test_2d_distributed_low((const int[2]){7, 20}, 37);
  errors += fft_test_2d_distributed_low((const int[2]){12, 14}, 23);

  errors += fft_test_3d_distributed_low((const int[3]){8, 8, 8}, 19);
  errors += fft_test_3d_distributed_low((const int[3]){3, 4, 5}, 13);
  errors += fft_test_3d_distributed_low((const int[3]){4, 8, 2}, 17);
  errors += fft_test_3d_distributed_low((const int[3]){7, 5, 3}, 11);

  return errors;
}

/*******************************************************************************
 * \brief Function to test the local transposition operation.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_transpose() {
  const int my_process = grid_mpi_comm_rank(grid_mpi_comm_world);
  // Check a few fft sizes
  const int fft_sizes[2] = {16, 18};

  int max_size = fft_sizes[0] * fft_sizes[1];

  double complex *input_array = calloc(max_size, sizeof(double complex));
  double complex *output_array = calloc(max_size, sizeof(double complex));

#pragma omp parallel for default(none) shared(input_array, fft_sizes)          \
    collapse(2)
  for (int index_1 = 0; index_1 < fft_sizes[0]; index_1++) {
    for (int index_2 = 0; index_2 < fft_sizes[1]; index_2++) {
      input_array[index_1 * fft_sizes[1] + index_2] =
          1.0 * index_1 - index_2 * I;
    }
  }

  transpose_local(input_array, output_array, fft_sizes[1], fft_sizes[0]);

  double error = 0.0;

#pragma omp parallel for default(none) shared(output_array, fft_sizes)         \
    reduction(max : error) collapse(2)
  for (int index_1 = 0; index_1 < fft_sizes[0]; index_1++) {
    for (int index_2 = 0; index_2 < fft_sizes[1]; index_2++) {
      error = fmax(error, cabs(output_array[index_2 * fft_sizes[0] + index_1] -
                               (1.0 * index_1 - index_2 * I)));
    }
  }

  free(input_array);
  free(output_array);

  if (error > 1e-12) {
    if (my_process == 0)
      printf("The low-level transpose does not work correctly: %f!\n", error);
    return 1;
  } else {
    if (my_process == 0)
      printf("The local transpose does work correctly!\n");
    return 0;
  }
}

// EOF
