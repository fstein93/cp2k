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

/*******************************************************************************
 * \brief Function to test the local FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_1d_local_low(const int fft_size, const int number_of_ffts,
                          const int transpose_rs, const int transpose_gs) {
  const int my_process = grid_mpi_comm_rank(grid_mpi_comm_world);

  int errors = 0;

  const double pi = acos(-1);

  double complex *input_array =
      calloc(fft_size * number_of_ffts, sizeof(double complex));
  double complex *output_array =
      calloc(fft_size * number_of_ffts, sizeof(double complex));

  double max_error = 0.0;
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

  if (max_error > 1.0e-12) {
    if (my_process == 0)
      printf("The fw 1D-FFT does not work correctly (%i %i): %f!\n", fft_size,
             number_of_ffts, max_error);
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

  free(input_array);
  free(output_array);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The bw 1D FFT does not work correctly (%i %i): %f!\n", fft_size,
             number_of_ffts, max_error);
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
int fft_test_2d_local_low(const int fft_size[2], const int number_of_ffts,
                          const int transpose_rs, const int transpose_gs) {
  const int my_process = grid_mpi_comm_rank(grid_mpi_comm_world);

  int errors = 0;

  const double pi = acos(-1);

  double complex *input_array = calloc(
      fft_size[0] * fft_size[1] * number_of_ffts, sizeof(double complex));
  double complex *output_array = calloc(
      fft_size[0] * fft_size[1] * number_of_ffts, sizeof(double complex));

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

  if (max_error > 1.0e-12) {
    if (my_process == 0)
      printf("The fw 2D-FFT does not work correctly (%i %i/%i): %f!\n",
             fft_size[0], fft_size[1], number_of_ffts, max_error);
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
            printf("Error %i %i %i: %f %f\n", index_1, index_2, number_of_fft,
                   cabs(my_value), cabs(ref_value));
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
            printf("Error %i %i %i: %f %f\n", index_1, index_2, number_of_fft,
                   cabs(my_value), cabs(ref_value));
          max_error = fmax(max_error, current_error);
        }
      }
    }
  }

  free(input_array);
  free(output_array);

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
int fft_test_3d_local_low(const int fft_size[3]) {
  const int my_process = grid_mpi_comm_rank(grid_mpi_comm_world);

  int errors = 0;

  const double pi = acos(-1);

  double complex *input_array =
      calloc(fft_size[0] * fft_size[1] * fft_size[2], sizeof(double complex));
  double complex *output_array =
      calloc(fft_size[0] * fft_size[1] * fft_size[2], sizeof(double complex));

  double max_error = 0.0;
  for (int mx = 0; mx < fft_size[0]; mx++) {
    for (int my = 0; my < fft_size[1]; my++) {
      for (int mz = 0; mz < fft_size[2]; mz++) {
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

  if (max_error > 1.0e-12) {
    if (my_process == 0)
      printf("The fw 3D-FFT does not work correctly (%i %i %i): %f!\n",
             fft_size[0], fft_size[1], fft_size[2], max_error);
    errors++;
  }

  max_error = 0.0;
  for (int mx = 0; mx < fft_size[0]; mx++) {
    for (int my = 0; my < fft_size[1]; my++) {
      for (int mz = 0; mz < fft_size[2]; mz++) {
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

  free(input_array);
  free(output_array);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The bw 3D-FFT does not work correctly (%i %i %i): %f!\n",
             fft_size[0], fft_size[1], fft_size[2], max_error);
    errors++;
  }

  if (errors == 0 && my_process == 0)
    printf("The 3D FFT does work correctly (%i %i %i)!\n", fft_size[0],
           fft_size[1], fft_size[2]);
  return errors;
}

/*******************************************************************************
 * \brief Function to test the local FFT backend (1-3D).
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_local() {
  int errors = 0;

  errors += fft_test_1d_local_low(16, 36, true, true);
  errors += fft_test_1d_local_low(18, 32, true, false);
  errors += fft_test_1d_local_low(20, 28, false, true);
  errors += fft_test_1d_local_low(12, 20, false, false);

  errors += fft_test_2d_local_low((const int[2]){10, 10}, 20, true, true);
  errors += fft_test_2d_local_low((const int[2]){16, 9}, 150, true, false);
  errors += fft_test_2d_local_low((const int[2]){7, 20}, 90, false, true);
  errors += fft_test_2d_local_low((const int[2]){12, 14}, 80, false, false);

  errors += fft_test_3d_local_low((const int[3]){8, 8, 8});
  errors += fft_test_3d_local_low((const int[3]){3, 4, 5});
  errors += fft_test_3d_local_low((const int[3]){4, 8, 2});
  errors += fft_test_3d_local_low((const int[3]){7, 5, 3});

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
