/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "fft_grid_layout.h"
#include "fft_driver.h"
#include "fft_lib.h"
#include "fft_lib_fftw.h"
#include "fft_utils.h"

#include "../base/cp_assert.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int current_grid_id = 1;

// Could be reformulated with Lapack or calculated
// For orthorhombic cells, this is at the order of
// 3*eps(multiplication)+6*eps(addition) For non-orthorhombic cells, this
// depends on the cell shape
const double max_rel_error_for_equivalence_g_squared = 1e-12;

typedef struct {
  double value;
  int index;
} double_index_pair;

static inline int shifted_index_from_c_index(const fft_grid_layout *g,
                                             const int c_index,
                                             const int direction) {
  if (direction == 0 && g->use_halfspace)
    return c_index == 0 ? 0 : -c_index;
  return convert_c_index_to_shifted_index(c_index, g->npts_global[direction]);
}

double squared_length_of_g_vector(const int g[3], double h_inv[3][3]) {
  if (g[0] == 0 && g[1] == 0 && g[2] == 0) {
    return 0.0;
  }
  const double two_pi = 2.0 * acos(-1.0);
  const double gx = g[2];
  const double gy = g[1];
  const double gz = g[0];
  double length_x = gx * h_inv[2][2] + gy * h_inv[2][1] + gz * h_inv[2][0];
  double length_y = gx * h_inv[1][2] + gy * h_inv[1][1] + gz * h_inv[1][0];
  double length_z = gx * h_inv[0][2] + gy * h_inv[0][1] + gz * h_inv[0][0];
  length_x *= two_pi;
  length_y *= two_pi;
  length_z *= two_pi;
  return length_x * length_x + length_y * length_y + length_z * length_z;
}

int compare_double(const void *a, const void *b) {
  const double a_value = ((const double_index_pair *)a)->value;
  const double b_value = ((const double_index_pair *)b)->value;
  return (a_value > b_value ? 1 : (a_value < b_value ? -1 : 0));
}

int compare_shell(const void *a, const void *b) {
  for (int index = 0; index < 3; index++) {
    const int a_value = ((const int *)a)[index];
    const int b_value = ((const int *)b)[index];
    const double max_allowed_error =
        max_rel_error_for_equivalence_g_squared * fmax(a_value, b_value);
    if (a_value > b_value + max_allowed_error) {
      return 1;
    } else if (a_value + max_allowed_error < b_value) {
      return -1;
    }
  }
  return 0;
}

void sort_shell(int (*shell)[3], const int shell_size) {
  qsort(shell, shell_size, sizeof(int[3]), compare_shell);
}

void sort_g_vectors(fft_grid_layout *my_fft_grid) {
  cp_assert(my_fft_grid != NULL);
  cp_assert(my_fft_grid->npts_gs_local >= 0);

  if (my_fft_grid->npts_gs_local == 0)
    return;

  double *local_index2g_squared =
      calloc(my_fft_grid->npts_gs_local, sizeof(double));
#pragma omp parallel for default(none)                                         \
    shared(my_fft_grid, local_index2g_squared)
  for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
    local_index2g_squared[index] = squared_length_of_g_vector(
        (const int[3]){
            shifted_index_from_c_index(my_fft_grid,
                                       my_fft_grid->index_to_g[index][0], 0),
            shifted_index_from_c_index(my_fft_grid,
                                       my_fft_grid->index_to_g[index][1], 1),
            shifted_index_from_c_index(my_fft_grid,
                                       my_fft_grid->index_to_g[index][2], 2)},
        my_fft_grid->cell_info.dhmat);
  }

  // Sort the indices according to the length of the vectors
  double_index_pair *g_square_index_pair =
      calloc(my_fft_grid->npts_gs_local, sizeof(double_index_pair));
#pragma omp parallel for default(none)                                         \
    shared(my_fft_grid, g_square_index_pair, local_index2g_squared)
  for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
    g_square_index_pair[index].value = local_index2g_squared[index];
    g_square_index_pair[index].index = index;
  }
  qsort(g_square_index_pair, my_fft_grid->npts_gs_local,
        sizeof(double_index_pair), compare_double);

  // Apply the sorting to the index_to_g array
  {
    int (*index_to_g_sorted)[3] =
        calloc(my_fft_grid->npts_gs_local, sizeof(int[3]));
    double *gsquared_sorted =
        calloc(my_fft_grid->npts_gs_local, sizeof(double));
#pragma omp parallel for default(none)                                         \
    shared(my_fft_grid, index_to_g_sorted, gsquared_sorted,                    \
               g_square_index_pair)
    for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
      memcpy(index_to_g_sorted[index],
             my_fft_grid->index_to_g[g_square_index_pair[index].index],
             3 * sizeof(int));
      gsquared_sorted[index] = g_square_index_pair[index].value;
    }
    memcpy(my_fft_grid->index_to_g, &index_to_g_sorted[0][0],
           my_fft_grid->npts_gs_local * sizeof(int[3]));
    if (my_fft_grid->index_to_gsquared != NULL)
      memcpy(my_fft_grid->index_to_gsquared, gsquared_sorted,
             my_fft_grid->npts_gs_local * sizeof(double));
    free(index_to_g_sorted);
    free(gsquared_sorted);
  }

  // Sort the vectors with the same length according to the x-, then y-, then
  // z-coordinate
  {
    double last_g_squared = g_square_index_pair[0].value;
    int start_index = 0;
    for (int end_index = 1; end_index < my_fft_grid->npts_gs_local;
         end_index++) {
      if (fabs(g_square_index_pair[end_index].value - last_g_squared) >
          fmax(g_square_index_pair[end_index].value, last_g_squared) *
              max_rel_error_for_equivalence_g_squared) {
        // If the length of the current vector is different from the previous
        // one, sort the vectors with the same length according to the x-, then
        // y-, then z-coordinate
        sort_shell(my_fft_grid->index_to_g + start_index,
                   end_index - start_index);
        start_index = end_index;
        last_g_squared = g_square_index_pair[end_index].value;
      }
    }
    // At the end, we need to sort the last shell
    sort_shell(my_fft_grid->index_to_g + start_index,
               my_fft_grid->npts_gs_local - start_index);
  }
  free(g_square_index_pair);
  free(local_index2g_squared);

  if (my_fft_grid->index_to_gsquared != NULL) {
#pragma omp parallel for default(none) shared(my_fft_grid)
    for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
      my_fft_grid->index_to_gsquared[index] = squared_length_of_g_vector(
          (const int[3]){
              shifted_index_from_c_index(my_fft_grid,
                                         my_fft_grid->index_to_g[index][0], 0),
              shifted_index_from_c_index(my_fft_grid,
                                         my_fft_grid->index_to_g[index][1], 1),
              shifted_index_from_c_index(my_fft_grid,
                                         my_fft_grid->index_to_g[index][2], 2)},
          my_fft_grid->cell_info.dhmat);
    }
  }
}

void grid_free_fft_grid_layout(fft_grid_layout *fft_grid) {
  if (fft_grid == NULL) {
    return;
  }
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "free_grid_layout");
  const int handle = fft_start_timer(routine_name);
  if (fft_grid != NULL) {
    cp_assert((fft_grid->ref_counter) > 0);
    fft_grid->ref_counter--;
    if (fft_grid->ref_counter == 0) {
      fft_fftw_free_distributed_plans(fft_grid->comm);
      fft_fftw_free_distributed_plans(fft_grid->sub_comm[0]);
      fft_fftw_free_distributed_plans(fft_grid->sub_comm[1]);
      cp_mpi_comm_free(&fft_grid->comm);
      cp_mpi_comm_free(&fft_grid->sub_comm[0]);
      cp_mpi_comm_free(&fft_grid->sub_comm[1]);
      cleanup_redistribution(fft_grid->redistribution);
      free(fft_grid->redistribution);
      free(fft_grid->proc2local_rs);
      free(fft_grid->proc2local_ms);
      free(fft_grid->proc2local_gs);
      free(fft_grid->proc2local_y_rs);
      free(fft_grid->proc2local_z_rs);
      free(fft_grid->proc2local_x_gs);
      free(fft_grid->proc2local_y_gs);
      free(fft_grid->xy_to_process);
      free(fft_grid->yz_to_process);
      free(fft_grid->yzp);
      free(fft_grid->ray_to_xy);
      free(fft_grid->xy_to_ray);
      free(fft_grid->rays_per_process);
      free(fft_grid->index_to_g);
      free(fft_grid->local_index_to_ref_grid);
      free(fft_grid->index_to_cart);
      free(fft_grid->index_to_cart_neg);
      free(fft_grid->index_to_cart_pos);
      free(fft_grid->index_to_ray);
      free(fft_grid->index_to_ray_neg);
      free(fft_grid->index_to_ray_pos);
      free(fft_grid->index_to_gsquared);
      free(fft_grid);
    }
  }
  fft_stop_timer(handle);
}

void setup_proc2local(fft_grid_layout *my_fft_grid,
                      const int *external_local_bounds) {
  const int number_of_processes = cp_mpi_comm_size(my_fft_grid->comm);
  const int my_process = cp_mpi_comm_rank(my_fft_grid->comm);

  my_fft_grid->proc2local_rs = calloc(6 * number_of_processes, sizeof(int));
  my_fft_grid->proc2local_ms = calloc(6 * number_of_processes, sizeof(int));
  my_fft_grid->proc2local_gs = calloc(6 * number_of_processes, sizeof(int));
  my_fft_grid->proc2local_y_rs =
      calloc(2 * my_fft_grid->proc_grid[1], sizeof(int));
  my_fft_grid->proc2local_z_rs =
      calloc(2 * my_fft_grid->proc_grid[0], sizeof(int));
  my_fft_grid->proc2local_x_gs =
      calloc(2 * my_fft_grid->proc_grid[1], sizeof(int));
  my_fft_grid->proc2local_y_gs =
      calloc(2 * my_fft_grid->proc_grid[0], sizeof(int));

  my_fft_grid->use_mpi_backend = fft_lib_use_mpi() &&
                                 cp_mpi_comm_size(my_fft_grid->comm) > 1 &&
                                 !external_local_bounds;

  // With distributed FFT libraries, we thereby determine a buffer size which is
  // refined later
  int buffer_size = 0, my_bounds[2] = {0, 0};
  if (my_fft_grid->use_mpi_backend) {
    // The data distribution is taken optimized for use with FFTW
    // The first index is distributed, the others are not
    // We ask for output data with the first two indices swapped (transposed
    // mode) to save communication within the library Several distributed 2D
    // FFTs require the distance between elements of adjacent FFTs to be 1
    // Starting from the order (z, y, x), we have to distribute the first index
    // and, in case of a pencil distribution, the LAST index (this should be
    // related to improved vectorization within the library)

    // As a first step, we determine the data distribution in each process
    // dimension
    if (my_fft_grid->proc_grid[1] > 1) {
      // Start with a distributed FFT using the first sub-communicator in y- and
      // z-direction
      int local_y_rs, local_y_start_rs, local_x_gs, local_x_start_gs;
      // We need to pre-distribute the z-direction
      const int block_size_z_rs =
          (my_fft_grid->npts_global_gspace[2] + my_fft_grid->proc_grid[0] - 1) /
          my_fft_grid->proc_grid[0];
      // In the last step, y is distributed in the second direction
      const int block_size_y_gs =
          (my_fft_grid->npts_global_gspace[1] + my_fft_grid->proc_grid[0] - 1) /
          my_fft_grid->proc_grid[0];
      // Determine a maximum buffer size
      // With half-space, we need different routines
      if (my_fft_grid->use_halfspace) {
        my_fft_grid->buffer_size = fft_2d_distributed_sizes_r2c(
            (const int[2]){my_fft_grid->npts_global[1],
                           my_fft_grid->npts_global[0]},
            block_size_z_rs, my_fft_grid->sub_comm[1], &local_y_rs,
            &local_y_start_rs, &local_x_gs, &local_x_start_gs);
      } else {
        my_fft_grid->buffer_size = fft_2d_distributed_sizes(
            (const int[2]){my_fft_grid->npts_global[1],
                           my_fft_grid->npts_global[0]},
            block_size_z_rs, my_fft_grid->sub_comm[1], &local_y_rs,
            &local_y_start_rs, &local_x_gs, &local_x_start_gs);
      }
      // Setup the bounds in real space
      // In z-direction, we need to define them ourselves
      // The distributions in x- and y-direction are provided by the FFT library
      // With y (second index) required to be locally available
      my_bounds[0] = local_y_start_rs;
      my_bounds[1] = local_y_rs;
      // Exchange the distribution with the other processes
      cp_mpi_allgather_int((const int *)my_bounds, 2,
                           (int *)my_fft_grid->proc2local_y_rs, 2,
                           my_fft_grid->sub_comm[1]);
      my_bounds[0] = local_x_start_gs;
      my_bounds[1] = local_x_gs;
      // Exchange the distribution with the other processes
      cp_mpi_allgather_int((const int *)my_bounds, 2,
                           (int *)my_fft_grid->proc2local_x_gs, 2,
                           my_fft_grid->sub_comm[1]);
      for (int process = 0; process < my_fft_grid->proc_grid[0]; process++) {
        my_fft_grid->proc2local_z_rs[process][0] =
            imin(process * block_size_z_rs, my_fft_grid->npts_global_gspace[2]);
        my_fft_grid->proc2local_z_rs[process][1] =
            imin((process + 1) * block_size_z_rs - 1,
                 my_fft_grid->npts_global_gspace[2] - 1) -
            imin(process * block_size_z_rs,
                 my_fft_grid->npts_global_gspace[2]) +
            1;
        my_fft_grid->proc2local_y_gs[process][0] =
            imin(process * block_size_y_gs, my_fft_grid->npts_global_gspace[1]);
        my_fft_grid->proc2local_y_gs[process][1] =
            imin((process + 1) * block_size_y_gs - 1,
                 my_fft_grid->npts_global_gspace[1] - 1) -
            imin(process * block_size_y_gs,
                 my_fft_grid->npts_global_gspace[1]) +
            1;
      }
    } else {
      // With distributed 3D FFTs, we ask the library to perform all FFT steps
      // This data distribution is obtained from the 2D case without data
      // distribution in the second process direction
      int local_z_rs = 0, local_z_start_rs = 0, local_y_gs = 0,
          local_y_start_gs = 0;
      // With ray distribution, we perform only a local 2D FFT to be able
      // to perform the final local FFT of the own rays only
      // So, we need to define the distribution in z-direction ourselves
      if (my_fft_grid->ray_distribution) {
        const int block_size_z_rs = (my_fft_grid->npts_global_gspace[2] +
                                     my_fft_grid->proc_grid[0] - 1) /
                                    my_fft_grid->proc_grid[0];
        const int block_size_y_gs = (my_fft_grid->npts_global_gspace[1] +
                                     my_fft_grid->proc_grid[0] - 1) /
                                    my_fft_grid->proc_grid[0];
        for (int process = 0; process < my_fft_grid->proc_grid[0]; process++) {
          my_fft_grid->proc2local_z_rs[process][0] = imin(
              process * block_size_z_rs, my_fft_grid->npts_global_gspace[2]);
          my_fft_grid->proc2local_z_rs[process][1] =
              imin((process + 1) * block_size_z_rs - 1,
                   my_fft_grid->npts_global_gspace[2] - 1) -
              imin(process * block_size_z_rs,
                   my_fft_grid->npts_global_gspace[2]) +
              1;
          my_fft_grid->proc2local_y_gs[process][0] = imin(
              process * block_size_y_gs, my_fft_grid->npts_global_gspace[1]);
          my_fft_grid->proc2local_y_gs[process][1] =
              imin((process + 1) * block_size_y_gs - 1,
                   my_fft_grid->npts_global_gspace[1] - 1) -
              imin(process * block_size_y_gs,
                   my_fft_grid->npts_global_gspace[1]) +
              1;
        }
      } else {
        // In blocked distribution, we use a distributed 3d FFT
        if (my_fft_grid->use_halfspace) {
          my_fft_grid->buffer_size = fft_3d_distributed_sizes_r2c(
              (const int[3]){my_fft_grid->npts_global[2],
                             my_fft_grid->npts_global[1],
                             my_fft_grid->npts_global[0]},
              my_fft_grid->sub_comm[0], &local_z_rs, &local_z_start_rs,
              &local_y_gs, &local_y_start_gs);
        } else {
          my_fft_grid->buffer_size = fft_3d_distributed_sizes(
              (const int[3]){my_fft_grid->npts_global[2],
                             my_fft_grid->npts_global[1],
                             my_fft_grid->npts_global[0]},
              my_fft_grid->sub_comm[0], &local_z_rs, &local_z_start_rs,
              &local_y_gs, &local_y_start_gs);
        }
        my_bounds[0] = local_z_start_rs;
        my_bounds[1] = local_z_rs;
        // Exchange the distribution with the other processes
        cp_mpi_allgather_int((const int *)my_bounds, 2,
                             (int *)my_fft_grid->proc2local_z_rs, 2,
                             my_fft_grid->sub_comm[0]);
        my_bounds[0] = local_y_start_gs;
        my_bounds[1] = local_y_gs;
        // Exchange the distribution with the other processes
        cp_mpi_allgather_int((const int *)my_bounds, 2,
                             (int *)my_fft_grid->proc2local_y_gs, 2,
                             my_fft_grid->sub_comm[0]);
      }
      for (int process = 0; process < my_fft_grid->proc_grid[1]; process++) {
        my_fft_grid->proc2local_y_rs[process][0] = 0;
        my_fft_grid->proc2local_y_rs[process][1] =
            my_fft_grid->npts_global_gspace[1];
        my_fft_grid->proc2local_x_gs[process][0] = 0;
        my_fft_grid->proc2local_x_gs[process][1] =
            my_fft_grid->npts_global_gspace[0];
      }
    }
  } else if (external_local_bounds != NULL) {
    // Determine the distribution in g-space
    const int block_size_x_gs =
        (my_fft_grid->npts_global_gspace[0] + my_fft_grid->proc_grid[1] - 1) /
        my_fft_grid->proc_grid[1];
    const int block_size_y_gs =
        (my_fft_grid->npts_global_gspace[1] + my_fft_grid->proc_grid[0] - 1) /
        my_fft_grid->proc_grid[0];
    for (int process = 0; process < my_fft_grid->proc_grid[1]; process++) {
      int proc_coord[2] = {0, process};
      const int global_rank = cp_mpi_cart_rank(my_fft_grid->comm, proc_coord);
      my_fft_grid->proc2local_y_rs[process][0] =
          external_local_bounds[6 * global_rank + 2];
      my_fft_grid->proc2local_y_rs[process][1] =
          external_local_bounds[6 * global_rank + 3] -
          external_local_bounds[6 * global_rank + 2] + 1;
      my_fft_grid->proc2local_x_gs[process][0] =
          imin(process * block_size_x_gs, my_fft_grid->npts_global_gspace[0]);
      my_fft_grid->proc2local_x_gs[process][1] =
          imin((process + 1) * block_size_x_gs - 1,
               my_fft_grid->npts_global_gspace[0] - 1) -
          imin(process * block_size_x_gs, my_fft_grid->npts_global_gspace[0]) +
          1;
    }
    for (int process = 0; process < my_fft_grid->proc_grid[0]; process++) {
      int proc_coord[2] = {process, 0};
      const int global_rank = cp_mpi_cart_rank(my_fft_grid->comm, proc_coord);
      my_fft_grid->proc2local_z_rs[process][0] =
          external_local_bounds[6 * global_rank + 4];
      my_fft_grid->proc2local_z_rs[process][1] =
          external_local_bounds[6 * global_rank + 5] -
          external_local_bounds[6 * global_rank + 4] + 1;
      my_fft_grid->proc2local_y_gs[process][0] =
          imin(process * block_size_y_gs, my_fft_grid->npts_global_gspace[1]);
      my_fft_grid->proc2local_y_gs[process][1] =
          imin((process + 1) * block_size_y_gs - 1,
               my_fft_grid->npts_global_gspace[1] - 1) -
          imin(process * block_size_y_gs, my_fft_grid->npts_global_gspace[1]) +
          1;
    }
  } else {
    // Serial case or without distributed FFT
    const int block_size_y_rs =
        (my_fft_grid->npts_global_gspace[1] + my_fft_grid->proc_grid[1] - 1) /
        my_fft_grid->proc_grid[1];
    const int block_size_z_rs =
        (my_fft_grid->npts_global_gspace[2] + my_fft_grid->proc_grid[0] - 1) /
        my_fft_grid->proc_grid[0];
    const int block_size_x_gs =
        (my_fft_grid->npts_global_gspace[0] + my_fft_grid->proc_grid[1] - 1) /
        my_fft_grid->proc_grid[1];
    const int block_size_y_gs =
        (my_fft_grid->npts_global_gspace[1] + my_fft_grid->proc_grid[0] - 1) /
        my_fft_grid->proc_grid[0];
    for (int process = 0; process < my_fft_grid->proc_grid[1]; process++) {
      my_fft_grid->proc2local_y_rs[process][0] =
          imin(process * block_size_y_rs, my_fft_grid->npts_global_gspace[1]);
      my_fft_grid->proc2local_y_rs[process][1] =
          imin((process + 1) * block_size_y_rs - 1,
               my_fft_grid->npts_global_gspace[1] - 1) -
          imin(process * block_size_y_rs, my_fft_grid->npts_global_gspace[1]) +
          1;
      my_fft_grid->proc2local_x_gs[process][0] =
          imin(process * block_size_x_gs, my_fft_grid->npts_global_gspace[0]);
      my_fft_grid->proc2local_x_gs[process][1] =
          imin((process + 1) * block_size_x_gs - 1,
               my_fft_grid->npts_global_gspace[0] - 1) -
          imin(process * block_size_x_gs, my_fft_grid->npts_global_gspace[0]) +
          1;
    }
    for (int process = 0; process < my_fft_grid->proc_grid[0]; process++) {
      my_fft_grid->proc2local_z_rs[process][0] =
          imin(process * block_size_z_rs, my_fft_grid->npts_global_gspace[2]);
      my_fft_grid->proc2local_z_rs[process][1] =
          imin((process + 1) * block_size_z_rs - 1,
               my_fft_grid->npts_global_gspace[2] - 1) -
          imin(process * block_size_z_rs, my_fft_grid->npts_global_gspace[2]) +
          1;
      my_fft_grid->proc2local_y_gs[process][0] =
          imin(process * block_size_y_gs, my_fft_grid->npts_global_gspace[1]);
      my_fft_grid->proc2local_y_gs[process][1] =
          imin((process + 1) * block_size_y_gs - 1,
               my_fft_grid->npts_global_gspace[1] - 1) -
          imin(process * block_size_y_gs, my_fft_grid->npts_global_gspace[1]) +
          1;
    }
  }
  // Then, we collect the mappings of the ranges in each representation
  // RS/MS/GS
  for (int process = 0; process < number_of_processes; process++) {
    int proc_coords[2];
    cp_mpi_cart_coords(my_fft_grid->comm, process, 2, proc_coords);
    // Compile the distribution in real space
    my_fft_grid->proc2local_rs[process][0][0] = 0;
    my_fft_grid->proc2local_rs[process][0][1] = my_fft_grid->npts_global[0];
    my_fft_grid->proc2local_rs[process][1][0] =
        my_fft_grid->proc2local_y_rs[proc_coords[1]][0];
    my_fft_grid->proc2local_rs[process][1][1] =
        my_fft_grid->proc2local_y_rs[proc_coords[1]][1];
    my_fft_grid->proc2local_rs[process][2][0] =
        my_fft_grid->proc2local_z_rs[proc_coords[0]][0];
    my_fft_grid->proc2local_rs[process][2][1] =
        my_fft_grid->proc2local_z_rs[proc_coords[0]][1];
    // Compile the distribution in reciprocal space
    my_fft_grid->proc2local_gs[process][0][0] =
        my_fft_grid->proc2local_x_gs[proc_coords[1]][0];
    my_fft_grid->proc2local_gs[process][0][1] =
        my_fft_grid->proc2local_x_gs[proc_coords[1]][1];
    my_fft_grid->proc2local_gs[process][1][0] =
        my_fft_grid->proc2local_y_gs[proc_coords[0]][0];
    my_fft_grid->proc2local_gs[process][1][1] =
        my_fft_grid->proc2local_y_gs[proc_coords[0]][1];
    my_fft_grid->proc2local_gs[process][2][0] = 0;
    my_fft_grid->proc2local_gs[process][2][1] = my_fft_grid->npts_global[2];
    // Compile the distribution in mixed space
    my_fft_grid->proc2local_ms[process][0][0] =
        my_fft_grid->proc2local_gs[process][0][0];
    my_fft_grid->proc2local_ms[process][0][1] =
        my_fft_grid->proc2local_gs[process][0][1];
    my_fft_grid->proc2local_ms[process][1][0] = 0;
    my_fft_grid->proc2local_ms[process][1][1] = my_fft_grid->npts_global[1];
    my_fft_grid->proc2local_ms[process][2][0] =
        my_fft_grid->proc2local_rs[process][2][0];
    my_fft_grid->proc2local_ms[process][2][1] =
        my_fft_grid->proc2local_rs[process][2][1];
  }

  // Finally, we determine the buffer size
  buffer_size =
      imax(buffer_size, my_fft_grid->npts_global_gspace[0] *
                            my_fft_grid->proc2local_rs[my_process][1][1] *
                            my_fft_grid->proc2local_rs[my_process][2][1]);
  buffer_size =
      imax(buffer_size, my_fft_grid->proc2local_ms[my_process][0][1] *
                            my_fft_grid->npts_global_gspace[1] *
                            my_fft_grid->proc2local_ms[my_process][2][1]);
  buffer_size =
      imax(buffer_size, my_fft_grid->proc2local_gs[my_process][0][1] *
                            my_fft_grid->proc2local_gs[my_process][1][1] *
                            my_fft_grid->npts_global_gspace[2]);
  my_fft_grid->buffer_size = buffer_size;

  if (false && my_process == 0) {
    printf("Proc2local RS\n");
    for (int process = 0; process < number_of_processes; process++) {
      printf("%i: %i %i / %i %i / %i %i\n", process,
             my_fft_grid->proc2local_rs[process][0][0],
             my_fft_grid->proc2local_rs[process][0][0] +
                 my_fft_grid->proc2local_rs[process][0][1] - 1,
             my_fft_grid->proc2local_rs[process][1][0],
             my_fft_grid->proc2local_rs[process][1][0] +
                 my_fft_grid->proc2local_rs[process][1][1] - 1,
             my_fft_grid->proc2local_rs[process][2][0],
             my_fft_grid->proc2local_rs[process][2][0] +
                 my_fft_grid->proc2local_rs[process][2][1] - 1);
    }
    printf("\n");
    printf("Proc2local MS\n");
    for (int process = 0; process < number_of_processes; process++) {
      printf("%i: %i %i / %i %i / %i %i\n", process,
             my_fft_grid->proc2local_ms[process][0][0],
             my_fft_grid->proc2local_ms[process][0][0] +
                 my_fft_grid->proc2local_ms[process][0][1] - 1,
             my_fft_grid->proc2local_ms[process][1][0],
             my_fft_grid->proc2local_ms[process][1][0] +
                 my_fft_grid->proc2local_ms[process][1][1] - 1,
             my_fft_grid->proc2local_ms[process][2][0],
             my_fft_grid->proc2local_ms[process][2][0] +
                 my_fft_grid->proc2local_ms[process][2][1] - 1);
    }
    printf("\n");
    printf("Proc2local GS\n");
    for (int process = 0; process < number_of_processes; process++) {
      printf("%i: %i %i / %i %i / %i %i\n", process,
             my_fft_grid->proc2local_gs[process][0][0],
             my_fft_grid->proc2local_gs[process][0][0] +
                 my_fft_grid->proc2local_gs[process][0][1] - 1,
             my_fft_grid->proc2local_gs[process][1][0],
             my_fft_grid->proc2local_gs[process][1][0] +
                 my_fft_grid->proc2local_gs[process][1][1] - 1,
             my_fft_grid->proc2local_gs[process][2][0],
             my_fft_grid->proc2local_gs[process][2][0] +
                 my_fft_grid->proc2local_gs[process][2][1] - 1);
    }
    printf("\n");
  }
}

/*******************************************************************************
 * \brief Registers all required FFTs for a grid layout
 * \param grid_rs complex-valued data in real space.
 * \param grid_gs complex data in reciprocal space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_register_3d(const fft_grid_layout *grid_layout) {

  ensure_buffer_size(grid_layout->buffer_size);

  if (grid_layout->use_halfspace) {
    if (grid_layout->ray_distribution) {
      fft_register_3d_r2c_ray(
          grid_layout->npts_global, grid_layout->npts_global_gspace,
          grid_layout->proc2local_rs, grid_layout->proc2local_ms,
          grid_layout->rays_per_process, grid_layout->comm,
          grid_layout->sub_comm);
    } else {
      fft_register_3d_r2c_blocked(
          grid_layout->npts_global, grid_layout->npts_global_gspace,
          grid_layout->proc2local_rs, grid_layout->proc2local_ms,
          grid_layout->proc2local_gs, grid_layout->comm,
          grid_layout->sub_comm);
    }
  } else {
    if (grid_layout->ray_distribution) {
      fft_register_3d_ray(grid_layout->npts_global, grid_layout->proc2local_rs,
                          grid_layout->proc2local_ms,
                          grid_layout->rays_per_process, grid_layout->comm,
                          grid_layout->sub_comm);
    } else {
      fft_register_3d_blocked(
          grid_layout->npts_global, grid_layout->proc2local_rs,
          grid_layout->proc2local_ms, grid_layout->proc2local_gs,
          grid_layout->comm, grid_layout->sub_comm);
    }
  }
}


/*******************************************************************************
 * \brief Choose the Cartesian process grid exactly as grid creation does.
 ******************************************************************************/
void fft_choose_proc_grid(const int number_of_processes,
                          const int npts_global[3], const int *pgrid_guess,
                          int proc_grid[2]) {
  proc_grid[0] = -1;
  proc_grid[1] = -1;
  if (pgrid_guess)
    memcpy(proc_grid, pgrid_guess, 2 * sizeof(int));
  if (proc_grid[0]) {
    if (proc_grid[1]) {
      if (proc_grid[0] * proc_grid[1] != number_of_processes) {
        proc_grid[0] = -1;
        proc_grid[1] = -1;
      }
    } else if (number_of_processes % proc_grid[0] == 0) {
      proc_grid[1] = number_of_processes / proc_grid[0];
    } else {
      proc_grid[0] = -1;
      proc_grid[1] = -1;
    }
  } else if (proc_grid[1]) {
    if (number_of_processes % proc_grid[1] == 0) {
      proc_grid[0] = number_of_processes / proc_grid[1];
    } else {
      proc_grid[1] = -1;
    }
  }
  if (proc_grid[0] <= 0 || proc_grid[1] <= 0 ||
      proc_grid[0] * proc_grid[1] != number_of_processes) {
    if (npts_global[2] < number_of_processes) {
      proc_grid[0] = 0;
      proc_grid[1] = 0;
      cp_mpi_dims_create(number_of_processes, 2, proc_grid);
      if ((npts_global[2] - npts_global[1]) *
              (proc_grid[0] - proc_grid[1]) <
          0) {
        const int proc_grid_0 = proc_grid[0];
        proc_grid[0] = proc_grid[1];
        proc_grid[1] = proc_grid_0;
      }
    } else {
      proc_grid[0] = number_of_processes;
      proc_grid[1] = 1;
    }
  }
}


/*******************************************************************************
 * \brief Convert a layout into the ray distribution using caller data.
 ******************************************************************************/
static void build_yz_to_process(fft_grid_layout *g) {
  const int nz = g->npts_global[0];
  const int ny = g->npts_global[1];
  const size_t size = (size_t)nz * (size_t)ny;
  free(g->yz_to_process);
  g->yz_to_process = malloc(size * sizeof(int));
  cp_assert(g->yz_to_process != NULL);
  for (size_t i = 0; i < size; i++)
    g->yz_to_process[i] = -1;

  int offset = 0;
  for (int p = 0; p < cp_mpi_comm_size(g->comm); p++) {
    for (int r = 0; r < g->rays_per_process[p]; r++) {
      const int c0 = g->ray_to_xy[offset + r][0];
      const int c1 = g->ray_to_xy[offset + r][1];
      const int z = g->use_halfspace && c0 != 0 ? nz - c0 : c0;
      g->yz_to_process[(size_t)z * ny + c1] = p;
    }
    offset += g->rays_per_process[p];
  }
}

static void build_final_yzp(fft_grid_layout *g) {
  const int nz = g->npts_global[0];
  const int ny = g->npts_global[1];
  const int nproc = cp_mpi_comm_size(g->comm);
  const int me = cp_mpi_comm_rank(g->comm);
  int *yz = calloc((size_t)nz * ny, sizeof(int));
  cp_assert(yz != NULL);

  for (int idx = 0; idx < g->npts_gs_local; idx++) {
    const int c0 = g->index_to_g[idx][0]; // C z index (pos convention)
    const int c1 = g->index_to_g[idx][1]; // C y index (pos convention)
    yz[(size_t)c0 * ny + c1]++;
    if (g->use_halfspace) {
      const int s0 = c0 > nz / 2 ? c0 - nz : c0;
      const int s1 = c1 > ny / 2 ? c1 - ny : c1;
      const int n0 = s0 <= 0 ? -s0 : nz - s0;
      const int n1 = s1 <= 0 ? -s1 : ny - s1;
      yz[(size_t)n0 * ny + n1]++;
    }
  }

  int local_nray = 0;
  for (size_t k = 0; k < (size_t)nz * ny; k++)
    local_nray += yz[k] > 0;
  int max_nray = local_nray;
  cp_mpi_max_int(&max_nray, 1, g->comm);

  int *local_yzp = calloc((size_t)2 * max_nray * nproc, sizeof(int));
  cp_assert(local_yzp != NULL);
  int is = 0;
  for (int i = 1; i <= nz; i++) {
    for (int j = 1; j <= ny; j++) {
      if (yz[(size_t)(i - 1) * ny + j - 1] == 0)
        continue;
      is++;
      local_yzp[(size_t)2 * (me * max_nray + (is - 1))] = j;
      local_yzp[(size_t)2 * (me * max_nray + (is - 1)) + 1] = i;
    }
  }
  cp_mpi_sum_int(local_yzp, 2 * max_nray * nproc, g->comm);
  free(yz);
  free(g->yzp);
  g->yzp = local_yzp;
  g->yzp_max = max_nray;
}

static int grid_enable_ray_distribution_internal(
    fft_grid_layout *g, const int *xy_to_process, const int nlocal,
    const int (*index_to_g)[3], const double *gsquared,
    const bool register_plans) {
  cp_assert(g != NULL);
  cp_assert(xy_to_process != NULL);
  cp_assert(index_to_g != NULL);
  cp_assert(gsquared != NULL);
  const int number_of_processes = cp_mpi_comm_size(g->comm);
  const int n0 = g->npts_global_gspace[0];
  const int n1 = g->npts_global[1];
  const size_t map_size = (size_t)n0 * (size_t)n1;

  free(g->index_to_g);
  free(g->index_to_gsquared);
  free(g->local_index_to_ref_grid);
  free(g->index_to_cart);
  free(g->index_to_cart_pos);
  free(g->index_to_cart_neg);
  free(g->index_to_ray);
  free(g->index_to_ray_pos);
  free(g->index_to_ray_neg);
  free(g->xy_to_process);
  free(g->ray_to_xy);
  free(g->xy_to_ray);
  free(g->rays_per_process);
  if (g->redistribution != NULL) {
    cleanup_redistribution(g->redistribution);
    free(g->redistribution);
    g->redistribution = NULL;
  }

  g->ray_distribution = true;
  g->npts_gs_local = nlocal;
  g->xy_to_process = malloc(map_size * sizeof(int));
  memcpy(g->xy_to_process, xy_to_process, map_size * sizeof(int));
  g->rays_per_process = calloc(number_of_processes, sizeof(int));
  for (size_t i = 0; i < map_size; i++) {
    const int p = g->xy_to_process[i];
    cp_assert(p >= 0 && p < number_of_processes);
    g->rays_per_process[p]++;
  }
  int total_number_of_rays = 0;
  int *ray_offsets = calloc(number_of_processes, sizeof(int));
  int *ray_index_per_process = calloc(number_of_processes, sizeof(int));
  for (int p = 0; p < number_of_processes; p++) {
    ray_offsets[p] = total_number_of_rays;
    total_number_of_rays += g->rays_per_process[p];
  }
  g->ray_to_xy = malloc((size_t)total_number_of_rays * sizeof(int[2]));
  g->xy_to_ray = malloc(map_size * sizeof(int));
  for (int c0 = 0; c0 < n0; c0++) {
    for (int c1 = 0; c1 < n1; c1++) {
      const int p = g->xy_to_process[(size_t)c0 * n1 + c1];
      const int idx = ray_index_per_process[p]++;
      g->ray_to_xy[ray_offsets[p] + idx][0] = c0;
      g->ray_to_xy[ray_offsets[p] + idx][1] = c1;
      g->xy_to_ray[(size_t)c0 * n1 + c1] = idx;
    }
  }
  free(ray_offsets);
  free(ray_index_per_process);

  g->number_of_positive_gs_points = 0;
  if (g->use_halfspace) {
    for (int i = 0; i < nlocal; i++)
      g->number_of_positive_gs_points += index_to_g[i][0] < n0;
    g->number_of_negative_gs_points =
        nlocal - g->number_of_positive_gs_points;
  } else {
    g->number_of_positive_gs_points = nlocal;
    g->number_of_negative_gs_points = 0;
  }
  g->index_to_g = malloc((size_t)nlocal * sizeof(int[3]));
  memcpy(g->index_to_g, index_to_g, (size_t)nlocal * sizeof(int[3]));
  g->index_to_gsquared = malloc((size_t)nlocal * sizeof(double));
  memcpy(g->index_to_gsquared, gsquared, (size_t)nlocal * sizeof(double));
  g->local_index_to_ref_grid = malloc((size_t)nlocal * sizeof(int));
  for (int i = 0; i < nlocal; i++)
    g->local_index_to_ref_grid[i] = i;

  g->index_to_cart = NULL;
  g->index_to_cart_pos = NULL;
  g->index_to_cart_neg = NULL;
  g->index_to_ray = NULL;
  g->index_to_ray_pos = NULL;
  g->index_to_ray_neg = NULL;
  if (g->use_halfspace) {
    g->index_to_ray_pos =
        calloc(g->number_of_positive_gs_points, sizeof(int[2]));
    g->index_to_ray_neg =
        calloc(g->number_of_negative_gs_points, sizeof(int[2]));
  } else {
    g->index_to_ray = calloc(nlocal, sizeof(int));
  }

  sort_g_vectors(g);
  if (g->use_halfspace) {
    int positive_index = 0;
    int negative_index = 0;
    for (int index = 0; index < g->npts_gs_local; index++) {
      int *index_g = g->index_to_g[index];
      if (index_g[0] < g->npts_global_gspace[0]) {
        g->index_to_ray_pos[positive_index][0] = index;
        g->index_to_ray_pos[positive_index][1] =
            g->xy_to_ray[(size_t)index_g[0] * n1 + index_g[1]] *
                g->npts_global[2] +
            index_g[2];
        positive_index++;
      } else {
        g->index_to_ray_neg[negative_index][0] = index;
        g->index_to_ray_neg[negative_index][1] =
            g->xy_to_ray[((size_t)(g->npts_global[0] - index_g[0])) * n1 +
                         (g->npts_global[1] - index_g[1]) %
                             g->npts_global[1]] *
                g->npts_global[2] +
            (g->npts_global[2] - index_g[2]) % g->npts_global[2];
        negative_index++;
      }
    }
    cp_assert(positive_index == g->number_of_positive_gs_points);
    cp_assert(negative_index == g->number_of_negative_gs_points);
  } else {
    for (int index = 0; index < g->npts_gs_local; index++) {
      int *index_g = g->index_to_g[index];
      g->index_to_ray[index] =
          g->xy_to_ray[(size_t)index_g[0] * n1 + index_g[1]] *
              g->npts_global[2] +
          index_g[2];
    }
  }

  g->buffer_size = imax(g->buffer_size, nlocal);
  g->redistribution = calloc(1, sizeof(fft_redistribution_t));
  prepare_redistribution(
      g->redistribution, g->npts_global_gspace, g->proc2local_ms,
      g->proc2local_x_gs, g->proc2local_y_rs, g->proc2local_y_gs,
      g->proc2local_z_rs, g->rays_per_process, g->ray_to_xy, g->comm,
      g->sub_comm);
  if (register_plans)
    fft_register_3d(g);
  build_yz_to_process(g);
  build_final_yzp(g);
  return 0;
}

int grid_enable_ray_distribution(fft_grid_layout *g,
                                 const int *xy_to_process, const int nlocal,
                                 const int (*index_to_g)[3],
                                 const double *gsquared) {
  return grid_enable_ray_distribution_internal(
      g, xy_to_process, nlocal, index_to_g, gsquared, true);
}

/*******************************************************************************
 * \brief Finalize the distribution of a provisionally blocked layout.
 ******************************************************************************/
/*******************************************************************************
 * \brief Prepare a provisional ray-to-process map without building ray plans.
 * The map is refined/committed later by grid_finalize_distribution.
 ******************************************************************************/
void grid_prepare_ray_mapping(fft_grid_layout *g,
                              const fft_grid_layout *ref) {
  cp_assert(g != NULL);
  const int number_of_processes = cp_mpi_comm_size(g->comm);
  const int n0 = g->npts_global_gspace[0];
  const int n1 = g->npts_global[1];
  const size_t map_size = (size_t)n0 * (size_t)n1;

  free(g->xy_to_process);
  free(g->rays_per_process);
  g->xy_to_process = malloc(map_size * sizeof(int));
  g->rays_per_process = calloc(number_of_processes, sizeof(int));
  unsigned char *has_data = calloc(map_size, sizeof(unsigned char));
  cp_assert(g->xy_to_process != NULL && g->rays_per_process != NULL &&
            has_data != NULL);

  for (int c0 = 0; c0 < n0; c0++) {
    const int s0 = convert_c_index_to_shifted_index(c0, g->npts_global[0]);
    for (int c1 = 0; c1 < n1; c1++) {
      const int s1 = convert_c_index_to_shifted_index(c1, g->npts_global[1]);
      for (int c2 = 0; c2 < g->npts_global[2]; c2++) {
        const int s2 = convert_c_index_to_shifted_index(c2, g->npts_global[2]);
        if (g->use_halfspace && s0 == 0 &&
            ((s1 == 0 && s2 > 0) || s1 > 0))
          continue;
        if (g->spherical && g->cutoff > 0.0) {
          const double gsq = squared_length_of_g_vector(
              (const int[3]){s0, s1, s2}, g->cell_info.dhmat);
          if (0.5 * gsq > g->cutoff)
            continue;
        }
        has_data[(size_t)c0 * n1 + c1] = 1;
        break;
      }
    }
  }

  const int ref_n0 = ref != NULL ? ref->npts_global_gspace[0] : 0;
  const int ref_n1 = ref != NULL ? ref->npts_global[1] : 0;
  const int nz = g->npts_global[0];
  const int lbz = -(nz / 2);
  const int lby = -(n1 / 2);
  const int nz_ray = g->npts_global_gspace[0];

  int rr_data = 0;
  for (int i = 1; i <= nz; i++) {
    if (g->use_halfspace && i > nz_ray)
      continue;
    const int sz = i - 1 + lbz;
    const int c0 = g->use_halfspace ? nz_ray - i
                                    : (sz < 0 ? sz + nz : sz);
    for (int j = 1; j <= n1; j++) {
      const int sy = j - 1 + lby;
      const int c1 = sy < 0 ? sy + n1 : sy;
      const size_t index = (size_t)c0 * n1 + c1;
      int process = -1;
      if (ref != NULL) {
        int z_ref = sz;
        if (ref->use_halfspace && z_ref > 0)
          z_ref = -z_ref;
        const int ref_cz = z_ref < 0 ? z_ref + ref->npts_global[0] : z_ref;
        const int ref_cy = sy < 0 ? sy + ref_n1 : sy;
        if (ref_cz >= 0 && ref_cz < ref->npts_global[0] && ref_cy >= 0 &&
            ref_cy < ref_n1) {
          const int j_ref = ref_cy + 1;
          const int i_ref = ref_cz + 1;
          if (ref->yz_to_process != NULL) {
            process =
                ref->yz_to_process[(size_t)(i_ref - 1) * ref_n1 + j_ref - 1];
          } else {
            const int rc0 = ref->use_halfspace
                                ? abs(z_ref)
                                : (z_ref < 0 ? z_ref + ref->npts_global[0]
                                             : z_ref);
            const int rc1 = sy < 0 ? sy + ref_n1 : sy;
            if (rc0 >= 0 && rc0 < ref_n0 && rc1 >= 0 && rc1 < ref_n1) {
              if (ref->xy_to_process != NULL)
                process = ref->xy_to_process[(size_t)rc0 * ref_n1 + rc1];
              else if (ref->proc2local_gs != NULL) {
                for (int p = 0; p < number_of_processes; p++) {
                  if (rc0 >= ref->proc2local_gs[p][1][0] &&
                      rc0 < ref->proc2local_gs[p][1][0] +
                                ref->proc2local_gs[p][1][1] &&
                      rc1 >= ref->proc2local_gs[p][0][0] &&
                      rc1 < ref->proc2local_gs[p][0][0] +
                                ref->proc2local_gs[p][0][1]) {
                    process = p;
                    break;
                  }
                }
              }
            }
          }
        }
      }
      g->xy_to_process[index] = process;
      if (has_data[index] && process < 0) {
        process = rr_data % number_of_processes;
        rr_data++;
        g->xy_to_process[index] = process;
      }
      if (has_data[index] && process >= 0)
        g->rays_per_process[process]++;
    }
  }

  int rr_empty = 0;
  for (size_t index = 0; index < map_size; index++) {
    if (g->xy_to_process[index] >= 0)
      continue;
    g->xy_to_process[index] = rr_empty % number_of_processes;
    rr_empty++;
  }
  free(has_data);
}

int grid_get_yzp_max(const fft_grid_layout *g) {
  cp_assert(g != NULL);
  return g->yzp_max;
}

void grid_get_yzp(const fft_grid_layout *g, int *yzp) {
  cp_assert(g != NULL && yzp != NULL);
  const int nproc = cp_mpi_comm_size(g->comm);
  memcpy(yzp, g->yzp,
         (size_t)2 * g->yzp_max * nproc * sizeof(int));
}

int grid_get_npts_gs_local(const fft_grid_layout *g) {
  cp_assert(g != NULL);
  return g->npts_gs_local;
}

void grid_get_index_to_g(const fft_grid_layout *g, int *index_to_g) {
  cp_assert(g != NULL && index_to_g != NULL);
  memcpy(index_to_g, g->index_to_g,
         (size_t)g->npts_gs_local * sizeof(int[3]));
}

void grid_get_gsquared(const fft_grid_layout *g, double *gsquared) {
  cp_assert(g != NULL && gsquared != NULL);
  memcpy(gsquared, g->index_to_gsquared,
         (size_t)g->npts_gs_local * sizeof(double));
}

int grid_generate_g_space(fft_grid_layout *g) {
  cp_assert(g != NULL && g->xy_to_process != NULL);
  const int n0 = g->npts_global_gspace[0];
  const int n1 = g->npts_global[1];
  const int n2 = g->npts_global[2];
  const int me = cp_mpi_comm_rank(g->comm);
  const size_t map_size = (size_t)n0 * n1;

  int capacity = 1024;
  int nlocal = 0;
  int (*index_to_g)[3] = malloc((size_t)capacity * sizeof(int[3]));
  double *gsquared = malloc((size_t)capacity * sizeof(double));
  cp_assert(index_to_g != NULL && gsquared != NULL);

  for (int c0 = 0; c0 < n0; c0++) {
    for (int c1 = 0; c1 < n1; c1++) {
      if (g->xy_to_process[(size_t)c0 * n1 + c1] != me)
        continue;
      const int s0 = g->use_halfspace
                         ? (c0 == 0 ? 0 : -c0)
                         : convert_c_index_to_shifted_index(
                               c0, g->npts_global[0]);
      const int s1 = convert_c_index_to_shifted_index(c1, n1);
      const bool mirror_outside =
          g->use_halfspace &&
          ((g->npts_global[0] % 2 == 0 &&
            s0 == -(g->npts_global[0] / 2)) ||
           (n1 % 2 == 0 && s1 == -(n1 / 2)));
      if (mirror_outside)
        continue;
      for (int c2 = 0; c2 < n2; c2++) {
        const int s2 = convert_c_index_to_shifted_index(c2, n2);
        if (g->use_halfspace && s0 == 0 &&
            ((s1 == 0 && s2 > 0) || s1 > 0))
          continue;
        double gsq = 0.0;
        if (g->spherical && g->cutoff > 0.0) {
          gsq = squared_length_of_g_vector(
              (const int[3]){s0, s1, s2}, g->cell_info.dhmat);
          if (0.5 * gsq > g->cutoff)
            continue;
        } else {
          gsq = squared_length_of_g_vector(
              (const int[3]){s0, s1, s2}, g->cell_info.dhmat);
        }
        if (nlocal == capacity) {
          capacity *= 2;
          index_to_g = realloc(index_to_g, (size_t)capacity * sizeof(int[3]));
          gsquared = realloc(gsquared, (size_t)capacity * sizeof(double));
          cp_assert(index_to_g != NULL && gsquared != NULL);
        }
        index_to_g[nlocal][0] = c0;
        index_to_g[nlocal][1] = c1;
        index_to_g[nlocal][2] = c2;
        gsquared[nlocal] = gsq;
        nlocal++;
      }
    }
  }

  int *mapping = malloc(map_size * sizeof(int));
  cp_assert(mapping != NULL);
  memcpy(mapping, g->xy_to_process, map_size * sizeof(int));
  const int status = grid_enable_ray_distribution_internal(
      g, mapping, nlocal, (const int(*)[3])index_to_g, gsquared, false);
  free(mapping);
  free(index_to_g);
  free(gsquared);
  return status;
}

int grid_finalize_ray_distribution(fft_grid_layout *g, const int nlocal,
                                    const int (*index_to_g)[3],
                                    const double *gsquared) {
  cp_assert(g != NULL && g->xy_to_process != NULL);
  const size_t map_size =
      (size_t)g->npts_global_gspace[0] * (size_t)g->npts_global[1];
  int *mapping = malloc(map_size * sizeof(int));
  cp_assert(mapping != NULL);
  memcpy(mapping, g->xy_to_process, map_size * sizeof(int));
  const int status = grid_enable_ray_distribution(
      g, mapping, nlocal, index_to_g, gsquared);
  free(mapping);
  return status;
}

void grid_set_spherical(fft_grid_layout *g, const bool spherical) {
  cp_assert(g != NULL);
  g->spherical = spherical;
}

void grid_set_yz_mapping(fft_grid_layout *g, const int *yz_to_process) {
  cp_assert(g != NULL && yz_to_process != NULL);
  const size_t size =
      (size_t)g->npts_global[0] * (size_t)g->npts_global[1];
  free(g->yz_to_process);
  g->yz_to_process = malloc(size * sizeof(int));
  cp_assert(g->yz_to_process != NULL);
  memcpy(g->yz_to_process, yz_to_process, size * sizeof(int));
}

int grid_finalize_distribution(fft_grid_layout *g, const bool use_ray,
                               const int *xy_to_process, const int nlocal,
                               const int (*index_to_g)[3],
                               const double *gsquared) {
  if (!use_ray) {
    cp_assert(!g->ray_distribution &&
              "Blocked finalization requires a blocked layout");
    return 0;
  }
  cp_assert(!g->ray_distribution &&
            "The layout must be blocked before ray finalization");
  return grid_enable_ray_distribution(g, xy_to_process, nlocal, index_to_g,
                                      gsquared);
}

void grid_create_fft_grid_layout(fft_grid_layout **fft_grid,
                                 const cp_mpi_comm_t comm,
                                 const int npts_global[3],
                                 const double hmat[3][3],
                                 const bool use_halfspace, const double cutoff,
                                 const int *pgrid_guess,
                                 const int *external_local_bounds,
                                 const bool use_ray_distribution) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "create_grid_layout");
  const int handle = fft_start_timer(routine_name);
  fft_grid_layout *my_fft_grid = NULL;
  if (*fft_grid != NULL) {
    my_fft_grid = *fft_grid;
    grid_free_fft_grid_layout(*fft_grid);
  }
  my_fft_grid = calloc(1, sizeof(fft_grid_layout));

  const int number_of_processes = cp_mpi_comm_size(comm);
  const int my_process = cp_mpi_comm_rank(comm);

  my_fft_grid->grid_id = current_grid_id;
  my_fft_grid->ref_grid_id = current_grid_id;
  current_grid_id++;
  my_fft_grid->ref_counter = 1;
  my_fft_grid->ray_distribution = use_ray_distribution;

  my_fft_grid->use_halfspace = use_halfspace;
  memcpy(my_fft_grid->npts_global, npts_global, 3 * sizeof(int));
  memcpy(my_fft_grid->npts_global_gspace, npts_global, 3 * sizeof(int));
  if (my_fft_grid->use_halfspace)
    my_fft_grid->npts_global_gspace[0] = npts_global[0] / 2 + 1;
  my_fft_grid->cell_info.volume = hmat[0][0] * hmat[1][1] * hmat[2][2] +
                                  hmat[0][1] * hmat[1][2] * hmat[2][0] +
                                  hmat[0][2] * hmat[1][0] * hmat[2][1] -
                                  hmat[0][2] * hmat[1][1] * hmat[2][0] -
                                  hmat[0][1] * hmat[1][0] * hmat[2][2] -
                                  hmat[0][0] * hmat[1][2] * hmat[2][1];
  // The Fortran interface passes the inverse cell matrix (dh_inv).
  for (int dir = 0; dir < 3; dir++)
    for (int dir2 = 0; dir2 < 3; dir2++)
      my_fft_grid->cell_info.dhmat[dir][dir2] = hmat[dir][dir2];
  for (int dir = 0; dir < 3; dir++) {
    my_fft_grid->cell_info.dr[dir] =
        sqrt(my_fft_grid->cell_info.dhmat[dir][0] *
                 my_fft_grid->cell_info.dhmat[dir][0] +
             my_fft_grid->cell_info.dhmat[dir][1] *
                 my_fft_grid->cell_info.dhmat[dir][1] +
             my_fft_grid->cell_info.dhmat[dir][2] *
                 my_fft_grid->cell_info.dhmat[dir][2]);
  }
  my_fft_grid->cell_info.volume = fabs(my_fft_grid->cell_info.volume);
  my_fft_grid->cell_info.dvolume =
      my_fft_grid->cell_info.volume /
      ((double)npts_global[0] * (double)npts_global[1] *
       (double)npts_global[2]);
  my_fft_grid->cutoff = cutoff;

  int periodic[2] = {1, 1};

  if (external_local_bounds != NULL) {
    // Check whether the externally provided bounds are valid
    // First, collect the bounds
    int *all_bounds = malloc(6 * number_of_processes * sizeof(int));
    cp_mpi_allgather_int(external_local_bounds, 6, all_bounds, 6, comm);
    // Align the lower bound to 0
    int lower_bound[3] = {0, 0, 0};
    for (int dir = 0; dir < 3; dir++)
      lower_bound[dir] = all_bounds[2 * dir];
    for (int process = 0; process < number_of_processes; process++) {
      for (int dir = 0; dir < 3; dir++)
        lower_bound[dir] =
            imin(lower_bound[dir], all_bounds[6 * process + 2 * dir]);
    }
    // Shift the bounds by the lowest element in each direction
    // and ensure the size of each chunk to be at least 0
    // Check whether the bounds lay within of the global bounds and the
    // x-coordinate is local Check whether we cover the whole space
    long int total_number_of_elements = 0;
    for (int process = 0; process < number_of_processes; process++) {
      for (int dir = 0; dir < 3; dir++) {
        all_bounds[6 * process + 2 * dir] -= lower_bound[dir];
        all_bounds[6 * process + 2 * dir + 1] -= lower_bound[dir];
        if (all_bounds[6 * process + 2 * dir] >
            all_bounds[6 * process + 2 * dir + 1])
          all_bounds[6 * process + 2 * dir + 1] =
              all_bounds[6 * process + 2 * dir] - 1;
        cp_assert(all_bounds[6 * process + 2 * dir + 1] < npts_global[dir] &&
               "Some externally provided bounds do not lay within the actual "
               "range");
      }
      cp_assert(all_bounds[6 * process] == 0 &&
             all_bounds[6 * process + 1] == npts_global[0] - 1 &&
             "The most outer index needs to be local!");
      total_number_of_elements += ((long int)all_bounds[6 * process + 1] -
                                   all_bounds[6 * process] + 1) *
                                  ((long int)all_bounds[6 * process + 3] -
                                   all_bounds[6 * process + 2] + 1) *
                                  ((long int)all_bounds[6 * process + 5] -
                                   all_bounds[6 * process + 4] + 1);
    }
    cp_assert(total_number_of_elements ==
               (long int)npts_global[0] * npts_global[1] * npts_global[2] &&
           "In real-space, we must cover the whole 3D space");
    // Check whether the bounds of two processes either overlap or are different
    // in each direction
    for (int process1 = 0; process1 < number_of_processes; process1++) {
      for (int process2 = process1 + 1; process2 < number_of_processes;
           process2++) {
        for (int dir = 0; dir < 3; dir++) {
          cp_assert((all_bounds[6 * process1 + 2 * dir] >
                      all_bounds[6 * process2 + 2 * dir + 1] ||
                  all_bounds[6 * process1 + 2 * dir + 1] <
                      all_bounds[6 * process2 + 2 * dir] ||
                  (all_bounds[6 * process1 + 2 * dir] ==
                       all_bounds[6 * process2 + 2 * dir] &&
                   all_bounds[6 * process1 + 2 * dir + 1] ==
                       all_bounds[6 * process2 + 2 * dir + 1])) &&
                 "Bounds between two processes in each direction need to be "
                 "non-overlapping or identical!");
        }
        // Ensure bounds of different processes do not match or else the length
        // in at least one direction is zero
        cp_assert((memcmp(all_bounds + 6 * process1, all_bounds + 6 * process2,
                       6 * sizeof(int)) != 0 ||
                all_bounds[6 * process1] > all_bounds[6 * process1 + 1] ||
                all_bounds[6 * process1 + 2] > all_bounds[6 * process1 + 3] ||
                all_bounds[6 * process1 + 4] > all_bounds[6 * process1 + 5]) &&
               "Two different processes need to have different bounds and not "
               "carry any data!");
      }
    }
    // Determine the underlying process grid
    // We loop over all processes and find the ranges of each process row
    int *starts_first_dimension = malloc(number_of_processes * sizeof(int));
    int size_of_first_dimension = 0;
    int current_first_position = 0;
    while (current_first_position < npts_global[2]) {
      for (int process = 0; process < number_of_processes; process++) {
        if (current_first_position == all_bounds[6 * process + 4]) {
          cp_assert(size_of_first_dimension < number_of_processes);
          starts_first_dimension[size_of_first_dimension] =
              current_first_position;
          size_of_first_dimension++;
          current_first_position = all_bounds[6 * process + 5] + 1;
        }
      }
    }
    // Next, for the other direction
    int *starts_second_dimension = malloc(number_of_processes * sizeof(int));
    int size_of_second_dimension = 0;
    int current_second_position = 0;
    while (current_second_position < npts_global[1]) {
      for (int process = 0; process < number_of_processes; process++) {
        if (current_second_position == all_bounds[6 * process + 2]) {
          cp_assert(size_of_second_dimension < number_of_processes);
          starts_second_dimension[size_of_second_dimension] =
              current_second_position;
          size_of_second_dimension++;
          current_second_position = all_bounds[6 * process + 3] + 1;
        }
      }
    }
    cp_assert(size_of_first_dimension * size_of_second_dimension ==
               number_of_processes &&
           "Incorrectly determined process grid! This should not happen!");
    my_fft_grid->proc_grid[0] = size_of_first_dimension;
    my_fft_grid->proc_grid[1] = size_of_second_dimension;

    // Attempt a trivial mapping
    int proc_coord_1 = -1;
    int proc_coord_2 = -1;
    for (int proc_coord = 0; proc_coord < size_of_first_dimension;
         proc_coord++) {
      if (all_bounds[6 * my_process + 4] ==
          starts_first_dimension[proc_coord]) {
        proc_coord_1 = proc_coord;
        break;
      }
    }
    cp_assert(proc_coord_1 >= 0 && "Lower bound in z-direction not found");
    for (int proc_coord = 0; proc_coord < size_of_second_dimension;
         proc_coord++) {
      if (all_bounds[6 * my_process + 2] ==
          starts_second_dimension[proc_coord]) {
        proc_coord_2 = proc_coord;
        break;
      }
    }
    cp_assert(proc_coord_2 >= 0 && "Lower bound in y-direction not found");
    // Trivial mapping not successful, so we need a new communicator
    if (all_bounds[6 * my_process + 4] !=
            starts_first_dimension[proc_coord_1] ||
        all_bounds[6 * my_process + 2] !=
            starts_second_dimension[proc_coord_2]) {
      // We need to create a new communicator with the correct order without
      cp_assert(false && "Nontrivial mapping NYI");

      my_fft_grid->comm =
          cp_mpi_cart_create(comm, 2, my_fft_grid->proc_grid, periodic, true);
    } else {
      my_fft_grid->comm =
          cp_mpi_cart_create(comm, 2, my_fft_grid->proc_grid, periodic, true);
    }

    free(starts_first_dimension);
    free(starts_second_dimension);

    cp_mpi_cart_get(my_fft_grid->comm, 2, my_fft_grid->proc_grid, periodic,
                    my_fft_grid->proc_coords);

    my_fft_grid->sub_comm[0] =
        cp_mpi_cart_sub(my_fft_grid->comm, (const int[2]){1, 0});
    my_fft_grid->sub_comm[1] =
        cp_mpi_cart_sub(my_fft_grid->comm, (const int[2]){0, 1});
    cp_assert(cp_mpi_comm_size(my_fft_grid->sub_comm[0]) ==
           my_fft_grid->proc_grid[0]);
    cp_assert(cp_mpi_comm_size(my_fft_grid->sub_comm[1]) ==
           my_fft_grid->proc_grid[1]);

    setup_proc2local(my_fft_grid, all_bounds);
    free(all_bounds);
  } else {
    my_fft_grid->proc_grid[0] = -1;
    my_fft_grid->proc_grid[1] = -1;
    if (pgrid_guess)
      memcpy(my_fft_grid->proc_grid, pgrid_guess, 2 * sizeof(int));
    if (my_fft_grid->proc_grid[0]) {
      if (my_fft_grid->proc_grid[1]) {
        if (my_fft_grid->proc_grid[0] * my_fft_grid->proc_grid[1] !=
            number_of_processes) {
          my_fft_grid->proc_grid[0] = -1;
          my_fft_grid->proc_grid[1] = -1;
        }
      } else if (number_of_processes % my_fft_grid->proc_grid[0] == 0) {
        my_fft_grid->proc_grid[1] =
            number_of_processes / my_fft_grid->proc_grid[0];
      } else {
        my_fft_grid->proc_grid[0] = -1;
        my_fft_grid->proc_grid[1] = -1;
      }
    } else if (my_fft_grid->proc_grid[1]) {
      if (number_of_processes % my_fft_grid->proc_grid[1] == 0) {
        my_fft_grid->proc_grid[0] =
            number_of_processes / my_fft_grid->proc_grid[1];
      } else {
        my_fft_grid->proc_grid[1] = -1;
      }
    }
    if (my_fft_grid->proc_grid[0] <= 0 || my_fft_grid->proc_grid[1] <= 0 ||
        my_fft_grid->proc_grid[0] * my_fft_grid->proc_grid[1] !=
            number_of_processes) {
      // Split the last dimension in real-space
      if (npts_global[2] < number_of_processes) {
        // We only distribute in two directions if necessary to reduce
        // communication
        my_fft_grid->proc_grid[0] = 0;
        my_fft_grid->proc_grid[1] = 0;
        cp_mpi_dims_create(number_of_processes, 2, my_fft_grid->proc_grid);
        // Swap dimension if the large process dimension is not on the large
        // global dimension
        if ((npts_global[2] - npts_global[1]) *
                (my_fft_grid->proc_grid[0] - my_fft_grid->proc_grid[1]) <
            0) {
          const int proc_grid_0 = my_fft_grid->proc_grid[0];
          my_fft_grid->proc_grid[0] = my_fft_grid->proc_grid[1];
          my_fft_grid->proc_grid[1] = proc_grid_0;
        }
      } else {
        my_fft_grid->proc_grid[0] = number_of_processes;
        my_fft_grid->proc_grid[1] = 1;
      }
    }
    cp_assert(my_fft_grid->proc_grid[0] && my_fft_grid->proc_grid[1] &&
           my_fft_grid->proc_grid[0] * my_fft_grid->proc_grid[1] ==
               number_of_processes);

    my_fft_grid->comm =
        cp_mpi_cart_create(comm, 2, my_fft_grid->proc_grid, periodic, true);

    cp_mpi_cart_get(my_fft_grid->comm, 2, my_fft_grid->proc_grid, periodic,
                    my_fft_grid->proc_coords);

    my_fft_grid->sub_comm[0] =
        cp_mpi_cart_sub(my_fft_grid->comm, (const int[2]){1, 0});
    my_fft_grid->sub_comm[1] =
        cp_mpi_cart_sub(my_fft_grid->comm, (const int[2]){0, 1});
    cp_assert(cp_mpi_comm_size(my_fft_grid->sub_comm[0]) ==
           my_fft_grid->proc_grid[0]);
    cp_assert(cp_mpi_comm_size(my_fft_grid->sub_comm[1]) ==
           my_fft_grid->proc_grid[1]);

    setup_proc2local(my_fft_grid, NULL);
  }

  int (*bounds_gs)[2] = my_fft_grid->proc2local_gs[my_process];

  int number_of_positive_gs_points =
      bounds_gs[0][1] * bounds_gs[1][1] * bounds_gs[2][1];
  int number_of_negative_gs_points = 0;
  if (use_halfspace) {
    // If this process carries data from the first element or independently the
    // half index in x-direction, we need to subtract the number of related
    // elements
    int number_of_x_elements = bounds_gs[0][1];
    if (bounds_gs[0][1] > 0) {
      if (bounds_gs[0][0] == 0)
        number_of_x_elements--;
      if (npts_global[0] % 2 == 0 &&
          bounds_gs[0][0] + bounds_gs[0][1] - 1 == npts_global[0] / 2)
        number_of_x_elements--;
    }
    number_of_negative_gs_points =
        number_of_x_elements * bounds_gs[1][1] * bounds_gs[2][1];
  }
  my_fft_grid->number_of_positive_gs_points = number_of_positive_gs_points;
  my_fft_grid->number_of_negative_gs_points = number_of_negative_gs_points;

  my_fft_grid->npts_gs_local =
      number_of_positive_gs_points + number_of_negative_gs_points;
  my_fft_grid->buffer_size =
      imax(my_fft_grid->buffer_size, my_fft_grid->npts_gs_local);

  my_fft_grid->xy_to_process = NULL;
  my_fft_grid->ray_to_xy = NULL;
  my_fft_grid->xy_to_ray = NULL;
  // Even for blocked g-space distributions provide the number of yz rays per
  // process (y length * z length), so the counts can always be fetched.
  my_fft_grid->rays_per_process = calloc(number_of_processes, sizeof(int));
  for (int process = 0; process < number_of_processes; process++)
    my_fft_grid->rays_per_process[process] =
        my_fft_grid->proc2local_gs[process][0][1] *
        my_fft_grid->proc2local_gs[process][1][1];
  my_fft_grid->index_to_g = calloc(my_fft_grid->npts_gs_local, sizeof(int[3]));
  if (use_halfspace) {
    my_fft_grid->index_to_cart = NULL;
    my_fft_grid->index_to_cart_pos =
        calloc(number_of_positive_gs_points, sizeof(int[2]));
    my_fft_grid->index_to_cart_neg =
        calloc(number_of_negative_gs_points, sizeof(int[2]));
    my_fft_grid->index_to_ray = NULL;
    my_fft_grid->index_to_ray_pos =
        calloc(number_of_positive_gs_points, sizeof(int[2]));
    my_fft_grid->index_to_ray_neg =
        calloc(number_of_negative_gs_points, sizeof(int[2]));
  } else {
    my_fft_grid->index_to_cart =
        calloc(number_of_positive_gs_points, sizeof(int));
    my_fft_grid->index_to_cart_pos = NULL;
    my_fft_grid->index_to_cart_neg = NULL;
    my_fft_grid->index_to_ray =
        calloc(number_of_positive_gs_points, sizeof(int));
    my_fft_grid->index_to_ray_pos = NULL;
    my_fft_grid->index_to_ray_neg = NULL;
  }
  my_fft_grid->index_to_ray = NULL;
  my_fft_grid->index_to_ray_neg = NULL;
  my_fft_grid->index_to_ray_pos = NULL;
  const int local_size_y = bounds_gs[1][1];
  const int local_size_z = bounds_gs[2][1];
#pragma omp parallel for default(none)                                         \
    shared(my_fft_grid, bounds_gs, local_size_y, local_size_z,                 \
               number_of_positive_gs_points)
  for (int index = 0; index < number_of_positive_gs_points; index++) {
    my_fft_grid->index_to_g[index][0] =
        bounds_gs[0][0] + index / local_size_y / local_size_z;
    cp_assert(my_fft_grid->index_to_g[index][0] <
               my_fft_grid->npts_global_gspace[0] &&
           my_fft_grid->index_to_g[index][0] >= 0);
    my_fft_grid->index_to_g[index][1] =
        bounds_gs[1][0] + (index / local_size_z) % local_size_y;
    my_fft_grid->index_to_g[index][2] = bounds_gs[2][0] + index % local_size_z;
  }
  if (use_halfspace) {
    int negative_index = number_of_positive_gs_points;
    for (int index = 0; index < number_of_positive_gs_points; index++) {
      const int index_x = my_fft_grid->index_to_g[index][0];
      if (index_x == 0 ||
          (npts_global[0] % 2 == 0 && index_x == npts_global[0] / 2))
        continue;
      my_fft_grid->index_to_g[negative_index][0] = npts_global[0] - index_x;
      my_fft_grid->index_to_g[negative_index][1] =
          (npts_global[1] - my_fft_grid->index_to_g[index][1]) % npts_global[1];
      my_fft_grid->index_to_g[negative_index][2] =
          (npts_global[2] - my_fft_grid->index_to_g[index][2]) % npts_global[2];
      negative_index++;
    }
    cp_assert(negative_index == my_fft_grid->npts_gs_local);
  }
  // Ensure that my_fft_grid->npts_gs_local is nonnegative to prevent compiler
  // warnings
  my_fft_grid->npts_gs_local =
      my_fft_grid->npts_gs_local >= 0 ? my_fft_grid->npts_gs_local : 0;
  my_fft_grid->local_index_to_ref_grid =
      calloc((size_t)my_fft_grid->npts_gs_local, sizeof(int));
  for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
    my_fft_grid->local_index_to_ref_grid[index] = index;
  }

  sort_g_vectors(my_fft_grid);

  if (use_halfspace) {
    int positive_index = 0;
    int negative_index = 0;
    if (fft_lib_use_mpi() && my_fft_grid->proc_grid[0] > 1 &&
        my_fft_grid->proc_grid[1] == 1) {
      for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
        int *index_g = my_fft_grid->index_to_g[index];
        if (index_g[0] < my_fft_grid->npts_global_gspace[0]) {
          my_fft_grid->index_to_cart_pos[positive_index][0] = index;
          my_fft_grid->index_to_cart_pos[positive_index][1] =
              ((index_g[1] - bounds_gs[1][0]) * bounds_gs[2][1] +
               (index_g[2] - bounds_gs[2][0])) *
                  bounds_gs[0][1] +
              (index_g[0] - bounds_gs[0][0]);
          positive_index++;
        } else {
          my_fft_grid->index_to_cart_neg[negative_index][0] = index;
          my_fft_grid->index_to_cart_neg[negative_index][1] =
              (((npts_global[1] - index_g[1]) % npts_global[1] -
                bounds_gs[1][0]) *
                   bounds_gs[2][1] +
               ((npts_global[2] - index_g[2]) % npts_global[2] -
                bounds_gs[2][0])) *
                  bounds_gs[0][1] +
              ((npts_global[0] - index_g[0]) % npts_global[0] -
               bounds_gs[0][0]);
          negative_index++;
        }
      }
    } else {
      for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
        int *index_g = my_fft_grid->index_to_g[index];
        if (index_g[0] < my_fft_grid->npts_global_gspace[0]) {
          my_fft_grid->index_to_cart_pos[positive_index][0] = index;
          my_fft_grid->index_to_cart_pos[positive_index][1] =
              ((index_g[0] - bounds_gs[0][0]) * bounds_gs[1][1] +
               (index_g[1] - bounds_gs[1][0])) *
                  bounds_gs[2][1] +
              (index_g[2] - bounds_gs[2][0]);
          positive_index++;
        } else {
          my_fft_grid->index_to_cart_neg[negative_index][0] = index;
          my_fft_grid->index_to_cart_neg[negative_index][1] =
              (((npts_global[0] - index_g[0]) % npts_global[0] -
                bounds_gs[0][0]) *
                   bounds_gs[1][1] +
               ((npts_global[1] - index_g[1]) % npts_global[1] -
                bounds_gs[1][0])) *
                  bounds_gs[2][1] +
              ((npts_global[2] - index_g[2]) % npts_global[2] -
               bounds_gs[2][0]);
          negative_index++;
        }
      }
    }
    cp_assert(positive_index == my_fft_grid->number_of_positive_gs_points);
    cp_assert(negative_index == my_fft_grid->number_of_negative_gs_points);
  } else {
    if (fft_lib_use_mpi() && my_fft_grid->proc_grid[0] > 1 &&
        my_fft_grid->proc_grid[1] == 1) {
      for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
        int *index_g = my_fft_grid->index_to_g[index];
        my_fft_grid->index_to_cart[index] =
            ((index_g[1] - bounds_gs[1][0]) * bounds_gs[2][1] +
             (index_g[2] - bounds_gs[2][0])) *
                bounds_gs[0][1] +
            (index_g[0] - bounds_gs[0][0]);
      }
    } else {
      for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
        int *index_g = my_fft_grid->index_to_g[index];
        my_fft_grid->index_to_cart[index] =
            ((index_g[0] - bounds_gs[0][0]) * bounds_gs[1][1] +
             (index_g[1] - bounds_gs[1][0])) *
                bounds_gs[2][1] +
            (index_g[2] - bounds_gs[2][0]);
      }
    }
  }

  my_fft_grid->index_to_gsquared =
      calloc(my_fft_grid->npts_gs_local, sizeof(double));
#pragma omp parallel for default(none) shared(my_fft_grid)
  for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
    my_fft_grid->index_to_gsquared[index] = squared_length_of_g_vector(
        (const int[3]){
            shifted_index_from_c_index(my_fft_grid,
                                       my_fft_grid->index_to_g[index][0], 0),
            shifted_index_from_c_index(my_fft_grid,
                                       my_fft_grid->index_to_g[index][1], 1),
            shifted_index_from_c_index(my_fft_grid,
                                       my_fft_grid->index_to_g[index][2], 2)},
        my_fft_grid->cell_info.dhmat);
  }

  my_fft_grid->redistribution = calloc(1, sizeof(fft_redistribution_t));
  prepare_redistribution(
      my_fft_grid->redistribution, my_fft_grid->npts_global_gspace,
      my_fft_grid->proc2local_ms, my_fft_grid->proc2local_x_gs,
      my_fft_grid->proc2local_y_rs, my_fft_grid->proc2local_y_gs,
      my_fft_grid->proc2local_z_rs, my_fft_grid->rays_per_process,
      my_fft_grid->ray_to_xy, my_fft_grid->comm, my_fft_grid->sub_comm);

  fft_register_3d(my_fft_grid);

  if (use_ray_distribution) {
    const int n0 = my_fft_grid->npts_global_gspace[0];
    const int n1 = my_fft_grid->npts_global[1];
    const size_t map_size = (size_t)n0 * (size_t)n1;
    int *xy_to_process = malloc(map_size * sizeof(int));
    for (int c0 = 0; c0 < n0; c0++)
      for (int c1 = 0; c1 < n1; c1++)
        xy_to_process[(size_t)c0 * n1 + c1] =
            (int)(((size_t)c0 * n1 + c1) % (size_t)number_of_processes);

    int nlocal = 0;
    for (int c0 = 0; c0 < n0; c0++) {
      for (int c1 = 0; c1 < n1; c1++) {
        if (xy_to_process[(size_t)c0 * n1 + c1] != my_process)
          continue;
        for (int c2 = 0; c2 < my_fft_grid->npts_global[2]; c2++) {
          const int shifted[3] = {
              convert_c_index_to_shifted_index(c0, my_fft_grid->npts_global[0]),
              convert_c_index_to_shifted_index(c1, my_fft_grid->npts_global[1]),
              convert_c_index_to_shifted_index(c2,
                                               my_fft_grid->npts_global[2])};
          if (my_fft_grid->use_halfspace && shifted[0] == 0 &&
              ((shifted[1] == 0 && shifted[2] > 0) || shifted[1] > 0))
            continue;
          if (cutoff > 0.0) {
            const double gsq = squared_length_of_g_vector(
                (const int[3]){shifted[0], shifted[1], shifted[2]},
                my_fft_grid->cell_info.dhmat);
            if (0.5 * gsq > cutoff)
              continue;
          }
          nlocal++;
        }
      }
    }
    int (*index_to_g)[3] = malloc((size_t)nlocal * sizeof(int[3]));
    double *gsquared = malloc((size_t)nlocal * sizeof(double));
    int index = 0;
    for (int c0 = 0; c0 < n0; c0++) {
      for (int c1 = 0; c1 < n1; c1++) {
        if (xy_to_process[(size_t)c0 * n1 + c1] != my_process)
          continue;
        for (int c2 = 0; c2 < my_fft_grid->npts_global[2]; c2++) {
          const int shifted[3] = {
              convert_c_index_to_shifted_index(c0, my_fft_grid->npts_global[0]),
              convert_c_index_to_shifted_index(c1, my_fft_grid->npts_global[1]),
              convert_c_index_to_shifted_index(c2,
                                               my_fft_grid->npts_global[2])};
          if (my_fft_grid->use_halfspace && shifted[0] == 0 &&
              ((shifted[1] == 0 && shifted[2] > 0) || shifted[1] > 0))
            continue;
          double gsq = 0.0;
          if (cutoff > 0.0) {
            gsq = squared_length_of_g_vector(
                (const int[3]){shifted[0], shifted[1], shifted[2]},
                my_fft_grid->cell_info.dhmat);
            if (0.5 * gsq > cutoff)
              continue;
          }
          index_to_g[index][0] = c0;
          index_to_g[index][1] = c1;
          index_to_g[index][2] = c2;
          gsquared[index] =
              cutoff > 0.0
                  ? gsq
                  : squared_length_of_g_vector(
                        (const int[3]){shifted[0], shifted[1], shifted[2]},
                        my_fft_grid->cell_info.dhmat);
          index++;
        }
      }
    }
    cp_assert(index == nlocal);
    grid_enable_ray_distribution(my_fft_grid, xy_to_process, nlocal, index_to_g,
                                 gsquared);
    free(xy_to_process);
    free(index_to_g);
    free(gsquared);
  }

  *fft_grid = my_fft_grid;

  fft_stop_timer(handle);
}

void grid_create_fft_grid_layout_from_reference(
    fft_grid_layout **fft_grid, const int npts_global[3], const double cutoff,
    const int *external_local_bounds, const fft_grid_layout *fft_grid_ref,
    const bool use_ray_distribution) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH,
           "create_grid_layout_from_ref");
  const int handle = fft_start_timer(routine_name);
  cp_assert(fft_grid_ref != NULL &&
         "Grid creation from reference grid requires a valid reference grid!");
  // Current restriction of the code.
  // cp_assert(!fft_grid_ref->ray_distribution &&
  //"The reference grid has to have a blocked distribution!");
  // We will use the reference grid to collect the data from other grids. To
  // prevent loss of accuracy, we require the new grid to be coarser than or as
  // coarse as the reference grid.
  cp_assert(npts_global[0] <= fft_grid_ref->npts_global[0] &&
         npts_global[1] <= fft_grid_ref->npts_global[1] &&
         npts_global[2] <= fft_grid_ref->npts_global[2] &&
         "The new grid cannot have more grid points in any direction than the "
         "reference grid!");

  const int number_of_processes = cp_mpi_comm_size(fft_grid_ref->comm);
  // const int my_process = cp_mpi_comm_rank(fft_grid_ref->comm);

  fft_grid_layout *my_fft_grid = NULL;
  if (*fft_grid != NULL) {
    my_fft_grid = *fft_grid;
    grid_free_fft_grid_layout(*fft_grid);
  }
  my_fft_grid = calloc(1, sizeof(fft_grid_layout));

  my_fft_grid->grid_id = current_grid_id;
  my_fft_grid->ref_grid_id = fft_grid_ref->grid_id;
  current_grid_id++;
  my_fft_grid->ref_counter = 1;

  my_fft_grid->ray_distribution = use_ray_distribution;

  if (npts_global[2] < number_of_processes) {
    // We only distribute in two directions if necessary to reduce communication
    my_fft_grid->proc_grid[0] = 0;
    my_fft_grid->proc_grid[1] = 0;
    cp_mpi_dims_create(number_of_processes, 2, my_fft_grid->proc_grid);
    // Swap dimension if the large process dimension is not on the large global
    // dimension
    if ((npts_global[2] - npts_global[1]) *
                (my_fft_grid->proc_grid[0] - my_fft_grid->proc_grid[1]) <
            0 &&
        my_fft_grid->proc_grid[1] > 1) {
      const int proc_grid_0 = my_fft_grid->proc_grid[0];
      my_fft_grid->proc_grid[0] = my_fft_grid->proc_grid[1];
      my_fft_grid->proc_grid[1] = proc_grid_0;
    }
  } else {
    my_fft_grid->proc_grid[0] = number_of_processes;
    my_fft_grid->proc_grid[1] = 1;
  }

  my_fft_grid->use_halfspace = fft_grid_ref->use_halfspace;
  memcpy(my_fft_grid->npts_global, npts_global, 3 * sizeof(int));
  memcpy(my_fft_grid->npts_global_gspace, npts_global, 3 * sizeof(int));
  if (my_fft_grid->use_halfspace)
    my_fft_grid->npts_global_gspace[0] = npts_global[0] / 2 + 1;
  for (int dir = 0; dir < 3; dir++) {
    for (int dir2 = 0; dir2 < 3; dir2++) {
      my_fft_grid->cell_info.dhmat[dir][dir2] =
          fft_grid_ref->cell_info.dhmat[dir][dir2] *
          ((double)fft_grid_ref->npts_global[dir2]) /
          ((double)npts_global[dir2]);
    }
  }
  my_fft_grid->cutoff = cutoff ? cutoff : fft_grid_ref->cutoff;

  int periodic[2] = {1, 1};
  my_fft_grid->comm = cp_mpi_cart_create(
      fft_grid_ref->comm, 2, my_fft_grid->proc_grid, periodic, false);

  cp_mpi_cart_get(my_fft_grid->comm, 2, my_fft_grid->proc_grid, periodic,
                  my_fft_grid->proc_coords);

  my_fft_grid->sub_comm[0] =
      cp_mpi_cart_sub(my_fft_grid->comm, (const int[2]){1, 0});
  my_fft_grid->sub_comm[1] =
      cp_mpi_cart_sub(my_fft_grid->comm, (const int[2]){0, 1});
  cp_assert(cp_mpi_comm_size(my_fft_grid->sub_comm[0]) ==
         my_fft_grid->proc_grid[0]);
  cp_assert(cp_mpi_comm_size(my_fft_grid->sub_comm[1]) ==
         my_fft_grid->proc_grid[1]);

  setup_proc2local(my_fft_grid, external_local_bounds);

  // Assign the (xy)-rays of the reference grid which are also on the current
  // grid to each process
  my_fft_grid->xy_to_process =
      malloc(my_fft_grid->npts_global_gspace[0] *
             my_fft_grid->npts_global_gspace[1] * sizeof(int));
  memset(my_fft_grid->xy_to_process, -1,
         my_fft_grid->npts_global_gspace[0] *
             my_fft_grid->npts_global_gspace[1] * sizeof(int));
  // Count the number of rays on each process
  my_fft_grid->rays_per_process = calloc(number_of_processes, sizeof(int));
  int total_number_of_rays = 0;
#pragma omp parallel for default(none)                                         \
    shared(my_fft_grid, fft_grid_ref, npts_global, number_of_processes,        \
               stdout) reduction(+ : total_number_of_rays)
  for (int process = 0; process < number_of_processes; process++) {
    for (int index_x = fft_grid_ref->proc2local_gs[process][0][0];
         index_x <= fft_grid_ref->proc2local_gs[process][0][0] +
                        fft_grid_ref->proc2local_gs[process][0][1] - 1;
         index_x++) {
      // The right half of the indices is shifted
      const int index_x_shifted = convert_c_index_to_shifted_index(
          index_x, fft_grid_ref->npts_global[0]);
      // Compare the shifted index with the allowed subset of shifted indices of
      // the new grid The allowed set is given by -(n-1)//2...n//2 (these are
      // always n elements)
      if (!is_on_grid(index_x_shifted, npts_global[0]))
        continue;
      const int index_x_new =
          convert_shifted_index_to_c_index(index_x_shifted, npts_global[0]);
      for (int index_y = fft_grid_ref->proc2local_gs[process][1][0];
           index_y <= fft_grid_ref->proc2local_gs[process][1][0] +
                          fft_grid_ref->proc2local_gs[process][1][1] - 1;
           index_y++) {
        // The right half of the indices is shifted
        const int index_y_shifted = convert_c_index_to_shifted_index(
            index_y, fft_grid_ref->npts_global[1]);
        // Same check for z-coordinate
        if (!is_on_grid(index_y_shifted, npts_global[1]))
          continue;
        const int index_y_new =
            convert_shifted_index_to_c_index(index_y_shifted, npts_global[1]);
        const int xy_index =
            index_x_new * my_fft_grid->npts_global_gspace[1] + index_y_new;
        cp_assert(my_fft_grid->xy_to_process[xy_index] < 0);
        my_fft_grid->xy_to_process[xy_index] = process;
        my_fft_grid->rays_per_process[process]++;
        total_number_of_rays++;
      }
    }
  }
  int own_index = 0;
  for (int ref_index = 0; ref_index < fft_grid_ref->npts_gs_local;
       ref_index++) {
    const int shifted_indices[3] = {
        convert_c_index_to_shifted_index(fft_grid_ref->index_to_g[ref_index][0],
                                         fft_grid_ref->npts_global[0]),
        convert_c_index_to_shifted_index(fft_grid_ref->index_to_g[ref_index][1],
                                         fft_grid_ref->npts_global[1]),
        convert_c_index_to_shifted_index(fft_grid_ref->index_to_g[ref_index][2],
                                         fft_grid_ref->npts_global[2])};
    if (is_on_grid(shifted_indices[0], my_fft_grid->npts_global[0]) &&
        is_on_grid(shifted_indices[1], my_fft_grid->npts_global[1]) &&
        is_on_grid(shifted_indices[2], my_fft_grid->npts_global[2])) {
      my_fft_grid->npts_gs_local++;
      my_fft_grid->number_of_positive_gs_points += shifted_indices[0] >= 0;
      my_fft_grid->number_of_negative_gs_points += shifted_indices[0] < 0;
    }
  }
  cp_assert(my_fft_grid->npts_gs_local ==
         my_fft_grid->number_of_positive_gs_points +
             my_fft_grid->number_of_negative_gs_points);
  my_fft_grid->buffer_size =
      imax(my_fft_grid->buffer_size, my_fft_grid->npts_gs_local);

  int *ray_offsets = calloc(number_of_processes, sizeof(int));
  int *ray_index_per_process = calloc(number_of_processes, sizeof(int));
  for (int process = 1; process < number_of_processes; process++) {
    ray_offsets[process] =
        ray_offsets[process - 1] + my_fft_grid->rays_per_process[process - 1];
  }
  cp_assert(ray_offsets[number_of_processes - 1] +
             my_fft_grid->rays_per_process[number_of_processes - 1] ==
         total_number_of_rays);

  // Create the map of yz index to the yz coordinates and the x-values required
  // for the mixed space
  my_fft_grid->ray_to_xy = malloc(total_number_of_rays * sizeof(int[2]));
  my_fft_grid->xy_to_ray =
      malloc(my_fft_grid->npts_global_gspace[0] *
             my_fft_grid->npts_global_gspace[1] * sizeof(int[2]));
  memset(my_fft_grid->ray_to_xy, -1, total_number_of_rays * sizeof(int[2]));
  for (int index_x = 0; index_x < fft_grid_ref->npts_global_gspace[0];
       index_x++) {
    const int index_x_shifted =
        convert_c_index_to_shifted_index(index_x, fft_grid_ref->npts_global[0]);
    if (!is_on_grid(index_x_shifted, npts_global[0]))
      continue;
    const int index_x_new =
        convert_shifted_index_to_c_index(index_x_shifted, npts_global[0]);
    for (int index_y = 0; index_y < fft_grid_ref->npts_global_gspace[1];
         index_y++) {
      const int index_y_shifted = convert_c_index_to_shifted_index(
          index_y, fft_grid_ref->npts_global[1]);
      // Same check for y-coordinate
      if (!is_on_grid(index_y_shifted, npts_global[1]))
        continue;
      const int index_y_new =
          convert_shifted_index_to_c_index(index_y_shifted, npts_global[1]);
      const int current_process =
          my_fft_grid
              ->xy_to_process[index_x_new * my_fft_grid->npts_global_gspace[1] +
                              index_y_new];
      const int current_ray_index = ray_index_per_process[current_process];
      const int current_ray_offset = ray_offsets[current_process];
      my_fft_grid->ray_to_xy[current_ray_offset + current_ray_index][0] =
          index_x_new;
      my_fft_grid->ray_to_xy[current_ray_offset + current_ray_index][1] =
          index_y_new;
      my_fft_grid->xy_to_ray[index_x_new * my_fft_grid->npts_global_gspace[1] +
                             index_y_new] = current_ray_index;
      ray_index_per_process[current_process]++;
    }
  }
  for (int process = 0; process < number_of_processes; process++) {
    cp_assert(ray_index_per_process[process] ==
               my_fft_grid->rays_per_process[process] &&
           "The number of rays does not match the expected number of rays!");
    const int current_ray_offset = ray_offsets[process];
    for (int ray_index = 0; ray_index < my_fft_grid->rays_per_process[process];
         ray_index++) {
      cp_assert(my_fft_grid->ray_to_xy[current_ray_offset + ray_index][0] >= 0 &&
             my_fft_grid->ray_to_xy[current_ray_offset + ray_index][1] >= 0 &&
             "The ray has to be assigned to a valid yz index!");
      cp_assert(my_fft_grid->ray_to_xy[current_ray_offset + ray_index][0] <
                 my_fft_grid->npts_global_gspace[0] &&
             my_fft_grid->ray_to_xy[current_ray_offset + ray_index][1] <
                 my_fft_grid->npts_global_gspace[1] &&
             "The ray has to be assigned to a valid yz index!");
    }
  }

  free(ray_offsets);
  free(ray_index_per_process);

  my_fft_grid->index_to_g = calloc(my_fft_grid->npts_gs_local, sizeof(int[3]));
  // This grid is smaller in all directions such that all points of the new grid
  // should be available on the reference grid
  my_fft_grid->local_index_to_ref_grid =
      calloc(my_fft_grid->npts_gs_local, sizeof(int));

  own_index = 0;
  for (int ref_index = 0; ref_index < fft_grid_ref->npts_gs_local;
       ref_index++) {
    const int shifted_indices[3] = {
        convert_c_index_to_shifted_index(fft_grid_ref->index_to_g[ref_index][0],
                                         fft_grid_ref->npts_global[0]),
        convert_c_index_to_shifted_index(fft_grid_ref->index_to_g[ref_index][1],
                                         fft_grid_ref->npts_global[1]),
        convert_c_index_to_shifted_index(fft_grid_ref->index_to_g[ref_index][2],
                                         fft_grid_ref->npts_global[2])};
    if (is_on_grid(shifted_indices[0], my_fft_grid->npts_global[0]) &&
        is_on_grid(shifted_indices[1], my_fft_grid->npts_global[1]) &&
        is_on_grid(shifted_indices[2], my_fft_grid->npts_global[2])) {
      for (int dir = 0; dir < 3; dir++) {
        my_fft_grid->index_to_g[own_index][dir] =
            convert_shifted_index_to_c_index(shifted_indices[dir],
                                             my_fft_grid->npts_global[dir]);
        my_fft_grid->local_index_to_ref_grid[own_index] = ref_index;
      }
      own_index++;
    }
  }
  cp_assert(own_index == my_fft_grid->npts_gs_local);

  if (my_fft_grid->use_halfspace) {
    my_fft_grid->index_to_ray = NULL;
    my_fft_grid->index_to_ray_pos =
        calloc(my_fft_grid->number_of_positive_gs_points, sizeof(int[2]));
    my_fft_grid->index_to_ray_neg =
        calloc(my_fft_grid->number_of_negative_gs_points, sizeof(int[2]));
    my_fft_grid->index_to_cart = NULL;
    my_fft_grid->index_to_cart_pos =
        calloc(my_fft_grid->number_of_positive_gs_points, sizeof(int[2]));
    my_fft_grid->index_to_cart_neg =
        calloc(my_fft_grid->number_of_negative_gs_points, sizeof(int[2]));
  } else {
    my_fft_grid->index_to_ray = calloc(my_fft_grid->npts_gs_local, sizeof(int));
    my_fft_grid->index_to_ray_pos = NULL;
    my_fft_grid->index_to_ray_neg = NULL;
    my_fft_grid->index_to_cart =
        calloc(my_fft_grid->npts_gs_local, sizeof(int));
    my_fft_grid->index_to_cart_pos = NULL;
    my_fft_grid->index_to_cart_neg = NULL;
  }
  my_fft_grid->index_to_cart = NULL;
  my_fft_grid->index_to_cart_neg = NULL;
  my_fft_grid->index_to_cart_pos = NULL;
  if (my_fft_grid->use_halfspace) {
    int positive_index = 0;
    int negative_index = 0;
    for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
      int *index_g = my_fft_grid->index_to_g[index];
      if (index_g[0] < my_fft_grid->npts_global_gspace[0]) {
        my_fft_grid->index_to_ray_pos[positive_index][0] = index;
        my_fft_grid->index_to_ray_pos[positive_index][1] =
            my_fft_grid->xy_to_ray[index_g[0] * npts_global[1] + index_g[1]] *
                npts_global[2] +
            index_g[2];
        positive_index++;
      } else {
        my_fft_grid->index_to_ray_neg[negative_index][0] = index;
        my_fft_grid->index_to_ray_neg[negative_index][1] =
            my_fft_grid
                    ->xy_to_ray[(npts_global[0] - index_g[0]) * npts_global[1] +
                                (npts_global[1] - index_g[1]) %
                                    npts_global[1]] *
                npts_global[2] +
            (npts_global[2] - index_g[2]) % npts_global[2];
        negative_index++;
      }
    }
  } else {
    for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
      int *index_g = my_fft_grid->index_to_g[index];
      my_fft_grid->index_to_ray[index] =
          my_fft_grid->xy_to_ray[index_g[0] * npts_global[1] + index_g[1]] *
              npts_global[2] +
          index_g[2];
    }
  }

  my_fft_grid->index_to_gsquared =
      calloc(my_fft_grid->npts_gs_local, sizeof(double));
#pragma omp parallel for default(none) shared(my_fft_grid)
  for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
    my_fft_grid->index_to_gsquared[index] = squared_length_of_g_vector(
        (const int[3]){
            shifted_index_from_c_index(my_fft_grid,
                                       my_fft_grid->index_to_g[index][0], 0),
            shifted_index_from_c_index(my_fft_grid,
                                       my_fft_grid->index_to_g[index][1], 1),
            shifted_index_from_c_index(my_fft_grid,
                                       my_fft_grid->index_to_g[index][2], 2)},
        my_fft_grid->cell_info.dhmat);
  }

  my_fft_grid->redistribution = calloc(1, sizeof(fft_redistribution_t));
  prepare_redistribution(
      my_fft_grid->redistribution, my_fft_grid->npts_global_gspace,
      my_fft_grid->proc2local_ms, my_fft_grid->proc2local_x_gs,
      my_fft_grid->proc2local_y_rs, my_fft_grid->proc2local_y_gs,
      my_fft_grid->proc2local_z_rs, my_fft_grid->rays_per_process,
      my_fft_grid->ray_to_xy, my_fft_grid->comm, my_fft_grid->sub_comm);

  fft_register_3d(my_fft_grid);

  *fft_grid = my_fft_grid;
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Retains a grid layout.
 * \author Frederick Stein
 ******************************************************************************/
void grid_retain_fft_grid_layout(fft_grid_layout *fft_grid) {
  cp_assert(fft_grid != NULL);
  cp_assert(fft_grid->ref_counter > 0);
  fft_grid->ref_counter++;
}

/*******************************************************************************
 * \brief Set the h-matrix of a grid layout.
 * \author Frederick Stein
 ******************************************************************************/
void fft_grid_set_hmat(fft_grid_layout *fft_grid, const double hmat[3][3]) {
  cp_assert(fft_grid != NULL);
  for (int dir = 0; dir < 3; dir++) {
    for (int dir2 = 0; dir2 < 3; dir2++) {
      fft_grid->cell_info.dhmat[dir][dir2] =
          hmat[dir][dir2] * fft_grid->npts_global[dir2];
    }
  }
}

/*******************************************************************************
 * \brief Print some information on a grid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_print_grid_layout_info(const fft_grid_layout *layout,
                                 bool print_distribution) {
  if (cp_mpi_comm_rank(layout->comm) == 0) {
    fprintf(stdout, "Grid ID: %i\n", layout->grid_id);
    fprintf(stdout, "Reference Grid ID: %i\n", layout->ref_grid_id);
    fprintf(stdout, "Global sizes: %i %i %i\n", layout->npts_global[0],
            layout->npts_global[1], layout->npts_global[2]);
    for (int dir = 0; dir < 3; dir++)
      fprintf(stdout, "Grid spacing %i: %f %f %f\n", dir,
              layout->cell_info.dhmat[dir][0], layout->cell_info.dhmat[dir][1],
              layout->cell_info.dhmat[dir][2]);
    fprintf(stdout, "Use half space: %i\n", layout->use_halfspace);
    fprintf(stdout, "Use ray distribution: %i\n", layout->ray_distribution);
    fprintf(stdout, "Process grid: %i %i\n", layout->proc_grid[0],
            layout->proc_grid[1]);
    if (print_distribution) {
      for (int process = 0; process < cp_mpi_comm_size(layout->comm);
           process++) {
        fprintf(stdout, "Local dimensions RS %i: %i %i/%i %i/%i %i\n", process,
                layout->proc2local_rs[process][0][0],
                layout->proc2local_rs[process][0][0] +
                    layout->proc2local_rs[process][0][1] - 1,
                layout->proc2local_rs[process][1][0],
                layout->proc2local_rs[process][1][0] +
                    layout->proc2local_rs[process][1][1] - 1,
                layout->proc2local_rs[process][2][0],
                layout->proc2local_rs[process][2][0] +
                    layout->proc2local_rs[process][2][1] - 1);
      }
      for (int process = 0; process < cp_mpi_comm_size(layout->comm);
           process++) {
        fprintf(stdout, "Local dimensions MS %i: %i %i/%i %i/%i %i\n", process,
                layout->proc2local_ms[process][0][0],
                layout->proc2local_ms[process][0][0] +
                    layout->proc2local_ms[process][0][1] - 1,
                layout->proc2local_ms[process][1][0],
                layout->proc2local_ms[process][1][0] +
                    layout->proc2local_ms[process][1][1] - 1,
                layout->proc2local_ms[process][2][0],
                layout->proc2local_ms[process][2][0] +
                    layout->proc2local_ms[process][2][1] - 1);
      }
      for (int process = 0; process < cp_mpi_comm_size(layout->comm);
           process++) {
        fprintf(stdout, "Local dimensions GS %i: %i %i/%i %i/%i %i\n", process,
                layout->proc2local_gs[process][0][0],
                layout->proc2local_gs[process][0][0] +
                    layout->proc2local_gs[process][0][1] - 1,
                layout->proc2local_gs[process][1][0],
                layout->proc2local_gs[process][1][0] +
                    layout->proc2local_gs[process][1][1] - 1,
                layout->proc2local_gs[process][2][0],
                layout->proc2local_gs[process][2][0] +
                    layout->proc2local_gs[process][2][1] - 1);
      }
      if (layout->ray_distribution) {
        for (int index_x = 0; index_x < layout->npts_global_gspace[0];
             index_x++) {
          for (int index_y = 0; index_y < layout->npts_global_gspace[0];
               index_y++) {
            fprintf(
                stdout, "xy-pair %i %i is on process %i\n", index_x, index_y,
                layout->xy_to_process[index_x * layout->npts_global_gspace[1] +
                                      index_y]);
          }
        }
      }
    }
    fflush(stdout);
  }
  cp_mpi_barrier(layout->comm);
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT to the sorted format.
 * \param grid_rs complex-valued data in real space.
 * \param grid_gs complex data in reciprocal space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_with_layout(const double complex *restrict grid_rs,
                           double complex *restrict grid_gs,
                           const fft_grid_layout *grid_layout) {
  cp_assert(grid_rs != NULL);
  cp_assert(grid_gs != NULL);
  cp_assert(grid_layout != NULL);
  cp_assert(grid_layout->ref_counter > 0);

  ensure_buffer_size(grid_layout->buffer_size);

  if (grid_layout->ray_distribution) {
    fft_3d_fw_ray(grid_rs, true, grid_gs, grid_layout->npts_gs_local,
                  grid_layout->npts_global, grid_layout->index_to_ray,
                  grid_layout->proc2local_rs, grid_layout->proc2local_ms,
                  grid_layout->proc2local_x_gs, grid_layout->rays_per_process,
                  grid_layout->redistribution, grid_layout->comm,
                  grid_layout->sub_comm);
  } else {
    fft_3d_fw_blocked(grid_rs, true, grid_gs, grid_layout->index_to_cart,
                      grid_layout->npts_gs_local, grid_layout->npts_global,
                      grid_layout->proc2local_rs, grid_layout->proc2local_ms,
                      grid_layout->proc2local_gs, grid_layout->proc2local_x_gs,
                      grid_layout->proc2local_y_gs, grid_layout->redistribution,
                      grid_layout->comm, grid_layout->sub_comm);
  }
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT to the blocked format.
 * \param grid_rs complex-valued data in real space.
 * \param grid_gs complex data in reciprocal space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_with_layout_to_cart(const double complex *restrict grid_rs,
                                   double complex *restrict grid_gs,
                                   const fft_grid_layout *grid_layout) {
  cp_assert(grid_rs != NULL);
  cp_assert(grid_gs != NULL);
  cp_assert(grid_layout != NULL);
  cp_assert(grid_layout->ref_counter > 0);

  ensure_buffer_size(grid_layout->buffer_size);

  fft_3d_fw_blocked(grid_rs, true, grid_gs, NULL, 0, grid_layout->npts_global,
                    grid_layout->proc2local_rs, grid_layout->proc2local_ms,
                    grid_layout->proc2local_gs, grid_layout->proc2local_x_gs,
                    grid_layout->proc2local_y_gs, grid_layout->redistribution,
                    grid_layout->comm, grid_layout->sub_comm);
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT to the sorted format.
 * \param grid_rs real-valued data in real space.
 * \param grid_gs complex data in reciprocal space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_r2c_with_layout_to_cart(const double *restrict grid_rs,
                                       double complex *restrict grid_gs,
                                       const fft_grid_layout *grid_layout) {
  cp_assert(grid_rs != NULL);
  cp_assert(grid_gs != NULL);
  cp_assert(grid_layout != NULL);
  cp_assert(grid_layout->ref_counter > 0);

  ensure_buffer_size(grid_layout->buffer_size);

  if (grid_layout->use_halfspace) {
    fft_3d_fw_r2c_blocked(
        grid_rs, grid_gs, NULL, NULL, 0, 0, grid_layout->npts_global,
        grid_layout->npts_global_gspace, grid_layout->proc2local_rs,
        grid_layout->proc2local_ms, grid_layout->proc2local_gs,
        grid_layout->proc2local_x_gs, grid_layout->proc2local_y_gs,
        grid_layout->redistribution, grid_layout->comm, grid_layout->sub_comm);
  } else {
    fft_3d_fw_blocked((const double complex *)grid_rs, false, grid_gs, NULL, 0,
                      grid_layout->npts_global, grid_layout->proc2local_rs,
                      grid_layout->proc2local_ms, grid_layout->proc2local_gs,
                      grid_layout->proc2local_x_gs,
                      grid_layout->proc2local_y_gs, grid_layout->redistribution,
                      grid_layout->comm, grid_layout->sub_comm);
  }
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT to the sorted format.
 * \param grid_rs real-valued data in real space.
 * \param grid_gs complex data in reciprocal space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_r2c_with_layout(const double *restrict grid_rs,
                               double complex *restrict grid_gs,
                               const fft_grid_layout *grid_layout) {
  cp_assert(grid_rs != NULL);
  cp_assert(grid_gs != NULL);
  cp_assert(grid_layout != NULL);
  cp_assert(grid_layout->ref_counter > 0);

  ensure_buffer_size(grid_layout->buffer_size);

  if (grid_layout->use_halfspace) {
    if (grid_layout->ray_distribution) {
      fft_3d_fw_r2c_ray(
          grid_rs, grid_gs, grid_layout->index_to_ray_pos,
          grid_layout->index_to_ray_neg,
          grid_layout->number_of_positive_gs_points,
          grid_layout->number_of_negative_gs_points, grid_layout->npts_global,
          grid_layout->npts_global_gspace, grid_layout->proc2local_rs,
          grid_layout->proc2local_ms, grid_layout->proc2local_x_gs,
          grid_layout->rays_per_process, grid_layout->redistribution,
          grid_layout->comm, grid_layout->sub_comm);
    } else {
      fft_3d_fw_r2c_blocked(
          grid_rs, grid_gs, grid_layout->index_to_cart_pos,
          grid_layout->index_to_cart_neg,
          grid_layout->number_of_positive_gs_points,
          grid_layout->number_of_negative_gs_points, grid_layout->npts_global,
          grid_layout->npts_global_gspace, grid_layout->proc2local_rs,
          grid_layout->proc2local_ms, grid_layout->proc2local_gs,
          grid_layout->proc2local_x_gs, grid_layout->proc2local_y_gs,
          grid_layout->redistribution, grid_layout->comm,
          grid_layout->sub_comm);
    }
  } else {
    if (grid_layout->ray_distribution) {
      fft_3d_fw_ray((const double complex *)grid_rs, false, grid_gs,
                    grid_layout->npts_gs_local, grid_layout->npts_global,
                    grid_layout->index_to_ray, grid_layout->proc2local_rs,
                    grid_layout->proc2local_ms, grid_layout->proc2local_x_gs,
                    grid_layout->rays_per_process, grid_layout->redistribution,
                    grid_layout->comm, grid_layout->sub_comm);
    } else {
      fft_3d_fw_blocked((const double complex *)grid_rs, false, grid_gs,
                        grid_layout->index_to_cart, grid_layout->npts_gs_local,
                        grid_layout->npts_global, grid_layout->proc2local_rs,
                        grid_layout->proc2local_ms, grid_layout->proc2local_gs,
                        grid_layout->proc2local_x_gs,
                        grid_layout->proc2local_y_gs,
                        grid_layout->redistribution, grid_layout->comm,
                        grid_layout->sub_comm);
    }
  }
  cp_assert(grid_rs != NULL);
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT from data sorted in g-space.
 * \param grid_layout FFT grid layout object.
 * \param grid_gs complex-valued data in reciprocal space.
 * \param grid_rs complex-valued data in real space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_with_layout(const double complex *restrict grid_gs,
                           double complex *restrict grid_rs,
                           const fft_grid_layout *grid_layout) {
  cp_assert(grid_gs != NULL);
  cp_assert(grid_rs != NULL);
  cp_assert(grid_layout != NULL);
  cp_assert(grid_layout->ref_counter > 0);

  ensure_buffer_size(grid_layout->buffer_size);

  if (grid_layout->ray_distribution) {
    fft_3d_bw_ray(grid_gs, grid_layout->index_to_ray,
                  grid_layout->npts_gs_local, grid_rs, true,
                  grid_layout->npts_global, grid_layout->proc2local_rs,
                  grid_layout->proc2local_ms, grid_layout->proc2local_x_gs,
                  grid_layout->rays_per_process, grid_layout->redistribution,
                  grid_layout->comm, grid_layout->sub_comm);
  } else {
    fft_3d_bw_blocked(
        grid_gs, grid_layout->index_to_cart, grid_layout->npts_gs_local,
        grid_rs, true, grid_layout->npts_global, grid_layout->proc2local_rs,
        grid_layout->proc2local_ms, grid_layout->proc2local_gs,
        grid_layout->proc2local_x_gs, grid_layout->proc2local_y_gs,
        grid_layout->redistribution, grid_layout->comm, grid_layout->sub_comm);
  }
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT from data sorted in g-space.
 * \param grid_layout FFT grid layout object.
 * \param grid_gs complex-valued data in reciprocal space.
 * \param grid_rs complex-valued data in real space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_with_layout_from_cart(const double complex *restrict grid_gs,
                                     double complex *restrict grid_rs,
                                     const fft_grid_layout *grid_layout) {
  cp_assert(grid_gs != NULL);
  cp_assert(grid_rs != NULL);
  cp_assert(grid_layout != NULL);
  cp_assert(grid_layout->ref_counter > 0);

  ensure_buffer_size(grid_layout->buffer_size);

  fft_3d_bw_blocked(grid_gs, NULL, 0, grid_rs, true, grid_layout->npts_global,
                    grid_layout->proc2local_rs, grid_layout->proc2local_ms,
                    grid_layout->proc2local_gs, grid_layout->proc2local_x_gs,
                    grid_layout->proc2local_y_gs, grid_layout->redistribution,
                    grid_layout->comm, grid_layout->sub_comm);
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT from data sorted in g-space.
 * \param grid_layout FFT grid layout object.
 * \param grid_gs complex data in reciprocal space.
 * \param grid_rs real-valued data in real space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_c2r_with_layout(const double complex *restrict grid_gs,
                               double *restrict grid_rs,
                               const fft_grid_layout *grid_layout) {
  cp_assert(grid_gs != NULL);
  cp_assert(grid_rs != NULL);
  cp_assert(grid_layout != NULL);
  cp_assert(grid_layout->ref_counter > 0);

  ensure_buffer_size(grid_layout->buffer_size);

  if (grid_layout->use_halfspace) {
    if (grid_layout->ray_distribution) {
      fft_3d_bw_c2r_ray(
          grid_gs, grid_layout->index_to_ray_pos,
          grid_layout->number_of_positive_gs_points, grid_rs,
          grid_layout->npts_global, grid_layout->npts_global_gspace,
          grid_layout->proc2local_rs, grid_layout->proc2local_ms,
          grid_layout->proc2local_x_gs, grid_layout->rays_per_process,
          grid_layout->redistribution, grid_layout->comm,
          grid_layout->sub_comm);
    } else {
      fft_3d_bw_c2r_blocked(
          grid_gs, grid_layout->index_to_cart_pos,
          grid_layout->number_of_positive_gs_points, grid_rs,
          grid_layout->npts_global, grid_layout->npts_global_gspace,
          grid_layout->proc2local_rs, grid_layout->proc2local_ms,
          grid_layout->proc2local_gs, grid_layout->proc2local_x_gs,
          grid_layout->proc2local_y_gs, grid_layout->redistribution,
          grid_layout->comm, grid_layout->sub_comm);
    }
  } else {
    if (grid_layout->ray_distribution) {
      fft_3d_bw_ray(grid_gs, grid_layout->index_to_ray,
                    grid_layout->npts_gs_local, (double complex *)grid_rs,
                    false, grid_layout->npts_global, grid_layout->proc2local_rs,
                    grid_layout->proc2local_ms, grid_layout->proc2local_x_gs,
                    grid_layout->rays_per_process, grid_layout->redistribution,
                    grid_layout->comm, grid_layout->sub_comm);
    } else {
      fft_3d_bw_blocked(
          grid_gs, grid_layout->index_to_cart, grid_layout->npts_gs_local,
          (double complex *)grid_rs, false, grid_layout->npts_global,
          grid_layout->proc2local_rs, grid_layout->proc2local_ms,
          grid_layout->proc2local_gs, grid_layout->proc2local_x_gs,
          grid_layout->proc2local_y_gs, grid_layout->redistribution,
          grid_layout->comm, grid_layout->sub_comm);
    }
  }
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT from data sorted in g-space.
 * \param grid_layout FFT grid layout object.
 * \param grid_gs complex data in reciprocal space.
 * \param grid_rs real-valued data in real space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_c2r_with_layout_from_cart(const double complex *restrict grid_gs,
                                         double *restrict grid_rs,
                                         const fft_grid_layout *grid_layout) {
  cp_assert(grid_gs != NULL);
  cp_assert(grid_rs != NULL);
  cp_assert(grid_layout != NULL);
  cp_assert(grid_layout->ref_counter > 0);

  ensure_buffer_size(grid_layout->buffer_size);

  if (grid_layout->use_halfspace) {
    fft_3d_bw_c2r_blocked(
        grid_gs, NULL, 0, grid_rs, grid_layout->npts_global,
        grid_layout->npts_global_gspace, grid_layout->proc2local_rs,
        grid_layout->proc2local_ms, grid_layout->proc2local_gs,
        grid_layout->proc2local_x_gs, grid_layout->proc2local_y_gs,
        grid_layout->redistribution, grid_layout->comm, grid_layout->sub_comm);
  } else {
    fft_3d_bw_blocked(grid_gs, NULL, 0, (double complex *)grid_rs, false,
                      grid_layout->npts_global, grid_layout->proc2local_rs,
                      grid_layout->proc2local_ms, grid_layout->proc2local_gs,
                      grid_layout->proc2local_x_gs,
                      grid_layout->proc2local_y_gs, grid_layout->redistribution,
                      grid_layout->comm, grid_layout->sub_comm);
  }
}

// EOF
