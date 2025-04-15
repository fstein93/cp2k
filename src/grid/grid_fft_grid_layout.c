/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "grid_fft_grid_layout.h"
#include "common/grid_common.h"
#include "grid_fft_grid.h"
#include "grid_fft_lib.h"
#include "grid_fft_reorder.h"

#include <assert.h>
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

double squared_length_of_g_vector(const int g[3], const double h_inv[3][3]) {
  if (g[0] == 0 && g[1] == 0 && g[2] == 0) {
    return 0.0;
  }
  const double two_pi = 2.0 * acos(-1.0);
  double length_g_squared = 0.0;
  for (int dir = 0; dir < 3; dir++) {
    double length_g_dir = 0.0;
    for (int dir2 = 0; dir2 < 3; dir2++) {
      length_g_dir += g[dir] * h_inv[dir2][dir];
    }
    length_g_dir *= two_pi;
    length_g_squared += length_g_dir * length_g_dir;
  }
  return length_g_squared;
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
    if (a_value > b_value + max_rel_error_for_equivalence_g_squared) {
      return 1;
    } else if (a_value + max_rel_error_for_equivalence_g_squared < b_value) {
      return -1;
    }
  }
  return 0;
}

void sort_shell(int (*shell)[3], const int shell_size) {
  qsort(shell, shell_size, sizeof(int[3]), compare_shell);
}

void sort_g_vectors(grid_fft_grid_layout *my_fft_grid) {
  assert(my_fft_grid != NULL);
  assert(my_fft_grid->npts_gs_local >= 0);

  int *local_index2g_squared = calloc(my_fft_grid->npts_gs_local, sizeof(int));
#pragma omp parallel for default(none)                                         \
    shared(my_fft_grid, local_index2g_squared)
  for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
    local_index2g_squared[index] = squared_length_of_g_vector(
        my_fft_grid->index_to_g[index], my_fft_grid->h_inv);
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
    int(*index_to_g_sorted)[3] =
        calloc(my_fft_grid->npts_gs_local, sizeof(int[3]));
#pragma omp parallel for default(none)                                         \
    shared(my_fft_grid, index_to_g_sorted, g_square_index_pair,                \
               local_index2g_squared)
    for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
      memcpy(index_to_g_sorted[index],
             my_fft_grid->index_to_g[g_square_index_pair[index].index],
             3 * sizeof(int));
      local_index2g_squared[index] = g_square_index_pair[index].value;
    }
    memcpy(my_fft_grid->index_to_g, &index_to_g_sorted[0][0],
           my_fft_grid->npts_gs_local * sizeof(int[3]));
    free(index_to_g_sorted);
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
}

void grid_free_fft_grid_layout(grid_fft_grid_layout *fft_grid) {
  if (fft_grid != NULL) {
    if (grid_mpi_comm_rank(fft_grid->comm) == 0)
      assert((fft_grid->ref_counter) > 0);
    fft_grid->ref_counter--;
    if (fft_grid->ref_counter == 0) {
      grid_mpi_comm_free(&fft_grid->comm);
      grid_mpi_comm_free(&fft_grid->sub_comm[0]);
      grid_mpi_comm_free(&fft_grid->sub_comm[1]);
      free(fft_grid->proc2local_rs);
      free(fft_grid->proc2local_ms);
      free(fft_grid->proc2local_gs);
      fft_free_complex(fft_grid->buffer_1);
      fft_free_complex(fft_grid->buffer_2);
      free(fft_grid->yz_to_process);
      free(fft_grid->ray_to_yz);
      free(fft_grid->rays_per_process);
      free(fft_grid->index_to_g);
      free(fft_grid->local_index_to_ref_grid);
      free(fft_grid);
    }
  }
}

void setup_proc2local(grid_fft_grid_layout *my_fft_grid,
                      const int npts_global[3]) {
  const int number_of_processes = grid_mpi_comm_size(my_fft_grid->comm);

  my_fft_grid->proc2local_rs = malloc(number_of_processes * sizeof(int[3][2]));
  my_fft_grid->proc2local_ms = malloc(number_of_processes * sizeof(int[3][2]));
  my_fft_grid->proc2local_gs = malloc(number_of_processes * sizeof(int[3][2]));
// OMP parallelization requires a multi-threaded MPI
#pragma omp parallel for default(none)                                         \
    shared(my_fft_grid, number_of_processes,                                   \
               npts_global) if (grid_mpi_query() >= grid_mpi_thread_multiple)
  for (int proc = 0; proc < number_of_processes; proc++) {
    int proc_coords[2];
    grid_mpi_cart_coords(my_fft_grid->comm, proc, 2, proc_coords);
    // Determine the bounds in real space
    my_fft_grid->proc2local_rs[proc][0][0] =
        proc_coords[0] * npts_global[0] / my_fft_grid->proc_grid[0];
    my_fft_grid->proc2local_rs[proc][0][1] =
        (proc_coords[0] + 1) * npts_global[0] / my_fft_grid->proc_grid[0] - 1;
    my_fft_grid->proc2local_rs[proc][1][0] =
        proc_coords[1] * npts_global[1] / my_fft_grid->proc_grid[1];
    my_fft_grid->proc2local_rs[proc][1][1] =
        (proc_coords[1] + 1) * npts_global[1] / my_fft_grid->proc_grid[1] - 1;
    my_fft_grid->proc2local_rs[proc][2][0] = 0;
    my_fft_grid->proc2local_rs[proc][2][1] = npts_global[2] - 1;
    // Determine the bounds in mixed space: we keep the distribution in the
    // first direction to reduce communication
    my_fft_grid->proc2local_ms[proc][0][0] =
        proc_coords[0] * npts_global[0] / my_fft_grid->proc_grid[0];
    my_fft_grid->proc2local_ms[proc][0][1] =
        (proc_coords[0] + 1) * npts_global[0] / my_fft_grid->proc_grid[0] - 1;
    my_fft_grid->proc2local_ms[proc][1][0] = 0;
    my_fft_grid->proc2local_ms[proc][1][1] = npts_global[1] - 1;
    my_fft_grid->proc2local_ms[proc][2][0] =
        proc_coords[1] * npts_global[2] / my_fft_grid->proc_grid[1];
    my_fft_grid->proc2local_ms[proc][2][1] =
        (proc_coords[1] + 1) * npts_global[2] / my_fft_grid->proc_grid[1] - 1;
    // Determine the bounds in mixed space: we keep the distribution in the
    // third direction to reduce communication
    my_fft_grid->proc2local_gs[proc][0][0] = 0;
    my_fft_grid->proc2local_gs[proc][0][1] = npts_global[0] - 1;
    my_fft_grid->proc2local_gs[proc][1][0] =
        proc_coords[0] * npts_global[1] / my_fft_grid->proc_grid[0];
    my_fft_grid->proc2local_gs[proc][1][1] =
        (proc_coords[0] + 1) * npts_global[1] / my_fft_grid->proc_grid[0] - 1;
    my_fft_grid->proc2local_gs[proc][2][0] =
        proc_coords[1] * npts_global[2] / my_fft_grid->proc_grid[1];
    my_fft_grid->proc2local_gs[proc][2][1] =
        (proc_coords[1] + 1) * npts_global[2] / my_fft_grid->proc_grid[1] - 1;
  }
}

void allocate_fft_buffers(grid_fft_grid_layout *my_fft_grid) {
  const int my_process = grid_mpi_comm_rank(my_fft_grid->comm);

  // Determine the maximum buffer size
  int buffer_size = 0;
  buffer_size =
      imax(buffer_size, (my_fft_grid->proc2local_rs[my_process][0][1] -
                         my_fft_grid->proc2local_rs[my_process][0][0] + 1) *
                            (my_fft_grid->proc2local_rs[my_process][1][1] -
                             my_fft_grid->proc2local_rs[my_process][1][0] + 1) *
                            (my_fft_grid->proc2local_rs[my_process][2][1] -
                             my_fft_grid->proc2local_rs[my_process][2][0] + 1));
  buffer_size =
      imax(buffer_size, (my_fft_grid->proc2local_ms[my_process][0][1] -
                         my_fft_grid->proc2local_ms[my_process][0][0] + 1) *
                            (my_fft_grid->proc2local_ms[my_process][1][1] -
                             my_fft_grid->proc2local_ms[my_process][1][0] + 1) *
                            (my_fft_grid->proc2local_ms[my_process][2][1] -
                             my_fft_grid->proc2local_ms[my_process][2][0] + 1));
  buffer_size =
      imax(buffer_size, (my_fft_grid->proc2local_gs[my_process][0][1] -
                         my_fft_grid->proc2local_gs[my_process][0][0] + 1) *
                            (my_fft_grid->proc2local_gs[my_process][1][1] -
                             my_fft_grid->proc2local_gs[my_process][1][0] + 1) *
                            (my_fft_grid->proc2local_gs[my_process][2][1] -
                             my_fft_grid->proc2local_gs[my_process][2][0] + 1));
  buffer_size = imax(buffer_size, my_fft_grid->npts_gs_local);
  // Allocate the buffers
  my_fft_grid->buffer_1 = NULL;
  my_fft_grid->buffer_2 = NULL;
  fft_allocate_complex(buffer_size, &my_fft_grid->buffer_1);
  fft_allocate_complex(buffer_size, &my_fft_grid->buffer_2);
}

void grid_create_fft_grid_layout(grid_fft_grid_layout **fft_grid,
                                 const grid_mpi_comm comm,
                                 const int npts_global[3],
                                 const double dh_inv[3][3]) {
  grid_fft_grid_layout *my_fft_grid = NULL;
  if (*fft_grid != NULL) {
    my_fft_grid = *fft_grid;
    grid_free_fft_grid_layout(*fft_grid);
  }
  my_fft_grid = calloc(1, sizeof(grid_fft_grid_layout));

  const int number_of_processes = grid_mpi_comm_size(comm);
  const int my_process = grid_mpi_comm_rank(comm);

  my_fft_grid->grid_id = current_grid_id;
  my_fft_grid->ref_grid_id = current_grid_id;
  current_grid_id++;
  my_fft_grid->ref_counter = 1;
  my_fft_grid->ray_distribution = false;

  if (npts_global[0] < number_of_processes) {
    // We only distribute in two directions if necessary to reduce communication
    grid_mpi_dims_create(number_of_processes, 2, &my_fft_grid->proc_grid[0]);
  } else {
    my_fft_grid->proc_grid[0] = number_of_processes;
    my_fft_grid->proc_grid[1] = 1;
  }

  memcpy(my_fft_grid->npts_global, npts_global, 3 * sizeof(int));
  for (int dir = 0; dir < 3; dir++) {
    for (int dir2 = 0; dir2 < 3; dir2++) {
      my_fft_grid->h_inv[dir][dir2] =
          dh_inv[dir][dir2] / ((double)npts_global[dir2]);
    }
  }

  my_fft_grid->periodic[0] = 1;
  my_fft_grid->periodic[1] = 1;
  grid_mpi_cart_create(comm, 2, my_fft_grid->proc_grid, my_fft_grid->periodic,
                       true, &my_fft_grid->comm);

  grid_mpi_cart_get(my_fft_grid->comm, 2, my_fft_grid->proc_grid,
                    my_fft_grid->periodic, my_fft_grid->proc_coords);

  grid_mpi_cart_sub(my_fft_grid->comm, (const int[2]){1, 0},
                    &my_fft_grid->sub_comm[0]);
  grid_mpi_cart_sub(my_fft_grid->comm, (const int[2]){0, 1},
                    &my_fft_grid->sub_comm[1]);

  setup_proc2local(my_fft_grid, npts_global);

  my_fft_grid->npts_gs_local =
      (my_fft_grid->proc2local_gs[my_process][0][1] -
       my_fft_grid->proc2local_gs[my_process][0][0] + 1) *
      (my_fft_grid->proc2local_gs[my_process][1][1] -
       my_fft_grid->proc2local_gs[my_process][1][0] + 1) *
      (my_fft_grid->proc2local_gs[my_process][2][1] -
       my_fft_grid->proc2local_gs[my_process][2][0] + 1);

  allocate_fft_buffers(my_fft_grid);

  my_fft_grid->yz_to_process = NULL;
  my_fft_grid->ray_to_yz = NULL;
  my_fft_grid->rays_per_process = NULL;
  my_fft_grid->index_to_g = calloc(my_fft_grid->npts_gs_local, sizeof(int[3]));
#pragma omp parallel for default(none) shared(my_fft_grid, my_process)
  for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
    my_fft_grid->index_to_g[index][0] =
        my_fft_grid->proc2local_gs[my_process][0][0] +
        index % (my_fft_grid->proc2local_gs[my_process][0][1] -
                 my_fft_grid->proc2local_gs[my_process][0][0] + 1);
    my_fft_grid->index_to_g[index][1] =
        my_fft_grid->proc2local_gs[my_process][1][0] +
        (index / (my_fft_grid->proc2local_gs[my_process][0][1] -
                  my_fft_grid->proc2local_gs[my_process][0][0] + 1)) %
            (my_fft_grid->proc2local_gs[my_process][1][1] -
             my_fft_grid->proc2local_gs[my_process][1][0] + 1);
    my_fft_grid->index_to_g[index][2] =
        my_fft_grid->proc2local_gs[my_process][2][0] +
        index / ((my_fft_grid->proc2local_gs[my_process][0][1] -
                  my_fft_grid->proc2local_gs[my_process][0][0] + 1) *
                 (my_fft_grid->proc2local_gs[my_process][1][1] -
                  my_fft_grid->proc2local_gs[my_process][1][0] + 1));
  }

  my_fft_grid->local_index_to_ref_grid =
      calloc(my_fft_grid->npts_gs_local, sizeof(int));
  for (int index = 0; index < my_fft_grid->npts_gs_local; index++) {
    my_fft_grid->local_index_to_ref_grid[index] = index;
  }

  sort_g_vectors(my_fft_grid);

  *fft_grid = my_fft_grid;
}

void grid_create_fft_grid_layout_from_reference(
    grid_fft_grid_layout **fft_grid, const int npts_global[3],
    const grid_fft_grid_layout *fft_grid_ref) {
  assert(fft_grid_ref != NULL &&
         "Grid creation from reference grid requires a valid reference grid!");
  // Current restriction of the code.
  assert(!fft_grid_ref->ray_distribution &&
         "The reference grid has to have a blocked distribution!");
  // We will use the reference grid to collect the data from other grids. To
  // prevent loss of accuracy, we require the new grid to be coarser than or as
  // coarse as the reference grid.
  assert(npts_global[0] <= fft_grid_ref->npts_global[0] &&
         npts_global[1] <= fft_grid_ref->npts_global[1] &&
         npts_global[2] <= fft_grid_ref->npts_global[2] &&
         "The new grid cannot have more grid points in any direction than the "
         "reference grid!");

  const int number_of_processes = grid_mpi_comm_size(fft_grid_ref->comm);
  const int my_process = grid_mpi_comm_rank(fft_grid_ref->comm);

  grid_fft_grid_layout *my_fft_grid = NULL;
  if (*fft_grid != NULL) {
    my_fft_grid = *fft_grid;
    grid_free_fft_grid_layout(*fft_grid);
  }
  my_fft_grid = calloc(1, sizeof(grid_fft_grid_layout));

  my_fft_grid->grid_id = current_grid_id;
  my_fft_grid->ref_grid_id = fft_grid_ref->grid_id;
  current_grid_id++;
  my_fft_grid->ref_counter = 1;

  my_fft_grid->ray_distribution = true;

  if (npts_global[0] < number_of_processes) {
    // We only distribute in two directions if necessary to reduce communication
    grid_mpi_dims_create(number_of_processes, 2, my_fft_grid->proc_grid);
  } else {
    my_fft_grid->proc_grid[0] = number_of_processes;
    my_fft_grid->proc_grid[1] = 1;
  }

  memcpy(my_fft_grid->npts_global, npts_global, 3 * sizeof(int));

  my_fft_grid->periodic[0] = 1;
  my_fft_grid->periodic[1] = 1;
  grid_mpi_cart_create(fft_grid_ref->comm, 2, &my_fft_grid->proc_grid[0],
                       &my_fft_grid->periodic[0], false, &my_fft_grid->comm);

  grid_mpi_cart_get(my_fft_grid->comm, 2, my_fft_grid->proc_grid,
                    my_fft_grid->periodic, my_fft_grid->proc_coords);

  grid_mpi_cart_sub(my_fft_grid->comm, (const int[2]){1, 0},
                    &my_fft_grid->sub_comm[0]);
  grid_mpi_cart_sub(my_fft_grid->comm, (const int[2]){0, 1},
                    &my_fft_grid->sub_comm[1]);

  setup_proc2local(my_fft_grid, npts_global);

  // Assign the (yz)-rays of the reference grid which are also on the current
  // grid to each process
  my_fft_grid->yz_to_process =
      malloc(npts_global[1] * npts_global[2] * sizeof(int));
  memset(my_fft_grid->yz_to_process, -1,
         npts_global[1] * npts_global[2] * sizeof(int));
  // Count the number of rays on each process
  my_fft_grid->rays_per_process = calloc(number_of_processes, sizeof(int));
  int total_number_of_rays = 0;
#pragma omp parallel for default(none)                                         \
    shared(my_fft_grid, fft_grid_ref, npts_global, number_of_processes)        \
    reduction(+ : total_number_of_rays)
  for (int process = 0; process < number_of_processes; process++) {
    for (int index_y = fft_grid_ref->proc2local_gs[process][1][0];
         index_y <= fft_grid_ref->proc2local_gs[process][1][1]; index_y++) {
      // The right half of the indices is shifted
      const int index_y_shifted = convert_c_index_to_shifted_index(
          index_y, fft_grid_ref->npts_global[1]);
      // Compare the shifted index with the allowed subset of shifted indices of
      // the new grid The allowed set is given by -(n-1)//2...n//2 (these are
      // always n elements)
      if (!is_on_grid(index_y_shifted, npts_global[1]))
        continue;
      const int index_y_new =
          convert_shifted_index_to_c_index(index_y_shifted, npts_global[1]);
      for (int index_z = fft_grid_ref->proc2local_gs[process][2][0];
           index_z <= fft_grid_ref->proc2local_gs[process][2][1]; index_z++) {
        // The right half of the indices is shifted
        const int index_z_shifted = convert_c_index_to_shifted_index(
            index_z, fft_grid_ref->npts_global[2]);
        // Same check for z-coordinate
        if (!is_on_grid(index_z_shifted, npts_global[2]))
          continue;
        const int index_z_new =
            convert_shifted_index_to_c_index(index_z_shifted, npts_global[2]);
        assert(my_fft_grid->yz_to_process[index_y_new * npts_global[2] +
                                          index_z_new] < 0);
        my_fft_grid->yz_to_process[index_y_new * npts_global[2] + index_z_new] =
            process;
        my_fft_grid->rays_per_process[process]++;
        total_number_of_rays++;
      }
    }
  }
  my_fft_grid->npts_gs_local =
      npts_global[0] * my_fft_grid->rays_per_process[my_process];

  allocate_fft_buffers(my_fft_grid);

  int *ray_offsets = calloc(number_of_processes, sizeof(int));
  int *ray_index_per_process = calloc(number_of_processes, sizeof(int));
  for (int process = 1; process < number_of_processes; process++) {
    ray_offsets[process] =
        ray_offsets[process - 1] + my_fft_grid->rays_per_process[process - 1];
  }
  assert(ray_offsets[number_of_processes - 1] +
             my_fft_grid->rays_per_process[number_of_processes - 1] ==
         total_number_of_rays);

  // Create the map of yz index to the yz coordinates and the z-values required
  // for the mixed space
  my_fft_grid->ray_to_yz = malloc(total_number_of_rays * sizeof(int[2]));
  memset(my_fft_grid->ray_to_yz, -1, total_number_of_rays * sizeof(int[2]));
  for (int index_y = 0; index_y < fft_grid_ref->npts_global[1]; index_y++) {
    const int index_y_shifted =
        convert_c_index_to_shifted_index(index_y, fft_grid_ref->npts_global[1]);
    if (!is_on_grid(index_y_shifted, npts_global[1]))
      continue;
    const int index_y_new =
        convert_shifted_index_to_c_index(index_y_shifted, npts_global[1]);
    for (int index_z = 0; index_z < fft_grid_ref->npts_global[2]; index_z++) {
      const int index_z_shifted = convert_c_index_to_shifted_index(
          index_z, fft_grid_ref->npts_global[2]);
      // Same check for z-coordinate
      if (!is_on_grid(index_z_shifted, npts_global[2]))
        continue;
      const int index_z_new =
          convert_shifted_index_to_c_index(index_z_shifted, npts_global[2]);
      const int current_process =
          my_fft_grid
              ->yz_to_process[index_y_new * npts_global[2] + index_z_new];
      const int current_ray_index = ray_index_per_process[current_process];
      const int current_ray_offset = ray_offsets[current_process];
      my_fft_grid->ray_to_yz[current_ray_offset + current_ray_index][0] =
          index_y_new;
      my_fft_grid->ray_to_yz[current_ray_offset + current_ray_index][1] =
          index_z_new;
      ray_index_per_process[current_process]++;
    }
  }
  for (int process = 0; process < number_of_processes; process++) {
    assert(ray_index_per_process[process] ==
               my_fft_grid->rays_per_process[process] &&
           "The number of rays does not match the expected number of rays!");
    const int current_ray_offset = ray_offsets[process];
    for (int ray_index = 0; ray_index < my_fft_grid->rays_per_process[process];
         ray_index++) {
      assert(my_fft_grid->ray_to_yz[current_ray_offset + ray_index][0] >= 0 &&
             my_fft_grid->ray_to_yz[current_ray_offset + ray_index][1] >= 0 &&
             "The ray has to be assigned to a valid yz index!");
      assert(my_fft_grid->ray_to_yz[current_ray_offset + ray_index][0] <
                 npts_global[1] &&
             my_fft_grid->ray_to_yz[current_ray_offset + ray_index][1] <
                 npts_global[2] &&
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
      for (int dir = 0; dir < 3; dir++) {
        my_fft_grid->index_to_g[own_index][dir] =
            convert_shifted_index_to_c_index(shifted_indices[dir],
                                             my_fft_grid->npts_global[dir]);
        my_fft_grid->local_index_to_ref_grid[own_index] = ref_index;
      }
      own_index++;
    }
  }
  assert(own_index == my_fft_grid->npts_gs_local);

  *fft_grid = my_fft_grid;
}

/*******************************************************************************
 * \brief Retains a grid layout.
 * \author Frederick Stein
 ******************************************************************************/
void grid_retain_fft_grid_layout(grid_fft_grid_layout *fft_grid) {
  assert(fft_grid != NULL);
  assert(fft_grid->ref_counter > 0);
  fft_grid->ref_counter++;
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a blocked distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_blocked(double *grid_rs, double complex *grid_gs,
                       const int npts_global[3],
                       const int (*proc2local_rs)[3][2],
                       const int (*proc2local_ms)[3][2],
                       const int (*proc2local_gs)[3][2],
                       const grid_mpi_comm comm,
                       const grid_mpi_comm sub_comm[2]) {
  const int my_process = grid_mpi_comm_rank(comm);

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {
      proc2local_rs[my_process][0][1] - proc2local_rs[my_process][0][0] + 1,
      proc2local_rs[my_process][1][1] - proc2local_rs[my_process][1][0] + 1,
      proc2local_rs[my_process][2][1] - proc2local_rs[my_process][2][0] + 1};
  const int number_of_elements_rs = product3(fft_sizes_rs);
  int fft_sizes_ms[3] = {
      proc2local_ms[my_process][0][1] - proc2local_ms[my_process][0][0] + 1,
      proc2local_ms[my_process][1][1] - proc2local_ms[my_process][1][0] + 1,
      proc2local_ms[my_process][2][1] - proc2local_ms[my_process][2][0] + 1};
  const int number_of_elements_ms = product3(fft_sizes_ms);
  int fft_sizes_gs[3] = {
      proc2local_gs[my_process][0][1] - proc2local_gs[my_process][0][0] + 1,
      proc2local_gs[my_process][1][1] - proc2local_gs[my_process][1][0] + 1,
      proc2local_gs[my_process][2][1] - proc2local_gs[my_process][2][0] + 1};
  const int number_of_elements_gs = product3(fft_sizes_gs);
  const int size_of_buffer =
      imax(imax(number_of_elements_rs, number_of_elements_ms),
           number_of_elements_gs);
  double complex *grid_buffer_1 =
      (double complex *)malloc(size_of_buffer * sizeof(double complex));
  double complex *grid_buffer_2 =
      (double complex *)malloc(size_of_buffer * sizeof(double complex));

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  grid_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  // Copy real array to complex buffer
#pragma omp parallel for default(none)                                         \
    shared(grid_buffer_1, grid_rs, number_of_elements_rs)
  for (int i = 0; i < number_of_elements_rs; i++) {
    grid_buffer_1[i] = grid_rs[i];
  }

  if (proc_grid[0] > 1 && proc_grid[1] > 1) {
    // Perform the first FFT
    fft_1d_fw_local(npts_global[2], fft_sizes_rs[0] * fft_sizes_rs[1],
                    grid_buffer_1, grid_buffer_2);

    // Perform transpose
    collect_y_and_distribute_z_blocked(grid_buffer_2, grid_buffer_1,
                                       npts_global, proc2local_rs,
                                       proc2local_ms, comm, sub_comm);

    // Perform the second FFT
    fft_1d_fw_local(npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2],
                    grid_buffer_1, grid_buffer_2);

    // Perform second transpose
    collect_x_and_distribute_y_blocked(grid_buffer_2, grid_buffer_1,
                                       npts_global, proc2local_ms,
                                       proc2local_gs, comm, sub_comm);

    // Perform the third FFT
    fft_1d_fw_local(npts_global[0], fft_sizes_gs[1] * fft_sizes_gs[2],
                    grid_buffer_1, grid_gs);
  } else if (proc_grid[0] > 1) {
    // Perform the first FFT
    fft_2d_fw_local((const int[2]){npts_global[2], npts_global[1]},
                    fft_sizes_ms[0], grid_buffer_1, grid_buffer_2);

    // Perform second transpose
    collect_x_and_distribute_y_blocked(grid_buffer_2, grid_buffer_1,
                                       npts_global, proc2local_ms,
                                       proc2local_gs, comm, sub_comm);

    // Perform the third FFT
    fft_1d_fw_local(npts_global[0], fft_sizes_gs[1] * fft_sizes_gs[2],
                    grid_buffer_1, grid_gs);
  } else {
    fft_3d_fw_local(npts_global, grid_buffer_1, grid_gs);
  }

  free(grid_buffer_1);
  free(grid_buffer_2);
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT using a blocked distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_blocked(double complex *grid_gs, double *grid_rs,
                       const int npts_global[3],
                       const int (*proc2local_rs)[3][2],
                       const int (*proc2local_ms)[3][2],
                       const int (*proc2local_gs)[3][2],
                       const grid_mpi_comm comm,
                       const grid_mpi_comm sub_comm[2]) {
  const int my_process = grid_mpi_comm_rank(comm);

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {
      proc2local_rs[my_process][0][1] - proc2local_rs[my_process][0][0] + 1,
      proc2local_rs[my_process][1][1] - proc2local_rs[my_process][1][0] + 1,
      proc2local_rs[my_process][2][1] - proc2local_rs[my_process][2][0] + 1};
  const int number_of_elements_rs = product3(fft_sizes_rs);
  int fft_sizes_ms[3] = {
      proc2local_ms[my_process][0][1] - proc2local_ms[my_process][0][0] + 1,
      proc2local_ms[my_process][1][1] - proc2local_ms[my_process][1][0] + 1,
      proc2local_ms[my_process][2][1] - proc2local_ms[my_process][2][0] + 1};
  const int number_of_elements_ms = product3(fft_sizes_ms);
  int fft_sizes_gs[3] = {
      proc2local_gs[my_process][0][1] - proc2local_gs[my_process][0][0] + 1,
      proc2local_gs[my_process][1][1] - proc2local_gs[my_process][1][0] + 1,
      proc2local_gs[my_process][2][1] - proc2local_gs[my_process][2][0] + 1};
  const int number_of_elements_gs = product3(fft_sizes_gs);
  const int size_of_buffer =
      imax(imax(number_of_elements_rs, number_of_elements_ms),
           number_of_elements_gs);
  double complex *grid_buffer_1 =
      (double complex *)malloc(size_of_buffer * sizeof(double complex));
  double complex *grid_buffer_2 =
      (double complex *)malloc(size_of_buffer * sizeof(double complex));

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  grid_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  if (proc_grid[0] > 1 && proc_grid[1] > 1) {
    // Perform the first FFT and one transposition (z,y,x)->(x,z,y)
    fft_1d_bw_local(npts_global[0], fft_sizes_gs[1] * fft_sizes_gs[2], grid_gs,
                    grid_buffer_1);

    // Collect data in y-direction and distribute x-direction
    collect_y_and_distribute_x_blocked(grid_buffer_1, grid_buffer_2,
                                       npts_global, proc2local_gs,
                                       proc2local_ms, comm, sub_comm);

    // Perform the second FFT and one transposition (x,z,y)->(y,x,z)
    fft_1d_bw_local(npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2],
                    grid_buffer_2, grid_buffer_1);

    // Collect data in z-direction and distribute y-direction
    collect_z_and_distribute_y_blocked(grid_buffer_1, grid_buffer_2,
                                       npts_global, proc2local_ms,
                                       proc2local_rs, comm, sub_comm);

    // Perform the third FFT and one transposition (y,x,z)->(z,y,x)
    fft_1d_bw_local(npts_global[2], fft_sizes_rs[0] * fft_sizes_rs[1],
                    grid_buffer_2, grid_buffer_1);
  } else if (proc_grid[0] > 1) {
    // Perform the first FFT and one transposition (z,y,x)->(x,z,y)
    fft_1d_bw_local(npts_global[0], fft_sizes_gs[1] * fft_sizes_gs[2], grid_gs,
                    grid_buffer_1);

    // Collect data in y-direction and distribute x-direction
    collect_y_and_distribute_x_blocked(grid_buffer_1, grid_buffer_2,
                                       npts_global, proc2local_gs,
                                       proc2local_ms, comm, sub_comm);

    // Perform the second FFT and one transposition (x,z,y)->(y,x,z)
    fft_2d_bw_local((const int[2]){npts_global[2], npts_global[1]},
                    fft_sizes_ms[0], grid_buffer_2, grid_buffer_1);
  } else {
    fft_3d_bw_local(npts_global, grid_gs, grid_buffer_1);
  }

// Copy real array to complex buffer
#pragma omp parallel default(none)                                             \
    shared(grid_buffer_1, grid_rs, number_of_elements_rs)
  for (int i = 0; i < number_of_elements_rs; i++) {
    grid_rs[i] = creal(grid_buffer_1[i]);
  }

  free(grid_buffer_1);
  free(grid_buffer_2);
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a ray distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_ray(double *grid_rs, double complex *grid_gs,
                   const int npts_global[3], const int (*proc2local_rs)[3][2],
                   const int (*proc2local_ms)[3][2], const int *yz_to_process,
                   const int *rays_per_process, const int (*ray_to_yz)[2],
                   const grid_mpi_comm comm, const grid_mpi_comm sub_comm[2]) {
  const int my_process = grid_mpi_comm_rank(comm);

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {
      proc2local_rs[my_process][0][1] - proc2local_rs[my_process][0][0] + 1,
      proc2local_rs[my_process][1][1] - proc2local_rs[my_process][1][0] + 1,
      proc2local_rs[my_process][2][1] - proc2local_rs[my_process][2][0] + 1};
  const int number_of_elements_rs = product3(fft_sizes_rs);
  int fft_sizes_ms[3] = {
      proc2local_ms[my_process][0][1] - proc2local_ms[my_process][0][0] + 1,
      proc2local_ms[my_process][1][1] - proc2local_ms[my_process][1][0] + 1,
      proc2local_ms[my_process][2][1] - proc2local_ms[my_process][2][0] + 1};
  const int number_of_elements_ms = product3(fft_sizes_ms);
  int number_of_local_yz_rays = 0;
#pragma omp parallel for default(none)                                         \
    shared(yz_to_process, my_process, npts_global)                             \
    reduction(+ : number_of_local_yz_rays)
  for (int yz_ray = 0; yz_ray < npts_global[1] * npts_global[2]; yz_ray++) {
    if (yz_to_process[yz_ray] == my_process)
      number_of_local_yz_rays++;
  }
  const int number_of_elements_gs = number_of_local_yz_rays * npts_global[0];
  const int size_of_buffer =
      imax(imax(number_of_elements_rs, number_of_elements_ms),
           number_of_elements_gs);
  double complex *grid_buffer_1 =
      (double complex *)malloc(size_of_buffer * sizeof(double complex));
  double complex *grid_buffer_2 =
      (double complex *)malloc(size_of_buffer * sizeof(double complex));

// Copy real array to complex buffer
#pragma omp parallel for default(none)                                         \
    shared(grid_buffer_1, grid_rs, number_of_elements_rs)
  for (int i = 0; i < number_of_elements_rs; i++) {
    grid_buffer_1[i] = grid_rs[i];
  }

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  grid_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  if (proc_grid[0] > 1 && proc_grid[1] > 1) {
    // Perform the first FFT
    fft_1d_fw_local(npts_global[2], fft_sizes_rs[0] * fft_sizes_rs[1],
                    grid_buffer_1, grid_buffer_2);

    // Perform transpose
    collect_y_and_distribute_z_blocked(grid_buffer_2, grid_buffer_1,
                                       npts_global, proc2local_rs,
                                       proc2local_ms, comm, sub_comm);

    // Perform the second FFT
    fft_1d_fw_local(npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2],
                    grid_buffer_1, grid_buffer_2);

    // Perform second transpose
    collect_x_and_distribute_y_ray(grid_buffer_2, grid_buffer_1, npts_global,
                                   proc2local_ms, rays_per_process, ray_to_yz,
                                   comm);

    // Perform the third FFT
    fft_1d_fw_local(npts_global[0], number_of_local_yz_rays, grid_buffer_1,
                    grid_gs);
  } else if (proc_grid[0] > 1) {
    // Perform the first FFT
    fft_2d_fw_local((const int[2]){npts_global[2], npts_global[1]},
                    fft_sizes_ms[0], grid_buffer_1, grid_buffer_2);

    // Perform second transpose
    collect_x_and_distribute_y_ray(grid_buffer_2, grid_buffer_1, npts_global,
                                   proc2local_ms, rays_per_process, ray_to_yz,
                                   comm);

    // Perform the third FFT
    fft_1d_fw_local(npts_global[0], number_of_local_yz_rays, grid_buffer_1,
                    grid_gs);
  } else {
    fft_3d_fw_local(npts_global, grid_buffer_1, grid_buffer_2);
    // Copy to the new format
    // Maybe, a 2D FFT, redistribution to rays and final FFT is faster
    int ray_index = 0;
    for (int index_y = 0; index_y < npts_global[1]; index_y++) {
      for (int index_z = 0; index_z < npts_global[2]; index_z++) {
        if (yz_to_process[index_y * npts_global[2] + index_z] == 0) {
          memcpy(&grid_gs[ray_index * npts_global[0]],
                 &grid_buffer_2[index_z * npts_global[0] * npts_global[1] +
                                index_y * npts_global[0]],
                 npts_global[0] * sizeof(double complex));
          ray_index++;
        }
      }
    }
  }

  free(grid_buffer_1);
  free(grid_buffer_2);
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT using a ray distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_ray(double complex *grid_gs, double *grid_rs,
                   const int npts_global[3], const int (*proc2local_rs)[3][2],
                   const int (*proc2local_ms)[3][2], const int *yz_to_process,
                   const int *rays_per_process, const int (*ray_to_yz)[2],
                   const grid_mpi_comm comm, const grid_mpi_comm sub_comm[2]) {
  const int my_process = grid_mpi_comm_rank(comm);

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {
      proc2local_rs[my_process][0][1] - proc2local_rs[my_process][0][0] + 1,
      proc2local_rs[my_process][1][1] - proc2local_rs[my_process][1][0] + 1,
      proc2local_rs[my_process][2][1] - proc2local_rs[my_process][2][0] + 1};
  const int number_of_elements_rs = product3(fft_sizes_rs);
  int fft_sizes_ms[3] = {
      proc2local_ms[my_process][0][1] - proc2local_ms[my_process][0][0] + 1,
      proc2local_ms[my_process][1][1] - proc2local_ms[my_process][1][0] + 1,
      proc2local_ms[my_process][2][1] - proc2local_ms[my_process][2][0] + 1};
  const int number_of_elements_ms = product3(fft_sizes_ms);
  int number_of_local_yz_rays = 0;
#pragma omp parallel for default(none)                                         \
    shared(yz_to_process, my_process, npts_global)                             \
    reduction(+ : number_of_local_yz_rays)
  for (int yz_ray = 0; yz_ray < npts_global[1] * npts_global[2]; yz_ray++) {
    if (yz_to_process[yz_ray] == my_process)
      number_of_local_yz_rays++;
  }
  const int number_of_elements_gs = number_of_local_yz_rays * npts_global[0];
  const int size_of_buffer =
      imax(imax(number_of_elements_rs, number_of_elements_ms),
           number_of_elements_gs);
  double complex *grid_buffer_1 =
      (double complex *)malloc(size_of_buffer * sizeof(double complex));
  double complex *grid_buffer_2 =
      (double complex *)malloc(size_of_buffer * sizeof(double complex));

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  grid_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  if (proc_grid[0] > 1 && proc_grid[1] > 1) {
    // Perform the first FFT
    fft_1d_bw_local(npts_global[0], number_of_local_yz_rays, grid_gs,
                    grid_buffer_1);

    // Perform transpose
    collect_y_and_distribute_x_ray(grid_buffer_1, grid_buffer_2, npts_global,
                                   proc2local_ms, rays_per_process, ray_to_yz,
                                   comm);

    // Perform the second FFT
    fft_1d_bw_local(npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2],
                    grid_buffer_2, grid_buffer_1);

    // Perform second transpose
    collect_z_and_distribute_y_blocked(grid_buffer_1, grid_buffer_2,
                                       npts_global, proc2local_ms,
                                       proc2local_rs, comm, sub_comm);

    // Perform the third FFT
    fft_1d_bw_local(npts_global[2], fft_sizes_rs[0] * fft_sizes_rs[1],
                    grid_buffer_2, grid_buffer_1);
  } else if (proc_grid[0] > 1) {
    // Perform the first FFT
    fft_1d_bw_local(npts_global[0], number_of_local_yz_rays, grid_gs,
                    grid_buffer_1);

    // Perform transpose
    collect_y_and_distribute_x_ray(grid_buffer_1, grid_buffer_2, npts_global,
                                   proc2local_ms, rays_per_process, ray_to_yz,
                                   comm);

    // Perform the second FFT
    fft_2d_bw_local((const int[2]){npts_global[2], npts_global[1]},
                    fft_sizes_ms[0], grid_buffer_2, grid_buffer_1);
  } else {
    // Copy to the new format
    // Maybe, the order 1D FFT, redistribution to blocks and 2D FFT is faster
    memset(grid_buffer_2, 0, product3(npts_global));
    int my_ray_offset = 0;
#pragma omp parallel for default(none)                                         \
    shared(yz_to_process, my_process, npts_global, rays_per_process,           \
               grid_buffer_2) reduction(+ : my_ray_offset)
    for (int process = 0; process < my_process; process++)
      my_ray_offset += rays_per_process[process];
#pragma omp parallel for default(none)                                         \
    shared(yz_to_process, my_process, npts_global, rays_per_process,           \
               grid_buffer_2, my_ray_offset, ray_to_yz, grid_gs)
    for (int yz_ray = 0; yz_ray < rays_per_process[my_process]; yz_ray++) {
      const int index_y = ray_to_yz[my_ray_offset + yz_ray][0];
      const int index_z = ray_to_yz[my_ray_offset + yz_ray][1];

      memcpy(&grid_buffer_2[index_z * npts_global[0] * npts_global[1] +
                            index_y * npts_global[0]],
             &grid_gs[yz_ray * npts_global[0]],
             npts_global[0] * sizeof(double complex));
    }
    fft_3d_bw_local(npts_global, grid_buffer_2, grid_buffer_1);
  }

// Copy real array to complex buffer
#pragma omp parallel for default(none)                                         \
    shared(grid_buffer_1, grid_rs, number_of_elements_rs)
  for (int i = 0; i < number_of_elements_rs; i++) {
    grid_rs[i] = creal(grid_buffer_1[i]);
  }

  free(grid_buffer_1);
  free(grid_buffer_2);
}

// EOF
