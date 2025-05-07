/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "grid_multigrid_test.h"
#include "common/grid_common.h"
#include "common/grid_mpi.h"
#include "grid_fft_grid.h"
#include "grid_multigrid.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int multigrid_reorder_grids_test_low(const int npts_global[3]) {
  //
  const grid_mpi_comm comm = grid_mpi_comm_world;
  const int number_of_processes = grid_mpi_comm_size(comm);
  const int my_process = grid_mpi_comm_rank(comm);

  int errors = 0;

  // Initial distribution: distributed in x
  int(*proc2local_x)[3][2] = calloc(number_of_processes, sizeof(int[3][2]));
  for (int process = 0; process < number_of_processes; process++) {
    proc2local_x[process][0][0] =
        process * npts_global[0] / number_of_processes;
    proc2local_x[process][0][1] =
        (process + 1) * npts_global[0] / number_of_processes - 1;
    proc2local_x[process][1][0] = 0;
    proc2local_x[process][1][1] = npts_global[1] - 1;
    proc2local_x[process][2][0] = 0;
    proc2local_x[process][2][1] = npts_global[2] - 1;
  }
  const int my_sizes_x[3] = {proc2local_x[my_process][0][1] -
                                 proc2local_x[my_process][0][0] + 1,
                             npts_global[1], npts_global[2]};
  const int number_of_local_elements_x = product3(my_sizes_x);
  double *grid_x = calloc(number_of_local_elements_x, sizeof(double));
  if (number_of_local_elements_x > 0) {
    for (int index = 0; index < number_of_local_elements_x; index++) {
      const int index_x =
          index % my_sizes_x[0] + proc2local_x[my_process][0][0];
      const int index_y = index / my_sizes_x[0] % my_sizes_x[1] +
                          proc2local_x[my_process][1][0];
      const int index_z = index / my_sizes_x[0] / my_sizes_x[1] +
                          proc2local_x[my_process][2][0];
      grid_x[index] = index_x * npts_global[1] * npts_global[2] +
                      index_y * npts_global[2] + index_z;
    }
  }

  // Convert to distribution in y
  int(*proc2local_y)[3][2] = calloc(number_of_processes, sizeof(int[3][2]));
  for (int process = 0; process < number_of_processes; process++) {
    proc2local_y[process][0][0] = 0;
    proc2local_y[process][0][1] = npts_global[0] - 1;
    proc2local_y[process][1][0] =
        process * npts_global[1] / number_of_processes;
    proc2local_y[process][1][1] =
        (process + 1) * npts_global[1] / number_of_processes - 1;
    proc2local_y[process][2][0] = 0;
    proc2local_y[process][2][1] = npts_global[2] - 1;
  }
  const int my_sizes_y[3] = {
      proc2local_y[my_process][0][1] - proc2local_y[my_process][0][0] + 1,
      proc2local_y[my_process][1][1] - proc2local_y[my_process][1][0] + 1,
      proc2local_y[my_process][2][1] - proc2local_y[my_process][2][0] + 1};
  const int number_of_local_elements_y = product3(my_sizes_y);
  double *grid_y = calloc(number_of_local_elements_y, sizeof(double));

  redistribute_grids(grid_x, grid_y, comm, comm, npts_global, proc2local_x,
                     proc2local_y);

  double max_error = 0.0;
  if (number_of_local_elements_x > 0) {
    for (int index = 0; index < number_of_local_elements_y; index++) {
      const int index_x =
          index % my_sizes_y[0] + proc2local_y[my_process][0][0];
      const int index_y = index / my_sizes_y[0] % my_sizes_y[1] +
                          proc2local_y[my_process][1][0];
      const int index_z = index / my_sizes_y[0] / my_sizes_y[1] +
                          proc2local_y[my_process][2][0];
      const double my_value = grid_y[index];
      const double ref_value = index_x * npts_global[1] * npts_global[2] +
                               index_y * npts_global[2] + index_z;
      const double current_error = fabs(my_value - ref_value);
      if (current_error > 1.0e-12 * ref_value)
        printf("%i Error %i %i %i: %f %f\n", my_process, index_x, index_y,
               index_z, my_value, ref_value);
      max_error = fmax(max_error, current_error);
    }
  }
  grid_mpi_max_double(&max_error, 1, comm);
  if (max_error > 1.0e-12) {
    if (my_process == 0)
      printf("Redistribution x->y does not work properly (sizes: %i %i %i, "
             "processes: %i): %f\n",
             npts_global[0], npts_global[1], npts_global[2],
             number_of_processes, max_error);
    errors++;
  }

  // Convert to distribution in z

  // Convert to distribution in x

  // If the number of processes is factorizable in two non-trivial factors

  // convert to distribution in xy

  // convert to distribution in xz

  // convert to distribution in yz

  // If the number of processes is factorizable in three non-trivial factors

  // convert to distribution in xyz

  if (errors == 0 && my_process == 0)
    printf(
        "The redistribution works correctly (sizes: %i %i %i, processes: %i)\n",
        npts_global[0], npts_global[1], npts_global[2], number_of_processes);

  free(proc2local_x);
  free(proc2local_y);
  free(grid_x);
  free(grid_y);

  return errors;
}

int multigrid_gather_scatter_halo_test() {
  //
  int errors = 0;
  return errors;
}

/*******************************************************************************
 * \brief Function to test the Multigrid backend.
 * \author Frederick Stein
 ******************************************************************************/
int multigrid_test() {

  const int number_of_processes = grid_mpi_comm_size(grid_mpi_comm_world);
  const int my_process = grid_mpi_comm_rank(grid_mpi_comm_world);
  const int npts_global[2][3] = {{4 * number_of_processes, 4, 4},
                                 {2 * number_of_processes, 2, 2}};
  const int npts_local[2][3] = {{8, 8, 8}, {4, 4, 4}};
  const int shift_local[2][3] = {{-2 + 4 * my_process, -2, -2},
                                 {-1 + 2 * my_process, -1, -1}};
  const int border_width[2][3] = {{2, 2, 2}, {1, 1, 1}};
  const double dh[2][3][3] = {
      {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}},
      {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
  const double dh_inv[2][3][3] = {
      {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}},
      {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
  const int pgrid_dims[2][3] = {{number_of_processes, 1, 1},
                                {number_of_processes, 1, 1}};
  grid_multigrid *multigrid = NULL;
  grid_create_multigrid(true, 2, npts_global, npts_local, shift_local,
                        border_width, dh, dh_inv, pgrid_dims,
                        grid_mpi_comm_world, &multigrid);
  for (int level = 0; level < multigrid->nlevels; level++) {
    assert(multigrid->fft_grid_layouts[level]->grid_id > 0);
  }

  grid_fft_grid_layout *fft_grid_layout = NULL;
  grid_create_fft_grid_layout(&fft_grid_layout, grid_mpi_comm_world,
                              npts_global[0], dh_inv[0]);

  grid_fft_real_rs_grid rs_grid;
  memset(&rs_grid, 0, sizeof(grid_fft_real_rs_grid));
  grid_create_real_rs_grid(&rs_grid, fft_grid_layout);

  grid_copy_to_multigrid_single(multigrid, rs_grid.data,
                                rs_grid.fft_grid_layout->comm,
                                rs_grid.fft_grid_layout->proc2local_rs);
  grid_copy_from_multigrid_single(multigrid, rs_grid.data,
                                  rs_grid.fft_grid_layout->comm,
                                  rs_grid.fft_grid_layout->proc2local_rs);

  grid_free_real_rs_grid(&rs_grid);
  grid_free_fft_grid_layout(fft_grid_layout);
  grid_free_multigrid(multigrid);

  int errors = 0;

  errors += multigrid_reorder_grids_test_low((const int[3]){1, 1, 1});
  errors += multigrid_reorder_grids_test_low((const int[3]){2, 2, 2});
  errors += multigrid_reorder_grids_test_low((const int[3]){5, 5, 5});
  errors += multigrid_reorder_grids_test_low((const int[3]){2, 3, 5});
  errors += multigrid_reorder_grids_test_low((const int[3]){3, 5, 2});
  errors += multigrid_reorder_grids_test_low((const int[3]){8, 5, 4});
  errors += multigrid_reorder_grids_test_low((const int[3]){7, 10, 9});
  errors += multigrid_reorder_grids_test_low((const int[3]){23, 29, 27});
  errors += multigrid_reorder_grids_test_low((const int[3]){100, 100, 100});

  return errors;
}

// EOF
