/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "grid_multigrid.h"
#include "common/grid_common.h"
#include "common/grid_library.h"
#include "grid_fft_grid.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool grid_get_multigrid_orthorhombic(const grid_multigrid *multigrid) {
  assert(multigrid != NULL);
  return multigrid->orthorhombic;
}

int grid_get_multigrid_nlevels(const grid_multigrid *multigrid) {
  assert(multigrid != NULL);
  return multigrid->nlevels;
}

void grid_get_multigrid_npts_global(const grid_multigrid *multigrid,
                                    int *nlevels, int **npts_global) {
  assert(multigrid != NULL);
  *nlevels = multigrid->nlevels;
  *npts_global = (int *)multigrid->npts_global;
}

void grid_get_multigrid_npts_local(const grid_multigrid *multigrid,
                                   int *nlevels, int **npts_local) {
  assert(multigrid != NULL);
  *nlevels = multigrid->nlevels;
  *npts_local = (int *)multigrid->npts_local;
}

void grid_get_multigrid_shift_local(const grid_multigrid *multigrid,
                                    int *nlevels, int **shift_local) {
  assert(multigrid != NULL);
  *nlevels = multigrid->nlevels;
  *shift_local = (int *)multigrid->shift_local;
}

void grid_get_multigrid_border_width(const grid_multigrid *multigrid,
                                     int *nlevels, int **border_width) {
  assert(multigrid != NULL);
  *nlevels = multigrid->nlevels;
  *border_width = (int *)multigrid->border_width;
}

void grid_get_multigrid_dh(const grid_multigrid *multigrid, int *nlevels,
                           double **dh) {
  assert(multigrid != NULL);
  *nlevels = multigrid->nlevels;
  *dh = (double *)multigrid->dh;
}

void grid_get_multigrid_dh_inv(const grid_multigrid *multigrid, int *nlevels,
                               double **dh_inv) {
  assert(multigrid != NULL);
  *nlevels = multigrid->nlevels;
  *dh_inv = (double *)multigrid->dh_inv;
}

grid_mpi_fint grid_get_multigrid_fortran_comm(const grid_multigrid *multigrid) {
  assert(multigrid != NULL);
  return grid_mpi_comm_c2f(multigrid->comm);
}

grid_mpi_comm grid_get_multigrid_comm(const grid_multigrid *multigrid) {
  assert(multigrid != NULL);
  return multigrid->comm;
}

void grid_copy_to_multigrid_local(const grid_multigrid *multigrid,
                                  const offload_buffer **grids) {
  for (int level = 0; level < multigrid->nlevels; level++) {
    memcpy(offload_get_buffer_host_pointer(multigrid->grids[level]),
           offload_get_buffer_host_pointer((offload_buffer *)grids[level]),
           sizeof(double) * multigrid->npts_local[level][0] *
               multigrid->npts_local[level][1] *
               multigrid->npts_local[level][2]);
  }
}

void grid_copy_from_multigrid_local(const grid_multigrid *multigrid,
                                    offload_buffer **grids) {
  for (int level = 0; level < multigrid->nlevels; level++) {
    memcpy(offload_get_buffer_host_pointer(grids[level]),
           offload_get_buffer_host_pointer(multigrid->grids[level]),
           sizeof(double) * multigrid->npts_local[level][0] *
               multigrid->npts_local[level][1] *
               multigrid->npts_local[level][2]);
  }
}

void grid_copy_to_multigrid_local_single(const grid_multigrid *multigrid,
                                         const double *grid, const int level) {
  memcpy(offload_get_buffer_host_pointer(multigrid->grids[level]), grid,
         sizeof(double) * multigrid->npts_local[level][0] *
             multigrid->npts_local[level][1] * multigrid->npts_local[level][2]);
}

void grid_copy_from_multigrid_local_single(const grid_multigrid *multigrid,
                                           double *grid, const int level) {
  memcpy(grid, offload_get_buffer_host_pointer(multigrid->grids[level]),
         sizeof(double) * multigrid->npts_local[level][0] *
             multigrid->npts_local[level][1] * multigrid->npts_local[level][2]);
}

void grid_copy_to_multigrid_single(const grid_multigrid *multigrid,
                                   const double *grid, const grid_mpi_comm comm,
                                   const int (*proc2local)[3][2]) {
  if (multigrid->nlevels > 1) {
    // Copy the data into our own grid
    redistribute_grids(grid, (double *)multigrid->fft_rs_grids->data, comm,
                       multigrid->fft_rs_grids->fft_grid_layout->comm,
                       multigrid->npts_global[0], proc2local,
                       (const int(*)[3][2])multigrid->fft_rs_grids
                           ->fft_grid_layout->proc2local_rs);
    // FFT the data
    fft_3d_fw(&multigrid->fft_rs_grids[0], &multigrid->fft_gs_grids[0]);
    grid_copy_to_multigrid_general_single(
        multigrid, 0, (double *)multigrid->fft_rs_grids->data,
        multigrid->fft_rs_grids->fft_grid_layout->comm,
        (const int *)multigrid->fft_rs_grids->fft_grid_layout->proc2local_rs);
    long int total_number_of_elements =
        ((long int)multigrid->npts_global[0][0]) *
        ((long int)multigrid->npts_global[0][1]) *
        ((long int)multigrid->npts_global[0][2]);
    // Create grids referencing the large one
    for (int level = 1; level < multigrid->nlevels; level++) {
      // Copy the data to the coarse grids
      grid_copy_to_coarse_grid(&multigrid->fft_gs_grids[0],
                               &multigrid->fft_gs_grids[level]);
      fft_3d_bw(&multigrid->fft_gs_grids[level],
                &multigrid->fft_rs_grids[level]);
      const double factor = (double)total_number_of_elements /
                            ((double)multigrid->npts_global[level][0]) /
                            ((double)multigrid->npts_global[level][1]) /
                            ((double)multigrid->npts_global[level][2]);
      const int(*my_bounds)[2] =
          multigrid->fft_rs_grids[level]
              .fft_grid_layout->proc2local_rs[grid_mpi_comm_rank(
                  multigrid->fft_rs_grids[level].fft_grid_layout->comm)];
      for (int index = 0; index < (my_bounds[0][1] - my_bounds[0][0] + 1) *
                                      (my_bounds[1][1] - my_bounds[1][0] + 1) *
                                      (my_bounds[2][1] - my_bounds[2][0] + 1);
           index++)
        multigrid->fft_rs_grids[level].data[index] *= factor;
      // Redistribute to the realspace grid
      grid_copy_to_multigrid_general_single(
          multigrid, level, (double *)multigrid->fft_rs_grids[level].data,
          multigrid->fft_rs_grids[level].fft_grid_layout->comm,
          (const int *)multigrid->fft_rs_grids[level]
              .fft_grid_layout->proc2local_rs);
    }
  } else {
    // Copy the data directly to the realspace grid
    grid_copy_to_multigrid_general_single(multigrid, 0, grid, comm,
                                          (const int *)proc2local);
  }
}

void grid_copy_from_multigrid_single(const grid_multigrid *multigrid,
                                     double *grid, const grid_mpi_comm comm,
                                     const int (*proc2local)[3][2]) {
  if (multigrid->nlevels > 1) {
    // Redistribute the data on the realspace grid to an FFT-optimal layout
    grid_copy_from_multigrid_general_single(
        multigrid, 0, multigrid->fft_rs_grids[0].data,
        multigrid->fft_rs_grids[0].fft_grid_layout->comm,
        (const int *)multigrid->fft_rs_grids[0].fft_grid_layout->proc2local_rs);
    // FFT the data
    fft_3d_fw(&multigrid->fft_rs_grids[0], &multigrid->fft_gs_grids[0]);
    for (int level = 1; level < multigrid->nlevels; level++) {
      // Redistribute the data on the coarse grids
      grid_copy_from_multigrid_general_single(
          multigrid, level, multigrid->fft_rs_grids[level].data,
          multigrid->fft_rs_grids[level].fft_grid_layout->comm,
          (const int *)multigrid->fft_rs_grids[level]
              .fft_grid_layout->proc2local_rs);
      // FFT the data
      fft_3d_fw(&multigrid->fft_rs_grids[level],
                &multigrid->fft_gs_grids[level]);
      // Add the data on the fine grid
      grid_add_to_fine_grid(&multigrid->fft_gs_grids[level],
                            &multigrid->fft_gs_grids[0]);
    }
    // FFT back to realspace
    fft_3d_bw(&multigrid->fft_gs_grids[0], &multigrid->fft_rs_grids[0]);
    // Copy the data back to the original grid
    redistribute_grids((double *)multigrid->fft_rs_grids[0].data, grid,
                       multigrid->fft_rs_grids[0].fft_grid_layout->comm, comm,
                       multigrid->npts_global[0],
                       (const int(*)[3][2])multigrid->fft_rs_grids[0]
                           .fft_grid_layout->proc2local_rs,
                       proc2local);
  } else {
    // Copy the data directly to the realspace grid
    grid_copy_from_multigrid_general_single(multigrid, 0, grid, multigrid->comm,
                                            (const int *)proc2local);
  }
}

void grid_copy_to_multigrid_single_f(const grid_multigrid *multigrid,
                                     const double *grid,
                                     const grid_mpi_fint comm,
                                     const int (*proc2local)[3][2]) {
  grid_copy_to_multigrid_single(multigrid, grid, grid_mpi_comm_f2c(comm),
                                proc2local);
}

void grid_copy_from_multigrid_single_f(const grid_multigrid *multigrid,
                                       double *grid, const grid_mpi_fint comm,
                                       const int (*proc2local)[3][2]) {
  grid_copy_from_multigrid_single(multigrid, grid, grid_mpi_comm_f2c(comm),
                                  proc2local);
}

void grid_copy_to_multigrid_local_single_f(const grid_multigrid *multigrid,
                                           const double *grid,
                                           const int level) {
  grid_copy_to_multigrid_local_single(multigrid, grid, level - 1);
}

void grid_copy_from_multigrid_local_single_f(const grid_multigrid *multigrid,
                                             double *grid, const int level) {
  grid_copy_from_multigrid_local_single(multigrid, grid, level - 1);
}

void grid_copy_to_multigrid_serial(double *grid_rs, const double *grid_pw,
                                   const int npts_rs[3],
                                   const int border_width[3]) {
  if (border_width[0] == 0 && border_width[1] == 0 && border_width[2] == 0) {
    memcpy(grid_rs, grid_pw,
           npts_rs[0] * npts_rs[1] * npts_rs[2] * sizeof(double));
  } else {
    int npts_pw[3];
    for (int dir = 0; dir < 3; dir++)
      npts_pw[dir] = npts_rs[dir] - 2 * border_width[dir];
#pragma omp parallel for default(none)                                         \
    shared(grid_rs, grid_pw, npts_rs, npts_pw, border_width)
    for (int iz = 0; iz < npts_rs[2]; iz++) {
      //
      int iz_pw = iz - border_width[2];
      if (iz < border_width[2])
        iz_pw += npts_pw[2];
      if (iz >= npts_pw[2] + border_width[2])
        iz_pw -= npts_pw[2];
      for (int iy = 0; iy < npts_rs[1]; iy++) {
        //
        int iy_pw = iy - border_width[1];
        if (iy < border_width[1])
          iy_pw += npts_pw[1];
        if (iy >= npts_pw[1] + border_width[1])
          iy_pw -= npts_pw[1];
        for (int ix = 0; ix < npts_rs[0]; ix++) {
          //
          int ix_pw = ix - border_width[0];
          if (ix < border_width[0])
            ix_pw += npts_pw[0];
          if (ix >= npts_pw[0] + border_width[0])
            ix_pw -= npts_pw[0];
          grid_rs[iz * npts_rs[0] * npts_rs[1] + iy * npts_rs[0] + ix] =
              grid_pw[iz_pw * npts_pw[1] * npts_pw[2] + iy_pw * npts_pw[0] +
                      ix_pw];
        }
      }
    }
  }
}

void grid_copy_to_multigrid_replicated(
    double *grid_rs, const double *grid_pw, const int npts_rs[3],
    const int border_width[3], const grid_mpi_comm comm,
    const int proc2local[grid_mpi_comm_size(comm)][3][2]) {

  const int number_of_processes = grid_mpi_comm_size(comm);
  const int my_process = grid_mpi_comm_rank(comm);

  memset(grid_rs, 0, npts_rs[0] * npts_rs[1] * npts_rs[2] * sizeof(double));

  // Determine the maximum number of grid points on a single rank
  int maximum_number_of_elements = 0;
  for (int process = 0; process < number_of_processes; process++) {
    const int current_number_of_elements =
        (proc2local[process][0][1] - proc2local[process][0][0] + 1) *
        (proc2local[process][1][1] - proc2local[process][1][0] + 1) *
        (proc2local[process][2][1] - proc2local[process][2][0] + 1);
    maximum_number_of_elements =
        (current_number_of_elements > maximum_number_of_elements
             ? current_number_of_elements
             : maximum_number_of_elements);
    for (int dir = 0; dir < 3; dir++) {
      assert(proc2local[process][dir][0] >= 0);
      assert(proc2local[process][dir][1] - proc2local[process][dir][0] + 1 >=
             0);
    }
  }

  // Allocate communication buffers
  double *send_buffer = calloc(maximum_number_of_elements, sizeof(double));
  double *recv_buffer = calloc(maximum_number_of_elements, sizeof(double));

  // Initialize send buffer with local data
  const int my_number_of_elements =
      (proc2local[my_process][0][1] - proc2local[my_process][0][0] + 1) *
      (proc2local[my_process][1][1] - proc2local[my_process][1][0] + 1) *
      (proc2local[my_process][2][1] - proc2local[my_process][2][0] + 1);

  memcpy(send_buffer, grid_pw, my_number_of_elements * sizeof(double));

  // We send and receive from our direct neighbor only
  const int send_process_static = modulo(my_process + 1, number_of_processes);
  const int recv_process_static = modulo(my_process - 1, number_of_processes);

  int send_size[3], recv_size[3];
  grid_mpi_request send_request = grid_mpi_request_null;
  grid_mpi_request recv_request = grid_mpi_request_null;

  // Pass local data to each process
  for (int process_shift = 0; process_shift < number_of_processes;
       process_shift++) {

    // Determine the process whose data we receive
    const int recv_process =
        modulo(my_process - process_shift - 1, number_of_processes);
    for (int dir = 0; dir < 3; dir++)
      recv_size[dir] = proc2local[recv_process][dir][1] -
                       proc2local[recv_process][dir][0] + 1;

    if (process_shift < number_of_processes) {
      grid_mpi_irecv_double(recv_buffer, product3(recv_size),
                            recv_process_static, process_shift, comm,
                            &recv_request);
    }

    // Determine the process whose data we send
    const int send_process =
        modulo(my_process - process_shift, number_of_processes);
    for (int dir = 0; dir < 3; dir++)
      send_size[dir] = proc2local[send_process][dir][1] -
                       proc2local[send_process][dir][0] + 1;

    if (process_shift < number_of_processes) {
      if (process_shift > 1)
        grid_mpi_wait(&send_request);
      grid_mpi_isend_double(send_buffer, product3(send_size),
                            send_process_static, process_shift, comm,
                            &send_request);
    }

    double *current_rs_grid =
        &grid_rs[(border_width[2] + proc2local[recv_process][2][0]) *
                     npts_rs[0] * npts_rs[1] +
                 (border_width[1] + proc2local[recv_process][1][0]) *
                     npts_rs[0] +
                 border_width[0] + proc2local[recv_process][0][0]];

    // Unpack recv
    grid_mpi_wait(&recv_request);
#pragma omp parallel for collapse(2) default(none)                             \
    shared(recv_size, current_rs_grid, npts_rs, recv_buffer)
    for (int iz = 0; iz < recv_size[2]; iz++) {
      for (int iy = 0; iy < recv_size[1]; iy++) {
        memcpy(
            &current_rs_grid[iz * npts_rs[0] * npts_rs[1] + iy * npts_rs[0]],
            &recv_buffer[iz * recv_size[0] * recv_size[1] + iy * recv_size[0]],
            recv_size[0] * sizeof(double));
      }
    }

    grid_mpi_wait(&send_request);

    // Swap buffers
    double *temp_pointer = send_buffer;
    send_buffer = recv_buffer;
    recv_buffer = temp_pointer;
  }

  // Deal with bounds
  if (border_width[0] != 0 || border_width[1] != 0 || border_width[2] != 0) {
    int shifts[3];
    for (int dir = 0; dir < 3; dir++)
      shifts[dir] = npts_rs[dir] - 2 * border_width[dir];
#pragma omp parallel for default(none)                                         \
    shared(grid_rs, grid_pw, npts_rs, border_width, shifts)
    for (int iz = 0; iz < npts_rs[2]; iz++) {
      int iz_orig = iz;
      if (iz < border_width[2])
        iz_orig += shifts[2];
      if (iz >= npts_rs[2] - border_width[2])
        iz_orig -= shifts[2];
      for (int iy = 0; iy < npts_rs[1]; iy++) {
        int iy_orig = iy;
        if (iy < border_width[1])
          iy_orig += shifts[1];
        if (iy >= npts_rs[1] - border_width[1])
          iy_orig -= shifts[1];
        for (int ix = 0; ix < npts_rs[0]; ix++) {
          int ix_orig = ix;
          if (ix < border_width[0])
            ix_orig += shifts[0];
          if (ix >= npts_rs[0] - border_width[0])
            ix_orig -= shifts[0];
          grid_rs[iz * npts_rs[0] * npts_rs[1] + iy * npts_rs[0] + ix] =
              grid_rs[iz_orig * npts_rs[0] * npts_rs[1] + iy_orig * npts_rs[0] +
                      ix_orig];
        }
      }
    }
  }
  free(send_buffer);
  free(recv_buffer);
}

void redistribute_grids(
    const double *grid_in, double *grid_out, const grid_mpi_comm comm_in,
    const grid_mpi_comm comm_out, const int npts_global[3],
    const int proc2local_in[grid_mpi_comm_size(comm_in)][3][2],
    const int proc2local_out[grid_mpi_comm_size(comm_out)][3][2]) {
  const int number_of_processes = grid_mpi_comm_size(comm_out);
  const int my_process_out = grid_mpi_comm_rank(comm_out);
  const int my_process_in = grid_mpi_comm_rank(comm_in);

  assert(grid_out != NULL);
  assert(grid_in != NULL);
  assert(!grid_mpi_comm_is_unequal(comm_in, comm_out));
  for (int process = 0; process < number_of_processes; process++) {
    for (int dir = 0; dir < 3; dir++) {
      assert(proc2local_out[process][dir][0] >= 0 &&
             "The bounds of the output grid cannot be lower than zero!");
      assert(proc2local_out[process][dir][1] < npts_global[dir] &&
             "The bounds of the output grid contains too many points!");
      assert(
          proc2local_out[process][dir][1] >=
              proc2local_out[process][dir][0] - 1 &&
          "The number of points on the output grid on one processor cannot be "
          "negative!");
      assert(proc2local_in[process][dir][0] >= 0 &&
             "The input grid is only allowed to have nonnegative indices!");
      assert(proc2local_in[process][dir][1] < npts_global[dir] &&
             "The input grid cannot have points outside of the inner RS grid!");
      assert(
          proc2local_in[process][dir][1] >=
              proc2local_in[process][dir][0] - 1 &&
          "The number of points on the input grid on one processor cannot be "
          "negative!");
    }
  }
  for (int dir = 0; dir < 3; dir++) {
    assert(npts_global[dir] >= 0 &&
           "Global number of points cannot be negative!");
  }

  // Prepare the intermediate buffer
  int my_bounds_out[3][2];
  int my_bounds_in[3][2];
  int my_sizes_out[3];
  int my_sizes_in[3];
  for (int dir = 0; dir < 3; dir++) {
    my_bounds_out[dir][0] = proc2local_out[my_process_out][dir][0];
    my_bounds_out[dir][1] = proc2local_out[my_process_out][dir][1];
    my_sizes_out[dir] = proc2local_out[my_process_out][dir][1] -
                        proc2local_out[my_process_out][dir][0] + 1;
    my_bounds_in[dir][0] = proc2local_in[my_process_in][dir][0];
    my_bounds_in[dir][1] = proc2local_in[my_process_in][dir][1];
    my_sizes_in[dir] = proc2local_in[my_process_in][dir][1] -
                       proc2local_in[my_process_in][dir][0] + 1;
  }
  const int my_number_of_elements_out = product3(my_sizes_out);

  int received_elements = 0;

  // Step A: Collect the inner local block
  int *map_in2out = malloc(number_of_processes * sizeof(int));
  grid_mpi_allgather_int(&my_process_out, 1, map_in2out, comm_in);

  // Determine from which processes we will receive data
  int number_of_elements_to_recv = 0;
  int number_of_processes_to_recv_from = 0;
  for (int process_shift = 1; process_shift < number_of_processes;
       process_shift++) {
    const int recv_process =
        modulo(my_process_in - process_shift, number_of_processes);
    int recv_size[3];
    for (int dir = 0; dir < 3; dir++)
      recv_size[dir] =
          imin(proc2local_in[recv_process][dir][1], my_bounds_out[dir][1]) -
          imax(proc2local_in[recv_process][dir][0], my_bounds_out[dir][0]) + 1;

    if (recv_size[0] <= 0 || recv_size[1] <= 0 || recv_size[2] <= 0)
      continue;
    number_of_elements_to_recv += product3(recv_size);
    number_of_processes_to_recv_from++;
  }

  // Setup arrays for recv data
  double *recv_buffer = calloc(number_of_elements_to_recv, sizeof(double));
  double **recv_buffers =
      calloc(number_of_processes_to_recv_from, sizeof(double *));
  grid_mpi_request *recv_requests =
      calloc(number_of_processes_to_recv_from, sizeof(grid_mpi_request));
  int *processes_to_recv_from =
      calloc(number_of_processes_to_recv_from, sizeof(int));

  // Initiate the receive operations
  int recv_offset = 0;
  int recv_counter = 0;
  for (int process_shift = 1; process_shift < number_of_processes;
       process_shift++) {
    const int recv_process =
        modulo(my_process_in - process_shift, number_of_processes);
    int recv_size[3];
    for (int dir = 0; dir < 3; dir++) {
      recv_size[dir] =
          imin(proc2local_in[recv_process][dir][1], my_bounds_out[dir][1]) -
          imax(proc2local_in[recv_process][dir][0], my_bounds_out[dir][0]) + 1;
    }

    if (recv_size[0] <= 0 || recv_size[1] <= 0 || recv_size[2] <= 0)
      continue;
    const int current_number_of_elements_to_recv = product3(recv_size);
    recv_buffers[recv_counter] = recv_buffer + recv_offset;
    processes_to_recv_from[recv_counter] = recv_process;

    grid_mpi_irecv_double(recv_buffers[recv_counter],
                          current_number_of_elements_to_recv, recv_process, 1,
                          comm_in, &recv_requests[recv_counter]);

    recv_offset += current_number_of_elements_to_recv;
    recv_counter++;
  }
  assert(recv_counter == number_of_processes_to_recv_from);
  assert(recv_offset == number_of_elements_to_recv);

  int number_of_elements_to_send = 0;
  int number_of_processes_to_send_to = 0;
  for (int process_shift = 1; process_shift < number_of_processes;
       process_shift++) {
    const int send_process =
        modulo(my_process_in + process_shift, number_of_processes);
    const int send_process_out = map_in2out[send_process];
    int send_size[3];
    for (int dir = 0; dir < 3; dir++)
      send_size[dir] =
          imin(my_bounds_in[dir][1], proc2local_out[send_process_out][dir][1]) -
          imax(my_bounds_in[dir][0], proc2local_out[send_process_out][dir][0]) +
          1;
    if (send_size[0] <= 0 || send_size[1] <= 0 || send_size[2] <= 0)
      continue;
    const int current_number_of_elements_to_send = product3(send_size);
    number_of_elements_to_send += current_number_of_elements_to_send;
    number_of_processes_to_send_to++;
  }

  double *send_buffer = calloc(number_of_elements_to_send, sizeof(double));
  double **send_buffers =
      calloc(number_of_processes_to_send_to, sizeof(double *));
  grid_mpi_request *send_requests =
      calloc(number_of_processes_to_send_to, sizeof(grid_mpi_request));

  int send_offset = 0;
  int send_counter = 0;
  for (int process_shift = 1; process_shift < number_of_processes;
       process_shift++) {
    const int send_process =
        modulo(my_process_in + process_shift, number_of_processes);
    const int send_process_out = map_in2out[send_process];
    int send_size[3];
    for (int dir = 0; dir < 3; dir++)
      send_size[dir] =
          imin(my_bounds_in[dir][1], proc2local_out[send_process_out][dir][1]) -
          imax(my_bounds_in[dir][0], proc2local_out[send_process_out][dir][0]) +
          1;
    if (send_size[0] <= 0 || send_size[1] <= 0 || send_size[2] <= 0)
      continue;
    const int current_number_of_elements_to_send = product3(send_size);
    send_buffers[send_counter] = send_buffer + send_offset;

    double *current_send_buffer = send_buffers[send_counter];
    const double *current_grid_in =
        &grid_in[imax(0, proc2local_out[send_process_out][2][0] -
                             my_bounds_in[2][0]) *
                     my_sizes_in[0] * my_sizes_in[1] +
                 imax(0, proc2local_out[send_process_out][1][0] -
                             my_bounds_in[1][0]) *
                     my_sizes_in[0] +
                 imax(0, proc2local_out[send_process_out][0][0] -
                             my_bounds_in[0][0])];

#pragma omp parallel for collapse(2) default(none)                             \
    shared(send_size, my_sizes_in, current_send_buffer, current_grid_in)
    for (int iz = 0; iz < send_size[2]; iz++) {
      for (int iy = 0; iy < send_size[1]; iy++) {
        memcpy(&current_send_buffer[iz * send_size[0] * send_size[1] +
                                    iy * send_size[0]],
               &current_grid_in[iz * my_sizes_in[0] * my_sizes_in[1] +
                                iy * my_sizes_in[0]],
               send_size[0] * sizeof(double));
      }
    }

    grid_mpi_isend_double(send_buffers[send_counter],
                          current_number_of_elements_to_send, send_process, 1,
                          comm_in, &send_requests[send_counter]);

    send_offset += current_number_of_elements_to_send;
    send_counter++;
  }
  assert(send_offset == number_of_elements_to_send);
  assert(send_counter == number_of_processes_to_send_to);

  // A2) Copy local data
  {
    int starts_out[3];
    for (int dir = 0; dir < 3; dir++)
      starts_out[dir] = imax(0, my_bounds_in[dir][0] - my_bounds_out[dir][0]);
    int ends_out[3];
    for (int dir = 0; dir < 3; dir++)
      ends_out[dir] = imin(my_sizes_out[dir] - 1,
                           my_bounds_in[dir][1] - my_bounds_out[dir][0]);
    int sizes_rs[3];
    for (int dir = 0; dir < 3; dir++)
      sizes_rs[dir] = imax(0, ends_out[dir] - starts_out[dir] + 1);
    double *current_rs_grid =
        &grid_out[starts_out[2] * my_sizes_out[0] * my_sizes_out[1] +
                  starts_out[1] * my_sizes_out[0] + starts_out[0]];
    const double *current_grid_in =
        &grid_in[(starts_out[2] + my_bounds_out[2][0] - my_bounds_in[2][0]) *
                     my_sizes_in[0] * my_sizes_in[1] +
                 (starts_out[1] + my_bounds_out[1][0] - my_bounds_in[1][0]) *
                     my_sizes_in[0] +
                 starts_out[0] + my_bounds_out[0][0] - my_bounds_in[0][0]];
#pragma omp parallel for collapse(2) default(none) shared(                     \
        sizes_rs, my_sizes_out, my_sizes_in, current_rs_grid, current_grid_in)
    for (int iz = 0; iz < sizes_rs[2]; iz++) {
      for (int iy = 0; iy < sizes_rs[1]; iy++) {
        memcpy(&current_rs_grid[iz * my_sizes_out[0] * my_sizes_out[1] +
                                iy * my_sizes_out[0]],
               &current_grid_in[iz * my_sizes_in[0] * my_sizes_in[1] +
                                iy * my_sizes_in[0]],
               sizes_rs[0] * sizeof(double));
      }
    }
    received_elements += product3(sizes_rs);
  }

  // A2) Send around local data of the input grid and copy it to our local
  // buffer
  for (int process_shift = 0; process_shift < number_of_processes_to_recv_from;
       process_shift++) {
    int recv_counter;
    grid_mpi_waitany(number_of_processes_to_recv_from, recv_requests,
                     &recv_counter);
    const int recv_process = processes_to_recv_from[recv_counter];

    int starts_out[3];
    for (int dir = 0; dir < 3; dir++)
      starts_out[dir] =
          imax(0, proc2local_in[recv_process][dir][0] - my_bounds_out[dir][0]);
    int ends_out[3];
    for (int dir = 0; dir < 3; dir++)
      ends_out[dir] =
          imin(my_sizes_out[dir] - 1,
               proc2local_in[recv_process][dir][1] - my_bounds_out[dir][0]);
    int recv_size[3];
    for (int dir = 0; dir < 3; dir++)
      recv_size[dir] = imax(0, ends_out[dir] - starts_out[dir] + 1);
    double *current_grid_out =
        &grid_out[starts_out[2] * my_sizes_out[0] * my_sizes_out[1] +
                  starts_out[1] * my_sizes_out[0] + starts_out[0]];
    const double *current_recv_buffer = recv_buffers[recv_counter];

#pragma omp parallel for collapse(2) default(none)                             \
    shared(my_sizes_out, recv_size, current_grid_out, current_recv_buffer)
    for (int iz = 0; iz < recv_size[2]; iz++) {
      for (int iy = 0; iy < recv_size[1]; iy++) {
        memcpy(&current_grid_out[iz * my_sizes_out[0] * my_sizes_out[1] +
                                 iy * my_sizes_out[0]],
               &current_recv_buffer[iz * recv_size[0] * recv_size[1] +
                                    iy * recv_size[0]],
               recv_size[0] * sizeof(double));
      }
    }
    received_elements += product3(recv_size);
  }

  grid_mpi_waitall(number_of_processes_to_send_to, send_requests);

  // Cleanup
  free(recv_buffer);
  free(recv_buffers);
  free(recv_requests);
  free(processes_to_recv_from);
  free(send_buffer);
  free(send_buffers);
  free(send_requests);
  free(map_in2out);

  assert(received_elements == my_number_of_elements_out &&
         "Not elements of the inner part of the RS grid were received");
}

void distribute_data_to_boundaries(
    double *grid_rs, const double *grid_rs_inner, const grid_mpi_comm comm_rs,
    const int npts_global[3],
    const int proc2local_rs[grid_mpi_comm_size(comm_rs)][3][2],
    const int border_width[3], const grid_redistribute *redistribute_rs) {
  const int number_of_processes = grid_mpi_comm_size(comm_rs);
  const int my_process_rs = grid_mpi_comm_rank(comm_rs);

  assert(grid_rs != NULL);
  assert(grid_rs_inner != NULL);
  assert(redistribute_rs != NULL);
  for (int process = 0; process < number_of_processes; process++) {
    for (int dir = 0; dir < 3; dir++) {
      assert(proc2local_rs[process][dir][0] >= -border_width[dir] &&
             "The inner part of the RS grid cannot be lower than zero!");
      assert(proc2local_rs[process][dir][1] <
                 npts_global[dir] + border_width[dir] &&
             "The inner part of the RS grid contains too many points!");
      assert(proc2local_rs[process][dir][1] - proc2local_rs[process][dir][0] +
                     1 >=
                 0 &&
             "The number of points on the RS grid on one processor cannot be "
             "negative!");
    }
  }
  for (int dir = 0; dir < 3; dir++) {
    assert(border_width[dir] >= 0 &&
           "The number of points on the boundary cannot be negative!");
    assert(npts_global[dir] >= 0 &&
           "Global number of points cannot be negative!");
  }

  // Prepare the intermediate buffer
  int my_sizes_rs[3];
  int my_sizes_rs_inner[3];
  for (int dir = 0; dir < 3; dir++) {
    my_sizes_rs[dir] = proc2local_rs[my_process_rs][dir][1] -
                       proc2local_rs[my_process_rs][dir][0] + 1;
    my_sizes_rs_inner[dir] = proc2local_rs[my_process_rs][dir][1] -
                             proc2local_rs[my_process_rs][dir][0] + 1 -
                             2 * border_width[dir];
  }
  const int my_number_of_elements_rs = product3(my_sizes_rs);
  const int my_number_of_inner_elements_rs = product3(my_sizes_rs_inner);

  int size_of_input_buffer = redistribute_rs->local_ranges[0][0][2] *
                             redistribute_rs->local_ranges[0][1][2] *
                             redistribute_rs->local_ranges[0][2][2];
  int size_of_output_buffer = redistribute_rs->local_ranges[0][0][2] *
                              redistribute_rs->local_ranges[0][1][2] *
                              redistribute_rs->local_ranges[0][2][2];

  // We send direction wise to cluster communication processes
  double *input_data = malloc(size_of_input_buffer * sizeof(double));
  double *output_data = malloc(size_of_output_buffer * sizeof(double));

  // We start with the own data
  memcpy(input_data, grid_rs_inner,
         my_number_of_inner_elements_rs * sizeof(double));

  double *memory_pool = calloc(redistribute_rs->size_of_buffer_to_halo +
                                   redistribute_rs->size_of_buffer_to_inner,
                               sizeof(double));
  double **recv_buffer = calloc(
      redistribute_rs->max_number_of_processes_to_inner, sizeof(double *));
  double **send_buffer = calloc(
      redistribute_rs->max_number_of_processes_to_halo, sizeof(double *));

  grid_mpi_request *recv_requests =
      malloc(redistribute_rs->max_number_of_processes_to_inner *
             sizeof(grid_mpi_request));
  grid_mpi_request *send_requests =
      malloc(redistribute_rs->max_number_of_processes_to_halo *
             sizeof(grid_mpi_request));

  // We go the reverse direction because the normal order of the predetermined
  // arrays assume the other direction
  for (int dir = 2; dir >= 0; dir--) {
    // Without border, there is nothing to exchange
    if (border_width[dir] == 0)
      continue;

    // We need (2-dir) because have the precalculated the arrays for the
    // direction RS2PW
    const int(*input_ranges)[3] = redistribute_rs->local_ranges[dir + 1];
    const int(*output_ranges)[3] = redistribute_rs->local_ranges[dir];

    const int number_of_output_elements =
        output_ranges[0][2] * output_ranges[1][2] * output_ranges[2][2];
    memset(output_data, 0, number_of_output_elements * sizeof(double));

    for (int process_index = 0;
         process_index < redistribute_rs->number_of_processes_to_inner[dir];
         process_index++) {
      recv_buffer[process_index] =
          memory_pool +
          redistribute_rs
              ->buffer_offsets_to_inner[redistribute_rs->offset_to_inner[dir] +
                                        process_index];
      recv_requests[process_index] = grid_mpi_request_null;
    }

    for (int process_index = 0;
         process_index < redistribute_rs->number_of_processes_to_halo[dir];
         process_index++) {
      send_buffer[process_index] =
          memory_pool + redistribute_rs->size_of_buffer_to_inner +
          redistribute_rs
              ->buffer_offsets_to_halo[redistribute_rs->offset_to_halo[dir] +
                                       process_index];
      send_requests[process_index] = grid_mpi_request_null;
    }

    // A2) Post receive requests
    for (int process_index = 0;
         process_index < redistribute_rs->number_of_processes_to_inner[dir];
         process_index++) {
      const int recv_process =
          redistribute_rs
              ->processes_to_inner[redistribute_rs->offset_to_inner[dir] +
                                   process_index];

      int number_of_elements_to_receive = 1;
      for (int dir2 = 0; dir2 < 3; dir2++) {
        number_of_elements_to_receive *=
            (dir2 == dir
                 ? redistribute_rs
                       ->sizes_to_inner[redistribute_rs->offset_to_inner[dir] +
                                        process_index]
                 : input_ranges[dir2][2]);
      }

      grid_mpi_irecv_double(recv_buffer[process_index],
                            number_of_elements_to_receive, recv_process, 1,
                            comm_rs, &recv_requests[process_index]);
    }

    // A2) Post send requests
    for (int process_index = 0;
         process_index < redistribute_rs->number_of_processes_to_halo[dir];
         process_index++) {
      const int send_process =
          redistribute_rs
              ->processes_to_halo[redistribute_rs->offset_to_halo[dir] +
                                  process_index];

      int send_sizes[3];
      for (int dir2 = 0; dir2 < 3; dir2++) {
        send_sizes[dir2] =
            (dir2 == dir
                 ? redistribute_rs
                       ->sizes_to_halo[redistribute_rs->offset_to_halo[dir] +
                                       process_index]
                 : input_ranges[dir2][2]);
      }
      const int number_of_elements_to_send = product3(send_sizes);
      const int *const send2local =
          (const int *const)redistribute_rs
              ->index2local_to_halo[redistribute_rs->offset_to_halo[dir] +
                                    process_index];
      double *current_send_buffer = send_buffer[process_index];
#pragma omp parallel for default(none)                                         \
    shared(send_sizes, send2local, input_ranges, output_ranges, input_data,    \
               current_send_buffer, process_index, comm_rs, dir)
      for (int iz_send = 0; iz_send < send_sizes[2]; iz_send++) {
        const int iz_local = (2 == dir ? send2local[iz_send] : iz_send);
        for (int iy_send = 0; iy_send < send_sizes[1]; iy_send++) {
          const int iy_local = (1 == dir ? send2local[iy_send] : iy_send);
          for (int ix_send = 0; ix_send < send_sizes[0]; ix_send++) {
            const int ix_local = (0 == dir ? send2local[ix_send] : ix_send);
            current_send_buffer[iz_send * send_sizes[0] * send_sizes[1] +
                                iy_send * send_sizes[0] + ix_send] =
                input_data[iz_local * input_ranges[0][2] * input_ranges[1][2] +
                           iy_local * input_ranges[0][2] + ix_local];
          }
        }
      }
      grid_mpi_isend_double(send_buffer[process_index],
                            number_of_elements_to_send, send_process, 1,
                            comm_rs, &send_requests[process_index]);
    }

    // Update the local data
    {
      // Do not forget the boundary outside of the main bound
      for (int iz = 0; iz < input_ranges[2][2]; iz++) {
        const int iz_orig =
            (dir == 2 ? modulo(iz + input_ranges[2][0], npts_global[2]) -
                            output_ranges[2][0]
                      : iz);
        if (iz_orig < 0 || iz_orig >= output_ranges[2][2])
          continue;
        for (int iy = 0; iy < input_ranges[1][2]; iy++) {
          const int iy_orig =
              (dir == 1 ? modulo(iy + input_ranges[1][0], npts_global[1]) -
                              output_ranges[1][0]
                        : iy);
          if (iy_orig < 0 || iy_orig >= output_ranges[1][2])
            continue;
          for (int ix = 0; ix < input_ranges[0][2]; ix++) {
            const int ix_orig =
                (dir == 0 ? modulo(ix + input_ranges[0][0], npts_global[0]) -
                                output_ranges[0][0]
                          : ix);
            if (ix_orig < 0 || ix_orig >= output_ranges[0][2])
              continue;
            output_data[iz_orig * output_ranges[0][2] * output_ranges[1][2] +
                        iy_orig * output_ranges[0][2] + ix_orig] +=
                input_data[iz * input_ranges[0][2] * input_ranges[1][2] +
                           iy * input_ranges[0][2] + ix];
          }
        }
      }
    }

    // A2) Wait for receive processes and add to local data
    for (int process_index = 0;
         process_index < redistribute_rs->number_of_processes_to_inner[dir];
         process_index++) {
      // Do not forget the boundary outside of the main bound
      int process = -1;
      grid_mpi_waitany(redistribute_rs->number_of_processes_to_inner[dir],
                       recv_requests, &process);
      int recv_sizes[3];
      for (int dir2 = 0; dir2 < 3; dir2++) {
        recv_sizes[dir2] =
            (dir2 == dir
                 ? redistribute_rs
                       ->sizes_to_inner[redistribute_rs->offset_to_inner[dir] +
                                        process]
                 : output_ranges[dir2][2]);
      }
      const int *const recv2local =
          (const int *const)redistribute_rs
              ->index2local_to_inner[redistribute_rs->offset_to_inner[dir] +
                                     process];
      for (int iz_recv = 0; iz_recv < recv_sizes[2]; iz_recv++) {
        const int iz_local = (2 == dir ? recv2local[iz_recv] : iz_recv);
        for (int iy_recv = 0; iy_recv < recv_sizes[1]; iy_recv++) {
          const int iy_local = (1 == dir ? recv2local[iy_recv] : iy_recv);
          for (int ix_recv = 0; ix_recv < recv_sizes[0]; ix_recv++) {
            const int ix_local = (0 == dir ? recv2local[ix_recv] : ix_recv);
            output_data[iz_local * output_ranges[0][2] * output_ranges[1][2] +
                        iy_local * output_ranges[0][2] + ix_local] +=
                recv_buffer[process][iz_recv * recv_sizes[0] * recv_sizes[1] +
                                     iy_recv * recv_sizes[0] + ix_recv];
          }
        }
      }
    }

    // A2) Wait for the send processes to finish
    grid_mpi_waitall(redistribute_rs->number_of_processes_to_halo[dir],
                     send_requests);
    // Swap pointers
    double *tmp = input_data;
    input_data = output_data;
    output_data = tmp;
  }

  memcpy(grid_rs, input_data, my_number_of_elements_rs * sizeof(double));

  free(recv_buffer);
  free(send_buffer);
  free(recv_requests);
  free(send_requests);
  free(memory_pool);
  free(input_data);
  free(output_data);
}

void grid_copy_to_multigrid_distributed(
    double *grid_rs, const double *grid_pw, const grid_mpi_comm comm_pw,
    const grid_mpi_comm comm_rs, const int npts_global[3],
    const int proc2local_rs[grid_mpi_comm_size(comm_rs)][3][2],
    const int proc2local_pw[grid_mpi_comm_size(comm_pw)][3][2],
    const int border_width[3], const grid_redistribute *redistribute_rs) {
  const int number_of_processes = grid_mpi_comm_size(comm_rs);
  const int my_process_rs = grid_mpi_comm_rank(comm_rs);

  assert(grid_rs != NULL);
  assert(grid_pw != NULL);
  assert(redistribute_rs != NULL);
  assert(!grid_mpi_comm_is_unequal(comm_pw, comm_rs));
  for (int process = 0; process < number_of_processes; process++) {
    for (int dir = 0; dir < 3; dir++) {
      assert(proc2local_rs[process][dir][0] >= -border_width[dir] &&
             "The inner part of the RS grid cannot be lower than zero!");
      assert(proc2local_rs[process][dir][1] <
                 npts_global[dir] + border_width[dir] &&
             "The inner part of the RS grid contains too many points!");
      assert(proc2local_rs[process][dir][1] - proc2local_rs[process][dir][0] +
                     1 >=
                 0 &&
             "The number of points on the RS grid on one processor cannot be "
             "negative!");
      assert(proc2local_pw[process][dir][0] >= 0 &&
             "The PW grid is only allowed to have nonnegative indices!");
      assert(proc2local_pw[process][dir][1] < npts_global[dir] &&
             "The PW grid cannot have points outside of the inner RS grid!");
      assert(proc2local_pw[process][dir][1] - proc2local_pw[process][dir][0] +
                     1 >=
                 0 &&
             "The number of points on the PW grid on one processor cannot be "
             "negative!");
    }
  }
  for (int dir = 0; dir < 3; dir++) {
    assert(border_width[dir] >= 0 &&
           "The number of points on the boundary cannot be negative!");
    assert(npts_global[dir] >= 0 &&
           "Global number of points cannot be negative!");
  }

  // Prepare the intermediate buffer
  int my_sizes_rs_inner[3];
  for (int dir = 0; dir < 3; dir++) {
    my_sizes_rs_inner[dir] = proc2local_rs[my_process_rs][dir][1] -
                             proc2local_rs[my_process_rs][dir][0] + 1 -
                             2 * border_width[dir];
  }
  const int my_number_of_inner_elements_rs = product3(my_sizes_rs_inner);

  double *grid_rs_inner =
      calloc(my_number_of_inner_elements_rs, sizeof(double));

  // Step A: Collect the inner local block
  int proc2local_rs_inner[number_of_processes][3][2];
  for (int process = 0; process < number_of_processes; process++) {
    for (int dir = 0; dir < 3; dir++) {
      proc2local_rs_inner[process][dir][0] =
          proc2local_rs[process][dir][0] + border_width[dir];
      proc2local_rs_inner[process][dir][1] =
          proc2local_rs[process][dir][1] - border_width[dir];
    }
  }
  redistribute_grids(grid_pw, grid_rs_inner, comm_pw, comm_rs, npts_global,
                     proc2local_pw, proc2local_rs_inner);

  // B) Distribute inner local block the everyone
  distribute_data_to_boundaries(grid_rs, grid_rs_inner, comm_rs, npts_global,
                                proc2local_rs, border_width, redistribute_rs);

  free(grid_rs_inner);
}

void grid_copy_to_multigrid_general(
    const grid_multigrid *multigrid, const double *grids[multigrid->nlevels],
    const grid_mpi_comm comm[multigrid->nlevels], const int *proc2local) {
  for (int level = 0; level < multigrid->nlevels; level++) {
    assert(!grid_mpi_comm_is_unequal(multigrid->comm, comm[level]));
    if (grid_mpi_comm_size(comm[level]) == 1) {
      grid_copy_to_multigrid_serial(multigrid->grids[level]->host_buffer,
                                    grids[level], multigrid->npts_local[level],
                                    multigrid->border_width[level]);
    } else {
      // The parallel case, we need to distinguish replicated grids from
      // distributed grids
      if (multigrid->pgrid_dims[level][0] * multigrid->pgrid_dims[level][1] *
              multigrid->pgrid_dims[level][2] ==
          1) {
        grid_copy_to_multigrid_replicated(
            multigrid->grids[level]->host_buffer, grids[level],
            multigrid->npts_local[level], multigrid->border_width[level],
            comm[level],
            (const int(*)[3][2]) &
                proc2local[level * grid_mpi_comm_size(comm[level]) * 6]);
      } else {
        grid_copy_to_multigrid_distributed(
            multigrid->grids[level]->host_buffer, grids[level], multigrid->comm,
            comm[level], multigrid->npts_global[level],
            (const int(*)[3][2]) &
                multigrid->proc2local[6 * level *
                                      grid_mpi_comm_size(multigrid->comm)],
            (const int(*)[3][2]) &
                proc2local[6 * level * grid_mpi_comm_size(comm[0])],
            multigrid->border_width[level], &multigrid->redistribute[level]);
      }
    }
  }
}

void grid_copy_to_multigrid_general_f(
    const grid_multigrid *multigrid, const double *grids[multigrid->nlevels],
    const grid_mpi_fint fortran_comm[multigrid->nlevels],
    const int *proc2local) {
  grid_mpi_comm comm[multigrid->nlevels];
  for (int level = 0; level < multigrid->nlevels; level++) {
    comm[level] = grid_mpi_comm_f2c(fortran_comm[level]);
  }

  grid_copy_to_multigrid_general(multigrid, grids, comm, proc2local);
}

void grid_copy_to_multigrid_general_single(const grid_multigrid *multigrid,
                                           const int level, const double *grid,
                                           const grid_mpi_comm comm,
                                           const int *proc2local) {
  assert(multigrid != NULL);
  assert(!grid_mpi_comm_is_unequal(multigrid->comm, comm));
  assert(grid != NULL);
  if (grid_mpi_comm_size(comm) == 1) {
    grid_copy_to_multigrid_serial(multigrid->grids[level]->host_buffer, grid,
                                  multigrid->npts_local[level],
                                  multigrid->border_width[level]);
  } else {
    // The parallel case, we need to distinguish replicated grids from
    // distributed grids
    if (multigrid->pgrid_dims[level][0] * multigrid->pgrid_dims[level][1] *
            multigrid->pgrid_dims[level][2] ==
        1) {
      grid_copy_to_multigrid_replicated(multigrid->grids[level]->host_buffer,
                                        grid, multigrid->npts_local[level],
                                        multigrid->border_width[level], comm,
                                        (const int(*)[3][2])proc2local);
    } else {
      grid_copy_to_multigrid_distributed(
          multigrid->grids[level]->host_buffer, grid, multigrid->comm, comm,
          multigrid->npts_global[level],
          (const int(*)[3][2]) &
              multigrid
                  ->proc2local[6 * level * grid_mpi_comm_size(multigrid->comm)],
          (const int(*)[3][2])proc2local, multigrid->border_width[level],
          &multigrid->redistribute[level]);
    }
  }
}

void grid_copy_to_multigrid_general_single_f(const grid_multigrid *multigrid,
                                             const int level,
                                             const double *grid,
                                             const grid_mpi_fint fortran_comm,
                                             const int *proc2local) {
  grid_copy_to_multigrid_general_single(
      multigrid, level - 1, grid, grid_mpi_comm_f2c(fortran_comm), proc2local);
}

void grid_copy_from_multigrid_serial(const double *grid_rs, double *grid_pw,
                                     const int npts_rs[3],
                                     const int border_width[3]) {
  if (border_width[0] == 0 && border_width[1] == 0 && border_width[2] == 0) {
    memcpy(grid_pw, grid_rs,
           npts_rs[0] * npts_rs[1] * npts_rs[2] * sizeof(double));
  } else {
    int npts_pw[3];
    for (int dir = 0; dir < 3; dir++)
      npts_pw[dir] = npts_rs[dir] - 2 * border_width[dir];
    for (int iz = border_width[2]; iz < npts_rs[2] - border_width[2]; iz++) {
      //
      int iz_pw = iz - border_width[2];
      for (int iy = border_width[1]; iy < npts_rs[1] - border_width[1]; iy++) {
        //
        int iy_pw = iy - border_width[1];
        for (int ix = border_width[0]; ix < npts_rs[0] - border_width[0];
             ix++) {
          //
          int ix_pw = ix - border_width[0];
          grid_pw[iz_pw * npts_pw[1] * npts_pw[2] + iy_pw * npts_pw[0] +
                  ix_pw] =
              grid_rs[iz * npts_rs[0] * npts_rs[1] + iy * npts_rs[0] + ix];
        }
      }
    }
  }
}

void grid_copy_from_multigrid_replicated(
    const double *grid_rs, double *grid_pw, const int npts_rs[3],
    const int border_width[3], const grid_mpi_comm comm,
    const int proc2local[grid_mpi_comm_size(comm)][3][2]) {

  const int number_of_processes = grid_mpi_comm_size(comm);
  const int my_process = grid_mpi_comm_rank(comm);

  // Determine the maximum number of grid points on a single rank
  int maximum_number_of_elements = 0;
  for (int process = 0; process < number_of_processes; process++) {
    const int current_number_of_elements =
        (proc2local[process][0][1] - proc2local[process][0][0] + 1) *
        (proc2local[process][1][1] - proc2local[process][1][0] + 1) *
        (proc2local[process][2][1] - proc2local[process][2][0] + 1);
    maximum_number_of_elements =
        imax(current_number_of_elements, maximum_number_of_elements);
  }
  const int my_number_of_elements =
      (proc2local[my_process][0][1] - proc2local[my_process][0][0] + 1) *
      (proc2local[my_process][1][1] - proc2local[my_process][1][0] + 1) *
      (proc2local[my_process][2][1] - proc2local[my_process][2][0] + 1);

  // Allocate communication buffers
  double *send_buffer = calloc(maximum_number_of_elements, sizeof(double));
  double *recv_buffer = calloc(maximum_number_of_elements, sizeof(double));

  // We actually send to the next neighbor
  const int process_to_send_to = modulo(my_process - 1, number_of_processes);
  const int process_to_recv_from = modulo(my_process + 1, number_of_processes);

  grid_mpi_request recv_request = grid_mpi_request_null;
  grid_mpi_request send_request = grid_mpi_request_null;

  int send_size[3], recv_size[3];

  // Send the data of the ip-th neighbor
  for (int process_shift = 1; process_shift <= number_of_processes;
       process_shift++) {
    // Load the send buffer for the next process to which the will be finally
    // sent (not necessarily the process to which we actually send)
    const int send_process =
        modulo(my_process + process_shift, number_of_processes);
    for (int dir = 0; dir < 3; dir++)
      send_size[dir] = proc2local[send_process][dir][1] -
                       proc2local[send_process][dir][0] + 1;

    // Wait until the sendbuffer (former recvbuffer) was sent (not required in
    // the very first iteration)
    if (process_shift > 1)
      grid_mpi_wait(&recv_request);

    const double *current_grid_rs =
        &grid_rs[(proc2local[send_process][2][0] + border_width[2]) *
                     npts_rs[0] * npts_rs[1] +
                 (proc2local[send_process][1][0] + border_width[1]) *
                     npts_rs[0] +
                 (proc2local[send_process][0][0] + border_width[0])];

    // Pack send_buffer
#pragma omp parallel for collapse(3) default(none)                             \
    shared(send_size, current_grid_rs, send_buffer, npts_rs)
    for (int iz = 0; iz < send_size[2]; iz++) {
      for (int iy = 0; iy < send_size[1]; iy++) {
        for (int ix = 0; ix < send_size[0]; ix++) {
          send_buffer[iz * send_size[0] * send_size[1] + iy * send_size[0] +
                      ix] += current_grid_rs[iz * npts_rs[0] * npts_rs[1] +
                                             iy * npts_rs[0] + ix];
        }
      }
    }

    if (process_shift == number_of_processes)
      break;

    // Load the recv buffer for the process to which the next data is sent
    const int recv_process =
        modulo(my_process + process_shift + 1, number_of_processes);
    for (int dir = 0; dir < 3; dir++)
      recv_size[dir] = proc2local[recv_process][dir][1] -
                       proc2local[recv_process][dir][0] + 1;

    // Communicate buffers
    if (process_shift > 1)
      grid_mpi_wait(&send_request);
    grid_mpi_irecv_double(recv_buffer, product3(recv_size),
                          process_to_recv_from, process_shift, comm,
                          &recv_request);
    grid_mpi_isend_double(send_buffer, product3(send_size), process_to_send_to,
                          process_shift, comm, &send_request);
    double *temp_pointer = send_buffer;
    send_buffer = recv_buffer;
    recv_buffer = temp_pointer;
  }

  grid_mpi_wait(&send_request);

  // Copy the final received data for yourself to the result
  memcpy(grid_pw, send_buffer, my_number_of_elements * sizeof(double));

  free(send_buffer);
  free(recv_buffer);
}

void collect_boundary_to_inner(
    const double *grid_rs, double *grid_rs_inner, const grid_mpi_comm comm_rs,
    const int npts_global[3],
    const int proc2local_rs[grid_mpi_comm_size(comm_rs)][3][2],
    const int border_width[3], const grid_redistribute *redistribute_rs) {
  const int number_of_processes = grid_mpi_comm_size(comm_rs);
  const int my_process_rs = grid_mpi_comm_rank(comm_rs);

  assert(grid_rs != NULL);
  for (int process = 0; process < number_of_processes; process++) {
    for (int dir = 0; dir < 3; dir++) {
      assert(proc2local_rs[process][dir][0] >= -border_width[dir] &&
             "The inner part of the RS grid cannot be lower than zero!");
      assert(proc2local_rs[process][dir][1] <
                 npts_global[dir] + border_width[dir] &&
             "The inner part of the RS grid contains too many points!");
      assert(proc2local_rs[process][dir][1] - proc2local_rs[process][dir][0] +
                     1 >=
                 0 &&
             "The number of points on the RS grid on one processor cannot be "
             "negative!");
    }
  }
  for (int dir = 0; dir < 3; dir++) {
    assert(border_width[dir] >= 0 &&
           "The number of points on the boundary cannot be negative!");
    assert(npts_global[dir] >= 0 &&
           "Global number of points cannot be negative!");
  }

  // Prepare the intermediate buffer
  int my_bounds_rs_inner[3][2];
  int my_sizes_rs_inner[3];
  for (int dir = 0; dir < 3; dir++) {
    my_bounds_rs_inner[dir][0] =
        proc2local_rs[my_process_rs][dir][0] + border_width[dir];
    my_bounds_rs_inner[dir][1] =
        proc2local_rs[my_process_rs][dir][1] - border_width[dir];
    my_sizes_rs_inner[dir] =
        my_bounds_rs_inner[dir][1] - my_bounds_rs_inner[dir][0] + 1;
  }
  const int my_number_of_inner_elements_rs = product3(my_sizes_rs_inner);

  int size_of_input_buffer = redistribute_rs->local_ranges[0][0][2] *
                             redistribute_rs->local_ranges[0][1][2] *
                             redistribute_rs->local_ranges[0][2][2];
  // int size_of_output_buffer =
  // redistribute_rs->local_ranges[1][0][2]*redistribute_rs->local_ranges[1][1][2]*redistribute_rs->local_ranges[1][2][2];

  // We send direction wise to cluster communication processes
  double *input_data = malloc(size_of_input_buffer * sizeof(double));
  double *output_data = malloc(size_of_input_buffer * sizeof(double));

  // We start with the own data
  memcpy(input_data, grid_rs, size_of_input_buffer * sizeof(double));

  double *memory_pool = calloc(redistribute_rs->size_of_buffer_to_halo +
                                   redistribute_rs->size_of_buffer_to_inner,
                               sizeof(double));
  double **recv_buffer = calloc(
      redistribute_rs->max_number_of_processes_to_halo, sizeof(double *));
  double **send_buffer = calloc(
      redistribute_rs->max_number_of_processes_to_inner, sizeof(double *));
  grid_mpi_request *recv_requests =
      malloc(redistribute_rs->max_number_of_processes_to_halo *
             sizeof(grid_mpi_request));
  grid_mpi_request *send_requests =
      malloc(redistribute_rs->max_number_of_processes_to_inner *
             sizeof(grid_mpi_request));

  for (int dir = 0; dir < 3; dir++) {
    // Without border, there is nothing to exchange
    if (border_width[dir] == 0)
      continue;

    const int(*input_ranges)[3] = redistribute_rs->local_ranges[dir];
    const int(*output_ranges)[3] = redistribute_rs->local_ranges[dir + 1];

    const int number_of_output_elements =
        output_ranges[0][2] * output_ranges[1][2] * output_ranges[2][2];
    memset(output_data, 0, number_of_output_elements * sizeof(double));

    for (int process_index = 0;
         process_index < redistribute_rs->number_of_processes_to_inner[dir];
         process_index++) {
      send_buffer[process_index] =
          memory_pool + redistribute_rs->size_of_buffer_to_halo +
          redistribute_rs
              ->buffer_offsets_to_halo[redistribute_rs->offset_to_halo[dir] +
                                       process_index];
      send_requests[process_index] = grid_mpi_request_null;
    }

    for (int process_index = 0;
         process_index < redistribute_rs->number_of_processes_to_halo[dir];
         process_index++) {
      recv_buffer[process_index] =
          memory_pool +
          redistribute_rs
              ->buffer_offsets_to_halo[redistribute_rs->offset_to_halo[dir] +
                                       process_index];
      recv_requests[process_index] = grid_mpi_request_null;
    }

    // A2) Post receive requests
    for (int process_index = 0;
         process_index < redistribute_rs->number_of_processes_to_halo[dir];
         process_index++) {
      const int recv_process =
          redistribute_rs
              ->processes_to_halo[redistribute_rs->offset_to_halo[dir] +
                                  process_index];

      int number_of_elements_to_receive = 1;
      for (int dir2 = 0; dir2 < 3; dir2++) {
        number_of_elements_to_receive *=
            (dir2 == dir
                 ? redistribute_rs
                       ->sizes_to_halo[redistribute_rs->offset_to_halo[dir] +
                                       process_index]
                 : input_ranges[dir2][2]);
      }

      grid_mpi_irecv_double(recv_buffer[process_index],
                            number_of_elements_to_receive, recv_process, 1,
                            comm_rs, &recv_requests[process_index]);
    }

    // A2) Post send reequests
    for (int process_index = 0;
         process_index < redistribute_rs->number_of_processes_to_inner[dir];
         process_index++) {
      const int send_process =
          redistribute_rs
              ->processes_to_inner[redistribute_rs->offset_to_inner[dir] +
                                   process_index];

      int send_sizes[3];
      for (int dir2 = 0; dir2 < 3; dir2++) {
        send_sizes[dir2] =
            (dir2 == dir
                 ? redistribute_rs
                       ->sizes_to_inner[redistribute_rs->offset_to_inner[dir] +
                                        process_index]
                 : input_ranges[dir2][2]);
      }
      const int number_of_elements_to_send = product3(send_sizes);
      const int *const send2local =
          (const int *const)redistribute_rs
              ->index2local_to_inner[redistribute_rs->offset_to_inner[dir] +
                                     process_index];
      for (int iz_send = 0; iz_send < send_sizes[2]; iz_send++) {
        const int iz_local = (2 == dir ? send2local[iz_send] : iz_send);
        for (int iy_send = 0; iy_send < send_sizes[1]; iy_send++) {
          const int iy_local = (1 == dir ? send2local[iy_send] : iy_send);
          for (int ix_send = 0; ix_send < send_sizes[0]; ix_send++) {
            const int ix_local = (0 == dir ? send2local[ix_send] : ix_send);
            send_buffer[process_index][iz_send * send_sizes[0] * send_sizes[1] +
                                       iy_send * send_sizes[0] + ix_send] =
                input_data[iz_local * input_ranges[0][2] * input_ranges[1][2] +
                           iy_local * input_ranges[0][2] + ix_local];
          }
        }
      }
      grid_mpi_isend_double(send_buffer[process_index],
                            number_of_elements_to_send, send_process, 1,
                            comm_rs, &send_requests[process_index]);
    }

    // Update the local data
    if (dir == 0) {
      // Do not forget the boundary outside of the main bound
#pragma omp parallel for default(none) collapse(2)                             \
    shared(input_ranges, output_ranges, npts_global, output_data, input_data)
      for (int iz = 0; iz < imin(input_ranges[2][2], output_ranges[2][2]);
           iz++) {
        for (int iy = 0; iy < imin(input_ranges[1][2], output_ranges[1][2]);
             iy++) {
          for (int ix = 0; ix < input_ranges[0][2]; ix++) {
            const int ix_orig =
                modulo(ix + input_ranges[0][0], npts_global[0]) -
                output_ranges[0][0];
            if (ix_orig < 0 || ix_orig >= output_ranges[0][2])
              continue;
            output_data[iz * output_ranges[0][2] * output_ranges[1][2] +
                        iy * output_ranges[0][2] + ix_orig] +=
                input_data[iz * input_ranges[0][2] * input_ranges[1][2] +
                           iy * input_ranges[0][2] + ix];
          }
        }
      }
    } else if (dir == 1) {
      // Do not forget the boundary outside of the main bound
#pragma omp parallel for default(none) collapse(2)                             \
    shared(input_ranges, output_ranges, npts_global, output_data, input_data)
      for (int iz = 0; iz < imin(input_ranges[2][2], output_ranges[1][2]);
           iz++) {
        for (int ix = 0; ix < imin(input_ranges[0][2], output_ranges[0][2]);
             ix++) {
          for (int iy = 0; iy < input_ranges[1][2]; iy++) {
            const int iy_orig =
                modulo(iy + input_ranges[1][0], npts_global[1]) -
                output_ranges[1][0];
            if (iy_orig < 0 || iy_orig >= output_ranges[1][2])
              continue;
            output_data[iz * output_ranges[0][2] * output_ranges[1][2] +
                        iy_orig * output_ranges[0][2] + ix] +=
                input_data[iz * input_ranges[0][2] * input_ranges[1][2] +
                           iy * input_ranges[0][2] + ix];
          }
        }
      }
    } else {
      // Do not forget the boundary outside of the main bound
#pragma omp parallel for default(none) collapse(2)                             \
    shared(input_ranges, output_ranges, npts_global, output_data, input_data)
      for (int iy = 0; iy < imin(input_ranges[1][2], output_ranges[1][2]);
           iy++) {
        for (int ix = 0; ix < imin(input_ranges[0][2], output_ranges[0][2]);
             ix++) {
          for (int iz = 0; iz < input_ranges[2][2]; iz++) {
            const int iz_orig =
                modulo(iz + input_ranges[2][0], npts_global[2]) -
                output_ranges[2][0];
            if (iz_orig < 0 || iz_orig >= output_ranges[2][2])
              continue;
            output_data[iz_orig * output_ranges[0][2] * output_ranges[1][2] +
                        iy * output_ranges[0][2] + ix] +=
                input_data[iz * input_ranges[0][2] * input_ranges[1][2] +
                           iy * input_ranges[0][2] + ix];
          }
        }
      }
    }

    // A2) Wait for receive processes and add to local data
    for (int process_index = 0;
         process_index < redistribute_rs->number_of_processes_to_halo[dir];
         process_index++) {
      // Do not forget the boundary outside of the main bound
      int process = -1;
      grid_mpi_waitany(redistribute_rs->number_of_processes_to_halo[dir],
                       recv_requests, &process);
      int recv_sizes[3];
      for (int dir2 = 0; dir2 < 3; dir2++) {
        recv_sizes[dir2] =
            (dir2 == dir
                 ? redistribute_rs
                       ->sizes_to_halo[redistribute_rs->offset_to_halo[dir] +
                                       process]
                 : output_ranges[dir2][2]);
      }
      const int *const recv2local =
          (const int *const)redistribute_rs
              ->index2local_to_halo[redistribute_rs->offset_to_halo[dir] +
                                    process];
      const double *current_recv_buffer = recv_buffer[process];
      if (dir == 0) {
        // Collapse(3) may lead to race conditions
        // Alternative would be atomic operations (expensive)
#pragma omp parallel for default(none) collapse(2)                             \
    shared(recv_sizes, recv2local, output_ranges, output_data,                 \
               current_recv_buffer)
        for (int iz_recv = 0; iz_recv < recv_sizes[2]; iz_recv++) {
          for (int iy_recv = 0; iy_recv < recv_sizes[1]; iy_recv++) {
            for (int ix_recv = 0; ix_recv < recv_sizes[0]; ix_recv++) {
              output_data[iz_recv * output_ranges[0][2] * output_ranges[1][2] +
                          iy_recv * output_ranges[0][2] +
                          recv2local[ix_recv]] +=
                  current_recv_buffer[iz_recv * recv_sizes[0] * recv_sizes[1] +
                                      iy_recv * recv_sizes[0] + ix_recv];
            }
          }
        }
      } else if (dir == 1) {
#pragma omp parallel for default(none) collapse(2)                             \
    shared(recv_sizes, recv2local, output_ranges, output_data,                 \
               current_recv_buffer)
        for (int iz_recv = 0; iz_recv < recv_sizes[2]; iz_recv++) {
          for (int ix_recv = 0; ix_recv < recv_sizes[0]; ix_recv++) {
            for (int iy_recv = 0; iy_recv < recv_sizes[1]; iy_recv++) {
              output_data[iz_recv * output_ranges[0][2] * output_ranges[1][2] +
                          recv2local[iy_recv] * output_ranges[0][2] +
                          ix_recv] +=
                  current_recv_buffer[iz_recv * recv_sizes[0] * recv_sizes[1] +
                                      iy_recv * recv_sizes[0] + ix_recv];
            }
          }
        }
      } else {
#pragma omp parallel for default(none) collapse(2)                             \
    shared(recv_sizes, recv2local, output_ranges, output_data,                 \
               current_recv_buffer)
        for (int iy_recv = 0; iy_recv < recv_sizes[1]; iy_recv++) {
          for (int ix_recv = 0; ix_recv < recv_sizes[0]; ix_recv++) {
            for (int iz_recv = 0; iz_recv < recv_sizes[2]; iz_recv++) {
              output_data[recv2local[iz_recv] * output_ranges[0][2] *
                              output_ranges[1][2] +
                          iy_recv * output_ranges[0][2] + ix_recv] +=
                  current_recv_buffer[iz_recv * recv_sizes[0] * recv_sizes[1] +
                                      iy_recv * recv_sizes[0] + ix_recv];
            }
          }
        }
      }
    }

    // A2) Wait for the send processes to finish
    grid_mpi_waitall(redistribute_rs->number_of_processes_to_inner[dir],
                     send_requests);
    // Swap pointers
    double *tmp = input_data;
    input_data = output_data;
    output_data = tmp;
  }

  memcpy(grid_rs_inner, input_data,
         my_number_of_inner_elements_rs * sizeof(double));

  free(recv_buffer);
  free(send_buffer);
  free(recv_requests);
  free(send_requests);
  free(memory_pool);
  free(input_data);
  free(output_data);
}

void grid_copy_from_multigrid_distributed(
    const double *grid_rs, double *grid_pw, const grid_mpi_comm comm_pw,
    const grid_mpi_comm comm_rs, const int npts_global[3],
    const int proc2local_rs[grid_mpi_comm_size(comm_rs)][3][2],
    const int proc2local_pw[grid_mpi_comm_size(comm_pw)][3][2],
    const int border_width[3], const grid_redistribute *redistribute_rs) {
  const int number_of_processes = grid_mpi_comm_size(comm_rs);
  const int my_process_rs = grid_mpi_comm_rank(comm_rs);

  assert(grid_rs != NULL);
  assert(grid_pw != NULL);
  assert(!grid_mpi_comm_is_unequal(comm_pw, comm_rs));
  for (int process = 0; process < number_of_processes; process++) {
    for (int dir = 0; dir < 3; dir++) {
      assert(proc2local_rs[process][dir][0] >= -border_width[dir] &&
             "The inner part of the RS grid cannot be lower than zero!");
      assert(proc2local_rs[process][dir][1] <
                 npts_global[dir] + border_width[dir] &&
             "The inner part of the RS grid contains too many points!");
      assert(proc2local_rs[process][dir][1] - proc2local_rs[process][dir][0] +
                     1 >=
                 0 &&
             "The number of points on the RS grid on one processor cannot be "
             "negative!");
      assert(proc2local_pw[process][dir][0] >= 0 &&
             "The PW grid is only allowed to have nonnegative indices!");
      assert(proc2local_pw[process][dir][1] < npts_global[dir] &&
             "The PW grid cannot have points outside of the inner RS grid!");
      assert(proc2local_pw[process][dir][1] - proc2local_pw[process][dir][0] +
                     1 >=
                 0 &&
             "The number of points on the PW grid on one processor cannot be "
             "negative!");
    }
  }
  for (int dir = 0; dir < 3; dir++) {
    assert(border_width[dir] >= 0 &&
           "The number of points on the boundary cannot be negative!");
    assert(npts_global[dir] >= 0 &&
           "Global number of points cannot be negative!");
  }

  // Prepare the intermediate buffer
  int my_bounds_rs_inner[3][2];
  int my_sizes_rs_inner[3];
  for (int dir = 0; dir < 3; dir++) {
    my_bounds_rs_inner[dir][0] =
        proc2local_rs[my_process_rs][dir][0] + border_width[dir];
    my_bounds_rs_inner[dir][1] =
        proc2local_rs[my_process_rs][dir][1] - border_width[dir];
    my_sizes_rs_inner[dir] =
        my_bounds_rs_inner[dir][1] - my_bounds_rs_inner[dir][0] + 1;
  }
  const int my_number_of_inner_elements_rs = product3(my_sizes_rs_inner);
  double *grid_rs_inner =
      calloc(my_number_of_inner_elements_rs, sizeof(double));

  // Step A: Collect the inner local block
  // From our redistribute container, we send to the inner part and recv the
  // halo
  collect_boundary_to_inner(grid_rs, grid_rs_inner, comm_rs, npts_global,
                            proc2local_rs, border_width, redistribute_rs);

  // Step B: Distribute inner local block to PW grids
  int proc2local_rs_inner[grid_mpi_comm_size(comm_rs)][3][2];
  for (int process = 0; process < number_of_processes; process++) {
    for (int dir = 0; dir < 3; dir++) {
      proc2local_rs_inner[process][dir][0] =
          proc2local_rs[process][dir][0] + border_width[dir];
      proc2local_rs_inner[process][dir][1] =
          proc2local_rs[process][dir][1] - border_width[dir];
    }
  }
  redistribute_grids(grid_rs_inner, grid_pw, comm_rs, comm_pw, npts_global,
                     proc2local_rs_inner, proc2local_pw);

  free(grid_rs_inner);
}

void grid_copy_from_multigrid_general(
    const grid_multigrid *multigrid, double *grids[multigrid->nlevels],
    const grid_mpi_comm comm[multigrid->nlevels], const int *proc2local) {
  for (int level = 0; level < multigrid->nlevels; level++) {
    assert(!grid_mpi_comm_is_unequal(multigrid->comm, comm[level]));
    if (grid_mpi_comm_size(comm[level]) == 1) {
      grid_copy_from_multigrid_serial(
          multigrid->grids[level]->host_buffer, grids[level],
          multigrid->npts_local[level], multigrid->border_width[level]);
    } else {
      // The parallel case, we need to distinguish replicated grids from
      // distributed grids
      if (multigrid->pgrid_dims[level][0] * multigrid->pgrid_dims[level][1] *
              multigrid->pgrid_dims[level][2] ==
          1) {
        grid_copy_from_multigrid_replicated(
            multigrid->grids[level]->host_buffer, grids[level],
            multigrid->npts_local[level], multigrid->border_width[level],
            comm[level],
            (const int(*)[3][2]) &
                proc2local[level * grid_mpi_comm_size(comm[level]) * 6]);
      } else {
        grid_copy_from_multigrid_distributed(
            multigrid->grids[level]->host_buffer, grids[level], multigrid->comm,
            comm[level], multigrid->npts_global[level],
            (const int(*)[3][2]) &
                multigrid->proc2local[6 * level *
                                      grid_mpi_comm_size(multigrid->comm)],
            (const int(*)[3][2]) &
                proc2local[6 * level * grid_mpi_comm_size(multigrid->comm)],
            multigrid->border_width[level], &multigrid->redistribute[level]);
      }
    }
  }
}

void grid_copy_from_multigrid_general_f(
    const grid_multigrid *multigrid, double *grids[multigrid->nlevels],
    const grid_mpi_fint fortran_comm[multigrid->nlevels],
    const int *proc2local) {
  grid_mpi_comm comm[multigrid->nlevels];
  for (int level = 0; level < multigrid->nlevels; level++)
    comm[level] = grid_mpi_comm_f2c(fortran_comm[level]);
  grid_copy_from_multigrid_general(multigrid, grids, comm, proc2local);
}

void grid_copy_from_multigrid_general_single(const grid_multigrid *multigrid,
                                             const int level, double *grid,
                                             const grid_mpi_comm comm,
                                             const int *proc2local) {
  assert(multigrid != NULL);
  assert(!grid_mpi_comm_is_unequal(multigrid->comm, comm));
  assert(grid != NULL);
  if (grid_mpi_comm_size(comm) == 1) {
    grid_copy_from_multigrid_serial(multigrid->grids[level]->host_buffer, grid,
                                    multigrid->npts_local[level],
                                    multigrid->border_width[level]);
  } else {
    // The parallel case, we need to distinguish replicated grids from
    // distributed grids
    if (multigrid->pgrid_dims[level][0] * multigrid->pgrid_dims[level][1] *
            multigrid->pgrid_dims[level][2] ==
        1) {
      grid_copy_from_multigrid_replicated(multigrid->grids[level]->host_buffer,
                                          grid, multigrid->npts_local[level],
                                          multigrid->border_width[level], comm,
                                          (const int(*)[3][2])proc2local);
    } else {
      grid_copy_from_multigrid_distributed(
          multigrid->grids[level]->host_buffer, grid, multigrid->comm, comm,
          multigrid->npts_global[level],
          (const int(*)[3][2]) &
              multigrid
                  ->proc2local[6 * level * grid_mpi_comm_size(multigrid->comm)],
          (const int(*)[3][2])proc2local, multigrid->border_width[level],
          &multigrid->redistribute[level]);
    }
  }
}

void grid_copy_from_multigrid_general_single_f(const grid_multigrid *multigrid,
                                               const int level, double *grid,
                                               const grid_mpi_fint fortran_comm,
                                               const int *proc2local) {
  grid_copy_from_multigrid_general_single(
      multigrid, level - 1, grid, grid_mpi_comm_f2c(fortran_comm), proc2local);
}

/*******************************************************************************
 * \brief Allocates a multigrid which is passed to task list-based and
 *pgf_product-based routines.
 *
 * \param orthorhombic     Whether simulation box is orthorhombic.
 * \param nlevels          Number of grid levels.
 * \param npts_global     Number of global grid points in each direction.
 * \param npts_local      Number of local grid points in each direction.
 * \param shift_local     Number of points local grid is shifted wrt global grid
 * \param border_width    Width of halo region in grid points in each direction.
 * \param dh              Incremental grid matrix.
 * \param dh_inv          Inverse incremental grid matrix.
 *
 * \param multigrid        Handle to the created multigrid.
 *
 * \author Frederick Stein
 ******************************************************************************/
void grid_create_multigrid_f(
    const bool orthorhombic, const int nlevels,
    const int npts_global[nlevels][3], const int npts_local[nlevels][3],
    const int shift_local[nlevels][3], const int border_width[nlevels][3],
    const double dh[nlevels][3][3], const double dh_inv[nlevels][3][3],
    const int grid_dims[nlevels][3], const grid_mpi_fint fortran_comm,
    grid_multigrid **multigrid_out) {
  grid_create_multigrid(orthorhombic, nlevels, npts_global, npts_local,
                        shift_local, border_width, dh, dh_inv, grid_dims,
                        grid_mpi_comm_f2c(fortran_comm), multigrid_out);
}

void grid_free_redistribute(grid_redistribute *redistribute) {
  free(redistribute->processes_to_inner);
  free(redistribute->processes_to_halo);
  free(redistribute->buffer_offsets_to_inner);
  free(redistribute->buffer_offsets_to_halo);
  for (int proc_count = 0;
       proc_count < redistribute->total_number_of_processes_to_inner;
       proc_count++) {
    free(redistribute->index2local_to_inner[proc_count]);
  }
  free(redistribute->index2local_to_inner);
  for (int proc_count = 0;
       proc_count < redistribute->total_number_of_processes_to_halo;
       proc_count++) {
    free(redistribute->index2local_to_halo[proc_count]);
  }
  free(redistribute->index2local_to_halo);
  free(redistribute->sizes_to_inner);
  free(redistribute->sizes_to_halo);
}

void grid_create_redistribute(
    const grid_mpi_comm comm, const int npts_global[3],
    const int proc2local[grid_mpi_comm_size(comm)][3][2],
    const int border_width[3], grid_redistribute *redistribute) {

  grid_free_redistribute(redistribute);

  const int number_of_processes = grid_mpi_comm_size(comm);
  redistribute->number_of_processes = number_of_processes;
  const int my_process = grid_mpi_comm_rank(comm);

  // Prepare the intermediate buffer
  int my_bounds[3][2];
  int my_bounds_inner[3][2];
  for (int dir = 0; dir < 3; dir++) {
    my_bounds[dir][0] = proc2local[my_process][dir][0];
    my_bounds[dir][1] = proc2local[my_process][dir][1];
    my_bounds_inner[dir][0] =
        proc2local[my_process][dir][0] + border_width[dir];
    my_bounds_inner[dir][1] =
        proc2local[my_process][dir][1] - border_width[dir];
  }

  for (int dir = 0; dir < 3; dir++) {
    // The current input covers the original ranges in all directions which we
    // haven't covered yet and the smaller directions from which we have
    redistribute->local_ranges[0][dir][0] = my_bounds[dir][0];
    redistribute->local_ranges[0][dir][1] = my_bounds[dir][1];
    redistribute->local_ranges[0][dir][2] =
        redistribute->local_ranges[0][dir][1] -
        redistribute->local_ranges[0][dir][0] + 1;
    assert(redistribute->local_ranges[0][dir][2] >= 0);
  }
  for (int dir = 0; dir < 3; dir++) {
    memcpy(redistribute->local_ranges[dir + 1], redistribute->local_ranges[dir],
           3 * 3 * sizeof(int));
    redistribute->local_ranges[dir + 1][dir][0] = my_bounds_inner[dir][0];
    redistribute->local_ranges[dir + 1][dir][1] = my_bounds_inner[dir][1];
    redistribute->local_ranges[dir + 1][dir][2] =
        redistribute->local_ranges[dir + 1][dir][1] -
        redistribute->local_ranges[dir + 1][dir][0] + 1;
    assert(redistribute->local_ranges[dir + 1][dir][2] >= 0);
  }

  redistribute->total_number_of_processes_to_inner = 0;
  redistribute->total_number_of_processes_to_inner = 0;
  redistribute->max_number_of_processes_to_inner = 0;
  redistribute->max_number_of_processes_to_halo = 0;
  for (int dir = 0; dir < 3; dir++) {
    redistribute->number_of_processes_to_inner[dir] = 0;
    redistribute->number_of_processes_to_halo[dir] = 0;
    if (border_width[dir] == 0)
      continue;

    const int(*input_ranges)[3] = redistribute->local_ranges[dir];
    const int(*output_ranges)[3] = redistribute->local_ranges[dir + 1];

    for (int process_shift = 0; process_shift < number_of_processes;
         process_shift++) {
      int recv_process = (my_process - process_shift + number_of_processes) %
                         number_of_processes;
      // We only need to recv from processes which have different bounds in the
      // exchange direction and the same in the other directions
      for (int dir2 = 0; dir2 < 3; dir2++) {
        if (dir2 == dir) {
          if (proc2local[recv_process][dir2][0] == my_bounds[dir2][0] &&
              proc2local[recv_process][dir2][1] == my_bounds[dir2][1]) {
            recv_process = grid_mpi_proc_null;
            break;
          }
          if (recv_process >= 0) {
            // Check if there is actually an element to recv from this process
            int recv_ranges[2];
            recv_ranges[0] = proc2local[recv_process][dir2][0];
            recv_ranges[1] = proc2local[recv_process][dir2][1];
            bool found_element = false;
            for (int recv_index = recv_ranges[0]; recv_index <= recv_ranges[1];
                 recv_index++) {
              const int local_index = modulo(recv_index, npts_global[dir2]);
              if (local_index >= output_ranges[dir2][0] &&
                  local_index <= output_ranges[dir2][1]) {
                found_element = true;
                break;
              }
            }
            if (!found_element) {
              recv_process = grid_mpi_proc_null;
              break;
            }
          }
        } else {
          if (proc2local[recv_process][dir2][0] != my_bounds[dir2][0] ||
              proc2local[recv_process][dir2][1] != my_bounds[dir2][1]) {
            recv_process = grid_mpi_proc_null;
            break;
          }
        }
      }
      if (recv_process >= 0)
        redistribute->number_of_processes_to_halo[dir]++;

      int send_process = (my_process + process_shift) % number_of_processes;
      // We only need to send to processes which have different bounds in the
      // exchange direction and the same in the other directions
      for (int dir2 = 0; dir2 < 3; dir2++) {
        if (dir2 == dir) {
          if (proc2local[send_process][dir][0] == my_bounds[dir][0] &&
              proc2local[send_process][dir][1] == my_bounds[dir][1]) {
            send_process = grid_mpi_proc_null;
            break;
          }
          if (send_process >= 0) {
            // Check whether there is any element to send to this process
            int send_ranges[2];
            send_ranges[0] =
                proc2local[send_process][dir][0] + border_width[dir];
            send_ranges[1] =
                proc2local[send_process][dir][1] - border_width[dir];
            bool found_element = false;
            for (int local_index = input_ranges[dir][0];
                 local_index <= input_ranges[dir][1]; local_index++) {
              const int send_index = modulo(local_index, npts_global[dir]);
              if (send_index >= send_ranges[0] &&
                  send_index <= send_ranges[1]) {
                found_element = true;
                break;
              }
            }
            if (!found_element) {
              send_process = grid_mpi_proc_null;
              break;
            }
          }
        } else {
          if (proc2local[send_process][dir2][0] != my_bounds[dir2][0] ||
              proc2local[send_process][dir2][1] != my_bounds[dir2][1]) {
            send_process = grid_mpi_proc_null;
            break;
          }
        }
      }
      if (send_process >= 0) {
        redistribute->number_of_processes_to_inner[dir]++;
      }
    }
    redistribute->max_number_of_processes_to_inner =
        imax(redistribute->max_number_of_processes_to_inner,
             redistribute->number_of_processes_to_inner[dir]);
    redistribute->max_number_of_processes_to_halo =
        imax(redistribute->max_number_of_processes_to_halo,
             redistribute->number_of_processes_to_halo[dir]);
    redistribute->total_number_of_processes_to_inner +=
        redistribute->number_of_processes_to_inner[dir];
    redistribute->total_number_of_processes_to_halo +=
        redistribute->number_of_processes_to_halo[dir];
  }

  redistribute->processes_to_inner =
      calloc(redistribute->total_number_of_processes_to_inner, sizeof(int));
  redistribute->processes_to_halo =
      calloc(redistribute->total_number_of_processes_to_halo, sizeof(int));

  // Determine the processes to send to and receive from and their offsets
  int send_proc_index = 0;
  int recv_proc_index = 0;
  for (int dir = 0; dir < 3; dir++) {
    redistribute->offset_to_inner[dir] =
        (dir > 0 ? redistribute->offset_to_inner[dir - 1] +
                       redistribute->number_of_processes_to_inner[dir - 1]
                 : 0);
    redistribute->offset_to_halo[dir] =
        (dir > 0 ? redistribute->offset_to_halo[dir - 1] +
                       redistribute->number_of_processes_to_halo[dir - 1]
                 : 0);
    if (border_width[dir] == 0)
      continue;

    const int(*input_ranges)[3] = redistribute->local_ranges[dir];
    const int(*output_ranges)[3] = redistribute->local_ranges[dir + 1];
    for (int process_shift = 0; process_shift < number_of_processes;
         process_shift++) {
      int recv_process = (my_process - process_shift + number_of_processes) %
                         number_of_processes;
      // We only need to recv from processes which have different bounds in the
      // exchange direction and the same in the other directions
      for (int dir2 = 0; dir2 < 3; dir2++) {
        if (dir2 == dir) {
          if (proc2local[recv_process][dir][0] == my_bounds[dir][0] &&
              proc2local[recv_process][dir][1] == my_bounds[dir][1]) {
            recv_process = grid_mpi_proc_null;
            break;
          }
          if (recv_process >= 0) {
            // Check if there is actually an element to recv from this process
            int recv_ranges[2];
            recv_ranges[0] = proc2local[recv_process][dir][0];
            recv_ranges[1] = proc2local[recv_process][dir][1];
            bool found_element = false;
            for (int recv_index = recv_ranges[0]; recv_index <= recv_ranges[1];
                 recv_index++) {
              const int local_index = modulo(recv_index, npts_global[dir]);
              if (local_index >= output_ranges[dir][0] &&
                  local_index <= output_ranges[dir][1]) {
                found_element = true;
                break;
              }
            }
            if (!found_element) {
              recv_process = grid_mpi_proc_null;
              break;
            }
          }
        } else {
          if (proc2local[recv_process][dir2][0] != my_bounds[dir2][0] ||
              proc2local[recv_process][dir2][1] != my_bounds[dir2][1]) {
            recv_process = grid_mpi_proc_null;
            break;
          }
        }
      }
      if (recv_process >= 0) {
        redistribute->processes_to_halo[recv_proc_index] = recv_process;
        recv_proc_index++;
      }
      int send_process = (my_process + process_shift) % number_of_processes;
      // We only need to send to processes which have different bounds in the
      // exchange direction and the same in the other directions
      for (int dir2 = 0; dir2 < 3; dir2++) {
        if (dir2 == dir) {
          if (proc2local[send_process][dir][0] == my_bounds[dir][0] &&
              proc2local[send_process][dir][1] == my_bounds[dir][1]) {
            send_process = grid_mpi_proc_null;
            break;
          }
          if (send_process >= 0) {
            int send_ranges[3];
            send_ranges[0] =
                proc2local[send_process][dir][0] + border_width[dir];
            send_ranges[1] =
                proc2local[send_process][dir][1] - border_width[dir];
            send_ranges[2] = send_ranges[1] - send_ranges[0] + 1;
            bool found_element = false;
            for (int local_index = input_ranges[dir][0];
                 local_index <= input_ranges[dir][1]; local_index++) {
              const int send_index = modulo(local_index, npts_global[dir]);
              if (send_index >= send_ranges[0] &&
                  send_index <= send_ranges[1]) {
                found_element = true;
                break;
              }
            }
            if (!found_element) {
              send_process = grid_mpi_proc_null;
              break;
            }
          }
        } else {
          if (proc2local[send_process][dir2][0] != my_bounds[dir2][0] ||
              proc2local[send_process][dir2][1] != my_bounds[dir2][1]) {
            send_process = grid_mpi_proc_null;
            break;
          }
        }
      }
      if (send_process >= 0) {
        redistribute->processes_to_inner[send_proc_index] = send_process;
        send_proc_index++;
      }
    }
    assert(recv_proc_index ==
           redistribute->offset_to_halo[dir] +
               redistribute->number_of_processes_to_halo[dir]);
    assert(send_proc_index ==
           redistribute->offset_to_inner[dir] +
               redistribute->number_of_processes_to_inner[dir]);
  }

  redistribute->size_of_buffer_to_inner = 0;
  redistribute->size_of_buffer_to_halo = 0;
  redistribute->buffer_offsets_to_inner =
      calloc(redistribute->total_number_of_processes_to_inner, sizeof(int));
  redistribute->buffer_offsets_to_halo =
      calloc(redistribute->total_number_of_processes_to_halo, sizeof(int));
  redistribute->index2local_to_inner =
      calloc(redistribute->total_number_of_processes_to_inner, sizeof(int **));
  redistribute->index2local_to_halo =
      calloc(redistribute->total_number_of_processes_to_halo, sizeof(int **));
  redistribute->sizes_to_inner =
      calloc(redistribute->total_number_of_processes_to_inner, sizeof(int));
  redistribute->sizes_to_halo =
      calloc(redistribute->total_number_of_processes_to_halo, sizeof(int));
  int proc_counter = 0;
  for (int dir = 0; dir < 3; dir++) {
    // Without border, there is nothing to exchange
    if (border_width[dir] == 0)
      continue;

    const int(*input_ranges)[3] = redistribute->local_ranges[dir];
    const int(*output_ranges)[3] = redistribute->local_ranges[dir + 1];

    // A2) Send around local data of the RS grid and copy it to our local buffer
    for (int process_shift = 0;
         process_shift < redistribute->number_of_processes_to_inner[dir];
         process_shift++) {
      const int send_process =
          redistribute->processes_to_inner[redistribute->offset_to_inner[dir] +
                                           process_shift];

      int number_of_elements_to_send = 0;
      if (send_process >= 0) {
        int send_ranges[3][3];
        number_of_elements_to_send = 1;
        for (int dir2 = 0; dir2 < 3; dir2++) {
          int tmp = 0;
          if (dir2 == dir) {
            send_ranges[dir][0] =
                proc2local[send_process][dir][0] + border_width[dir];
            send_ranges[dir][1] =
                proc2local[send_process][dir][1] - border_width[dir];
            send_ranges[dir][2] = send_ranges[dir][1] - send_ranges[dir][0] + 1;
            for (int local_index = input_ranges[dir][0];
                 local_index <= input_ranges[dir][1]; local_index++) {
              const int send_index = modulo(local_index, npts_global[dir]);
              if (send_index >= send_ranges[dir][0] &&
                  send_index <= send_ranges[dir][1])
                tmp++;
            }
            redistribute->sizes_to_inner[proc_counter] = tmp;
          } else {
            send_ranges[dir2][0] = output_ranges[dir2][0];
            send_ranges[dir2][1] = output_ranges[dir2][1];
            send_ranges[dir2][2] =
                send_ranges[dir2][1] - send_ranges[dir2][0] + 1;
            tmp = send_ranges[dir2][2];
          }
          number_of_elements_to_send *= tmp;
        }
        redistribute->index2local_to_inner[proc_counter] =
            calloc(redistribute->sizes_to_inner[proc_counter], sizeof(int));
        int local_send_index = 0;
        for (int local_index = 0; local_index < input_ranges[dir][2];
             local_index++) {
          const int send_index =
              modulo(local_index + input_ranges[dir][0], npts_global[dir]);
          if (send_index >= send_ranges[dir][0] &&
              send_index <= send_ranges[dir][1]) {
            redistribute->index2local_to_inner[proc_counter][local_send_index] =
                local_index;
            local_send_index++;
          }
        }
        redistribute->buffer_offsets_to_inner[proc_counter] =
            redistribute->size_of_buffer_to_inner;
        redistribute->size_of_buffer_to_inner += number_of_elements_to_send;
        proc_counter++;
      }
    }
  }

  proc_counter = 0;
  for (int dir = 0; dir < 3; dir++) {
    // Without border, there is nothing to exchange
    if (border_width[dir] == 0)
      continue;

    const int(*input_ranges)[3] = redistribute->local_ranges[dir];
    const int(*output_ranges)[3] = redistribute->local_ranges[dir + 1];

    // A2) Send around local data of the RS grid and copy it to our local buffer
    for (int process_shift = 0;
         process_shift < redistribute->number_of_processes_to_halo[dir];
         process_shift++) {
      const int recv_process =
          redistribute->processes_to_halo[redistribute->offset_to_halo[dir] +
                                          process_shift];

      int number_of_elements_to_receive = 0;
      if (recv_process >= 0) {
        int recv_ranges[3][3];
        number_of_elements_to_receive = 1;
        for (int dir2 = 0; dir2 < 3; dir2++) {
          if (dir2 == dir) {
            recv_ranges[dir][0] = proc2local[recv_process][dir][0];
            recv_ranges[dir][1] = proc2local[recv_process][dir][1];
            recv_ranges[dir][2] = recv_ranges[dir][1] - recv_ranges[dir][0] + 1;
            int received_elements_in_dir = 0;
            for (int recv_index = recv_ranges[dir][0];
                 recv_index <= recv_ranges[dir][1]; recv_index++) {
              const int local_index = modulo(recv_index, npts_global[dir]);
              if (local_index >= output_ranges[dir][0] &&
                  local_index <= output_ranges[dir][1])
                received_elements_in_dir++;
            }
            redistribute->sizes_to_halo[proc_counter] =
                received_elements_in_dir;
            number_of_elements_to_receive *= received_elements_in_dir;
          } else {
            recv_ranges[dir2][0] = input_ranges[dir2][0];
            recv_ranges[dir2][1] = input_ranges[dir2][1];
            recv_ranges[dir2][2] = input_ranges[dir2][2];
            number_of_elements_to_receive *= recv_ranges[dir2][2];
          }
        }
        redistribute->index2local_to_halo[proc_counter] =
            calloc(redistribute->sizes_to_halo[proc_counter], sizeof(int));
        int local_recv_index = 0;
        for (int recv_index = recv_ranges[dir][0];
             recv_index <= recv_ranges[dir][1]; recv_index++) {
          const int local_index =
              modulo(recv_index, npts_global[dir]) - output_ranges[dir][0];
          if (local_index >= 0 && local_index < output_ranges[dir][2]) {
            redistribute->index2local_to_halo[proc_counter][local_recv_index] =
                local_index;
            local_recv_index++;
          }
        }
        redistribute->buffer_offsets_to_halo[proc_counter] =
            redistribute->size_of_buffer_to_halo;
        redistribute->size_of_buffer_to_halo += number_of_elements_to_receive;
        proc_counter++;
      }
    }
  }
}

void grid_multigrid_allocate_buffers(grid_multigrid *multigrid,
                                     const int nlevels,
                                     const grid_mpi_comm comm) {
  const grid_library_config config = grid_library_get_config();

  multigrid->nlevels = nlevels;
  multigrid->npts_global = calloc(nlevels, sizeof(int[3]));
  multigrid->npts_local = calloc(nlevels, sizeof(int[3]));
  multigrid->shift_local = calloc(nlevels, sizeof(int[3]));
  multigrid->border_width = calloc(nlevels, sizeof(int[3]));
  multigrid->dh = calloc(nlevels, sizeof(double[3][3]));
  multigrid->dh_inv = calloc(nlevels, sizeof(double[3][3]));
  multigrid->grids = calloc(nlevels, sizeof(offload_buffer *));
  multigrid->pgrid_dims = calloc(nlevels, sizeof(int[3]));
  multigrid->proc2local =
      calloc(nlevels * grid_mpi_comm_size(comm), sizeof(int[3][2]));
  multigrid->redistribute = calloc(nlevels, sizeof(grid_redistribute));
  multigrid->fft_grid_layouts = calloc(nlevels, sizeof(grid_fft_grid_layout *));
  multigrid->fft_rs_grids = calloc(nlevels, sizeof(grid_fft_real_rs_grid));
  multigrid->fft_gs_grids = calloc(nlevels, sizeof(grid_fft_complex_gs_grid));

  // Resolve AUTO to a concrete backend.
  if (config.backend == GRID_BACKEND_AUTO) {
#if defined(__OFFLOAD_HIP) && !defined(__NO_OFFLOAD_GRID)
    multigrid->backend = GRID_BACKEND_HIP;
#elif defined(__OFFLOAD) && !defined(__NO_OFFLOAD_GRID)
    multigrid->backend = GRID_BACKEND_GPU;
#else
    multigrid->backend = GRID_BACKEND_CPU;
#endif
  } else {
    multigrid->backend = config.backend;
  }
}

void grid_multigrid_setup_distribution(
    grid_multigrid *multigrid, const bool orthorhombic, const int nlevels,
    const int npts_global[nlevels][3], const int npts_local[nlevels][3],
    const int shift_local[nlevels][3], const int border_width[nlevels][3],
    const double dh[nlevels][3][3], const double dh_inv[nlevels][3][3],
    const int pgrid_dims[nlevels][3], const grid_mpi_comm comm) {

  const int num_int = 3 * nlevels;
  const int num_double = 9 * nlevels;

  for (int level = 0; level < nlevels; level++) {
    offload_create_buffer(npts_local[level][0] * npts_local[level][1] *
                              npts_local[level][2],
                          &multigrid->grids[level]);
  }

  multigrid->orthorhombic = orthorhombic;
  memcpy(multigrid->npts_global, npts_global, num_int * sizeof(int));
  memcpy(multigrid->npts_local, npts_local, num_int * sizeof(int));
  memcpy(multigrid->shift_local, shift_local, num_int * sizeof(int));
  memcpy(multigrid->border_width, border_width, num_int * sizeof(int));
  memcpy(multigrid->dh, dh, num_double * sizeof(double));
  memcpy(multigrid->dh_inv, dh_inv, num_double * sizeof(double));
  memcpy(multigrid->pgrid_dims, pgrid_dims, num_int * sizeof(int));
  grid_mpi_comm_dup(comm, &multigrid->comm);

  for (int level = 0; level < nlevels; level++) {
    int local_bounds[3][2];
    for (int dir = 0; dir < 3; dir++) {
      local_bounds[dir][0] = shift_local[level][dir];
      local_bounds[dir][1] =
          shift_local[level][dir] + npts_local[level][dir] - 1;
    }
    grid_mpi_allgather_int(
        &local_bounds[0][0], 6,
        &multigrid->proc2local[level * 6 * grid_mpi_comm_size(comm)], comm);

    grid_create_redistribute(
        multigrid->comm, multigrid->npts_global[level],
        (const int(*)[3][2]) &
            multigrid->proc2local[level * 6 * grid_mpi_comm_size(comm)],
        multigrid->border_width[level], &(multigrid->redistribute[level]));
  }
}

void grid_multigrid_setup_fft_grids(grid_multigrid *multigrid) {
  grid_create_fft_grid_layout(&multigrid->fft_grid_layouts[0], multigrid->comm,
                              multigrid->npts_global[0], multigrid->dh_inv[0],
                              false);
  grid_create_real_rs_grid(&multigrid->fft_rs_grids[0],
                           multigrid->fft_grid_layouts[0]);
  grid_create_complex_gs_grid(&multigrid->fft_gs_grids[0],
                              multigrid->fft_grid_layouts[0]);
  for (int level = 1; level < multigrid->nlevels; level++) {
    grid_create_fft_grid_layout_from_reference(
        &multigrid->fft_grid_layouts[level], multigrid->npts_global[level],
        multigrid->fft_grid_layouts[0]);
    grid_create_real_rs_grid(&multigrid->fft_rs_grids[level],
                             multigrid->fft_grid_layouts[level]);
    grid_create_complex_gs_grid(&multigrid->fft_gs_grids[level],
                                multigrid->fft_grid_layouts[level]);
  }
}

/*******************************************************************************
 * \brief Allocates a multigrid which is passed to task list-based and
 *pgf_product-based routines.
 *
 * \param orthorhombic     Whether simulation box is orthorhombic.
 * \param nlevels          Number of grid levels.
 * \param npts_global     Number of global grid points in each direction.
 * \param npts_local      Number of local grid points in each direction.
 * \param shift_local     Number of points local grid is shifted wrt global grid
 * \param border_width    Width of halo region in grid points in each direction.
 * \param dh              Incremental grid matrix.
 * \param dh_inv          Inverse incremental grid matrix.
 *
 * \param multigrid        Handle to the created multigrid.
 *
 * \author Frederick Stein
 ******************************************************************************/
void grid_create_multigrid(
    const bool orthorhombic, const int nlevels,
    const int npts_global[nlevels][3], const int npts_local[nlevels][3],
    const int shift_local[nlevels][3], const int border_width[nlevels][3],
    const double dh[nlevels][3][3], const double dh_inv[nlevels][3][3],
    const int pgrid_dims[nlevels][3], const grid_mpi_comm comm,
    grid_multigrid **multigrid_out) {

  grid_multigrid *multigrid = NULL;

  assert(multigrid_out != NULL);
  for (int level = 0; level < nlevels; level++) {
    assert(pgrid_dims[level][0] * pgrid_dims[level][1] * pgrid_dims[level][2] ==
               grid_mpi_comm_size(comm) ||
           (pgrid_dims[level][0] == 1 && pgrid_dims[level][1] == 1 &&
            pgrid_dims[level][2] == 1));
  }

  if (*multigrid_out != NULL) {
    multigrid = *multigrid_out;
    free(multigrid->npts_global);
    free(multigrid->npts_local);
    free(multigrid->shift_local);
    free(multigrid->border_width);
    free(multigrid->dh);
    free(multigrid->dh_inv);
    if (multigrid->grids != NULL) {
      for (int level = 0; level < multigrid->nlevels; level++) {
        offload_free_buffer(multigrid->grids[level]);
      }
      free(multigrid->grids);
    }
    free(multigrid->pgrid_dims);
    free(multigrid->proc2local);
    if (multigrid->redistribute != NULL) {
      for (int level = 0; level < multigrid->nlevels; level++) {
        grid_free_redistribute(&multigrid->redistribute[level]);
      }
      free(multigrid->redistribute);
    }
    if (multigrid->fft_grid_layouts != NULL) {
      for (int level = 0; level < multigrid->nlevels; level++) {
        grid_free_fft_grid_layout(multigrid->fft_grid_layouts[level]);
      }
      free(multigrid->fft_grid_layouts);
    }
    if (multigrid->fft_rs_grids != NULL) {
      for (int level = 0; level < multigrid->nlevels; level++) {
        grid_free_real_rs_grid(&multigrid->fft_rs_grids[level]);
      }
      free(multigrid->fft_rs_grids);
    }
    if (multigrid->fft_gs_grids != NULL) {
      for (int level = 0; level < multigrid->nlevels; level++) {
        grid_free_complex_gs_grid(&multigrid->fft_gs_grids[level]);
      }
      free(multigrid->fft_gs_grids);
    }
    grid_mpi_comm_free(&multigrid->comm);
    grid_ref_free_multigrid(multigrid->ref);
    grid_cpu_free_multigrid(multigrid->cpu);
    multigrid->nlevels = -1;
    memset(multigrid, 0, sizeof(grid_multigrid));
  } else {
    multigrid = calloc(1, sizeof(grid_multigrid));
  }
  grid_multigrid_allocate_buffers(multigrid, nlevels, comm);

  grid_multigrid_setup_distribution(multigrid, orthorhombic, nlevels,
                                    npts_global, npts_local, shift_local,
                                    border_width, dh, dh_inv, pgrid_dims, comm);

  grid_multigrid_setup_fft_grids(multigrid);

  grid_ref_create_multigrid(orthorhombic, nlevels, npts_global, npts_local,
                            shift_local, border_width, dh, dh_inv, comm,
                            &(multigrid->ref));

  // We need it for collocation/integration of pgf_products
  grid_cpu_create_multigrid(orthorhombic, nlevels, npts_global, npts_local,
                            shift_local, border_width, dh, dh_inv, comm,
                            &(multigrid->cpu));

  switch (multigrid->backend) {
  case GRID_BACKEND_REF:
    break;
  case GRID_BACKEND_CPU:
    break;
  }

  for (int level = 0; level < nlevels; level++) {
    memset(offload_get_buffer_host_pointer(multigrid->grids[level]), 0,
           npts_local[level][0] * npts_local[level][1] * npts_local[level][2] *
               sizeof(double));
  }

  *multigrid_out = multigrid;
}

/*******************************************************************************
 * \brief Deallocates given multigrid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_free_multigrid(grid_multigrid *multigrid) {
  if (multigrid != NULL) {
    free(multigrid->npts_global);
    free(multigrid->npts_local);
    free(multigrid->shift_local);
    free(multigrid->border_width);
    free(multigrid->dh);
    free(multigrid->dh_inv);
    if (multigrid->grids != NULL) {
      for (int level = 0; level < multigrid->nlevels; level++) {
        offload_free_buffer(multigrid->grids[level]);
      }
      free(multigrid->grids);
    }
    free(multigrid->pgrid_dims);
    free(multigrid->proc2local);
    if (multigrid->redistribute != NULL) {
      for (int level = 0; level < multigrid->nlevels; level++) {
        grid_free_redistribute(&multigrid->redistribute[level]);
      }
      free(multigrid->redistribute);
    }
    if (multigrid->fft_grid_layouts != NULL) {
      for (int level = 0; level < multigrid->nlevels; level++) {
        grid_free_fft_grid_layout(multigrid->fft_grid_layouts[level]);
      }
      free(multigrid->fft_grid_layouts);
    }
    if (multigrid->fft_rs_grids != NULL) {
      for (int level = 0; level < multigrid->nlevels; level++) {
        grid_free_real_rs_grid(&multigrid->fft_rs_grids[level]);
      }
      free(multigrid->fft_rs_grids);
    }
    if (multigrid->fft_gs_grids != NULL) {
      for (int level = 0; level < multigrid->nlevels; level++) {
        grid_free_complex_gs_grid(&multigrid->fft_gs_grids[level]);
      }
      free(multigrid->fft_gs_grids);
    }
    grid_mpi_comm_free(&multigrid->comm);
    grid_ref_free_multigrid(multigrid->ref);
    grid_cpu_free_multigrid(multigrid->cpu);
    multigrid->nlevels = -1;
    free(multigrid);
  }
}

// EOF
