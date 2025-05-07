/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "grid_fft_reorder.h"
#include "common/grid_common.h"
#include "common/grid_mpi.h"

#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*******************************************************************************
 * \brief Performs a transposition of (x,y,z)->(z,x,y).
 * \author Frederick Stein
 ******************************************************************************/
void collect_y_and_distribute_z_blocked(
    const double complex *grid, double complex *transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int (*proc2local_transposed)[3][2], const grid_mpi_comm comm,
    const grid_mpi_comm sub_comm[2]) {
  const int my_process = grid_mpi_comm_rank(comm);

  int proc_coord[2];
  int dims[2];
  int periods[2];
  grid_mpi_cart_get(comm, 2, dims, periods, proc_coord);

  const int my_sizes[3] = {
      proc2local[my_process][0][1] - proc2local[my_process][0][0] + 1,
      proc2local[my_process][1][1] - proc2local[my_process][1][0] + 1,
      proc2local[my_process][2][1] - proc2local[my_process][2][0] + 1};
  assert(my_sizes[2] == npts_global[2]);
  const int my_sizes_transposed[3] = {
      proc2local_transposed[my_process][0][1] -
          proc2local_transposed[my_process][0][0] + 1,
      proc2local_transposed[my_process][1][1] -
          proc2local_transposed[my_process][1][0] + 1,
      proc2local_transposed[my_process][2][1] -
          proc2local_transposed[my_process][2][0] + 1};
  assert(my_sizes_transposed[1] == npts_global[1]);
  assert(my_sizes[0] == my_sizes_transposed[0]);

  int *send_displacements = calloc(dims[1], sizeof(int));
  int *recv_displacements = calloc(dims[1], sizeof(int));
  int *send_counts = calloc(dims[1], sizeof(int));
  int *recv_counts = calloc(dims[1], sizeof(int));
  double complex *send_buffer =
      calloc(product3(my_sizes), sizeof(double complex));

  int send_offset = 0;
  int recv_offset = 0;
  for (int process = 0; process < dims[1]; process++) {
    // Setup arrays
    send_displacements[process] = send_offset;
    recv_displacements[process] = recv_offset;
    int rank;
    grid_mpi_cart_rank(comm, (const int[2]){proc_coord[0], process}, &rank);
    const int current_send_count = my_sizes[0] * my_sizes[1] *
                                   (proc2local_transposed[rank][2][1] -
                                    proc2local_transposed[rank][2][0] + 1);
    send_counts[process] = current_send_count;
    const int current_recv_count =
        my_sizes_transposed[0] *
        (proc2local[rank][1][1] - proc2local[rank][1][0] + 1) *
        my_sizes_transposed[2];
    recv_counts[process] = current_recv_count;
    send_offset += current_send_count;
    recv_offset += current_recv_count;
// Copy the data to the send buffer
#pragma omp parallel for collapse(2) default(none)                             \
    shared(my_sizes, my_sizes_transposed, proc2local, proc2local_transposed,   \
               send_buffer, grid, send_displacements, process, rank)
    for (int index_y = 0; index_y < my_sizes[1]; index_y++) {
      for (int index_x = 0; index_x < my_sizes[0]; index_x++) {
        memcpy(send_buffer + send_displacements[process] +
                   (index_y * my_sizes[0] + index_x) *
                       (proc2local_transposed[rank][2][1] -
                        proc2local_transposed[rank][2][0] + 1),
               grid + (index_y * my_sizes[0] + index_x) * my_sizes[2] +
                   proc2local_transposed[rank][2][0],
               (proc2local_transposed[rank][2][1] -
                proc2local_transposed[rank][2][0] + 1) *
                   sizeof(double complex));
      }
    }
  }
  assert(send_offset == product3(my_sizes));
  assert(recv_offset == product3(my_sizes_transposed));

  // Use collective MPI communication
  grid_mpi_alltoallv_double_complex(send_buffer, send_counts,
                                    send_displacements, transposed, recv_counts,
                                    recv_displacements, sub_comm[1]);

  free(send_buffer);
  free(send_counts);
  free(send_displacements);
  free(recv_counts);
  free(recv_displacements);
}

/*******************************************************************************
 * \brief Performs a transposition of (x,y,z)->(z,x,y).
 * \author Frederick Stein
 ******************************************************************************/
void collect_z_and_distribute_y_blocked(
    const double complex *grid, double complex *transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int (*proc2local_transposed)[3][2], const grid_mpi_comm comm,
    const grid_mpi_comm sub_comm[2]) {
  const int my_process = grid_mpi_comm_rank(comm);

  int proc_coord[2];
  int dims[2];
  int periods[2];
  grid_mpi_cart_get(comm, 2, dims, periods, proc_coord);

  const int my_sizes[3] = {
      proc2local[my_process][0][1] - proc2local[my_process][0][0] + 1,
      proc2local[my_process][1][1] - proc2local[my_process][1][0] + 1,
      proc2local[my_process][2][1] - proc2local[my_process][2][0] + 1};
  assert(my_sizes[1] == npts_global[1]);
  const int my_sizes_transposed[3] = {
      proc2local_transposed[my_process][0][1] -
          proc2local_transposed[my_process][0][0] + 1,
      proc2local_transposed[my_process][1][1] -
          proc2local_transposed[my_process][1][0] + 1,
      proc2local_transposed[my_process][2][1] -
          proc2local_transposed[my_process][2][0] + 1};
  assert(my_sizes_transposed[2] == npts_global[2]);
  assert(my_sizes[0] == my_sizes_transposed[0]);

  int *send_displacements = calloc(dims[0], sizeof(int));
  int *recv_displacements = calloc(dims[0], sizeof(int));
  int *send_counts = calloc(dims[0], sizeof(int));
  int *recv_counts = calloc(dims[0], sizeof(int));
  double complex *recv_buffer =
      calloc(product3(my_sizes_transposed), sizeof(double complex));

  int send_offset = 0;
  int recv_offset = 0;
  for (int process = 0; process < dims[1]; process++) {
    // Setup arrays
    send_displacements[process] = send_offset;
    recv_displacements[process] = recv_offset;
    int rank;
    grid_mpi_cart_rank(comm, (const int[2]){proc_coord[0], process}, &rank);
    const int current_send_count = my_sizes[0] *
                                   (proc2local_transposed[rank][1][1] -
                                    proc2local_transposed[rank][1][0] + 1) *
                                   my_sizes[2];
    send_counts[process] = current_send_count;
    send_offset += current_send_count;
    const int current_recv_count =
        my_sizes_transposed[0] * my_sizes_transposed[1] *
        (proc2local[rank][2][1] - proc2local[rank][2][0] + 1);
    recv_counts[process] = current_recv_count;
    recv_offset += current_recv_count;
  }
  assert(send_offset == product3(my_sizes));
  assert(recv_offset == product3(my_sizes_transposed));

  // Use collective MPI communication
  grid_mpi_alltoallv_double_complex(grid, send_counts, send_displacements,
                                    recv_buffer, recv_counts,
                                    recv_displacements, sub_comm[1]);

  for (int process = 0; process < dims[1]; process++) {
    int rank;
    grid_mpi_cart_rank(comm, (const int[2]){proc_coord[0], process}, &rank);
// Copy the data to the output array
#pragma omp parallel for collapse(2) default(none)                             \
    shared(my_sizes, my_sizes_transposed, proc2local, proc2local_transposed,   \
               recv_buffer, transposed, recv_displacements, process, rank)
    for (int index_y = 0; index_y < my_sizes_transposed[1]; index_y++) {
      for (int index_x = 0; index_x < my_sizes_transposed[0]; index_x++) {
        memcpy(transposed +
                   (index_y * my_sizes_transposed[0] + index_x) *
                       my_sizes_transposed[2] +
                   proc2local[rank][2][0],
               recv_buffer + recv_displacements[process] +
                   (index_y * my_sizes[0] + index_x) *
                       (proc2local[rank][2][1] - proc2local[rank][2][0] + 1),
               (proc2local[rank][2][1] - proc2local[rank][2][0] + 1) *
                   sizeof(double complex));
      }
    }
  }

  free(recv_buffer);
  free(send_counts);
  free(send_displacements);
  free(recv_counts);
  free(recv_displacements);
}

/*******************************************************************************
 * \brief Performs a transposition of the kind (y_D,z,x_D)->(x,y_D,z_D).
 * \author Frederick Stein
 ******************************************************************************/
void collect_x_and_distribute_z_blocked_transpose(
    const double complex *grid, double complex *transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int (*proc2local_transposed)[3][2], const grid_mpi_comm comm,
    const grid_mpi_comm sub_comm[2]) {
  const int my_process = grid_mpi_comm_rank(comm);

  int proc_coord[2];
  int dims[2];
  int periods[2];
  grid_mpi_cart_get(comm, 2, dims, periods, proc_coord);

  const int my_sizes[3] = {
      proc2local[my_process][0][1] - proc2local[my_process][0][0] + 1,
      proc2local[my_process][1][1] - proc2local[my_process][1][0] + 1,
      proc2local[my_process][2][1] - proc2local[my_process][2][0] + 1};
  assert(my_sizes[2] == npts_global[2]);
  const int my_sizes_transposed[3] = {
      proc2local_transposed[my_process][0][1] -
          proc2local_transposed[my_process][0][0] + 1,
      proc2local_transposed[my_process][1][1] -
          proc2local_transposed[my_process][1][0] + 1,
      proc2local_transposed[my_process][2][1] -
          proc2local_transposed[my_process][2][0] + 1};
  assert(my_sizes_transposed[0] == npts_global[0]);
  assert(my_sizes[1] == my_sizes_transposed[1]);

  int *send_displacements = calloc(dims[1], sizeof(int));
  int *recv_displacements = calloc(dims[1], sizeof(int));
  int *send_counts = calloc(dims[1], sizeof(int));
  int *recv_counts = calloc(dims[1], sizeof(int));
  double complex *send_buffer =
      calloc(product3(my_sizes), sizeof(double complex));

  int send_offset = 0;
  int recv_offset = 0;
  for (int process = 0; process < dims[1]; process++) {
    // Setup arrays
    send_displacements[process] = send_offset;
    recv_displacements[process] = recv_offset;
    int rank;
    grid_mpi_cart_rank(comm, (const int[2]){proc_coord[0], process}, &rank);
    const int current_send_count = my_sizes[0] * my_sizes[1] *
                                   (proc2local_transposed[rank][2][1] -
                                    proc2local_transposed[rank][2][0] + 1);
    send_counts[process] = current_send_count;
    const int current_recv_count =
        (proc2local[rank][0][1] - proc2local[rank][0][0] + 1) *
        my_sizes_transposed[1] * my_sizes_transposed[2];
    recv_counts[process] = current_recv_count;
    send_offset += current_send_count;
    recv_offset += current_recv_count;
// Copy the data to the send buffer and exchange the last two indices
#pragma omp parallel for collapse(2) default(none)                             \
    shared(my_sizes, proc2local_transposed, send_buffer, grid,                 \
               send_displacements, process, rank)
    for (int index_z = 0; index_z < (proc2local_transposed[rank][2][1] -
                                     proc2local_transposed[rank][2][0] + 1);
         index_z++) {
      for (int index_y = 0; index_y < my_sizes[1]; index_y++) {
        for (int index_x = 0; index_x < my_sizes[0]; index_x++) {
          send_buffer[send_displacements[process] +
                      (index_x * (proc2local_transposed[rank][2][1] -
                                  proc2local_transposed[rank][2][0] + 1) +
                       index_z) *
                          my_sizes[1] +
                      index_y] =
              grid[(index_y * my_sizes[2] + proc2local_transposed[rank][2][0] +
                    index_z) *
                       my_sizes[0] +
                   index_x];
        }
      }
    }
  }
  assert(send_offset == product3(my_sizes));
  assert(recv_offset == product3(my_sizes_transposed));

  // Use collective MPI communication
  grid_mpi_alltoallv_double_complex(send_buffer, send_counts,
                                    send_displacements, transposed, recv_counts,
                                    recv_displacements, sub_comm[1]);

  free(send_buffer);
  free(send_counts);
  free(send_displacements);
  free(recv_counts);
  free(recv_displacements);
}

/*******************************************************************************
 * \brief Performs a transposition of the kind (x,y_D,z_D)->(y_D,z,x_D).
 * \author Frederick Stein
 ******************************************************************************/
void collect_z_and_distribute_x_blocked_transpose(
    const double complex *grid, double complex *transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int (*proc2local_transposed)[3][2], const grid_mpi_comm comm,
    const grid_mpi_comm sub_comm[2]) {
  const int my_process = grid_mpi_comm_rank(comm);

  int proc_coord[2];
  int dims[2];
  int periods[2];
  grid_mpi_cart_get(comm, 2, dims, periods, proc_coord);

  const int my_sizes[3] = {
      proc2local[my_process][0][1] - proc2local[my_process][0][0] + 1,
      proc2local[my_process][1][1] - proc2local[my_process][1][0] + 1,
      proc2local[my_process][2][1] - proc2local[my_process][2][0] + 1};
  assert(my_sizes[0] == npts_global[0]);
  const int my_sizes_transposed[3] = {
      proc2local_transposed[my_process][0][1] -
          proc2local_transposed[my_process][0][0] + 1,
      proc2local_transposed[my_process][1][1] -
          proc2local_transposed[my_process][1][0] + 1,
      proc2local_transposed[my_process][2][1] -
          proc2local_transposed[my_process][2][0] + 1};
  assert(my_sizes_transposed[2] == npts_global[2]);
  assert(my_sizes[1] == my_sizes_transposed[1]);

  int *send_displacements = calloc(dims[1], sizeof(int));
  int *recv_displacements = calloc(dims[1], sizeof(int));
  int *send_counts = calloc(dims[1], sizeof(int));
  int *recv_counts = calloc(dims[1], sizeof(int));
  double complex *recv_buffer =
      calloc(product3(my_sizes_transposed), sizeof(double complex));

  int send_offset = 0;
  int recv_offset = 0;
  for (int process = 0; process < dims[1]; process++) {
    // Setup arrays
    send_displacements[process] = send_offset;
    recv_displacements[process] = recv_offset;
    int rank;
    grid_mpi_cart_rank(comm, (const int[2]){proc_coord[0], process}, &rank);
    const int current_recv_count =
        my_sizes_transposed[0] * my_sizes_transposed[1] *
        (proc2local[rank][2][1] - proc2local[rank][2][0] + 1);
    recv_counts[process] = current_recv_count;
    const int current_send_count = (proc2local_transposed[rank][0][1] -
                                    proc2local_transposed[rank][0][0] + 1) *
                                   my_sizes[1] * my_sizes[2];
    send_counts[process] = current_send_count;
    send_offset += current_send_count;
    recv_offset += current_recv_count;
  }
  assert(send_offset == product3(my_sizes));
  assert(recv_offset == product3(my_sizes_transposed));

  // Use collective MPI communication
  grid_mpi_alltoallv_double_complex(grid, send_counts, send_displacements,
                                    recv_buffer, recv_counts,
                                    recv_displacements, sub_comm[1]);

  for (int process = 0; process < dims[1]; process++) {
    int rank;
    grid_mpi_cart_rank(comm, (const int[2]){proc_coord[0], process}, &rank);
// Copy the data to the send buffer and exchange the last two indices
#pragma omp parallel for collapse(2) default(none)                             \
    shared(my_sizes_transposed, proc2local, recv_buffer, transposed,           \
               recv_displacements, process, rank)
    for (int index_z = 0;
         index_z < (proc2local[rank][2][1] - proc2local[rank][2][0] + 1);
         index_z++) {
      for (int index_y = 0; index_y < my_sizes_transposed[1]; index_y++) {
        for (int index_x = 0; index_x < my_sizes_transposed[0]; index_x++) {
          transposed[(index_y * my_sizes_transposed[2] +
                      proc2local[rank][2][0] + index_z) *
                         my_sizes_transposed[0] +
                     index_x] =
              recv_buffer[recv_displacements[process] +
                          (index_x * (proc2local[rank][2][1] -
                                      proc2local[rank][2][0] + 1) +
                           index_z) *
                              my_sizes_transposed[1] +
                          index_y];
        }
      }
    }
  }

  free(recv_buffer);
  free(send_counts);
  free(send_displacements);
  free(recv_counts);
  free(recv_displacements);
}

/*******************************************************************************
 * \brief Performs a transposition of (z,x,y)->(y,z,x).
 * \author Frederick Stein
 ******************************************************************************/
void collect_x_and_distribute_y_blocked(
    const double complex *grid, double complex *transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int (*proc2local_transposed)[3][2], const grid_mpi_comm comm,
    const grid_mpi_comm sub_comm[2]) {
  const int my_process = grid_mpi_comm_rank(comm);

  int proc_coord[2];
  int dims[2];
  int periods[2];
  grid_mpi_cart_get(comm, 2, dims, periods, proc_coord);

  const int my_sizes[3] = {
      proc2local[my_process][0][1] - proc2local[my_process][0][0] + 1,
      proc2local[my_process][1][1] - proc2local[my_process][1][0] + 1,
      proc2local[my_process][2][1] - proc2local[my_process][2][0] + 1};
  assert(my_sizes[1] == npts_global[1]);
  const int my_sizes_transposed[3] = {
      proc2local_transposed[my_process][0][1] -
          proc2local_transposed[my_process][0][0] + 1,
      proc2local_transposed[my_process][1][1] -
          proc2local_transposed[my_process][1][0] + 1,
      proc2local_transposed[my_process][2][1] -
          proc2local_transposed[my_process][2][0] + 1};
  assert(my_sizes_transposed[0] == npts_global[0]);
  assert(my_sizes[2] == my_sizes_transposed[2]);

  int *send_displacements = calloc(dims[0], sizeof(int));
  int *recv_displacements = calloc(dims[0], sizeof(int));
  int *send_counts = calloc(dims[0], sizeof(int));
  int *recv_counts = calloc(dims[0], sizeof(int));
  double complex *send_buffer =
      calloc(product3(my_sizes), sizeof(double complex));

  int send_offset = 0;
  int recv_offset = 0;
  for (int process = 0; process < dims[0]; process++) {
    // Setup arrays
    send_displacements[process] = send_offset;
    recv_displacements[process] = recv_offset;
    int rank;
    grid_mpi_cart_rank(comm, (const int[2]){process, proc_coord[1]}, &rank);
    const int current_send_count = my_sizes[0] *
                                   (proc2local_transposed[rank][1][1] -
                                    proc2local_transposed[rank][1][0] + 1) *
                                   my_sizes[2];
    send_counts[process] = current_send_count;
    send_offset += current_send_count;
    const int current_recv_count =
        (proc2local[rank][0][1] - proc2local[rank][0][0] + 1) *
        my_sizes_transposed[1] * my_sizes_transposed[2];
    recv_counts[process] = current_recv_count;
    recv_offset += current_recv_count;
// Copy the data to the send buffer
#pragma omp parallel for collapse(2) default(none)                             \
    shared(my_sizes, my_sizes_transposed, proc2local, proc2local_transposed,   \
               send_buffer, grid, send_displacements, process, rank)
    for (int index_x = 0; index_x < my_sizes[0]; index_x++) {
      for (int index_z = 0; index_z < my_sizes[2]; index_z++) {
        memcpy(send_buffer + send_displacements[process] +
                   (index_x * my_sizes[2] + index_z) *
                       (proc2local_transposed[rank][1][1] -
                        proc2local_transposed[rank][1][0] + 1),
               grid + (index_x * my_sizes[2] + index_z) * my_sizes[1] +
                   proc2local_transposed[rank][1][0],
               (proc2local_transposed[rank][1][1] -
                proc2local_transposed[rank][1][0] + 1) *
                   sizeof(double complex));
      }
    }
  }
  assert(send_offset == product3(my_sizes));
  assert(recv_offset == product3(my_sizes_transposed));

  // Use collective MPI communication
  grid_mpi_alltoallv_double_complex(send_buffer, send_counts,
                                    send_displacements, transposed, recv_counts,
                                    recv_displacements, sub_comm[0]);

  free(send_buffer);
  free(send_counts);
  free(send_displacements);
  free(recv_counts);
  free(recv_displacements);
}

/*******************************************************************************
 * \brief Performs a transposition of (z,x,y)->(y,z,x).
 * \author Frederick Stein
 ******************************************************************************/
void collect_y_and_distribute_x_blocked(
    const double complex *grid, double complex *transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int (*proc2local_transposed)[3][2], const grid_mpi_comm comm,
    const grid_mpi_comm sub_comm[2]) {
  const int my_process = grid_mpi_comm_rank(comm);

  int proc_coord[2];
  int dims[2];
  int periods[2];
  grid_mpi_cart_get(comm, 2, dims, periods, proc_coord);

  const int my_sizes[3] = {
      proc2local[my_process][0][1] - proc2local[my_process][0][0] + 1,
      proc2local[my_process][1][1] - proc2local[my_process][1][0] + 1,
      proc2local[my_process][2][1] - proc2local[my_process][2][0] + 1};
  assert(my_sizes[0] == npts_global[0]);
  const int my_sizes_transposed[3] = {
      proc2local_transposed[my_process][0][1] -
          proc2local_transposed[my_process][0][0] + 1,
      proc2local_transposed[my_process][1][1] -
          proc2local_transposed[my_process][1][0] + 1,
      proc2local_transposed[my_process][2][1] -
          proc2local_transposed[my_process][2][0] + 1};
  assert(my_sizes_transposed[1] == npts_global[1]);
  assert(my_sizes[2] == my_sizes_transposed[2]);

  int *send_displacements = calloc(dims[0], sizeof(int));
  int *recv_displacements = calloc(dims[0], sizeof(int));
  int *send_counts = calloc(dims[0], sizeof(int));
  int *recv_counts = calloc(dims[0], sizeof(int));
  double complex *recv_buffer =
      calloc(product3(my_sizes_transposed), sizeof(double complex));

  int send_offset = 0;
  int recv_offset = 0;
  for (int process = 0; process < dims[0]; process++) {
    // Setup arrays
    send_displacements[process] = send_offset;
    recv_displacements[process] = recv_offset;
    int rank;
    grid_mpi_cart_rank(comm, (const int[2]){process, proc_coord[1]}, &rank);
    const int current_send_count = (proc2local_transposed[rank][0][1] -
                                    proc2local_transposed[rank][0][0] + 1) *
                                   my_sizes[1] * my_sizes[2];
    send_counts[process] = current_send_count;
    send_offset += current_send_count;
    const int current_recv_count =
        my_sizes_transposed[0] *
        (proc2local[rank][1][1] - proc2local[rank][1][0] + 1) *
        my_sizes_transposed[2];
    recv_counts[process] = current_recv_count;
    recv_offset += current_recv_count;
  }
  assert(send_offset == product3(my_sizes));
  assert(recv_offset == product3(my_sizes_transposed));

  // Use collective MPI communication
  grid_mpi_alltoallv_double_complex(grid, send_counts, send_displacements,
                                    recv_buffer, recv_counts,
                                    recv_displacements, sub_comm[0]);

  for (int process = 0; process < dims[0]; process++) {
    int rank;
    grid_mpi_cart_rank(comm, (const int[2]){process, proc_coord[1]}, &rank);
// Copy the data to the output array
#pragma omp parallel for collapse(2) default(none)                             \
    shared(my_sizes, my_sizes_transposed, proc2local, proc2local_transposed,   \
               recv_buffer, transposed, recv_displacements, process, rank)
    for (int index_x = 0; index_x < my_sizes_transposed[0]; index_x++) {
      for (int index_z = 0; index_z < my_sizes_transposed[2]; index_z++) {
        memcpy(transposed +
                   (index_x * my_sizes_transposed[2] + index_z) *
                       my_sizes_transposed[1] +
                   proc2local[rank][1][0],
               recv_buffer + recv_displacements[process] +
                   (index_x * my_sizes[2] + index_z) *
                       (proc2local[rank][1][1] - proc2local[rank][1][0] + 1),
               (proc2local[rank][1][1] - proc2local[rank][1][0] + 1) *
                   sizeof(double complex));
      }
    }
  }

  free(recv_buffer);
  free(send_counts);
  free(send_displacements);
  free(recv_counts);
  free(recv_displacements);
}

/*******************************************************************************
 * \brief Performs a redistribution (x_D, z_D, y) -> (x, zy_D).
 * \author Frederick Stein
 ******************************************************************************/
void collect_x_and_distribute_yz_ray(const double complex *grid,
                                     double complex *transposed,
                                     const int npts_global[3],
                                     const int (*proc2local)[3][2],
                                     const int *number_of_rays,
                                     const int (*ray_to_yz)[2],
                                     const grid_mpi_comm comm) {
  const int number_of_processes = grid_mpi_comm_size(comm);
  const int my_process = grid_mpi_comm_rank(comm);

  int my_ray_offset = 0;
  for (int process = 0; process < my_process; process++)
    my_ray_offset += number_of_rays[process];
  const int my_number_of_rays = number_of_rays[my_process];
  const int(*my_bounds)[2] = proc2local[my_process];
  const int my_sizes[3] = {my_bounds[0][1] - my_bounds[0][0] + 1,
                           my_bounds[1][1] - my_bounds[1][0] + 1,
                           my_bounds[2][1] - my_bounds[2][0] + 1};
  assert(my_sizes[1] == npts_global[1]);

  double complex *recv_buffer =
      malloc(my_number_of_rays * npts_global[0] * sizeof(double complex));
  double complex *send_buffer =
      malloc(product3(my_sizes) * sizeof(double complex));
  grid_mpi_request recv_request = grid_mpi_request_null,
                   send_request = grid_mpi_request_null;

  memset(transposed, 0,
         my_number_of_rays * npts_global[0] * sizeof(double complex));

// Copy and transpose the local data
#pragma omp parallel for default(none)                                         \
    shared(my_sizes, my_bounds, my_number_of_rays, grid, ray_to_yz,            \
               transposed, my_ray_offset, npts_global) collapse(2)
  for (int index_x = my_bounds[0][0]; index_x <= my_bounds[0][1]; index_x++) {
    for (int yz_ray = 0; yz_ray < my_number_of_rays; yz_ray++) {
      const int index_y = ray_to_yz[my_ray_offset + yz_ray][0];
      const int index_z = ray_to_yz[my_ray_offset + yz_ray][1];

      // Check whether we carry that ray after the transposition
      if (index_z >= my_bounds[2][0] && index_z <= my_bounds[2][1]) {
        // Copy the data
        transposed[index_x * my_number_of_rays + yz_ray] =
            grid[(index_x - my_bounds[0][0]) * my_sizes[1] * my_sizes[2] +
                 (index_z - my_bounds[2][0]) * my_sizes[1] + index_y];
      }
    }
  }

  for (int process_shift = 1; process_shift < number_of_processes;
       process_shift++) {
    const int send_process =
        modulo(my_process + process_shift, number_of_processes);
    const int recv_process =
        modulo(my_process - process_shift, number_of_processes);

    const int(*proc2local_recv)[2] = proc2local[recv_process];

    int number_of_rays_to_recv = 0;
#pragma omp parallel for default(none)                                         \
    shared(my_number_of_rays, my_ray_offset, number_of_rays, ray_to_yz,        \
               proc2local_recv) reduction(+ : number_of_rays_to_recv)
    for (int ray = my_ray_offset; ray < my_ray_offset + my_number_of_rays;
         ray++) {
      const int index_z = ray_to_yz[ray][1];
      if (index_z >= proc2local_recv[2][0] &&
          index_z <= proc2local_recv[2][1]) {
        number_of_rays_to_recv++;
      }
    }
    const int number_of_elements_to_recv =
        number_of_rays_to_recv *
        (proc2local_recv[0][1] - proc2local_recv[0][0] + 1);
    memset(recv_buffer, 0, number_of_elements_to_recv * sizeof(double complex));

    // Post receive request
    grid_mpi_irecv_double_complex(recv_buffer, number_of_elements_to_recv,
                                  recv_process, 1, comm, &recv_request);

    const int number_of_rays_send = number_of_rays[send_process];
    const int(*send_rays)[2] = ray_to_yz;
    for (int process = 0; process < send_process; process++)
      send_rays += number_of_rays[process];
    int number_of_rays_to_send = 0;
    for (int ray = 0; ray < number_of_rays_send; ray++) {
      const int index_z = send_rays[ray][1];
      if (index_z >= my_bounds[2][0] && index_z <= my_bounds[2][1]) {
        number_of_rays_to_send++;
      }
    }
    const int number_of_elements_to_send = number_of_rays_to_send * my_sizes[0];
    memset(send_buffer, 0, number_of_elements_to_send * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(my_sizes, my_bounds, number_of_rays_to_send, send_buffer, grid,     \
               send_rays, number_of_rays_send)
    for (int index_x = 0; index_x < my_sizes[0]; index_x++) {
      int ray_position = 0;
      for (int ray = 0; ray < number_of_rays_send; ray++) {
        const int index_y = send_rays[ray][0];
        const int index_z = send_rays[ray][1];
        if (index_z >= my_bounds[2][0] && index_z <= my_bounds[2][1]) {
          send_buffer[index_x * number_of_rays_to_send + ray_position] =
              grid[index_y + (index_z - my_bounds[2][0]) * my_sizes[1] +
                   index_x * my_sizes[1] * my_sizes[2]];
          ray_position++;
        }
      }
      assert(ray_position == number_of_rays_to_send);
    }

    // Post send request
    grid_mpi_isend_double_complex(send_buffer, number_of_elements_to_send,
                                  send_process, 1, comm, &send_request);

    // Wait for the receive process and copy the data
    grid_mpi_wait(&recv_request);

#pragma omp parallel for default(none) shared(                                 \
        my_number_of_rays, npts_global, recv_buffer, transposed,               \
            proc2local_recv, number_of_rays_to_recv, ray_to_yz, my_ray_offset)
    for (int index_x = 0;
         index_x < proc2local_recv[0][1] - proc2local_recv[0][0] + 1;
         index_x++) {
      int ray_position = 0;
      for (int ray = my_ray_offset; ray < my_ray_offset + my_number_of_rays;
           ray++) {
        const int index_z = ray_to_yz[ray][1];
        if (index_z >= proc2local_recv[2][0] &&
            index_z <= proc2local_recv[2][1]) {
          transposed[(index_x + proc2local_recv[0][0]) * my_number_of_rays +
                     (ray - my_ray_offset)] =
              recv_buffer[index_x * number_of_rays_to_recv + ray_position];
          ray_position++;
        }
      }
      assert(ray_position == number_of_rays_to_recv);
    }

    // Wait for the send request
    grid_mpi_wait(&send_request);
  }

  free(recv_buffer);
  free(send_buffer);
}

/*******************************************************************************
 * \brief Performs a transposition of (x, zy_D) -> (x_D, z_D, y).
 * \author Frederick Stein
 ******************************************************************************/
void collect_yz_and_distribute_x_ray(const double complex *grid,
                                     double complex *transposed,
                                     const int npts_global[3],
                                     const int (*proc2local_transposed)[3][2],
                                     const int *number_of_rays,
                                     const int (*ray_to_yz)[2],
                                     const grid_mpi_comm comm) {
  const int number_of_processes = grid_mpi_comm_size(comm);
  const int my_process = grid_mpi_comm_rank(comm);

  if (my_process == 0) {
    for (int process = 0; process < number_of_processes; process++) {
      printf("proc2local_transposed[%d] = (%d,%d,%d) -> (%d,%d,%d)\n", process,
             proc2local_transposed[process][0][0],
             proc2local_transposed[process][1][0],
             proc2local_transposed[process][2][0],
             proc2local_transposed[process][0][1],
             proc2local_transposed[process][1][1],
             proc2local_transposed[process][2][1]);
    }
    int offset = 0;
    for (int process = 0; process < number_of_processes; process++) {
      for (int ray = offset; ray < offset + number_of_rays[process]; ray++) {
        printf("ray_to_yz %d (%d,%d): %i\n", ray, ray_to_yz[ray][0],
               ray_to_yz[ray][1], process);
      }
      offset += number_of_rays[process];
    }
    fflush(stdout);
  }
  grid_mpi_barrier(comm);

  int max_number_of_rays = 0;
  for (int process = 0; process < number_of_processes; process++)
    max_number_of_rays = imax(max_number_of_rays, number_of_rays[process]);

  const int(*my_bounds)[2] = proc2local_transposed[my_process];
  int my_transposed_sizes[3];
  for (int dir = 0; dir < 3; dir++)
    my_transposed_sizes[dir] = my_bounds[dir][1] - my_bounds[dir][0] + 1;
  assert(my_transposed_sizes[1] == npts_global[1]);
  const int max_number_of_elements =
      imax(max_number_of_rays * npts_global[0], product3(my_transposed_sizes));
  const int my_number_of_rays = number_of_rays[my_process];

  double complex *recv_buffer =
      malloc(max_number_of_elements * sizeof(double complex));
  double complex *send_buffer =
      malloc(max_number_of_elements * sizeof(double complex));
  grid_mpi_request recv_request = grid_mpi_request_null,
                   send_request = grid_mpi_request_null;

  memset(transposed, 0, product3(my_transposed_sizes) * sizeof(double complex));

  // Copy and transpose the local data
  int number_of_received_rays = 0;
  const int(*my_rays)[2] = ray_to_yz;
  for (int process = 0; process < my_process; process++)
    my_rays += number_of_rays[process];
#pragma omp parallel for default(none)                                         \
    shared(my_transposed_sizes, my_bounds, my_rays, my_number_of_rays, grid,   \
               transposed, npts_global) reduction(+ : number_of_received_rays)
  for (int yz_ray = 0; yz_ray < my_number_of_rays; yz_ray++) {
    const int index_y = my_rays[yz_ray][0];
    const int index_z = my_rays[yz_ray][1];

    // Check whether we carry that ray after the transposition
    if (index_z < my_bounds[2][0] || index_z > my_bounds[2][1])
      continue;

    // Copy the data
    for (int index_x = my_bounds[0][0]; index_x <= my_bounds[0][1]; index_x++) {
      transposed[(index_x - my_bounds[0][0]) * my_transposed_sizes[1] *
                     my_transposed_sizes[2] +
                 (index_z - my_bounds[2][0]) * my_transposed_sizes[1] +
                 (index_y - my_bounds[1][0])] =
          grid[index_x * my_number_of_rays + yz_ray];
    }
    number_of_received_rays++;
  }

  for (int process_shift = 1; process_shift < number_of_processes;
       process_shift++) {
    const int send_process =
        modulo(my_process + process_shift, number_of_processes);
    const int recv_process =
        modulo(my_process - process_shift, number_of_processes);

    int number_of_rays_to_recv = 0;
    const int(*recv_rays)[2] = ray_to_yz;
    const int number_of_rays_recv = number_of_rays[recv_process];
    for (int process = 0; process < recv_process; process++)
      recv_rays += number_of_rays[process];
#pragma omp parallel for default(none)                                         \
    shared(number_of_rays_recv, recv_rays, proc2local_transposed, my_bounds)   \
    reduction(+ : number_of_rays_to_recv)
    for (int ray = 0; ray < number_of_rays_recv; ray++) {
      const int index_z = recv_rays[ray][1];
      if (index_z >= my_bounds[2][0] && index_z <= my_bounds[2][1]) {
        number_of_rays_to_recv++;
      }
    }
    memset(recv_buffer, 0, max_number_of_elements * sizeof(double complex));

    // Post receive request
    grid_mpi_irecv_double_complex(
        recv_buffer, my_transposed_sizes[0] * number_of_rays_to_recv,
        recv_process, 1, comm, &recv_request);

    memset(send_buffer, 0, max_number_of_elements * sizeof(double complex));
    const int(*proc2local_send)[2] = proc2local_transposed[send_process];
    int number_of_rays_to_send = 0;
#pragma omp parallel for default(none)                                         \
    shared(my_number_of_rays, my_rays, proc2local_send)                        \
    reduction(+ : number_of_rays_to_send)
    for (int ray = 0; ray < my_number_of_rays; ray++) {
      const int index_z = my_rays[ray][1];
      if (index_z >= proc2local_send[2][0] &&
          index_z <= proc2local_send[2][1]) {
        number_of_rays_to_send++;
      }
    }
#pragma omp parallel for default(none)                                         \
    shared(npts_global, number_of_rays_to_send, send_buffer, grid, my_rays,    \
               proc2local_send, my_number_of_rays)
    for (int index_x = proc2local_send[0][0]; index_x <= proc2local_send[0][1];
         index_x++) {
      int ray_position = 0;
      for (int ray = 0; ray < my_number_of_rays; ray++) {
        const int index_z = my_rays[ray][1];
        if (index_z >= proc2local_send[2][0] &&
            index_z <= proc2local_send[2][1]) {
          send_buffer[(index_x - proc2local_send[0][0]) *
                          number_of_rays_to_send +
                      ray_position] = grid[index_x * my_number_of_rays + ray];
          ray_position++;
        }
      }
      assert(ray_position == number_of_rays_to_send);
    }

    // Post send request
    grid_mpi_isend_double_complex(
        send_buffer,
        number_of_rays_to_send *
            (proc2local_send[0][1] - proc2local_send[0][0] + 1),
        send_process, 1, comm, &send_request);

    // Wait for the receive process and copy the data
    printf("%i Waiting for receive process %d\n", my_process, recv_process);
    fflush(stdout);
    grid_mpi_wait(&recv_request);
    printf("%i Received data from process %d\n", my_process, recv_process);
    fflush(stdout);

#pragma omp parallel for default(none) shared(                                 \
        my_transposed_sizes, my_bounds, number_of_rays_to_recv, recv_buffer,   \
            transposed, number_of_rays_recv, recv_rays, number_of_rays)
    for (int index_x = 0; index_x < my_transposed_sizes[0]; index_x++) {
      int ray_position = 0;
      for (int ray = 0; ray < number_of_rays_recv; ray++) {
        const int index_y = recv_rays[ray][0];
        const int index_z = recv_rays[ray][1];
        if (index_z >= my_bounds[2][0] && index_z <= my_bounds[2][1]) {
          transposed[index_x * my_transposed_sizes[1] * my_transposed_sizes[2] +
                     (index_z - my_bounds[2][0]) * my_transposed_sizes[1] +
                     index_y] =
              recv_buffer[index_x * number_of_rays_to_recv + ray_position];
          ray_position++;
        }
      }
      assert(ray_position == number_of_rays_to_recv);
    }

    // Wait for the send request
    printf("%i Waiting for send process %d\n", my_process, send_process);
    fflush(stdout);
    grid_mpi_wait(&send_request);
    printf("%i Sent data to process %d\n", my_process, send_process);
    fflush(stdout);
  }

  free(recv_buffer);
  free(send_buffer);
}

/*******************************************************************************
 * \brief Performs a redistribution (x_D, z_D, y) -> (y_D, z, x_D).
 * \author Frederick Stein
 ******************************************************************************/
void collect_x_and_distribute_yz_ray_transpose(const double complex *grid,
                                               double complex *transposed,
                                               const int npts_global[3],
                                               const int (*proc2local)[3][2],
                                               const int *number_of_rays,
                                               const int (*ray_to_yz)[2],
                                               const grid_mpi_comm comm) {
  const int number_of_processes = grid_mpi_comm_size(comm);
  const int my_process = grid_mpi_comm_rank(comm);

  int my_ray_offset = 0;
  for (int process = 0; process < my_process; process++)
    my_ray_offset += number_of_rays[process];
  const int my_number_of_rays = number_of_rays[my_process];
  const int(*my_bounds)[2] = proc2local[my_process];
  const int my_sizes[3] = {my_bounds[0][1] - my_bounds[0][0] + 1,
                           my_bounds[1][1] - my_bounds[1][0] + 1,
                           my_bounds[2][1] - my_bounds[2][0] + 1};
  assert(my_sizes[2] == npts_global[2]);

  double complex *recv_buffer =
      malloc(my_number_of_rays * npts_global[0] * sizeof(double complex));
  double complex *send_buffer =
      malloc(product3(my_sizes) * sizeof(double complex));
  grid_mpi_request recv_request = grid_mpi_request_null,
                   send_request = grid_mpi_request_null;

  memset(transposed, 0,
         my_number_of_rays * npts_global[0] * sizeof(double complex));

// Copy and transpose the local data
#pragma omp parallel for default(none)                                         \
    shared(my_sizes, my_bounds, my_number_of_rays, grid, ray_to_yz,            \
               transposed, my_ray_offset, npts_global) collapse(2)
  for (int index_x = my_bounds[0][0]; index_x <= my_bounds[0][1]; index_x++) {
    for (int yz_ray = 0; yz_ray < my_number_of_rays; yz_ray++) {
      const int index_y = ray_to_yz[my_ray_offset + yz_ray][0];
      const int index_z = ray_to_yz[my_ray_offset + yz_ray][1];

      // Check whether we carry that ray after the transposition
      if (index_y >= my_bounds[1][0] && index_y <= my_bounds[1][1]) {
        // Copy the data
        transposed[index_x * my_number_of_rays + yz_ray] =
            grid[(index_y - my_bounds[1][0]) * my_sizes[0] * my_sizes[2] +
                 (index_z - my_bounds[2][0]) * my_sizes[0] + index_x -
                 my_bounds[0][0]];
      }
    }
  }

  for (int process_shift = 1; process_shift < number_of_processes;
       process_shift++) {
    const int send_process =
        modulo(my_process + process_shift, number_of_processes);
    const int recv_process =
        modulo(my_process - process_shift, number_of_processes);

    const int(*proc2local_recv)[2] = proc2local[recv_process];

    int number_of_rays_to_recv = 0;
#pragma omp parallel for default(none)                                         \
    shared(my_number_of_rays, my_ray_offset, number_of_rays, ray_to_yz,        \
               proc2local_recv) reduction(+ : number_of_rays_to_recv)
    for (int ray = my_ray_offset; ray < my_ray_offset + my_number_of_rays;
         ray++) {
      const int index_y = ray_to_yz[ray][0];
      if (index_y >= proc2local_recv[1][0] &&
          index_y <= proc2local_recv[1][1]) {
        number_of_rays_to_recv++;
      }
    }
    const int number_of_elements_to_recv =
        number_of_rays_to_recv *
        (proc2local_recv[0][1] - proc2local_recv[0][0] + 1);
    memset(recv_buffer, 0, number_of_elements_to_recv * sizeof(double complex));

    // Post receive request
    grid_mpi_irecv_double_complex(recv_buffer, number_of_elements_to_recv,
                                  recv_process, 1, comm, &recv_request);

    const int number_of_rays_send = number_of_rays[send_process];
    const int(*send_rays)[2] = ray_to_yz;
    for (int process = 0; process < send_process; process++)
      send_rays += number_of_rays[process];
    int number_of_rays_to_send = 0;
    for (int ray = 0; ray < number_of_rays_send; ray++) {
      const int index_y = send_rays[ray][0];
      if (index_y >= my_bounds[1][0] && index_y <= my_bounds[1][1]) {
        number_of_rays_to_send++;
      }
    }
    const int number_of_elements_to_send = number_of_rays_to_send * my_sizes[0];
    memset(send_buffer, 0, number_of_elements_to_send * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(my_sizes, my_bounds, number_of_rays_to_send, send_buffer, grid,     \
               send_rays, number_of_rays_send)
    for (int index_x = 0; index_x < my_sizes[0]; index_x++) {
      int ray_position = 0;
      for (int ray = 0; ray < number_of_rays_send; ray++) {
        const int index_y = send_rays[ray][0];
        const int index_z = send_rays[ray][1];
        if (index_y >= my_bounds[1][0] && index_y <= my_bounds[1][1]) {
          send_buffer[index_x * number_of_rays_to_send + ray_position] =
              grid[index_x + (index_z - my_bounds[2][0]) * my_sizes[0] +
                   (index_y - my_bounds[1][0]) * my_sizes[0] * my_sizes[2]];
          ray_position++;
        }
      }
      assert(ray_position == number_of_rays_to_send);
    }

    // Post send request
    grid_mpi_isend_double_complex(send_buffer, number_of_elements_to_send,
                                  send_process, 1, comm, &send_request);

    // Wait for the receive process and copy the data
    grid_mpi_wait(&recv_request);

#pragma omp parallel for default(none) shared(                                 \
        my_number_of_rays, npts_global, recv_buffer, transposed,               \
            proc2local_recv, number_of_rays_to_recv, ray_to_yz, my_ray_offset)
    for (int index_x = 0;
         index_x < proc2local_recv[0][1] - proc2local_recv[0][0] + 1;
         index_x++) {
      int ray_position = 0;
      for (int ray = my_ray_offset; ray < my_ray_offset + my_number_of_rays;
           ray++) {
        const int index_y = ray_to_yz[ray][0];
        if (index_y >= proc2local_recv[1][0] &&
            index_y <= proc2local_recv[1][1]) {
          transposed[(index_x + proc2local_recv[0][0]) * my_number_of_rays +
                     (ray - my_ray_offset)] =
              recv_buffer[index_x * number_of_rays_to_recv + ray_position];
          ray_position++;
        }
      }
      assert(ray_position == number_of_rays_to_recv);
    }

    // Wait for the send request
    grid_mpi_wait(&send_request);
  }

  free(recv_buffer);
  free(send_buffer);
}

/*******************************************************************************
 * \brief Performs a transposition of (x, zy_D) -> (x_D, z_D, y).
 * \author Frederick Stein
 ******************************************************************************/
void collect_yz_and_distribute_x_ray_transpose(
    const double complex *grid, double complex *transposed,
    const int npts_global[3], const int (*proc2local_transposed)[3][2],
    const int *number_of_rays, const int (*ray_to_yz)[2],
    const grid_mpi_comm comm) {
  const int number_of_processes = grid_mpi_comm_size(comm);
  const int my_process = grid_mpi_comm_rank(comm);

  if (my_process == 0) {
    for (int process = 0; process < number_of_processes; process++) {
      printf("proc2local_transposed[%d] = (%d,%d,%d) -> (%d,%d,%d)\n", process,
             proc2local_transposed[process][0][0],
             proc2local_transposed[process][1][0],
             proc2local_transposed[process][2][0],
             proc2local_transposed[process][0][1],
             proc2local_transposed[process][1][1],
             proc2local_transposed[process][2][1]);
    }
    int offset = 0;
    for (int process = 0; process < number_of_processes; process++) {
      for (int ray = offset; ray < offset + number_of_rays[process]; ray++) {
        printf("ray_to_yz %d (%d,%d): %i\n", ray, ray_to_yz[ray][0],
               ray_to_yz[ray][1], process);
      }
      offset += number_of_rays[process];
    }
    fflush(stdout);
  }
  grid_mpi_barrier(comm);

  int max_number_of_rays = 0;
  for (int process = 0; process < number_of_processes; process++)
    max_number_of_rays = imax(max_number_of_rays, number_of_rays[process]);

  const int(*my_bounds)[2] = proc2local_transposed[my_process];
  int my_transposed_sizes[3];
  for (int dir = 0; dir < 3; dir++)
    my_transposed_sizes[dir] = my_bounds[dir][1] - my_bounds[dir][0] + 1;
  assert(my_transposed_sizes[2] == npts_global[2]);
  const int max_number_of_elements =
      imax(max_number_of_rays * npts_global[0], product3(my_transposed_sizes));
  const int my_number_of_rays = number_of_rays[my_process];

  double complex *recv_buffer =
      malloc(max_number_of_elements * sizeof(double complex));
  double complex *send_buffer =
      malloc(max_number_of_elements * sizeof(double complex));
  grid_mpi_request recv_request = grid_mpi_request_null,
                   send_request = grid_mpi_request_null;

  memset(transposed, 0, product3(my_transposed_sizes) * sizeof(double complex));

  // Copy and transpose the local data
  int number_of_received_rays = 0;
  const int(*my_rays)[2] = ray_to_yz;
  for (int process = 0; process < my_process; process++)
    my_rays += number_of_rays[process];
#pragma omp parallel for default(none)                                         \
    shared(my_transposed_sizes, my_bounds, my_rays, my_number_of_rays, grid,   \
               transposed, npts_global) reduction(+ : number_of_received_rays)
  for (int yz_ray = 0; yz_ray < my_number_of_rays; yz_ray++) {
    const int index_y = my_rays[yz_ray][0];
    const int index_z = my_rays[yz_ray][1];

    // Check whether we carry that ray after the transposition
    if (index_y < my_bounds[1][0] || index_y > my_bounds[1][1])
      continue;

    // Copy the data
    for (int index_x = my_bounds[0][0]; index_x <= my_bounds[0][1]; index_x++) {
      transposed[(index_y - my_bounds[1][0]) * my_transposed_sizes[0] *
                     my_transposed_sizes[2] +
                 (index_z - my_bounds[2][0]) * my_transposed_sizes[0] +
                 (index_x - my_bounds[0][0])] =
          grid[index_x * my_number_of_rays + yz_ray];
    }
    number_of_received_rays++;
  }

  for (int process_shift = 1; process_shift < number_of_processes;
       process_shift++) {
    const int send_process =
        modulo(my_process + process_shift, number_of_processes);
    const int recv_process =
        modulo(my_process - process_shift, number_of_processes);

    int number_of_rays_to_recv = 0;
    const int(*recv_rays)[2] = ray_to_yz;
    const int number_of_rays_recv = number_of_rays[recv_process];
    for (int process = 0; process < recv_process; process++)
      recv_rays += number_of_rays[process];
#pragma omp parallel for default(none)                                         \
    shared(number_of_rays_recv, recv_rays, proc2local_transposed, my_bounds)   \
    reduction(+ : number_of_rays_to_recv)
    for (int ray = 0; ray < number_of_rays_recv; ray++) {
      const int index_y = recv_rays[ray][0];
      if (index_y >= my_bounds[1][0] && index_y <= my_bounds[1][1]) {
        number_of_rays_to_recv++;
      }
    }
    memset(recv_buffer, 0, max_number_of_elements * sizeof(double complex));

    // Post receive request
    grid_mpi_irecv_double_complex(
        recv_buffer, my_transposed_sizes[0] * number_of_rays_to_recv,
        recv_process, 1, comm, &recv_request);

    memset(send_buffer, 0, max_number_of_elements * sizeof(double complex));
    const int(*proc2local_send)[2] = proc2local_transposed[send_process];
    int number_of_rays_to_send = 0;
#pragma omp parallel for default(none)                                         \
    shared(my_number_of_rays, my_rays, proc2local_send)                        \
    reduction(+ : number_of_rays_to_send)
    for (int ray = 0; ray < my_number_of_rays; ray++) {
      const int index_y = my_rays[ray][0];
      if (index_y >= proc2local_send[1][0] &&
          index_y <= proc2local_send[1][1]) {
        number_of_rays_to_send++;
      }
    }
#pragma omp parallel for default(none)                                         \
    shared(npts_global, number_of_rays_to_send, send_buffer, grid, my_rays,    \
               proc2local_send, my_number_of_rays)
    for (int index_x = proc2local_send[0][0]; index_x <= proc2local_send[0][1];
         index_x++) {
      int ray_position = 0;
      for (int ray = 0; ray < my_number_of_rays; ray++) {
        const int index_y = my_rays[ray][0];
        if (index_y >= proc2local_send[1][0] &&
            index_y <= proc2local_send[1][1]) {
          send_buffer[(index_x - proc2local_send[0][0]) *
                          number_of_rays_to_send +
                      ray_position] = grid[index_x * my_number_of_rays + ray];
          ray_position++;
        }
      }
      assert(ray_position == number_of_rays_to_send);
    }

    // Post send request
    grid_mpi_isend_double_complex(
        send_buffer,
        number_of_rays_to_send *
            (proc2local_send[0][1] - proc2local_send[0][0] + 1),
        send_process, 1, comm, &send_request);

    // Wait for the receive process and copy the data
    printf("%i Waiting for receive process %d\n", my_process, recv_process);
    fflush(stdout);
    grid_mpi_wait(&recv_request);
    printf("%i Received data from process %d\n", my_process, recv_process);
    fflush(stdout);

#pragma omp parallel for default(none) shared(                                 \
        my_transposed_sizes, my_bounds, number_of_rays_to_recv, recv_buffer,   \
            transposed, number_of_rays_recv, recv_rays, number_of_rays)
    for (int index_x = 0; index_x < my_transposed_sizes[0]; index_x++) {
      int ray_position = 0;
      for (int ray = 0; ray < number_of_rays_recv; ray++) {
        const int index_y = recv_rays[ray][0];
        const int index_z = recv_rays[ray][1];
        if (index_y >= my_bounds[1][0] && index_y <= my_bounds[1][1]) {
          transposed[(index_y - my_bounds[1][0]) * my_transposed_sizes[0] *
                         my_transposed_sizes[2] +
                     (index_z - my_bounds[2][0]) * my_transposed_sizes[0] +
                     index_x] =
              recv_buffer[index_x * number_of_rays_to_recv + ray_position];
          ray_position++;
        }
      }
      assert(ray_position == number_of_rays_to_recv);
    }

    // Wait for the send request
    printf("%i Waiting for send process %d\n", my_process, send_process);
    fflush(stdout);
    grid_mpi_wait(&send_request);
    printf("%i Sent data to process %d\n", my_process, send_process);
    fflush(stdout);
  }

  free(recv_buffer);
  free(send_buffer);
}

// EOF
