/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef GRID_MPI_H
#define GRID_MPI_H

#include <complex.h>
#include <stdbool.h>

#if defined(__parallel)
#include <mpi.h>

typedef MPI_Comm grid_mpi_comm;
typedef MPI_Request grid_mpi_request;
typedef MPI_Fint grid_mpi_fint;

static const grid_mpi_comm grid_mpi_comm_world = MPI_COMM_WORLD;
static const grid_mpi_comm grid_mpi_comm_null = MPI_COMM_NULL;
static const grid_mpi_comm grid_mpi_comm_self = MPI_COMM_SELF;
static const grid_mpi_request grid_mpi_request_null = MPI_REQUEST_NULL;
static const int grid_mpi_any_source = MPI_ANY_SOURCE;
static const int grid_mpi_proc_null = MPI_PROC_NULL;
static const int grid_mpi_any_tag = MPI_ANY_TAG;
#else
typedef int grid_mpi_comm;
typedef int grid_mpi_request;
typedef int grid_mpi_fint;

static const grid_mpi_comm grid_mpi_comm_world = -2;
static const grid_mpi_comm grid_mpi_comm_null = -3;
static const grid_mpi_comm grid_mpi_comm_self = -5;
static const grid_mpi_request grid_mpi_request_null = -7;
static const int grid_mpi_any_source = -11;
static const int grid_mpi_proc_null = -13;
static const int grid_mpi_any_tag = -17;
#endif

void grid_mpi_init(int *argc, char ***argv);

void grid_mpi_finalize(void);

int grid_mpi_comm_size(const grid_mpi_comm comm);

int grid_mpi_comm_rank(const grid_mpi_comm comm);

void grid_mpi_cart_get(const grid_mpi_comm comm, int maxdims, int *dims,
                       int *periods, int *coords);

void grid_mpi_cart_rank(const grid_mpi_comm comm, const int *coords, int *rank);

grid_mpi_comm grid_mpi_comm_f2c(const grid_mpi_fint fortran_comm);

grid_mpi_fint grid_mpi_comm_c2f(const grid_mpi_comm comm);

void grid_mpi_comm_dup(const grid_mpi_comm old_comm, grid_mpi_comm *new_comm);

void grid_mpi_comm_free(grid_mpi_comm *comm);

void grid_mpi_barrier(const grid_mpi_comm comm);

bool grid_mpi_comm_is_unequal(const grid_mpi_comm comm1,
                              const grid_mpi_comm comm2);

bool grid_mpi_comm_is_similar(const grid_mpi_comm comm1,
                              const grid_mpi_comm comm2);

bool grid_mpi_comm_is_congruent(const grid_mpi_comm comm1,
                                const grid_mpi_comm comm2);

bool grid_mpi_comm_is_ident(const grid_mpi_comm comm1,
                            const grid_mpi_comm comm2);

void grid_mpi_sendrecv_double(const double *sendbuffer, const int sendcount,
                              const int dest, const int sendtag,
                              double *recvbuffer, const int recvcount,
                              const int source, const int recvtag,
                              const grid_mpi_comm comm);

void grid_mpi_sendrecv_int(const int *sendbuffer, const int sendcount,
                           const int dest, const int sendtag, int *recvbuffer,
                           const int recvcount, const int source,
                           const int recvtag, const grid_mpi_comm comm);

void grid_mpi_isend_double(const double *sendbuffer, const int sendcount,
                           const int dest, const int sendtag,
                           const grid_mpi_comm comm, grid_mpi_request *request);

void grid_mpi_irecv_double(double *recvbuffer, const int recvcount,
                           const int source, const int recvtag,
                           const grid_mpi_comm comm, grid_mpi_request *request);

void grid_mpi_isend_double_complex(const double complex *sendbuffer,
                                   const int sendcount, const int dest,
                                   const int sendtag, const grid_mpi_comm comm,
                                   grid_mpi_request *request);

void grid_mpi_irecv_double_complex(double complex *recvbuffer,
                                   const int recvcount, const int source,
                                   const int recvtag, const grid_mpi_comm comm,
                                   grid_mpi_request *request);

void grid_mpi_wait(grid_mpi_request *request);

void grid_mpi_waitany(const int number_of_requests,
                      grid_mpi_request request[number_of_requests], int *idx);

void grid_mpi_waitall(const int number_of_requests,
                      grid_mpi_request request[number_of_requests]);

void grid_mpi_allgather_int(const int *sendbuffer, int sendcount,
                            int *recvbuffer, grid_mpi_comm comm);

void grid_mpi_sum_double(double *buffer, const int count,
                         const grid_mpi_comm comm);

void grid_mpi_sum_int(int *buffer, const int count, const grid_mpi_comm comm);

void grid_mpi_max_double(double *buffer, const int count,
                         const grid_mpi_comm comm);

void grid_mpi_alltoallv_double_complex(const double complex *send_buffer,
                                       const int *send_counts,
                                       const int *send_displacements,
                                       double complex *recv_buffer,
                                       const int *recv_count,
                                       const int *recv_displacements,
                                       const grid_mpi_comm comm);

void grid_mpi_dims_create(const int number_of_processes,
                          const int number_of_dimensions, int *dimensions);

void grid_mpi_cart_create(const grid_mpi_comm comm_old, const int ndims,
                          const int *dims, const int *periods,
                          const int reorder, grid_mpi_comm *comm_cart);

void grid_mpi_cart_sub(const grid_mpi_comm comm_old, const int *remain_dims,
                       grid_mpi_comm *sub_comm);

void grid_mpi_cart_coords(const grid_mpi_comm comm, const int rank, int maxdims,
                          int coords[]);

#endif

// EOF