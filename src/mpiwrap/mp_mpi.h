/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef mp_mpi_H
#define mp_mpi_H

#include <complex.h>
#include <stdbool.h>

#if defined(__parallel)
#include <mpi.h>

typedef MPI_Comm mp_mpi_comm;
typedef MPI_Request mp_mpi_request;
typedef MPI_Fint mp_mpi_fint;

static const mp_mpi_comm mp_mpi_comm_world = MPI_COMM_WORLD;
static const mp_mpi_comm mp_mpi_comm_null = MPI_COMM_NULL;
static const mp_mpi_comm mp_mpi_comm_self = MPI_COMM_SELF;
static const mp_mpi_request mp_mpi_request_null = MPI_REQUEST_NULL;
static const int mp_mpi_any_source = MPI_ANY_SOURCE;
static const int mp_mpi_proc_null = MPI_PROC_NULL;
static const int mp_mpi_any_tag = MPI_ANY_TAG;
static const int mp_mpi_thread_single = MPI_THREAD_SINGLE;
static const int mp_mpi_thread_funneled = MPI_THREAD_FUNNELED;
static const int mp_mpi_thread_serialized = MPI_THREAD_SERIALIZED;
static const int mp_mpi_thread_multiple = MPI_THREAD_MULTIPLE;
#else
typedef int mp_mpi_comm;
typedef int mp_mpi_request;
typedef int mp_mpi_fint;

static const mp_mpi_comm mp_mpi_comm_world = -2;
static const mp_mpi_comm mp_mpi_comm_null = -3;
static const mp_mpi_comm mp_mpi_comm_self = -5;
static const mp_mpi_request mp_mpi_request_null = -7;
static const int mp_mpi_any_source = -11;
static const int mp_mpi_proc_null = -13;
static const int mp_mpi_any_tag = -17;
static const int mp_mpi_thread_single = -7;
static const int mp_mpi_thread_funneled = -5;
static const int mp_mpi_thread_serialized = -3;
static const int mp_mpi_thread_multiple = -2;
#endif

/*******************************************************************************
 * \brief Initialize the MPI library (for unittesting).
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_init(int *argc, char ***argv);

/*******************************************************************************
 * \brief Finalize the MPI library (for unittesting).
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_finalize(void);

/*******************************************************************************
 * \brief Returns the available level of thread support.
 * \author Frederick Stein
 ******************************************************************************/
int mp_mpi_query(void);

/*******************************************************************************
 * \brief Return the number of ranks of a communicator.
 * \author Frederick Stein
 ******************************************************************************/
int mp_mpi_comm_size(const mp_mpi_comm comm);

/*******************************************************************************
 * \brief Return the own rank of a communicator.
 * \author Frederick Stein
 ******************************************************************************/
int mp_mpi_comm_rank(const mp_mpi_comm comm);

/*******************************************************************************
 * \brief Return info on Cartesian communicators.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_cart_get(const mp_mpi_comm comm, int maxdims, int *dims,
                       int *periods, int *coords);

/*******************************************************************************
 * \brief Return the rank of a process at a given coordinate.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_cart_rank(const mp_mpi_comm comm, const int *coords, int *rank);

/*******************************************************************************
 * \brief Convert a Fortran communicator handle to a C communicator.
 * \author Frederick Stein
 ******************************************************************************/
mp_mpi_comm mp_mpi_comm_f2c(const mp_mpi_fint fortran_comm);

/*******************************************************************************
 * \brief Convert a C communicator to a Fortran handle.
 * \author Frederick Stein
 ******************************************************************************/
mp_mpi_fint mp_mpi_comm_c2f(const mp_mpi_comm comm);

/*******************************************************************************
 * \brief Duplicate a communicator.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_comm_dup(const mp_mpi_comm old_comm, mp_mpi_comm *new_comm);

/*******************************************************************************
 * \brief Free a communicator.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_comm_free(mp_mpi_comm *comm);

/*******************************************************************************
 * \brief Wait until all ranks of a communicator reach this barrier.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_barrier(const mp_mpi_comm comm);

/*******************************************************************************
 * \brief Check whether two communicators are unequal.
 * \author Frederick Stein
 ******************************************************************************/
bool mp_mpi_comm_is_unequal(const mp_mpi_comm comm1,
                              const mp_mpi_comm comm2);

/*******************************************************************************
 * \brief Check whether two communicators are similar.
 * \author Frederick Stein
 ******************************************************************************/
bool mp_mpi_comm_is_similar(const mp_mpi_comm comm1,
                              const mp_mpi_comm comm2);

/*******************************************************************************
 * \brief Check whether two communicators are congruent.
 * \author Frederick Stein
 ******************************************************************************/
bool mp_mpi_comm_is_congruent(const mp_mpi_comm comm1,
                                const mp_mpi_comm comm2);

/*******************************************************************************
 * \brief Check whether two communicators are identical.
 * \author Frederick Stein
 ******************************************************************************/
bool mp_mpi_comm_is_ident(const mp_mpi_comm comm1,
                            const mp_mpi_comm comm2);

/*******************************************************************************
 * \brief Perform a blocking sendrecv of integers.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_sendrecv_int(const int *sendbuffer, const int sendcount,
                           const int dest, const int sendtag, int *recvbuffer,
                           const int recvcount, const int source,
                           const int recvtag, const mp_mpi_comm comm);

/*******************************************************************************
 * \brief Perform a blocking sendrecv of doubles.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_sendrecv_double(const double *sendbuffer, const int sendcount,
                              const int dest, const int sendtag,
                              double *recvbuffer, const int recvcount,
                              const int source, const int recvtag,
                              const mp_mpi_comm comm);

/*******************************************************************************
 * \brief Perform a blocing send of int.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_send_int(const int *sendbuffer, const int sendcount,
                       const int dest, const int sendtag,
                       const mp_mpi_comm comm);

/*******************************************************************************
 * \brief Perform a blocking recv of int.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_recv_int(int *recvbuffer, const int recvcount, const int source,
                       const int recvtag, const mp_mpi_comm comm);

/*******************************************************************************
 * \brief Perform a non-blocing send of doubles.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_isend_double(const double *sendbuffer, const int sendcount,
                           const int dest, const int sendtag,
                           const mp_mpi_comm comm, mp_mpi_request *request);

/*******************************************************************************
 * \brief Perform a non-blocking recv of doubles.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_irecv_double(double *recvbuffer, const int recvcount,
                           const int source, const int recvtag,
                           const mp_mpi_comm comm, mp_mpi_request *request);

/*******************************************************************************
 * \brief Perform a non-blocking send of double complex.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_isend_double_complex(const double complex *sendbuffer,
                                   const int sendcount, const int dest,
                                   const int sendtag, const mp_mpi_comm comm,
                                   mp_mpi_request *request);

/*******************************************************************************
 * \brief Perform a non-blocking recv of double complex.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_irecv_double_complex(double complex *recvbuffer,
                                   const int recvcount, const int source,
                                   const int recvtag, const mp_mpi_comm comm,
                                   mp_mpi_request *request);

/*******************************************************************************
 * \brief Wait for a request to finish.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_wait(mp_mpi_request *request);

/*******************************************************************************
 * \brief Wait for any/one of a set of requests to finish.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_waitany(const int number_of_requests,
                      mp_mpi_request request[number_of_requests], int *idx);

/*******************************************************************************
 * \brief Wait for all requests to finish.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_waitall(const int number_of_requests,
                      mp_mpi_request request[number_of_requests]);

/*******************************************************************************
 * \brief Gather integers from all processes.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_allgather_int(const int *sendbuffer, int sendcount,
                            int *recvbuffer, mp_mpi_comm comm);

/*******************************************************************************
 * \brief Sum doubles over all ranks of a communicator.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_sum_double(double *buffer, const int count,
                         const mp_mpi_comm comm);

/*******************************************************************************
 * \brief Sum integers over all ranks of a communicator.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_sum_int(int *buffer, const int count, const mp_mpi_comm comm);

/*******************************************************************************
 * \brief Determine the maximum of doubles over ranks of a communicator.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_max_double(double *buffer, const int count,
                         const mp_mpi_comm comm);

/*******************************************************************************
 * \brief Sum doubles over all ranks of a communicator.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_sum_double_root(double *buffer, const int count, const int root,
                              const mp_mpi_comm comm);

/*******************************************************************************
 * \brief Determine the maximum of doubles over ranks of a communicator.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_max_double_root(double *buffer, const int count, const int root,
                              const mp_mpi_comm comm);

/*******************************************************************************
 * \brief Perform an Alltoall of double complex.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_alltoallv_double_complex(const double complex *send_buffer,
                                       const int *send_counts,
                                       const int *send_displacements,
                                       double complex *recv_buffer,
                                       const int *recv_counts,
                                       const int *recv_displacements,
                                       const mp_mpi_comm comm);

/*******************************************************************************
 * \brief Broadcasts integers from a given root process.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_bcast_int(int *buffer, const int count, const int root,
                        const mp_mpi_comm comm);

/*******************************************************************************
 * \brief Broadcasts integers from a given root process.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_bcast_char(char *buffer, const int count, const int root,
                         const mp_mpi_comm comm);

/*******************************************************************************
 * \brief Determine good dimensions (wrapper to MPI_Dims_create).
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_dims_create(int number_of_processes, int number_of_dimensions,
                          int *dimensions);

/*******************************************************************************
 * \brief Create a Cartesian communicator.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_cart_create(mp_mpi_comm comm_old, int ndims, const int dims[],
                          const int periods[], int reorder,
                          mp_mpi_comm *comm_cart);

/*******************************************************************************
 * \brief Create sub communicators to a Cartesian communicator.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_cart_sub(const mp_mpi_comm comm_old, const int *remain_dims,
                       mp_mpi_comm *sub_comm);

/*******************************************************************************
 * \brief Determine the process coordinates of a rank in a Cartesian topology.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_cart_coords(const mp_mpi_comm comm, const int rank, int maxdims,
                          int coords[]);

#endif

// EOF