/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "mp_mpi.h"

#include <assert.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*******************************************************************************
 * \brief Check given MPI status and upon failure abort with a nice message.
 * \author Ole Schuett, Frederick Stein
 ******************************************************************************/
static inline void error_check(int error) {
#if defined(__parallel)
  if (error != MPI_SUCCESS) {
    int error_len, error_class;
    char error_string[MPI_MAX_ERROR_STRING];
    MPI_Error_class(error, &error_class);
    MPI_Error_string(error, error_string, &error_len);
    fprintf(stderr, "MPI Error %s (Class %i) in %s:%i\n", error_string,
            error_class, __FILE__, __LINE__);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
#else
  (void)error;
#endif
}

/*******************************************************************************
 * \brief Initialize the MPI library (for unittesting).
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_init(int *argc, char ***argv) {
#if defined(__parallel)
  int provided_thread_level;
  error_check(
      MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided_thread_level));
  assert(provided_thread_level >= MPI_THREAD_FUNNELED &&
         "Required thread level not supported by the MPI implementation.");
#else
  (void)argc;
  (void)argv;
#endif
}

/*******************************************************************************
 * \brief Finalize the MPI library (for unittesting).
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_finalize(void) {
#if defined(__parallel)
  error_check(MPI_Finalize());
#else
  // Nothing to do in the serial case
#endif
}

/*******************************************************************************
 * \brief Returns the available level of thread support.
 * \author Frederick Stein
 ******************************************************************************/
int mp_mpi_query(void) {
#if defined(__parallel)
  int provided;
  error_check(MPI_Query_thread(&provided));
  return provided;
#else
  return mp_mpi_thread_single;
#endif
}

/*******************************************************************************
 * \brief Return the number of ranks of a communicator.
 * \author Frederick Stein
 ******************************************************************************/
int mp_mpi_comm_size(const mp_mpi_comm comm) {
#if defined(__parallel)
  int comm_size;
  error_check(MPI_Comm_size(comm, &comm_size));
  return comm_size;
#else
  (void)comm;
  return 1;
#endif
}

/*******************************************************************************
 * \brief Return the own rank of a communicator.
 * \author Frederick Stein
 ******************************************************************************/
int mp_mpi_comm_rank(const mp_mpi_comm comm) {
#if defined(__parallel)
  int comm_rank;
  error_check(MPI_Comm_rank(comm, &comm_rank));
  return comm_rank;
#else
  (void)comm;
  return 0;
#endif
}

/*******************************************************************************
 * \brief Return info on Cartesian communicators.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_cart_get(const mp_mpi_comm comm, int maxdims, int *dims,
                       int *periods, int *coords) {
#if defined(__parallel)
  error_check(MPI_Cart_get(comm, maxdims, dims, periods, coords));
#else
  (void)comm;
  for (int dim = 0; dim < maxdims; dim++) {
    dims[dim] = 1;
    periods[dim] = 0;
    coords[dim] = 0;
  }
#endif
}

/*******************************************************************************
 * \brief Return the rank of a process at a given coordinate.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_cart_rank(const mp_mpi_comm comm, const int *coords,
                        int *rank) {
#if defined(__parallel)
  error_check(MPI_Cart_rank(comm, coords, rank));
#else
  (void)comm;
  (void)coords;
  *rank = 0;
#endif
}

/*******************************************************************************
 * \brief Convert a Fortran communicator handle to a C communicator.
 * \author Frederick Stein
 ******************************************************************************/
mp_mpi_comm mp_mpi_comm_f2c(const mp_mpi_fint fortran_comm) {
#if defined(__parallel)
  return MPI_Comm_f2c(fortran_comm);
#else
  return (mp_mpi_comm)fortran_comm;
#endif
}

/*******************************************************************************
 * \brief Convert a C communicator to a Fortran handle.
 * \author Frederick Stein
 ******************************************************************************/
mp_mpi_fint mp_mpi_comm_c2f(const mp_mpi_comm comm) {
#if defined(__parallel)
  return MPI_Comm_c2f(comm);
#else
  return (mp_mpi_fint)comm;
#endif
}

/*******************************************************************************
 * \brief Duplicate a communicator.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_comm_dup(const mp_mpi_comm old_comm, mp_mpi_comm *new_comm) {
#if defined(__parallel)
  error_check(MPI_Comm_dup(old_comm, new_comm));
#else
  *new_comm = old_comm;
#endif
}

/*******************************************************************************
 * \brief Free a communicator.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_comm_free(mp_mpi_comm *comm) {
#if defined(__parallel)
  error_check(MPI_Comm_free(comm));
#else
  *comm = mp_mpi_comm_null;
#endif
}

/*******************************************************************************
 * \brief Wait until all ranks of a communicator reach this barrier.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_barrier(const mp_mpi_comm comm) {
#if defined(__parallel)
  error_check(MPI_Barrier(comm));
#else
  // Nothing to do in the serial case
  (void)comm;
#endif
}

/*******************************************************************************
 * \brief Check whether two communicators are unequal.
 * \author Frederick Stein
 ******************************************************************************/
bool mp_mpi_comm_is_unequal(const mp_mpi_comm comm1,
                              const mp_mpi_comm comm2) {
#if defined(__parallel)
  int result = -1;
  error_check(MPI_Comm_compare(comm1, comm2, &result));
  return result == MPI_UNEQUAL;
#else
  return ((comm1 == mp_mpi_comm_null) && (comm2 != mp_mpi_comm_null)) ||
         ((comm1 != mp_mpi_comm_null) && (comm2 == mp_mpi_comm_null));
#endif
}

/*******************************************************************************
 * \brief Check whether two communicators are similar.
 * \author Frederick Stein
 ******************************************************************************/
bool mp_mpi_comm_is_similar(const mp_mpi_comm comm1,
                              const mp_mpi_comm comm2) {
#if defined(__parallel)
  int result = -1;
  error_check(MPI_Comm_compare(comm1, comm2, &result));
  return result == MPI_SIMILAR || result == MPI_CONGRUENT ||
         result == MPI_IDENT;
#else
  return ((comm1 == mp_mpi_comm_null) && (comm2 == mp_mpi_comm_null)) ||
         ((comm1 != mp_mpi_comm_null) && (comm2 != mp_mpi_comm_null));
#endif
}

/*******************************************************************************
 * \brief Check whether two communicators are congruent.
 * \author Frederick Stein
 ******************************************************************************/
bool mp_mpi_comm_is_congruent(const mp_mpi_comm comm1,
                                const mp_mpi_comm comm2) {
#if defined(__parallel)
  int result = -1;
  error_check(MPI_Comm_compare(comm1, comm2, &result));
  return result == MPI_CONGRUENT || result == MPI_IDENT;
#else
  return ((comm1 == mp_mpi_comm_null) && (comm2 == mp_mpi_comm_null)) ||
         ((comm1 != mp_mpi_comm_null) && (comm2 != mp_mpi_comm_null));
#endif
}

/*******************************************************************************
 * \brief Check whether two communicators are identical.
 * \author Frederick Stein
 ******************************************************************************/
bool mp_mpi_comm_is_ident(const mp_mpi_comm comm1,
                            const mp_mpi_comm comm2) {
#if defined(__parallel)
  int result = -1;
  error_check(MPI_Comm_compare(comm1, comm2, &result));
  return result == MPI_IDENT;
#else
  return ((comm1 == mp_mpi_comm_null) && (comm2 == mp_mpi_comm_null)) ||
         ((comm1 != mp_mpi_comm_null) && (comm2 != mp_mpi_comm_null));
#endif
}

/*******************************************************************************
 * \brief Perform a blocking sendrecv of integers.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_sendrecv_int(const int *sendbuffer, const int sendcount,
                           const int dest, const int sendtag, int *recvbuffer,
                           const int recvcount, const int source,
                           const int recvtag, const mp_mpi_comm comm) {
#if defined(__parallel)
  error_check(MPI_Sendrecv(sendbuffer, sendcount, MPI_DOUBLE, dest, sendtag,
                           recvbuffer, recvcount, MPI_DOUBLE, source, recvtag,
                           comm, MPI_STATUS_IGNORE));
#else
  (void)sendbuffer;
  (void)sendcount;
  (void)dest;
  (void)sendtag;
  (void)recvbuffer;
  (void)recvcount;
  (void)source;
  (void)recvtag;
  (void)comm;
  // Check the input for reasonable values in serial case
  assert(sendbuffer != NULL);
  assert(recvbuffer != NULL);
  assert((dest == 0 || dest == mp_mpi_any_source ||
          dest == mp_mpi_proc_null) &&
         "Invalid receive process");
  assert((source == 0 || source == mp_mpi_proc_null) &&
         "Invalid sent process");
  assert((recvtag == sendtag || recvtag == mp_mpi_any_tag) &&
         "Invalid send or receive tag");
  if (dest != mp_mpi_proc_null && source != mp_mpi_proc_null) {
    memcpy(recvbuffer, sendbuffer, sendcount * sizeof(double));
  }
#endif
}

/*******************************************************************************
 * \brief Perform a blocking sendrecv of doubles.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_sendrecv_double(const double *sendbuffer, const int sendcount,
                              const int dest, const int sendtag,
                              double *recvbuffer, const int recvcount,
                              const int source, const int recvtag,
                              const mp_mpi_comm comm) {
#if defined(__parallel)
  error_check(MPI_Sendrecv(sendbuffer, sendcount, MPI_DOUBLE, dest, sendtag,
                           recvbuffer, recvcount, MPI_DOUBLE, source, recvtag,
                           comm, MPI_STATUS_IGNORE));
#else
  (void)sendbuffer;
  (void)sendcount;
  (void)dest;
  (void)sendtag;
  (void)recvbuffer;
  (void)recvcount;
  (void)source;
  (void)recvtag;
  (void)comm;
  // Check the input for reasonable values in serial case
  assert(sendbuffer != NULL);
  assert(recvbuffer != NULL);
  assert((dest == 0 || dest == mp_mpi_any_source ||
          dest == mp_mpi_proc_null) &&
         "Invalid receive process");
  assert((source == 0 || source == mp_mpi_proc_null) &&
         "Invalid sent process");
  assert((recvtag == sendtag || recvtag == mp_mpi_any_tag) &&
         "Invalid send or receive tag");
  if (dest != mp_mpi_proc_null && source != mp_mpi_proc_null) {
    memcpy(recvbuffer, sendbuffer, sendcount * sizeof(double));
  }
#endif
}

/*******************************************************************************
 * \brief Perform a non-blocing send of doubles.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_send_int(const int *sendbuffer, const int sendcount,
                       const int dest, const int sendtag,
                       const mp_mpi_comm comm) {
#if defined(__parallel)
  assert(sendbuffer != NULL);
  assert(sendcount >= 0 && "Send count must be nonnegative!");
  assert(sendtag >= 0 && "Send tag must be nonnegative!");
  assert(dest >= 0 && "Send process must be nonnegative!");
  assert(dest < mp_mpi_comm_size(comm) &&
         "Send process must be lower than the number of processes!");
  error_check(MPI_Send(sendbuffer, sendcount, MPI_INT, dest, sendtag, comm));
#else
  (void)sendbuffer;
  (void)sendcount;
  (void)dest;
  (void)sendtag;
  (void)comm;
  assert(false && "Single send not allowed in serial mode");
#endif
}

/*******************************************************************************
 * \brief Perform a non-blocking recv of doubles.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_recv_int(int *recvbuffer, const int recvcount, const int source,
                       const int recvtag, const mp_mpi_comm comm) {
#if defined(__parallel)
  assert(recvbuffer != NULL);
  assert(recvcount >= 0 && "Receive count must be nonnegative!");
  assert(recvtag >= 0 && "Receive tag must be nonnegative!");
  assert(source >= 0 && "Receive process must be nonnegative!");
  assert(source < mp_mpi_comm_size(comm) &&
         "Receive process must be lower than the number of processes!");
  error_check(MPI_Recv(recvbuffer, recvcount, MPI_INT, source, recvtag, comm,
                       MPI_STATUS_IGNORE));
#else
  (void)recvbuffer;
  (void)recvcount;
  (void)source;
  (void)recvtag;
  (void)comm;
  assert(false && "Single receive not allowed in serial mode");
#endif
}

/*******************************************************************************
 * \brief Perform a non-blocing send of doubles.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_isend_double(const double *sendbuffer, const int sendcount,
                           const int dest, const int sendtag,
                           const mp_mpi_comm comm,
                           mp_mpi_request *request) {
#if defined(__parallel)
  assert(sendbuffer != NULL);
  assert(sendcount >= 0 && "Send count must be nonnegative!");
  assert(sendtag >= 0 && "Send tag must be nonnegative!");
  assert(dest >= 0 && "Send process must be nonnegative!");
  assert(dest < mp_mpi_comm_size(comm) &&
         "Send process must be lower than the number of processes!");
  error_check(MPI_Isend(sendbuffer, sendcount, MPI_DOUBLE, dest, sendtag, comm,
                        request));
#else
  (void)sendbuffer;
  (void)sendcount;
  (void)dest;
  (void)sendtag;
  (void)comm;
  *request = 2;
  assert(false && "Non-blocking send not allowed in serial mode");
#endif
}

/*******************************************************************************
 * \brief Perform a non-blocking recv of doubles.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_irecv_double(double *recvbuffer, const int recvcount,
                           const int source, const int recvtag,
                           const mp_mpi_comm comm,
                           mp_mpi_request *request) {
#if defined(__parallel)
  assert(recvbuffer != NULL);
  assert(recvcount >= 0 && "Receive count must be nonnegative!");
  assert(recvtag >= 0 && "Receive tag must be nonnegative!");
  assert(source >= 0 && "Receive process must be nonnegative!");
  assert(source < mp_mpi_comm_size(comm) &&
         "Receive process must be lower than the number of processes!");
  error_check(MPI_Irecv(recvbuffer, recvcount, MPI_DOUBLE, source, recvtag,
                        comm, request));
#else
  (void)recvbuffer;
  (void)recvcount;
  (void)source;
  (void)recvtag;
  (void)comm;
  *request = 3;
  assert(false && "Non-blocking receive not allowed in serial mode");
#endif
}

/*******************************************************************************
 * \brief Perform a non-blocking send of double complex.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_isend_double_complex(const double complex *sendbuffer,
                                   const int sendcount, const int dest,
                                   const int sendtag, const mp_mpi_comm comm,
                                   mp_mpi_request *request) {
#if defined(__parallel)
  assert(sendbuffer != NULL);
  assert(sendcount >= 0 && "Send count must be nonnegative!");
  assert(sendtag >= 0 && "Send tag must be nonnegative!");
  assert(dest >= 0 && "Send process must be nonnegative!");
  assert(dest < mp_mpi_comm_size(comm) &&
         "Send process must be lower than the number of processes!");
  error_check(MPI_Isend(sendbuffer, sendcount, MPI_C_DOUBLE_COMPLEX, dest,
                        sendtag, comm, request));
#else
  (void)sendbuffer;
  (void)sendcount;
  (void)dest;
  (void)sendtag;
  (void)comm;
  *request = 2;
  assert(false && "Non-blocking send not allowed in serial mode");
#endif
}

/*******************************************************************************
 * \brief Perform a non-blocking recv of double complex.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_irecv_double_complex(double complex *recvbuffer,
                                   const int recvcount, const int source,
                                   const int recvtag, const mp_mpi_comm comm,
                                   mp_mpi_request *request) {
#if defined(__parallel)
  assert(recvbuffer != NULL);
  assert(recvcount >= 0 && "Receive count must be nonnegative!");
  assert(recvtag >= 0 && "Receive tag must be nonnegative!");
  assert(source >= 0 && "Receive process must be nonnegative!");
  assert(source < mp_mpi_comm_size(comm) &&
         "Receive process must be lower than the number of processes!");
  error_check(MPI_Irecv(recvbuffer, recvcount, MPI_C_DOUBLE_COMPLEX, source,
                        recvtag, comm, request));
#else
  (void)recvbuffer;
  (void)recvcount;
  (void)source;
  (void)recvtag;
  (void)comm;
  *request = 3;
  assert(false && "Non-blocking receive not allowed in serial mode");
#endif
}

/*******************************************************************************
 * \brief Wait for a request to finish.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_wait(mp_mpi_request *request) {
  assert(request != NULL);
#if defined(__parallel)
  error_check(MPI_Wait(request, MPI_STATUS_IGNORE));
#else
  *request = mp_mpi_request_null;
#endif
}

/*******************************************************************************
 * \brief Wait for any/one of a set of requests to finish.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_waitany(const int number_of_requests,
                      mp_mpi_request request[number_of_requests], int *idx) {
  assert(idx != NULL);
#if defined(__parallel)
  error_check(MPI_Waitany(number_of_requests, request, idx, MPI_STATUS_IGNORE));
#else
  *idx = -1;
  for (int request_idx = 0; request_idx < number_of_requests; request_idx++) {
    if (request[request_idx] != mp_mpi_request_null) {
      *idx = request_idx;
      request[request_idx] = mp_mpi_request_null;
    }
  }
#endif
}

/*******************************************************************************
 * \brief Wait for all requests to finish.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_waitall(const int number_of_requests,
                      mp_mpi_request request[number_of_requests]) {
#if defined(__parallel)
  error_check(MPI_Waitall(number_of_requests, request, MPI_STATUSES_IGNORE));
#else
  for (int idx = 0; idx < number_of_requests; idx++) {
    request[idx] = mp_mpi_request_null;
  }
#endif
}

/*******************************************************************************
 * \brief Gather integers from all processes.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_allgather_int(const int *sendbuffer, int sendcount,
                            int *recvbuffer, mp_mpi_comm comm) {
#if defined(__parallel)
  assert(sendbuffer != NULL);
  assert(recvbuffer != NULL);
  assert(sendcount >= 0 && "Send count must be nonnegative!");
  error_check(MPI_Allgather(sendbuffer, sendcount, MPI_INT, recvbuffer,
                            sendcount, MPI_INT, comm));
#else
  (void)comm;
  memcpy(recvbuffer, sendbuffer, sendcount * sizeof(int));
#endif
}

/*******************************************************************************
 * \brief Sum doubles over all ranks of a communicator.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_sum_double(double *buffer, const int count,
                         const mp_mpi_comm comm) {
#if defined(__parallel)
  assert(buffer != NULL);
  assert(count >= 0 && "Send count must be nonnegative!");
  error_check(
      MPI_Allreduce(MPI_IN_PLACE, buffer, count, MPI_DOUBLE, MPI_SUM, comm));
#else
  assert(buffer != NULL);
  (void)comm;
  (void)buffer;
  (void)count;
#endif
}

/*******************************************************************************
 * \brief Sum integers over all ranks of a communicator.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_sum_int(int *buffer, const int count, const mp_mpi_comm comm) {
#if defined(__parallel)
  assert(buffer != NULL);
  assert(count >= 0 && "Send count must be nonnegative!");
  error_check(
      MPI_Allreduce(MPI_IN_PLACE, buffer, count, MPI_INT, MPI_SUM, comm));
#else
  assert(buffer != NULL);
  (void)comm;
  (void)buffer;
  (void)count;
#endif
}

/*******************************************************************************
 * \brief Determine the maximum of doubles over ranks of a communicator.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_max_double(double *buffer, const int count,
                         const mp_mpi_comm comm) {
#if defined(__parallel)
  assert(buffer != NULL);
  assert(count >= 0 && "Send count must be nonnegative!");
  error_check(
      MPI_Allreduce(MPI_IN_PLACE, buffer, count, MPI_DOUBLE, MPI_MAX, comm));
#else
  assert(buffer != NULL);
  (void)comm;
  (void)buffer;
  (void)count;
#endif
}

/*******************************************************************************
 * \brief Sum doubles over all ranks of a communicator and collect at root.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_sum_double_root(double *buffer, const int count, const int root,
                              const mp_mpi_comm comm) {
#if defined(__parallel)
  assert(buffer != NULL);
  assert(count >= 0 && "Send count must be nonnegative!");
  assert(root >= 0 && "Invalid root process!");
  double *result = malloc(count * sizeof(double));
  error_check(
      MPI_Reduce(buffer, result, count, MPI_DOUBLE, MPI_SUM, root, comm));
  memcpy(buffer, result, count * sizeof(double));
  free(result);
#else
  assert(buffer != NULL);
  (void)comm;
  (void)buffer;
  (void)root;
  (void)count;
#endif
}

/*******************************************************************************
 * \brief Determine the maximum of doubles over ranks of a communicator and
 *collect at root. \author Frederick Stein
 ******************************************************************************/
void mp_mpi_max_double_root(double *buffer, const int count, const int root,
                              const mp_mpi_comm comm) {
#if defined(__parallel)
  assert(buffer != NULL);
  assert(count >= 0 && "Send count must be nonnegative!");
  assert(root >= 0 && "Invalid root process!");
  double *result = malloc(count * sizeof(double));
  error_check(
      MPI_Reduce(buffer, result, count, MPI_DOUBLE, MPI_MAX, root, comm));
  if (root == mp_mpi_comm_rank(comm))
    memcpy(buffer, result, count * sizeof(double));
  free(result);
#else
  assert(buffer != NULL);
  (void)comm;
  (void)buffer;
  (void)root;
  (void)count;
#endif
}

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
                                       const mp_mpi_comm comm) {
  assert(send_buffer != NULL);
  assert(recv_buffer != NULL);
  assert(send_counts != NULL);
  assert(recv_counts != NULL);
  assert(send_displacements != NULL);
  assert(recv_displacements != NULL);
#if defined(__parallel)
  error_check(MPI_Alltoallv(send_buffer, send_counts, send_displacements,
                            MPI_C_DOUBLE_COMPLEX, recv_buffer, recv_counts,
                            recv_displacements, MPI_C_DOUBLE_COMPLEX, comm));
#else
  assert(*send_counts == *recv_counts);
  (void)comm;
  (void)recv_counts;
  memcpy(recv_buffer + (*recv_displacements),
         send_buffer + (*send_displacements),
         (*send_counts) * sizeof(double complex));
#endif
}

/*******************************************************************************
 * \brief Broadcasts integers from a given root process.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_bcast_int(int *buffer, const int count, const int root,
                        const mp_mpi_comm comm) {
  assert(buffer != NULL);
  assert(count >= 0);
  assert(root >= 0);
#if defined(__parallel)
  error_check(MPI_Bcast(buffer, count, MPI_INT, root, comm));
#else
  (void)buffer;
  (void)count;
  (void)root;
  (void)comm;
#endif
}

/*******************************************************************************
 * \brief Broadcasts integers from a given root process.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_bcast_char(char *buffer, const int count, const int root,
                         const mp_mpi_comm comm) {
  assert(buffer != NULL);
  assert(count >= 0);
  assert(root >= 0);
#if defined(__parallel)
  error_check(MPI_Bcast(buffer, count, MPI_CHAR, root, comm));
#else
  (void)buffer;
  (void)count;
  (void)root;
  (void)comm;
#endif
}

/*******************************************************************************
 * \brief Determine good dimensions (wrapper to MPI_Dims_create).
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_dims_create(int number_of_processes, int number_of_dimensions,
                          int *dimensions) {
#if defined(__parallel)
  assert(number_of_processes > 0 &&
         "The number of processes needs to be positive");
  assert(number_of_dimensions >= 0 &&
         "The number of dimensions needs to be positive!");
  assert(dimensions != NULL && "The target array needs to point to some data!");
  MPI_Dims_create(number_of_processes, number_of_dimensions, dimensions);
#else
  (void)number_of_processes;
  for (int dim = 0; dim < number_of_dimensions; dim++)
    dimensions[dim] = 1;
#endif
}

/*******************************************************************************
 * \brief Create a Cartesian communicator.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_cart_create(mp_mpi_comm comm_old, int ndims, const int dims[],
                          const int periods[], int reorder,
                          mp_mpi_comm *comm_cart) {
#if defined(__parallel)
  assert(ndims > 0 && "The number of processes needs to be positive");
  error_check(
      MPI_Cart_create(comm_old, ndims, dims, periods, reorder, comm_cart));
#else
  (void)ndims;
  (void)dims;
  (void)periods;
  (void)reorder;
  *comm_cart = comm_old - 43;
#endif
}

/*******************************************************************************
 * \brief Create sub communicators to a Cartesian communicator.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_cart_sub(const mp_mpi_comm comm_old, const int *remain_dims,
                       mp_mpi_comm *sub_comm) {
#if defined(__parallel)
  assert(remain_dims != NULL);
  assert(sub_comm != NULL);
  error_check(MPI_Cart_sub(comm_old, remain_dims, sub_comm));
#else
  (void)comm_old;
  (void)remain_dims;
  *sub_comm = comm_old - 47;
#endif
}

/*******************************************************************************
 * \brief Determine the process coordinates of a rank in a Cartesian topology.
 * \author Frederick Stein
 ******************************************************************************/
void mp_mpi_cart_coords(const mp_mpi_comm comm, const int rank, int maxdims,
                          int coords[]) {
#if defined(__parallel)
  assert(maxdims > 0 && "The number of processes needs to be positive");
  error_check(MPI_Cart_coords(comm, rank, maxdims, coords));
#else
  (void)comm;
  (void)rank;
  for (int dim = 0; dim < maxdims; dim++)
    coords[dim] = 0;
#endif
}

// EOF