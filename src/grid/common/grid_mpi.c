/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "grid_mpi.h"

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

int grid_mpi_comm_size(const grid_mpi_comm comm) {
#if defined(__parallel)
  int comm_size;
  error_check(MPI_Comm_size(comm, &comm_size));
  return comm_size;
#else
  (void)comm;
  return 1;
#endif
}

int grid_mpi_comm_rank(const grid_mpi_comm comm) {
#if defined(__parallel)
  int comm_rank;
  error_check(MPI_Comm_rank(comm, &comm_rank));
  return comm_rank;
#else
  (void)comm;
  return 0;
#endif
}

void grid_mpi_cart_get(const grid_mpi_comm comm, int maxdims, int *dims,
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

void grid_mpi_cart_rank(const grid_mpi_comm comm, const int *coords,
                        int *rank) {
#if defined(__parallel)
  error_check(MPI_Cart_rank(comm, coords, rank));
#else
  (void)comm;
  (void)coords;
  *rank = 0;
#endif
}

grid_mpi_comm grid_mpi_comm_f2c(const grid_mpi_fint fortran_comm) {
#if defined(__parallel)
  return MPI_Comm_f2c(fortran_comm);
#else
  return (grid_mpi_comm)fortran_comm;
#endif
}

grid_mpi_fint grid_mpi_comm_c2f(const grid_mpi_comm comm) {
#if defined(__parallel)
  return MPI_Comm_c2f(comm);
#else
  return (grid_mpi_fint)comm;
#endif
}

void grid_mpi_comm_dup(const grid_mpi_comm old_comm, grid_mpi_comm *new_comm) {
#if defined(__parallel)
  error_check(MPI_Comm_dup(old_comm, new_comm));
#else
  *new_comm = old_comm;
#endif
}

void grid_mpi_comm_free(grid_mpi_comm *comm) {
#if defined(__parallel)
  error_check(MPI_Comm_free(comm));
#else
  *comm = grid_mpi_comm_null;
#endif
}

void grid_mpi_barrier(const grid_mpi_comm comm) {
#if defined(__parallel)
  error_check(MPI_Barrier(comm));
#else
  // Nothing to do in the serial case
  (void)comm;
#endif
}

bool grid_mpi_comm_is_unequal(const grid_mpi_comm comm1,
                              const grid_mpi_comm comm2) {
#if defined(__parallel)
  int result = -1;
  error_check(MPI_Comm_compare(comm1, comm2, &result));
  return result == MPI_UNEQUAL;
#else
  return ((comm1 == grid_mpi_comm_null) && (comm2 != grid_mpi_comm_null)) ||
         ((comm1 != grid_mpi_comm_null) && (comm2 == grid_mpi_comm_null));
#endif
}

bool grid_mpi_comm_is_similar(const grid_mpi_comm comm1,
                              const grid_mpi_comm comm2) {
#if defined(__parallel)
  int result = -1;
  error_check(MPI_Comm_compare(comm1, comm2, &result));
  return result == MPI_SIMILAR || result == MPI_CONGRUENT ||
         result == MPI_IDENT;
#else
  return ((comm1 == grid_mpi_comm_null) && (comm2 == grid_mpi_comm_null)) ||
         ((comm1 != grid_mpi_comm_null) && (comm2 != grid_mpi_comm_null));
#endif
}

bool grid_mpi_comm_is_congruent(const grid_mpi_comm comm1,
                                const grid_mpi_comm comm2) {
#if defined(__parallel)
  int result = -1;
  error_check(MPI_Comm_compare(comm1, comm2, &result));
  return result == MPI_CONGRUENT || result == MPI_IDENT;
#else
  return ((comm1 == grid_mpi_comm_null) && (comm2 == grid_mpi_comm_null)) ||
         ((comm1 != grid_mpi_comm_null) && (comm2 != grid_mpi_comm_null));
#endif
}

bool grid_mpi_comm_is_ident(const grid_mpi_comm comm1,
                            const grid_mpi_comm comm2) {
#if defined(__parallel)
  int result = -1;
  error_check(MPI_Comm_compare(comm1, comm2, &result));
  return result == MPI_IDENT;
#else
  return ((comm1 == grid_mpi_comm_null) && (comm2 == grid_mpi_comm_null)) ||
         ((comm1 != grid_mpi_comm_null) && (comm2 != grid_mpi_comm_null));
#endif
}

void grid_mpi_sendrecv_int(const int *sendbuffer, const int sendcount,
                           const int dest, const int sendtag, int *recvbuffer,
                           const int recvcount, const int source,
                           const int recvtag, const grid_mpi_comm comm) {
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
  assert((dest == 0 || dest == grid_mpi_any_source ||
          dest == grid_mpi_proc_null) &&
         "Invalid receive process");
  assert((source == 0 || source == grid_mpi_proc_null) &&
         "Invalid sent process");
  assert((recvtag == sendtag || recvtag == grid_mpi_any_tag) &&
         "Invalid send or receive tag");
  if (dest != grid_mpi_proc_null && source != grid_mpi_proc_null) {
    memcpy(recvbuffer, sendbuffer, sendcount * sizeof(double));
  }
#endif
}

void grid_mpi_sendrecv_double(const double *sendbuffer, const int sendcount,
                              const int dest, const int sendtag,
                              double *recvbuffer, const int recvcount,
                              const int source, const int recvtag,
                              const grid_mpi_comm comm) {
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
  assert((dest == 0 || dest == grid_mpi_any_source ||
          dest == grid_mpi_proc_null) &&
         "Invalid receive process");
  assert((source == 0 || source == grid_mpi_proc_null) &&
         "Invalid sent process");
  assert((recvtag == sendtag || recvtag == grid_mpi_any_tag) &&
         "Invalid send or receive tag");
  if (dest != grid_mpi_proc_null && source != grid_mpi_proc_null) {
    memcpy(recvbuffer, sendbuffer, sendcount * sizeof(double));
  }
#endif
}

void grid_mpi_sendrecv_double_complex(
    const double complex *sendbuffer, const int sendcount, const int dest,
    const int sendtag, double complex *recvbuffer, const int recvcount,
    const int source, const int recvtag, const grid_mpi_comm comm) {
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
  assert((dest == 0 || dest == grid_mpi_any_source ||
          dest == grid_mpi_proc_null) &&
         "Invalid receive process");
  assert((source == 0 || source == grid_mpi_proc_null) &&
         "Invalid sent process");
  assert((recvtag == sendtag || recvtag == grid_mpi_any_tag) &&
         "Invalid send or receive tag");
  if (dest != grid_mpi_proc_null && source != grid_mpi_proc_null) {
    memcpy(recvbuffer, sendbuffer, sendcount * sizeof(double));
  }
#endif
}

void grid_mpi_isend_double(const double *sendbuffer, const int sendcount,
                           const int dest, const int sendtag,
                           const grid_mpi_comm comm,
                           grid_mpi_request *request) {
#if defined(__parallel)
  assert(sendbuffer != NULL);
  assert(sendcount >= 0 && "Send count must be nonnegative!");
  assert(sendtag >= 0 && "Send tag must be nonnegative!");
  assert(dest >= 0 && "Send process must be nonnegative!");
  assert(dest < grid_mpi_comm_size(comm) &&
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

void grid_mpi_irecv_double(double *recvbuffer, const int recvcount,
                           const int source, const int recvtag,
                           const grid_mpi_comm comm,
                           grid_mpi_request *request) {
#if defined(__parallel)
  assert(recvbuffer != NULL);
  assert(recvcount >= 0 && "Receive count must be nonnegative!");
  assert(recvtag >= 0 && "Receive tag must be nonnegative!");
  assert(source >= 0 && "Receive process must be nonnegative!");
  assert(source < grid_mpi_comm_size(comm) &&
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

void grid_mpi_isend_double_complex(const double complex *sendbuffer,
                                   const int sendcount, const int dest,
                                   const int sendtag, const grid_mpi_comm comm,
                                   grid_mpi_request *request) {
#if defined(__parallel)
  assert(sendbuffer != NULL);
  assert(sendcount >= 0 && "Send count must be nonnegative!");
  assert(sendtag >= 0 && "Send tag must be nonnegative!");
  assert(dest >= 0 && "Send process must be nonnegative!");
  assert(dest < grid_mpi_comm_size(comm) &&
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

void grid_mpi_irecv_double_complex(double complex *recvbuffer,
                                   const int recvcount, const int source,
                                   const int recvtag, const grid_mpi_comm comm,
                                   grid_mpi_request *request) {
#if defined(__parallel)
  assert(recvbuffer != NULL);
  assert(recvcount >= 0 && "Receive count must be nonnegative!");
  assert(recvtag >= 0 && "Receive tag must be nonnegative!");
  assert(source >= 0 && "Receive process must be nonnegative!");
  assert(source < grid_mpi_comm_size(comm) &&
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

void grid_mpi_wait(grid_mpi_request *request) {
  assert(request != NULL);
#if defined(__parallel)
  error_check(MPI_Wait(request, MPI_STATUS_IGNORE));
#else
  *request = grid_mpi_request_null;
#endif
}

void grid_mpi_waitany(const int number_of_requests,
                      grid_mpi_request request[number_of_requests], int *idx) {
  assert(idx != NULL);
#if defined(__parallel)
  error_check(MPI_Waitany(number_of_requests, request, idx, MPI_STATUS_IGNORE));
#else
  *idx = -1;
  for (int request_idx = 0; request_idx < number_of_requests; request_idx++) {
    if (request[request_idx] != grid_mpi_request_null) {
      *idx = request_idx;
      request[request_idx] = grid_mpi_request_null;
    }
  }
#endif
}

void grid_mpi_waitall(const int number_of_requests,
                      grid_mpi_request request[number_of_requests]) {
#if defined(__parallel)
  error_check(MPI_Waitall(number_of_requests, request, MPI_STATUSES_IGNORE));
#else
  for (int idx = 0; idx < number_of_requests; idx++) {
    request[idx] = grid_mpi_request_null;
  }
#endif
}

void grid_mpi_allgather_int(const int *sendbuffer, int sendcount,
                            int *recvbuffer, grid_mpi_comm comm) {
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

void grid_mpi_sum_double(double *buffer, const int count,
                         const grid_mpi_comm comm) {
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

void grid_mpi_sum_int(int *buffer, const int count, const grid_mpi_comm comm) {
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

void grid_mpi_max_double(double *buffer, const int count,
                         const grid_mpi_comm comm) {
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

void grid_mpi_alltoallv_double_complex(const double complex *send_buffer,
                                       const int *send_counts,
                                       const int *send_displacements,
                                       double complex *recv_buffer,
                                       const int *recv_counts,
                                       const int *recv_displacements,
                                       const grid_mpi_comm comm) {
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

void grid_mpi_dims_create(int number_of_processes, int number_of_dimensions,
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

void grid_mpi_cart_create(grid_mpi_comm comm_old, int ndims, const int dims[],
                          const int periods[], int reorder,
                          grid_mpi_comm *comm_cart) {
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

void grid_mpi_cart_sub(const grid_mpi_comm comm_old, const int *remain_dims,
                       grid_mpi_comm *sub_comm) {
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

void grid_mpi_cart_coords(const grid_mpi_comm comm, const int rank, int maxdims,
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