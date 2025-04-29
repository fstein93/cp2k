/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "grid_fft_lib_fftw.h"

#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(__FFTW3)
#include <fftw3.h>
#if defined(__parallel) && defined(__FFTW3_MPI)
#include <fftw3-mpi.h>
#endif

/*******************************************************************************
 * \brief Static variables for retaining objects that are expensive to create.
 * \author Ole Schuett
 ******************************************************************************/
typedef struct {
  // The key contains
  // 0: rank, transposition (see below)
  // 1: associated communicator
  // 2: direction
  // 3, 4, 5: FFT sizes (or FFT sizes and number of FFTs)
  int key[6];
  fftw_plan *plan;
} cache_entry;

#define FFTW_CACHE_SIZE 32
static cache_entry cache[FFTW_CACHE_SIZE];
static int cache_oldest_entry = 0; // used for LRU eviction

static bool is_initialized = false;

static int fftw_planning_mode = -1;
static bool use_fftw_mpi = false;

// These constants encode transposition and MPI usage into the key to cache the
// plans 1, 2, 3 is the rank of the transposition 4 == 2^2
#define FFTW_TRANSPOSE_RS 4
// 8 == 2^3
#define FFTW_TRANSPOSE_GS 8

/*******************************************************************************
 * \brief Fetches an fft plan from the cache. Returns NULL if not found.
 * \author Ole Schuett, Frederick Stein
 ******************************************************************************/
static fftw_plan *lookup_plan_from_cache(const int key[6]) {
  assert(is_initialized);
  for (int i = 0; i < FFTW_CACHE_SIZE; i++) {
    const int *x = cache[i].key;
    if (x[0] == key[0] && x[1] == key[1] && x[2] == key[2] && x[3] == key[3] &&
        x[4] == key[4] && x[5] == key[5]) {
      return cache[i].plan;
    }
  }
  return NULL;
}

/*******************************************************************************
 * \brief Adds an fft plan to the cache. Assumes ownership of plan's memory.
 * \author Ole Schuett, Frederick Stein
 ******************************************************************************/
static void add_plan_to_cache(const int key[6], fftw_plan *plan) {
  const int i = cache_oldest_entry;
  cache_oldest_entry = (cache_oldest_entry + 1) % FFTW_CACHE_SIZE;
  if (cache[i].plan != NULL) {
    fftw_destroy_plan(*cache[i].plan);
    free(cache[i].plan);
  }
  cache[i].key[0] = key[0];
  cache[i].key[1] = key[1];
  cache[i].key[2] = key[2];
  cache[i].key[3] = key[3];
  cache[i].key[4] = key[4];
  cache[i].key[5] = key[5];
  cache[i].plan = plan;
}
#endif

/*******************************************************************************
 * \brief Initialize the FFT library (if not done externally).
 * \author Frederick Stein, Ole Schuett
 ******************************************************************************/
void fft_fftw_init_lib(const fftw_plan_type fftw_planning_flag,
                       const bool use_fft_mpi) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  if (is_initialized) {
    return;
  }
  memset(cache, 0, sizeof(cache_entry) * FFTW_CACHE_SIZE);
  cache_oldest_entry = 0;

  is_initialized = true;
  fftw_init_threads();

  fftw_planning_mode = fftw_planning_flag;
  switch (fftw_planning_flag) {
  case FFT_ESTIMATE:
    fftw_planning_mode = FFTW_ESTIMATE;
    break;
  case FFT_MEASURE:
    fftw_planning_mode = FFTW_MEASURE;
    break;
  case FFT_PATIENT:
    fftw_planning_mode = FFTW_PATIENT;
    break;
  case FFT_EXHAUSTIVE:
    fftw_planning_mode = FFTW_EXHAUSTIVE;
    break;
  default:
    assert(0 && "Unknown FFTW planning flag.");
  }

  // FIXME:
  // Some systems apparently do not support memory alignment
#if defined(__FFTW3_UNALIGNED)
  fftw_planning_mode += FFTW_UNALIGNED
#endif

#if defined(__parallel) && defined(__FFTW3_MPI)
      use_fftw_mpi = use_fft_mpi;
  fftw_mpi_init();
#else
  (void)use_fft_mpi;
  use_fftw_mpi = false;
#endif
  if (use_fftw_mpi)
    printf("Using FFTW MPI\n");
#else
  (void)fftw_planning_flag;
  (void)use_fft_mpi;
#endif
}

/*******************************************************************************
 * \brief Finalize the FFT library (if not done externally).
 * \author Frederick Stein, Ole Schuett
 ******************************************************************************/
void fft_fftw_finalize_lib() {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  if (!is_initialized) {
    return;
  }
  for (int i = 0; i < FFTW_CACHE_SIZE; i++) {
    if (cache[i].plan != NULL) {
      fftw_destroy_plan(*cache[i].plan);
      free(cache[i].plan);
    }
  }
  is_initialized = false;
  fftw_planning_mode = -1;
#if defined(__parallel) && defined(__FFTW3_MPI)
  fftw_mpi_cleanup();
#else
  fftw_cleanup();
#endif
#endif
}

/*******************************************************************************
 * \brief Whether a compound MPI implementation of FFT is available.
 * \author Frederick Stein
 ******************************************************************************/
bool fft_fftw_lib_use_mpi() {
#if defined(__FFTW3) && defined(__parallel) && defined(__FFTW3_MPI)
  return use_fftw_mpi;
#else
  return false;
#endif
}

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_allocate_double(const int length, double **buffer) {
#if defined(__FFTW3)
  assert(buffer != NULL);
  assert(*buffer == NULL);
  *buffer = fftw_alloc_real(length);
#else
  (void)length;
  (void)buffer;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_allocate_complex(const int length, double complex **buffer) {
#if defined(__FFTW3)
  assert(buffer != NULL);
  assert(*buffer == NULL);
  *buffer = fftw_alloc_complex(length);
#else
  (void)length;
  (void)buffer;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Free buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_free_double(double *buffer) {
#if defined(__FFTW3)
  fftw_free(buffer);
#else
  (void)buffer;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Free buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_free_complex(double complex *buffer) {
#if defined(__FFTW3)
  fftw_free(buffer);
#else
  (void)buffer;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

#if defined(__FFTW3)
/*******************************************************************************
 * \brief Create plan of a 1D FFT.
 * \author Frederick Stein
 ******************************************************************************/
fftw_plan *fft_fftw_create_1d_plan(const int direction, const int fft_size,
                                   const int number_of_ffts,
                                   const bool transpose_rs,
                                   const bool transpose_gs) {
  const int key[6] = {1 + FFTW_TRANSPOSE_RS * transpose_rs +
                          FFTW_TRANSPOSE_GS * transpose_gs,
                      grid_mpi_comm_c2f(grid_mpi_comm_null),
                      direction,
                      fft_size,
                      number_of_ffts,
                      0};
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    const int nthreads = omp_get_max_threads();
    fftw_plan_with_nthreads(nthreads);
    const int rank = 1;
    const int n[] = {fft_size};
    const int howmany = number_of_ffts;
    const int *inembed = n;
    const int *onembed = n;
    int idist = fft_size;
    int odist = fft_size;
    int istride = 1;
    int ostride = 1;
    if (transpose_rs) {
      istride = number_of_ffts;
      idist = 1;
    }
    if (transpose_gs) {
      ostride = number_of_ffts;
      odist = 1;
    }
    double complex *buffer_1 = fftw_alloc_complex(fft_size * number_of_ffts);
    double complex *buffer_2 = fftw_alloc_complex(fft_size * number_of_ffts);
    plan = malloc(sizeof(fftw_plan));
    if (direction == FFTW_FORWARD) {
      *plan = fftw_plan_many_dft(rank, n, howmany, buffer_1, inembed, istride,
                                 idist, buffer_2, onembed, ostride, odist,
                                 FFTW_FORWARD, fftw_planning_mode);
    } else {
      *plan = fftw_plan_many_dft(rank, n, howmany, buffer_1, onembed, ostride,
                                 odist, buffer_2, inembed, istride, idist,
                                 FFTW_BACKWARD, fftw_planning_mode);
    }
    add_plan_to_cache(key, plan);
    fftw_free(buffer_1);
    fftw_free(buffer_2);
  }
  return plan;
}

/*******************************************************************************
 * \brief Create plan of a 1D FFT.
 * \author Frederick Stein
 ******************************************************************************/
fftw_plan *fft_fftw_create_2d_plan(const int direction, const int fft_size[2],
                                   const int number_of_ffts,
                                   const bool transpose_rs,
                                   const bool transpose_gs) {
  const int key[6] = {2 + FFTW_TRANSPOSE_RS * transpose_rs +
                          FFTW_TRANSPOSE_GS * transpose_gs,
                      grid_mpi_comm_c2f(grid_mpi_comm_null),
                      direction,
                      fft_size[1],
                      fft_size[0],
                      number_of_ffts};
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    const int nthreads = omp_get_max_threads();
    fftw_plan_with_nthreads(nthreads);
    const int rank = 2;
    const int *n = &fft_size[0];
    const int howmany = number_of_ffts;
    const int *inembed = n;
    const int *onembed = n;
    int idist = fft_size[0] * fft_size[1];
    int odist = fft_size[0] * fft_size[1];
    int istride = 1;
    int ostride = 1;
    if (transpose_rs) {
      istride = number_of_ffts;
      idist = 1;
    }
    if (transpose_gs) {
      ostride = number_of_ffts;
      odist = 1;
    }
    double complex *buffer_1 =
        fftw_alloc_complex(fft_size[0] * fft_size[1] * number_of_ffts);
    double complex *buffer_2 =
        fftw_alloc_complex(fft_size[0] * fft_size[1] * number_of_ffts);
    plan = malloc(sizeof(fftw_plan));
    if (direction == FFTW_FORWARD) {
      *plan = fftw_plan_many_dft(rank, n, howmany, buffer_1, inembed, istride,
                                 idist, buffer_2, onembed, ostride, odist,
                                 FFTW_FORWARD, fftw_planning_mode);
    } else {
      *plan = fftw_plan_many_dft(rank, n, howmany, buffer_1, onembed, ostride,
                                 odist, buffer_2, inembed, istride, idist,
                                 FFTW_BACKWARD, fftw_planning_mode);
    }
    add_plan_to_cache(key, plan);
    fftw_free(buffer_1);
    fftw_free(buffer_2);
  }
  return plan;
}

/*******************************************************************************
 * \brief Create plan of a 1D FFT.
 * \author Frederick Stein
 ******************************************************************************/
fftw_plan *fft_fftw_create_3d_plan(const int direction, const int fft_size[3]) {
  // add
  const int key[6] = {
      3,           direction,   grid_mpi_comm_c2f(grid_mpi_comm_null),
      fft_size[2], fft_size[1], fft_size[0]};
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    const int nthreads = omp_get_max_threads();
    fftw_plan_with_nthreads(nthreads);
    double complex *buffer_1 =
        fftw_alloc_complex(fft_size[0] * fft_size[1] * fft_size[2]);
    double complex *buffer_2 =
        fftw_alloc_complex(fft_size[0] * fft_size[1] * fft_size[2]);
    plan = malloc(sizeof(fftw_plan));
    *plan = fftw_plan_dft_3d(fft_size[2], fft_size[1], fft_size[0], buffer_1,
                             buffer_2, direction, fftw_planning_mode);
    add_plan_to_cache(key, plan);
    fftw_free(buffer_1);
    fftw_free(buffer_2);
  }
  return plan;
}

#if defined(__parallel) && defined(__FFTW3_MPI)
/*******************************************************************************
 * \brief Create plan of a 1D FFT.
 * \author Frederick Stein
 ******************************************************************************/
fftw_plan *fft_fftw_create_distributed_3d_plan(const int direction,
                                               const int fft_size[3],
                                               const grid_mpi_comm comm) {
  const int key[6] = {3,           grid_mpi_comm_c2f(comm),
                      direction,   fft_size[2],
                      fft_size[1], fft_size[0]};
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    const int nthreads = omp_get_max_threads();
    fftw_plan_with_nthreads(nthreads);
    const int block_size_1 =
        (fft_size[1] + grid_mpi_comm_size(comm) - 1) / grid_mpi_comm_size(comm);
    const int block_size_2 =
        (fft_size[2] + grid_mpi_comm_size(comm) - 1) / grid_mpi_comm_size(comm);
    ptrdiff_t local_n2, local_2_start;
    ptrdiff_t local_n1, local_1_start;
    const ptrdiff_t n[3] = {fft_size[2], fft_size[1], fft_size[0]};
    const int buffer_size = fftw_mpi_local_size_many_transposed(
        3, n, 1, block_size_2, block_size_1, comm, &local_n2, &local_2_start,
        &local_n1, &local_1_start);
    double complex *buffer_1 = fftw_alloc_complex(buffer_size);
    double complex *buffer_2 = fftw_alloc_complex(buffer_size);
    plan = malloc(sizeof(fftw_plan));
    if (direction == FFTW_FORWARD) {
      *plan = fftw_mpi_plan_many_dft(
          3, n, 1, block_size_2, block_size_1, buffer_1, buffer_2, comm,
          direction, fftw_planning_mode + FFTW_MPI_TRANSPOSED_OUT);
    } else {
      *plan = fftw_mpi_plan_many_dft(
          3, n, 1, block_size_1, block_size_2, buffer_1, buffer_2, comm,
          direction, fftw_planning_mode + FFTW_MPI_TRANSPOSED_IN);
    }
    add_plan_to_cache(key, plan);
    fftw_free(buffer_1);
    fftw_free(buffer_2);
  }
  return plan;
}
#endif
#endif

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_fftw_1d_fw_local(const int fft_size, const int number_of_ffts,
                          const bool transpose_rs, const bool transpose_gs,
                          double complex *grid_in, double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  fftw_plan *plan = fft_fftw_create_1d_plan(
      FFTW_FORWARD, fft_size, number_of_ffts, transpose_rs, transpose_gs);
  fftw_execute_dft(*plan, grid_in, grid_out);
#else
  (void)fft_size;
  (void)number_of_ffts;
  (void)grid_in;
  (void)grid_out;
  (void)transpose_rs;
  (void)transpose_gs;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Naive implementation of backwards FFT to transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_fftw_1d_bw_local(const int fft_size, const int number_of_ffts,
                          const bool transpose_rs, const bool transpose_gs,
                          double complex *grid_in, double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  fftw_plan *plan = fft_fftw_create_1d_plan(
      FFTW_BACKWARD, fft_size, number_of_ffts, transpose_rs, transpose_gs);
  fftw_execute_dft(*plan, grid_in, grid_out);
#else
  (void)fft_size;
  (void)number_of_ffts;
  (void)grid_in;
  (void)grid_out;
  (void)transpose_rs;
  (void)transpose_gs;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Local transposition.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_transpose_local(double complex *grid,
                              double complex *grid_transposed,
                              const int number_of_columns_grid,
                              const int number_of_rows_grid) {
#pragma omp parallel for collapse(2) default(none)                             \
    shared(grid, grid_transposed, number_of_columns_grid, number_of_rows_grid)
  for (int column_index = 0; column_index < number_of_columns_grid;
       column_index++) {
    for (int row_index = 0; row_index < number_of_rows_grid; row_index++) {
      grid_transposed[column_index * number_of_rows_grid + row_index] =
          grid[row_index * number_of_columns_grid + column_index];
    }
  }
}

/*******************************************************************************
 * \brief Naive implementation of 2D FFT (transposed format, no normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_2d_fw_local(const int fft_size[2], const int number_of_ffts,
                          const bool transpose_rs, const bool transpose_gs,
                          double complex *grid_in, double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  fftw_plan *plan = fft_fftw_create_2d_plan(
      FFTW_FORWARD, fft_size, number_of_ffts, transpose_rs, transpose_gs);
  fftw_execute_dft(*plan, grid_in, grid_out);
#else
  (void)fft_size;
  (void)number_of_ffts;
  (void)grid_in;
  (void)grid_out;
  (void)transpose_rs;
  (void)transpose_gs;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs local 2D FFT (reverse to fw routine, no normalization).
 * \note fft_2d_bw_local(grid_gs, grid_rs, n1, n2, m) is the reverse to
 * fft_2d_rw_local(grid_rs, grid_gs, n1, n2, m) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_2d_bw_local(const int fft_size[2], const int number_of_ffts,
                          const bool transpose_rs, const bool transpose_gs,
                          double complex *grid_in, double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  fftw_plan *plan = fft_fftw_create_2d_plan(
      FFTW_BACKWARD, fft_size, number_of_ffts, transpose_rs, transpose_gs);
  fftw_execute_dft(*plan, grid_in, grid_out);
#else
  (void)fft_size;
  (void)number_of_ffts;
  (void)grid_in;
  (void)grid_out;
  (void)transpose_rs;
  (void)transpose_gs;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs local 3D FFT (no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_3d_fw_local(const int fft_size[3], double complex *grid_in,
                          double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  fftw_plan *plan = fft_fftw_create_3d_plan(FFTW_FORWARD, fft_size);
  fftw_execute_dft(*plan, grid_in, grid_out);
#else
  (void)fft_size;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs local 3D FFT (reverse to fw routine, no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_3d_bw_local(const int fft_size[3], double complex *grid_in,
                          double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  fftw_plan *plan = fft_fftw_create_3d_plan(FFTW_BACKWARD, fft_size);
  fftw_execute_dft(*plan, grid_in, grid_out);
#else
  (void)fft_size;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Return buffer size and local sizes and start for distributed 2D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_fftw_2d_distributed_sizes(const int npts_global[2],
                                  const int number_of_ffts,
                                  const grid_mpi_comm comm, int *local_n0,
                                  int *local_n0_start) {
#if defined(__FFTW3) && defined(__parallel) && defined(__FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(use_fftw_mpi);
  assert(npts_global[0] > 0);
  assert(npts_global[1] > 0);
  assert(number_of_ffts > 0);
  const ptrdiff_t n[2] = {npts_global[0], npts_global[1]};
  const ptrdiff_t howmany = number_of_ffts;
  const ptrdiff_t block_size = (npts_global[0] + grid_mpi_comm_size(comm) - 1) /
                               grid_mpi_comm_size(comm);
  ptrdiff_t my_local_n0, my_local_n0_start;
  const ptrdiff_t buffer_size = fftw_mpi_local_size_many(
      2, n, howmany, block_size, comm, &my_local_n0, &my_local_n0_start);
  *local_n0 = my_local_n0;
  *local_n0_start = my_local_n0_start;
  return buffer_size;
#else
  (void)npts_global;
  (void)number_of_ffts;
  (void)comm;
  (void)local_n0;
  (void)local_n0_start;
  assert(0 && "The grid library was not compiled with FFTW support.");
  return -1;
#endif
}

/*******************************************************************************
 * \brief Return buffer size and local sizes and start for distributed 3D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_fftw_3d_distributed_sizes(const int npts_global[3],
                                  const grid_mpi_comm comm, int *local_n2,
                                  int *local_n2_start, int *local_n1,
                                  int *local_n1_start) {
#if defined(__FFTW3) && defined(__parallel) && defined(__FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(use_fftw_mpi);
  assert(npts_global[0] > 0);
  assert(npts_global[1] > 0);
  assert(npts_global[2] > 0);
  const ptrdiff_t n[3] = {npts_global[2], npts_global[1], npts_global[0]};
  ptrdiff_t my_local_n2, my_local_n2_start;
  ptrdiff_t my_local_n1, my_local_n1_start;
  const ptrdiff_t block_size_2 =
      (npts_global[2] + grid_mpi_comm_size(comm) - 1) /
      grid_mpi_comm_size(comm);
  const ptrdiff_t block_size_1 =
      (npts_global[1] + grid_mpi_comm_size(comm) - 1) /
      grid_mpi_comm_size(comm);
  const ptrdiff_t my_buffer_size = fftw_mpi_local_size_many_transposed(
      3, n, 1, block_size_2, block_size_1, comm, &my_local_n2,
      &my_local_n2_start, &my_local_n1, &my_local_n1_start);
  *local_n2 = my_local_n2;
  *local_n2_start = my_local_n2_start;
  *local_n1 = my_local_n1;
  *local_n1_start = my_local_n1_start;
  return my_buffer_size;
#else
  (void)npts_global;
  (void)comm;
  (void)local_n2;
  (void)local_n2_start;
  (void)local_n1;
  (void)local_n1_start;
  assert(0 && "The grid library was not compiled with FFTW support.");
  return -1;
#endif
}

/*******************************************************************************
 * \brief Performs a distributed 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_2d_fw_distributed(const int npts_global[2],
                                const int number_of_ffts,
                                const grid_mpi_comm comm,
                                double complex *grid_in,
                                double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  assert(use_fftw_mpi);
  assert(grid_in != NULL);
  assert(grid_out != NULL);
  assert(npts_global[0] > 0);
  assert(npts_global[1] > 0);
  assert(number_of_ffts > 0);
  (void)comm;
#else
  (void)npts_global;
  (void)number_of_ffts;
  (void)comm;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs a distributed 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_3d_fw_distributed(const int npts_global[3],
                                const grid_mpi_comm comm,
                                double complex *grid_in,
                                double complex *grid_out) {
#if defined(__FFTW3) && defined(__parallel) && defined(__FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(use_fftw_mpi);
  fftw_plan *plan =
      fft_fftw_create_distributed_3d_plan(FFTW_FORWARD, npts_global, comm);
  fftw_mpi_execute_dft(*plan, grid_in, grid_out);
#else
  (void)npts_global;
  (void)comm;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with FFTW and MPI support.");
#endif
}

/*******************************************************************************
 * \brief Performs a distributed 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_3d_bw_distributed(const int npts_global[3],
                                const grid_mpi_comm comm,
                                double complex *grid_in,
                                double complex *grid_out) {
#if defined(__FFTW3) && defined(__parallel) && defined(__FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(use_fftw_mpi);
  fftw_plan *plan =
      fft_fftw_create_distributed_3d_plan(FFTW_BACKWARD, npts_global, comm);
  fftw_mpi_execute_dft(*plan, grid_in, grid_out);
#else
  (void)npts_global;
  (void)comm;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with FFTW and MPI support.");
#endif
}

// EOF
