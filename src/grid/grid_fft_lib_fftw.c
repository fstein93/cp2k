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
#if defined(__parallel)
#include <fftw3-mpi.h>
#endif

/*******************************************************************************
 * \brief Static variables for retaining objects that are expensive to create.
 * \author Ole Schuett
 ******************************************************************************/
typedef struct {
  int key[5];
  fftw_plan *plan;
} cache_entry;

#define FFTW_CACHE_SIZE 32
static cache_entry cache[FFTW_CACHE_SIZE];
static int cache_oldest_entry = 0; // used for LRU eviction

static bool is_initialized = false;

static int fftw_planning_mode = -1;
static bool use_fftw_mpi = false;

/*******************************************************************************
 * \brief Fetches an fft plan from the cache. Returns NULL if not found.
 * \author Ole Schuett, Frederick Stein
 ******************************************************************************/
static fftw_plan *lookup_plan_from_cache(const int key[5]) {
  assert(is_initialized);
  for (int i = 0; i < FFTW_CACHE_SIZE; i++) {
    const int *x = cache[i].key;
    if (x[0] == key[0] && x[1] == key[1] && x[2] == key[2] && x[3] == key[3] &&
        x[4] == key[4]) {
      return cache[i].plan;
    }
  }
  return NULL;
}

/*******************************************************************************
 * \brief Adds an fft plan to the cache. Assumes ownership of plan's memory.
 * \author Ole Schuett, Frederick Stein
 ******************************************************************************/
static void add_plan_to_cache(const int key[5], fftw_plan *plan) {
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

#if defined(__FFTW3_UNALIGNED)
  fftw_planning_mode += FFTW_UNALIGNED
#endif

#if defined(__parallel)
      use_fftw_mpi = use_fft_mpi;
  fftw_mpi_init();
#else
  (void)use_fft_mpi;
  use_fftw_mpi = false;
#endif
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
#if defined(__parallel)
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
#if defined(__FFTW3)
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
                                   const int number_of_ffts) {
  const int key[5] = {1, direction, fft_size, number_of_ffts, 0};
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    const int nthreads = omp_get_max_threads();
    fftw_plan_with_nthreads(nthreads);
    const int rank = 1;
    const int n[] = {fft_size};
    const int howmany = number_of_ffts;
    const int idist = 1;
    const int odist = fft_size;
    const int istride = number_of_ffts;
    const int ostride = 1;
    const int *inembed = n;
    const int *onembed = n;
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
                                   const int number_of_ffts) {
  const int key[5] = {2, direction, fft_size[1], fft_size[0], number_of_ffts};
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    const int nthreads = omp_get_max_threads();
    fftw_plan_with_nthreads(nthreads);
    const int rank = 2;
    const int *n = &fft_size[0];
    const int howmany = number_of_ffts;
    const int *inembed = n;
    const int *onembed = n;
    const int idist = 1;
    const int odist = fft_size[0] * fft_size[1];
    const int istride = number_of_ffts;
    const int ostride = 1;
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
  const int key[5] = {3, direction, fft_size[2], fft_size[1], fft_size[0]};
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
#endif

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_fftw_1d_fw_local(const int fft_size, const int number_of_ffts,
                          double complex *grid_in, double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  fftw_plan *plan =
      fft_fftw_create_1d_plan(FFTW_FORWARD, fft_size, number_of_ffts);
  fftw_execute_dft(*plan, grid_in, grid_out);
#else
  (void)fft_size;
  (void)number_of_ffts;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Naive implementation of backwards FFT to transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_fftw_1d_bw_local(const int fft_size, const int number_of_ffts,
                          double complex *grid_in, double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  fftw_plan *plan =
      fft_fftw_create_1d_plan(FFTW_BACKWARD, fft_size, number_of_ffts);
  fftw_execute_dft(*plan, grid_in, grid_out);
#else
  (void)fft_size;
  (void)number_of_ffts;
  (void)grid_in;
  (void)grid_out;
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
                          double complex *grid_in, double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  fftw_plan *plan =
      fft_fftw_create_2d_plan(FFTW_FORWARD, fft_size, number_of_ffts);
  fftw_execute_dft(*plan, grid_in, grid_out);
#else
  (void)fft_size;
  (void)number_of_ffts;
  (void)grid_in;
  (void)grid_out;
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
                          double complex *grid_in, double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  fftw_plan *plan =
      fft_fftw_create_2d_plan(FFTW_BACKWARD, fft_size, number_of_ffts);
  fftw_execute_dft(*plan, grid_in, grid_out);
#else
  (void)fft_size;
  (void)number_of_ffts;
  (void)grid_in;
  (void)grid_out;
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

// EOF
