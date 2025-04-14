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

/*******************************************************************************
 * \brief Static variables for retaining objects that are expensive to create.
 * \author Ole Schuett
 ******************************************************************************/
typedef struct {
  int key[4];
  fftw_plan *plan;
} cache_entry;

#define FFTW_CACHE_SIZE 32
static cache_entry cache[FFTW_CACHE_SIZE];
static int cache_oldest_entry = 0; // used for LRU eviction

static bool is_initialized = false;

/*******************************************************************************
 * \brief Fetches an fft plan from the cache. Returns NULL if not found.
 * \author Ole Schuett
 ******************************************************************************/
static fftw_plan *lookup_plan_from_cache(const int key[4]) {
  assert(is_initialized);
  for (int i = 0; i < FFTW_CACHE_SIZE; i++) {
    const int *x = cache[i].key;
    if (x[0] == key[0] && x[1] == key[1] && x[2] == key[2] && x[3] == key[3]) {
      return cache[i].plan;
    }
  }
  return NULL;
}

/*******************************************************************************
 * \brief Adds an fft plan to the cache. Assumes ownership of plan's memory.
 * \author Ole Schuett
 ******************************************************************************/
static void add_plan_to_cache(const int key[4], fftw_plan *plan) {
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
  cache[i].plan = plan;
}
#endif

/*******************************************************************************
 * \brief Initialize the FFT library (if not done externally).
 * \author Frederick Stein, Ole Schuett
 ******************************************************************************/
void fft_fftw_init_lib() {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  if (is_initialized) {
    return;
  }
  memset(cache, 0, sizeof(cache_entry) * FFTW_CACHE_SIZE);
  cache_oldest_entry = 0;

  is_initialized = true;
  fftw_init_threads();
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
  fftw_cleanup();
#endif
}

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_allocate_double(const int length, double **buffer) {
  assert(buffer != NULL);
  assert(*buffer == NULL);
#if defined(__FFTW3)
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
  assert(buffer != NULL);
  assert(*buffer == NULL);
#if defined(__FFTW3)
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

/*******************************************************************************
 * \brief Create plan of a 1D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_create_1d_plan(double complex *grid_rs, double complex *grid_gs,
                             const int fft_size, const int number_of_ffts,
                             grid_fft_fftw_plan *plan_fw,
                             grid_fft_fftw_plan *plan_bw) {
#if defined(__FFTW3)
if (false) plan_fw = lookup_plan_from_cache(
    (const int[4]){fft_size, number_of_ffts, 0, 0});
  const int rank = 1;
  const int n[] = {fft_size};
  const int howmany = number_of_ffts;
  const int idist = 1;
  const int odist = fft_size;
  const int istride = number_of_ffts;
  const int ostride = 1;
  const int *inembed = n;
  const int *onembed = n;
  *plan_fw = fftw_plan_many_dft(rank, n, howmany, grid_rs, inembed, istride,
                                idist, grid_gs, onembed, ostride, odist,
                                FFTW_FORWARD, FFTW_ESTIMATE);
  *plan_bw = fftw_plan_many_dft(rank, n, howmany, grid_gs, onembed, ostride,
                                odist, grid_rs, inembed, istride, idist,
                                FFTW_BACKWARD, FFTW_ESTIMATE);
                                if (false) add_plan_to_cache(
                                    (const int[4]){fft_size, number_of_ffts, 0, 0},
                                    plan_fw);
#else
  (void)grid_rs;
  (void)grid_gs;
  (void)fft_size;
  (void)number_of_ffts;
  (void)plan_fw;
  (void)plan_bw;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Create plan of a 1D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_create_2d_plan(double complex *grid_rs, double complex *grid_gs,
                             const int fft_size[2], const int number_of_ffts,
                             grid_fft_fftw_plan *plan_fw,
                             grid_fft_fftw_plan *plan_bw) {
#if defined(__FFTW3)
  const int rank = 2;
  const int n[] = {fft_size[1], fft_size[0]};
  const int howmany = number_of_ffts;
  const int idist = 1;
  const int odist = fft_size[0] * fft_size[1];
  const int istride = number_of_ffts;
  const int ostride = 1;
  const int *inembed = n;
  const int *onembed = n;
  *plan_fw = fftw_plan_many_dft(rank, n, howmany, grid_rs, inembed, istride,
                                idist, grid_gs, onembed, ostride, odist,
                                FFTW_FORWARD, FFTW_ESTIMATE);
  *plan_bw = fftw_plan_many_dft(rank, n, howmany, grid_gs, onembed, ostride,
                                odist, grid_rs, inembed, istride, idist,
                                FFTW_BACKWARD, FFTW_ESTIMATE);
#else
  (void)grid_rs;
  (void)grid_gs;
  (void)fft_size;
  (void)number_of_ffts;
  (void)plan_fw;
  (void)plan_bw;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Create plan of a 1D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_create_3d_plan(double complex *grid_rs, double complex *grid_gs,
                             const int fft_size[3], grid_fft_fftw_plan *plan_fw,
                             grid_fft_fftw_plan *plan_bw) {
#if defined(__FFTW3)
  *plan_fw = fftw_plan_dft_3d(fft_size[2], fft_size[1], fft_size[0], grid_rs,
                              grid_gs, FFTW_FORWARD, FFTW_ESTIMATE);
  *plan_bw = fftw_plan_dft_3d(fft_size[2], fft_size[1], fft_size[0], grid_gs,
                              grid_rs, FFTW_BACKWARD, FFTW_ESTIMATE);
#else
  (void)grid_rs;
  (void)grid_gs;
  (void)fft_size;
  (void)plan_fw;
  (void)plan_bw;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Create plan of a 1D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_free_plan(grid_fft_fftw_plan *plan_fw,
                        grid_fft_fftw_plan *plan_bw) {
#if defined(__FFTW3)
  if (plan_fw != NULL)
    fftw_destroy_plan(*plan_fw);
  if (plan_bw != NULL)
    fftw_destroy_plan(*plan_bw);
#else
  (void)plan_fw;
  (void)plan_bw;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_fftw_1d_fw_local(const grid_fft_fftw_plan plan_fw,
                          double complex *grid_in, double complex *grid_out) {
#if defined(__FFTW3)
  fftw_execute_dft(plan_fw, grid_in, grid_out);
#else
  (void)plan_fw;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Naive implementation of backwards FFT to transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_fftw_1d_bw_local(const grid_fft_fftw_plan plan_bw,
                          double complex *grid_in, double complex *grid_out) {
#if defined(__FFTW3)
  fftw_execute_dft(plan_bw, grid_in, grid_out);
#else
  (void)plan_bw;
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
void fft_fftw_2d_fw_local(const grid_fft_fftw_plan plan_fw,
                          double complex *grid_in, double complex *grid_out) {
#if defined(__FFTW3)
  fftw_execute_dft(plan_fw, grid_in, grid_out);
#else
  (void)plan_fw;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs local 2D FFT (reverse to fw routine, no normalization).
 * \note fft_2d_bw_local(grid_gs, grid_rs, n1, n2, m) is the reverse to
 * fft_2d_rw_local(grid_rs, grid_gs, n1, n2, m) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_2d_bw_local(const grid_fft_fftw_plan plan_bw,
                          double complex *grid_in, double complex *grid_out) {
#if defined(__FFTW3)
  fftw_execute_dft(plan_bw, grid_in, grid_out);
#else
  (void)plan_bw;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs local 3D FFT (no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_3d_fw_local(const grid_fft_fftw_plan plan_fw,
                          double complex *grid_in, double complex *grid_out) {
#if defined(__FFTW3)
  fftw_execute_dft(plan_fw, grid_in, grid_out);
#else
  (void)plan_fw;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs local 3D FFT (reverse to fw routine, no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_3d_bw_local(const grid_fft_fftw_plan plan_bw,
                          double complex *grid_in, double complex *grid_out) {
#if defined(__FFTW3)
  fftw_execute_dft(plan_bw, grid_in, grid_out);
#else
  (void)plan_bw;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

// EOF
