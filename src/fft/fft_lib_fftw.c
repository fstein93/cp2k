/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "fft_lib_fftw.h"
#include "fft_timer.h"
#include "fft_utils.h"

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
#define __USE_FFTW3_MPI
#endif

#define KEY_SIZE 13

/*******************************************************************************
 * \brief Static variables for retaining objects that are expensive to create.
 * \author Ole Schuett, Frederick Stein
 ******************************************************************************/
typedef struct {
  // The key contains
  // 0: rank (see below)
  // 1: associated Fortran communicator handle (to store it as an integer)
  // 2: Number of associated OpenMP threads
  // 3: direction (forward/backward)
  // 4, 5, 6: FFT sizes (or FFT sizes and number of FFTs)
  // 7, 8, 9: input stride
  // 10, 11, 12: output strides
  int key[KEY_SIZE];
  fftw_plan *plan;
} cache_entry;

// We need to reserve more space because of the different combinations
// (local/distributed, C2C/R2C) This works to run all tests
#define FFTW_CACHE_SIZE 256
static cache_entry cache[FFTW_CACHE_SIZE];
static int cache_oldest_entry = 0; // used for LRU eviction

static bool is_initialized = false;

static int fftw_planning_mode = -1;

// These constants encode transposition and MPI usage into the key to cache the
// plans
// Modulo 4 encodes the rank (1, 2, 3)
// 4 == 2^2
#define FFTW_R2C 4
// 8 == 2^3
#define FFTW_INPLACE 8

/*******************************************************************************
 * \brief Fetches an fft plan from the cache. Returns NULL if not found.
 * \author Ole Schuett, Frederick Stein
 ******************************************************************************/
static fftw_plan *lookup_plan_from_cache(const int key[KEY_SIZE]) {
  assert(is_initialized);
  for (int i = 0; i < FFTW_CACHE_SIZE; i++) {
    const int *x = cache[i].key;
    if (memcmp(key, x, KEY_SIZE * sizeof(int)) == 0) {
      return cache[i].plan;
    }
  }
  return NULL;
}

/*******************************************************************************
 * \brief Adds an fft plan to the cache. Assumes ownership of plan's memory.
 * \author Ole Schuett, Frederick Stein
 ******************************************************************************/
static void add_plan_to_cache(const int key[KEY_SIZE], fftw_plan *plan) {
  const int i = cache_oldest_entry;
  cache_oldest_entry = (cache_oldest_entry + 1) % FFTW_CACHE_SIZE;
  if (cache[i].plan != NULL) {
    fprintf(stderr,
            "Storage to cache FFTW plans is full. Delete an old plan...\n");
    fftw_destroy_plan(*cache[i].plan);
    free(cache[i].plan);
  }
  memcpy(cache[i].key, key, KEY_SIZE * sizeof(int));
  cache[i].plan = plan;
}
#endif

static bool use_fftw_mpi = false;
static bool has_guru_interface = false;

#if defined(__FFTW3)
bool is_guru_interface_available() {
  fftw_iodim dims[1];
  dims[0].n = 1;
  dims[0].is = 1;
  dims[0].os = 1;
  fftw_iodim howmany_dims[2];
  howmany_dims[0].n = 1;
  howmany_dims[0].is = 1;
  howmany_dims[0].os = 1;
  howmany_dims[1].n = 1;
  howmany_dims[1].is = 1;
  howmany_dims[1].os = 1;
  double complex buffer1;
  fftw_plan plan =
      fftw_plan_guru_dft(1, dims, 2, howmany_dims, &buffer1, &buffer1,
                         FFTW_FORWARD, fftw_planning_mode);

  if (plan != NULL) {
    fftw_destroy_plan(plan);
    return true;
  } else {
    return false;
  }
}
#endif

#if defined(__USE_FFTW3_MPI)
bool fft_fftw_test_mpi_backend() {
  const int nthreads = omp_get_max_threads();
  const cp_mpi_comm_t comm = cp_mpi_get_comm_world();
  const int fft_size[3] = {2, 2, 2};
  fftw_plan_with_nthreads(nthreads);
  const int block_size_0 =
      (fft_size[0] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
  const int block_size_1 =
      (fft_size[1] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
  ptrdiff_t local_n0, local_0_start;
  ptrdiff_t local_n1, local_1_start;
  const ptrdiff_t n[3] = {fft_size[0], fft_size[1], fft_size[2]};
  const int buffer_size = fftw_mpi_local_size_many_transposed(
      3, n, 1, block_size_0, block_size_1, comm, &local_n0, &local_0_start,
      &local_n1, &local_1_start);
  double complex *buffer_1 = fftw_alloc_complex(buffer_size);
  double complex *buffer_2 = fftw_alloc_complex(buffer_size);
  fftw_plan plan = fftw_mpi_plan_many_dft(
      3, n, 1, block_size_0, block_size_1, buffer_1, buffer_2, comm,
      FFTW_FORWARD, fftw_planning_mode + FFTW_MPI_TRANSPOSED_OUT);
  fftw_free(buffer_1);
  fftw_free(buffer_2);
  if (plan != NULL) {
    fftw_destroy_plan(plan);
    return true;
  } else {
    return false;
  }
}
#endif

/*******************************************************************************
 * \brief Initialize the FFT library (if not done externally).
 * \author Frederick Stein, Ole Schuett
 ******************************************************************************/
void fft_fftw_init_lib(const fftw_plan_type fftw_planning_flag,
                       const bool use_fft_mpi, const bool use_guru_interface,
                       const char *wisdom_file) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  if (is_initialized) {
    return;
  }
  const bool is_print_rank = cp_mpi_comm_rank(cp_mpi_get_comm_world()) == 0;
  memset(cache, 0, sizeof(cache_entry) * FFTW_CACHE_SIZE);
  cache_oldest_entry = 0;

  is_initialized = true;
  // We need a threaded library!
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

  has_guru_interface = use_guru_interface;
  if (has_guru_interface) {
    has_guru_interface = is_guru_interface_available();
    if (is_print_rank) {
      if (has_guru_interface) {
        fprintf(stdout, "Guru interface is available!\n");
      } else {
        fprintf(stdout, "Guru interface is not available!\n");
      }
    }
  } else {
    if (is_print_rank) {
      fprintf(stdout, "Guru interface not in use!\n");
    }
  }

#if defined(__USE_FFTW3_MPI)
  use_fftw_mpi = use_fft_mpi;
  if (use_fftw_mpi) {
    fftw_mpi_init();
    use_fftw_mpi = fft_fftw_test_mpi_backend();
    if (is_print_rank) {
      if (!use_fftw_mpi) {
        fprintf(stderr,
                "Creation of a MPI plan failed! FFTW-MPI will not be used!\n");
      } else {
        fprintf(stdout, "Use FFTW-MPI!\n");
      }
    }
  } else if (is_print_rank) {
    fprintf(stdout, "FFTW-MPI not requested!\n");
  }
  if (is_print_rank) fflush(stdout);
  cp_mpi_barrier(cp_mpi_get_comm_world());
#else
  (void)use_fft_mpi;
  use_fftw_mpi = false;
#endif
  // Export wisdom after intializing the library to ensure correct threading
  // etc.
  if (wisdom_file != NULL) {
    const int error = fftw_import_wisdom_from_filename(wisdom_file);
    if (error != 0 && is_print_rank)
      fprintf(stderr,
              "Importing wisdom failed! Maybe the file does not exist.");
  }
#else
  (void)fftw_planning_flag;
  (void)use_fft_mpi;
  (void)wisdom_file;
#endif
}

/*******************************************************************************
 * \brief Finalize the FFT library (if not done externally).
 * \author Frederick Stein, Ole Schuett
 ******************************************************************************/
void fft_fftw_finalize_lib(const char *wisdom_file) {
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
  // Export wisdom before finalizing the library to ensure storing the correct
  // threading etc.
  if (wisdom_file != NULL) {
    const int error = fftw_export_wisdom_to_filename(wisdom_file);
    if (error != 0 && cp_mpi_comm_rank(cp_mpi_get_comm_world()))
      fprintf(stderr,
              "Exporting wisdom failed! Maybe writing access is missing.");
  }
  is_initialized = false;
  fftw_planning_mode = -1;
#if defined(__USE_FFTW3_MPI)
  fftw_mpi_cleanup();
#else
  fftw_cleanup();
#endif
#else
  (void)wisdom_file;
#endif
}

/*******************************************************************************
 * \brief Whether a distributed FFT implementation is available.
 * \author Frederick Stein
 ******************************************************************************/
bool fft_fftw_lib_use_mpi() {
#if defined(__USE_FFTW3_MPI)
  return use_fftw_mpi;
#else
  return false;
#endif
}

/*******************************************************************************
 * \brief Whether a distributed FFT implementation is available.
 * \author Frederick Stein
 ******************************************************************************/
bool fft_fftw_lib_has_guru_interface() { return has_guru_interface; }

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_allocate_double(const int length, double **buffer) {
#if defined(__FFTW3)
  assert(is_initialized);
  assert(buffer != NULL);
  assert(*buffer == NULL);
  *buffer = fftw_alloc_real(imax(1, length));
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
  assert(is_initialized);
  assert(buffer != NULL);
  assert(*buffer == NULL);
  *buffer = fftw_alloc_complex(imax(1, length));
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
  assert(is_initialized);
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
  assert(is_initialized);
  fftw_free(buffer);
#else
  (void)buffer;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

#if defined(__FFTW3)
/*******************************************************************************
 * \brief Get key from FFT input parameters (1D, C2C-case)
 * \author Frederick Stein
 ******************************************************************************/
void get_key_1d(const int direction, const int fft_size,
                        const int number_of_ffts,
                        const bool transpose_rs,
                        const bool transpose_gs, 
                     const int leading_dimension_rs, const int leading_dimension_gs,
                        const int number_of_threads, const bool inplace, int *key) {
  key[0] = 1 + FFTW_INPLACE * inplace;
  key[1] = cp_mpi_comm_c2f(cp_mpi_get_comm_self());
  key[2] = number_of_threads;
  key[3] = direction;
  key[4] = fft_size;
  key[5] = number_of_ffts;
  key[6] = 0;
  key[7] = transpose_rs ? leading_dimension_rs : 1;
  key[8] = transpose_rs ? 1 : leading_dimension_rs;
  key[9] = 0;
  key[10] = transpose_gs ? leading_dimension_gs : 1;
  key[11] = transpose_gs ? 1 : leading_dimension_gs;
  key[12] = 0;
  }

/*******************************************************************************
 * \brief Get key from FFT input parameters (1D, R2C-case)
 * \author Frederick Stein
 ******************************************************************************/
void get_key_1d_r2c(const int direction, const int fft_size,
                            const int number_of_ffts, const bool transpose_rs,
                            const bool transpose_gs,
                     const int leading_dimension_rs, const int leading_dimension_gs,
                            const int number_of_threads, const bool inplace, int *key) {
  key[0] = 1 + FFTW_R2C + FFTW_INPLACE * inplace;
  key[1] = cp_mpi_comm_c2f(cp_mpi_get_comm_self());
  key[2] = number_of_threads;
  key[3] = direction;
  key[4] = fft_size;
  key[5] = number_of_ffts;
  key[6] = 0;
  key[7] = transpose_rs ? leading_dimension_rs : 1;
  key[8] = transpose_rs ? 1 : leading_dimension_rs;
  key[9] = 0;
  key[10] = transpose_gs ? leading_dimension_gs : 1;
  key[11] = transpose_gs ? 1 : leading_dimension_gs;
  key[12] = 0;
  }

/*******************************************************************************
 * \brief Get key from FFT input parameters (2D, C2C-case)
 * \author Frederick Stein
 ******************************************************************************/
void get_key_2d(const int direction, const int fft_size[2],
                        const int number_of_ffts, const bool transpose_rs,
                        const bool transpose_gs,
                        const int number_of_threads, const bool inplace, int *key) {
  key[0] = 2 + FFTW_INPLACE * inplace;
  key[1] = cp_mpi_comm_c2f(cp_mpi_get_comm_self());
  key[2] = number_of_threads;
  key[3] = direction;
  key[4] = fft_size[0];
  key[5] = fft_size[1];
  key[6] = number_of_ffts;
  key[7] = (transpose_rs ? number_of_ffts : 1) * fft_size[1];
  key[8] = transpose_rs ? number_of_ffts : 1;
  key[9] = transpose_rs ? 1 : fft_size[0] * fft_size[1];
  key[10] = (transpose_gs ? number_of_ffts : 1) * fft_size[1];
  key[11] = transpose_gs ? number_of_ffts : 1;
  key[12] = transpose_gs ? 1 : fft_size[0] * fft_size[1];
                        }

/*******************************************************************************
 * \brief Get key from FFT input parameters (2D, R2C-case)
 * \author Frederick Stein
 ******************************************************************************/
void get_key_2d_r2c(const int direction, const int fft_size[2],
                            const int number_of_ffts, const bool transpose_rs,
                            const bool transpose_gs,
                            const int number_of_threads, const bool inplace, int *key) {
  key[0] = 2 + FFTW_R2C + FFTW_INPLACE * inplace;
  key[1] = cp_mpi_comm_c2f(cp_mpi_get_comm_self());
  key[2] = number_of_threads;
  key[3] = direction;
  key[4] = fft_size[0];
  key[5] = fft_size[1];
  key[6] = number_of_ffts;
  key[7] = (transpose_rs ? number_of_ffts : 1) * fft_size[1];
  key[8] = transpose_rs ? number_of_ffts : 1;
  key[9] = transpose_rs ? 1 : fft_size[0] * fft_size[1];
  key[10] = (transpose_gs ? number_of_ffts : 1) * (fft_size[1] / 2 + 1);
  key[11] = transpose_gs ? number_of_ffts : 1;
  key[12] = transpose_gs ? 1 : fft_size[0] * (fft_size[1] / 2 + 1);
  }

/*******************************************************************************
 * \brief Get key from FFT input parameters (3D, C2C-case)
 * \author Frederick Stein
 ******************************************************************************/
void get_key_3d(const int direction, const int fft_size[3],
                                   const int number_of_threads,
                                   const bool inplace, int *key) {
  key[0] = 3 + FFTW_INPLACE * inplace;
  key[1] = cp_mpi_comm_c2f(cp_mpi_get_comm_self());
  key[2] = number_of_threads;
  key[3] = direction;
  key[4] = fft_size[0];
  key[5] = fft_size[1];
  key[6] = fft_size[2];
  key[7] = fft_size[1] * fft_size[2];
  key[8] = fft_size[2];
  key[9] = 1;
  key[10] = fft_size[1] * fft_size[2];
  key[11] = fft_size[2];
  key[12] = 1;
}

/*******************************************************************************
 * \brief Get key from FFT input parameters (3D, R2C-case)
 * \author Frederick Stein
 ******************************************************************************/
void get_key_3d_r2c(const int direction,
                                       const int fft_size[3],
                                       const int number_of_threads,
                                       const bool inplace, int *key) {
  key[0] = 3 + FFTW_R2C + FFTW_INPLACE * inplace;
  key[1] = cp_mpi_comm_c2f(cp_mpi_get_comm_self());
  key[2] = number_of_threads;
  key[3] = direction;
  key[4] = fft_size[0];
  key[5] = fft_size[1];
  key[6] = fft_size[2];
  key[7] = fft_size[1] * fft_size[2];
  key[8] = fft_size[2];
  key[9] = 1;
  key[10] = fft_size[1] * (fft_size[2]/2+1);
  key[11] = fft_size[2]/2+1;
  key[12] = 1;
}

/*******************************************************************************
 * \brief Get key from FFT input parameters (Guru, C2C-case)
 * \author Frederick Stein
 ******************************************************************************/
void get_key_guru(const int direction, int rank,
                                     const fft_iodim *dims, int howmany_rank,
                                     const fft_iodim *howmany_dims,
                                     const int number_of_threads,
                                     const bool inplace, int *key) {
  assert(rank + howmany_rank <= 3 &&
         "Larger combined ranks than 3 are not implemented\n");

  key[0] = rank + FFTW_INPLACE * inplace;
  key[1] = cp_mpi_comm_c2f(cp_mpi_get_comm_self());
  key[2] = number_of_threads;
  key[3] = direction;
  key[4] = rank > 0 ? dims[0].n : (rank + howmany_rank > 0 ? howmany_dims[0].n : 0);
  key[5] = rank > 1 ? dims[1].n
               : (rank + howmany_rank > 1 ? howmany_dims[1 - rank].n : 0);
  key[6] = rank > 2 ? dims[2].n
               : (rank + howmany_rank > 2 ? howmany_dims[2 - rank].n : 0);
  key[7] = rank > 0 ? dims[0].is
               : (rank + howmany_rank > 0 ? howmany_dims[0].is : 0);
  key[8] = rank > 1 ? dims[1].is
               : (rank + howmany_rank > 1 ? howmany_dims[1 - rank].is : 0);
  key[9] = rank > 2 ? dims[2].is
               : (rank + howmany_rank > 2 ? howmany_dims[2 - rank].is : 0);
  key[10] = rank > 0 ? dims[0].os
               : (rank + howmany_rank > 0 ? howmany_dims[0].os : 0);
  key[11] = rank > 1 ? dims[1].os
               : (rank + howmany_rank > 1 ? howmany_dims[1 - rank].os : 0);
  key[12] = rank > 2 ? dims[2].os
               : (rank + howmany_rank > 2 ? howmany_dims[2 - rank].os : 0);
}

/*******************************************************************************
 * \brief Get key from FFT input parameters (Guru, R2C-case)
 * \author Frederick Stein
 ******************************************************************************/
void get_key_guru_r2c(
    const int direction, int rank, const fft_iodim *dims, int howmany_rank,
    const fft_iodim *howmany_dims, const int number_of_threads,
    const bool inplace, int *key) {
  assert(rank + howmany_rank <= 3 &&
         "Larger combined ranks than 3 are not implemented\n");

  key[0] = rank + FFTW_R2C + FFTW_INPLACE * inplace;
  key[1] = cp_mpi_comm_c2f(cp_mpi_get_comm_self());
  key[2] = number_of_threads;
  key[3] = direction;
  key[4] = rank > 0 ? dims[0].n : (rank + howmany_rank > 0 ? howmany_dims[0].n : 0);
  key[5] = rank > 1 ? dims[1].n
               : (rank + howmany_rank > 1 ? howmany_dims[1 - rank].n : 0);
  key[6] = rank > 2 ? dims[2].n
               : (rank + howmany_rank > 2 ? howmany_dims[2 - rank].n : 0);
  key[7] = rank > 0 ? dims[0].is
               : (rank + howmany_rank > 0 ? howmany_dims[0].is : 0);
  key[8] = rank > 1 ? dims[1].is
               : (rank + howmany_rank > 1 ? howmany_dims[1 - rank].is : 0);
  key[9] = rank > 2 ? dims[2].is
               : (rank + howmany_rank > 2 ? howmany_dims[2 - rank].is : 0);
  key[10] = rank > 0 ? dims[0].os
               : (rank + howmany_rank > 0 ? howmany_dims[0].os : 0);
  key[11] = rank > 1 ? dims[1].os
               : (rank + howmany_rank > 1 ? howmany_dims[1 - rank].os : 0);
  key[12] = rank > 2 ? dims[2].os
               : (rank + howmany_rank > 2 ? howmany_dims[2 - rank].os : 0);
}

#if defined(__USE_FFTW3_MPI)

/*******************************************************************************
 * \brief Get key from FFT input parameters (2D, C2C-case, distributed)
 * \author Frederick Stein
 ******************************************************************************/
void get_key_2d_distributed(const int direction,
                                               const int fft_size[2],
                                               const int number_of_ffts,
                                               const cp_mpi_comm_t comm,
                                                  const int number_of_threads, int *key) {
  key[0] = 2;
  key[1] = cp_mpi_comm_c2f(comm);
  key[2] = number_of_threads;
  key[3] = direction;
  key[4] = fft_size[0];
  key[5] = fft_size[1];
  key[6] = number_of_ffts;
  key[7] = fft_size[1] * number_of_ffts;
  key[8] = number_of_ffts;
  key[9] = 1;
  key[10] = number_of_ffts;
  key[11] = fft_size[0] * number_of_ffts;
  key[12] = 1;
}

/*******************************************************************************
 * \brief Get key from FFT input parameters (2D, R2C-case, distributed)
 * \author Frederick Stein
 ******************************************************************************/
void get_key_2d_r2c_distributed(const int direction,
                                                   const int fft_size[2],
                                                   const int number_of_ffts,
                                                   const cp_mpi_comm_t comm,
                                                  const int number_of_threads, int *key) {
  key[0] = 2 + FFTW_R2C;
  key[1] = cp_mpi_comm_c2f(comm);
  key[2] = number_of_threads;
  key[3] = direction;
  key[4] = fft_size[0];
  key[5] = fft_size[1];
  key[6] = number_of_ffts;
  key[7] = fft_size[1] * number_of_ffts;
  key[8] = number_of_ffts;
  key[9] = 1;
  key[10] = number_of_ffts;
  key[11] = (fft_size[0]/2+1) * number_of_ffts;
  key[12] = 1;
}

/*******************************************************************************
 * \brief Get key from FFT input parameters (3D, C2C-case)
 * \author Frederick Stein
 ******************************************************************************/
void get_key_3d_distributed(const int direction,
                                               const int fft_size[3],
                                               const cp_mpi_comm_t comm,
                                                  const int number_of_threads, int *key) {
  key[0] = 3;
  key[1] = cp_mpi_comm_c2f(comm);
  key[2] = number_of_threads;
  key[3] = direction;
  key[4] = fft_size[0];
  key[5] = fft_size[1];
  key[6] = fft_size[2];
  key[7] = fft_size[1] * fft_size[2];
  key[8] = fft_size[2];
  key[9] = 1;
  key[10] = fft_size[2];
  key[11] = fft_size[0] * fft_size[2];
  key[12] = 1;
}

/*******************************************************************************
 * \brief Get key from FFT input parameters (3D, R2C-case, distributed)
 * \author Frederick Stein
 ******************************************************************************/
void get_key_3d_r2c_distributed(const int direction,
                                                   const int fft_size[3],
                                                   const cp_mpi_comm_t comm,
                                                  const int number_of_threads, int *key) {
  key[0] = 3 + FFTW_R2C;
  key[1] = cp_mpi_comm_c2f(comm);
  key[2] = number_of_threads;
  key[3] = direction;
  key[4] = fft_size[0];
  key[5] = fft_size[1];
  key[6] = fft_size[2];
  key[7] = fft_size[1] * 2*(fft_size[2]/2+1);
  key[8] = 2*(fft_size[2]/2+1);
  key[9] = 1;
  key[10] = 2*(fft_size[2]/2+1);
  key[11] = fft_size[0] * 2*(fft_size[2]/2+1);
  key[12] = 1;
}

/*******************************************************************************
 * \brief Determine buffer size for a local FFT from a key
 * \author Frederick Stein
 ******************************************************************************/
int get_buffer_size_from_key(const int key[KEY_SIZE]) {
  int buffer_size = 0;
  for (int r = 0; r < 3; r++) {
    if (r != key[0]%4-1 || ((key[0] & FFTW_R2C) != FFTW_R2C)) {
      buffer_size += key[4+r]*key[7+r];
    } else {
      buffer_size += (key[4+r]/2+1)*key[7+r];
    }
  }
  int buffer_size_out = 0;
  for (int r = 0; r < 3; r++) {
    if (r != key[0]%4-1 || ((key[0] & FFTW_R2C) != FFTW_R2C)) {
      buffer_size_out += key[4+r]*key[10+r];
    } else {
      buffer_size_out += (key[4+r]/2+1)*key[10+r];
    }
  }
  return imax(buffer_size, buffer_size_out);
}
#endif

/*******************************************************************************
 * \brief Create plan of a local C2C 1D FFT.
 * \author Frederick Stein
 ******************************************************************************/
fftw_plan *
fft_fftw_create_1d_plan(const int key[KEY_SIZE], double complex* grid_in, double complex *grid_out) {
  const int direction = key[3];
  const int fft_size = key[4];
  const int number_of_threads = key[2];
  const int number_of_ffts = key[5];
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_1d_%cw_c2c_Plocal",
           direction == FFTW_FORWARD ? 'f' : 'b');
  const int handle = fft_start_timer(routine_name);
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH,
           "fft_1d_%cw_c2c_Plocal_%i_%i",
           direction == FFTW_FORWARD ? 'f' : 'b', fft_size, number_of_ffts);
  const int handle2 = fft_start_timer(routine_name);
  fftw_plan_with_nthreads(number_of_threads);
  const int rank = 1;
  const int n[] = {key[4]};
  const int howmany = key[5];
  const int *inembed = n;
  const int *onembed = n;
  const int idist = key[8];
  const int odist = key[11];
  const int istride = key[7];
  const int ostride = key[10];
  fftw_plan *plan = malloc(sizeof(fftw_plan));
  if (key[3] == FFTW_FORWARD) {
    *plan = fftw_plan_many_dft(rank, n, howmany, grid_in, inembed, istride,
                                idist, grid_out, onembed, ostride, odist,
                                FFTW_FORWARD, fftw_planning_mode);
  } else {
    *plan = fftw_plan_many_dft(rank, n, howmany, grid_in, onembed, ostride,
                                odist, grid_out, inembed, istride, idist,
                                FFTW_BACKWARD, fftw_planning_mode);
  }
  assert(plan != NULL);
  add_plan_to_cache(key, plan);
  fft_stop_timer(handle2);
  fft_stop_timer(handle);
  return plan;
}
/*******************************************************************************
 * \brief Create plan of a local R2C/C2R 1D FFT.
 * \author Frederick Stein
 ******************************************************************************/
fftw_plan *
fft_fftw_create_1d_plan_r2c(const int key[KEY_SIZE], double *grid_rs, double complex *grid_gs) {
  const int direction = key[3];
  const int fft_size = key[4];
  const int number_of_threads = key[2];
  const int number_of_ffts = key[5];
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_1d_%s_Plocal_%i_%i",
           direction == FFTW_FORWARD ? "fw_r2c" : "bw_c2r", fft_size,
           number_of_ffts);
  const int handle = fft_start_timer(routine_name);
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_1d_%s_Plocal",
           direction == FFTW_FORWARD ? "fw_r2c" : "bw_c2r");
  const int handle2 = fft_start_timer(routine_name);
    fftw_plan_with_nthreads(number_of_threads);
    const int rank = key[0]%4;
    const int n[] = {key[4]};
    const int howmany = key[5];
    const int *inembed = NULL;
    const int *onembed = NULL;
    const int idist = key[8];
    const int odist = key[11];
    const int istride = key[7];
    const int ostride = key[10];
    fftw_plan *plan = malloc(sizeof(fftw_plan));
    if (key[3] == FFTW_FORWARD) {
      *plan = fftw_plan_many_dft_r2c(rank, n, howmany, grid_rs, inembed,
                                     istride, idist, grid_gs, onembed, ostride,
                                     odist, fftw_planning_mode);
    } else {
      // We use the buffers the other way around to prevent
      // out-of-bounds-accesses of the planner if the output array has only the
      // minimum size
      *plan = fftw_plan_many_dft_c2r(
          rank, n, howmany, grid_gs, onembed, ostride, odist,
          grid_rs, inembed, istride, idist, fftw_planning_mode);
    }
    assert(plan != NULL);
    add_plan_to_cache(key, plan);
  fft_stop_timer(handle2);
  fft_stop_timer(handle);
  return plan;
}

/*******************************************************************************
 * \brief Create plan of a local C2C 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
fftw_plan *
fft_fftw_create_2d_plan(const int key[KEY_SIZE], double complex *grid_in, double complex *grid_out) {
  const int direction = key[3];
  const int *fft_size = key+4;
  const int number_of_threads = key[2];
  const int number_of_ffts = key[6];
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_2d_%cw_c2c_Plocal",
           direction == FFTW_FORWARD ? 'f' : 'b');
  const int handle = fft_start_timer(routine_name);
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH,
           "fft_2d_%cw_c2c_Plocal_%i_%i_%i",
           direction == FFTW_FORWARD ? 'f' : 'b', fft_size[0], fft_size[1],
           number_of_ffts);
  const int handle2 = fft_start_timer(routine_name);
    fftw_plan_with_nthreads(number_of_threads);
    const int rank = 2;
    const int *n = fft_size;
    const int howmany = number_of_ffts;
    const int *inembed = n;
    const int *onembed = n;
    const int idist = key[9];
    const int odist = key[12];
    const int istride = key[8];
    const int ostride = key[11];
    fftw_plan *plan = malloc(sizeof(fftw_plan));
    if (key[3] == FFTW_FORWARD) {
      *plan = fftw_plan_many_dft(rank, n, howmany, grid_in, inembed, istride,
                                 idist, grid_out, onembed, ostride, odist,
                                 FFTW_FORWARD, fftw_planning_mode);
    } else {
      *plan = fftw_plan_many_dft(rank, n, howmany, grid_in, onembed, ostride,
                                 odist, grid_out, inembed, istride, idist,
                                 FFTW_BACKWARD, fftw_planning_mode);
    }
    assert(plan != NULL);
    add_plan_to_cache(key, plan);
  fft_stop_timer(handle2);
  fft_stop_timer(handle);
  return plan;
}

/*******************************************************************************
 * \brief Create plan of a local R2C/C2R 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
fftw_plan *
fft_fftw_create_2d_plan_r2c(const int key[KEY_SIZE], double *grid_rs, double complex *grid_gs) {
  const int direction = key[3];
  const int *fft_size = key+4;
  const int number_of_threads = key[2];
  const int number_of_ffts = key[6];
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_2d_%s_Plocal",
           direction == FFTW_FORWARD ? "fw_r2c" : "bw_c2r");
  const int handle = fft_start_timer(routine_name);
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_2d_%s_Plocal_%i_%i_%i",
           direction == FFTW_FORWARD ? "fw_r2c" : "bw_c2r", fft_size[0],
           fft_size[1], number_of_ffts);
  const int handle2 = fft_start_timer(routine_name);
    fftw_plan_with_nthreads(number_of_threads);
    // We need the guru interface here because cuts the last dimension in half
    // whereas we want the first dimension
    const int rank = key[0]%4;
    const int *n = &key[4];
    const int howmany = key[6];
    const int *inembed = NULL; // = fft_size;
    const int *onembed = NULL; // = {fft_size[0],fft_size[1]/2+1};
    const int idist = key[9];
    const int odist = key[12];
    const int istride = key[8];
    const int ostride = key[11];
    fftw_plan *plan = malloc(sizeof(fftw_plan));
    if (key[3] == FFTW_FORWARD) {
      *plan = fftw_plan_many_dft_r2c(rank, n, howmany, grid_rs, inembed,
                                     istride, idist, grid_gs, onembed,
                                     ostride, odist, fftw_planning_mode);
    } else {
      // We use the buffers the other way around to prevent
      // out-of-bounds-accesses of the planner if the output array has only the
      // minimum size
      *plan = fftw_plan_many_dft_c2r(
          rank, n, howmany, grid_gs, onembed, ostride,
          odist, grid_rs, inembed, istride, idist,
          fftw_planning_mode);
    }
    assert(plan != NULL);
    add_plan_to_cache(key, plan);
  fft_stop_timer(handle2);
  fft_stop_timer(handle);
  return plan;
}

/*******************************************************************************
 * \brief Create plan of a local C2C 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
fftw_plan *fft_fftw_create_3d_plan(const int key[KEY_SIZE],
                                   double complex *grid_in,
                                   double complex *grid_out) {
  const int direction = key[3];
  const int *fft_size = key+4;
  const int number_of_threads = key[2];
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_%cw_c2c_Plocal",
           direction == FFTW_FORWARD ? 'f' : 'b');
  const int handle = fft_start_timer(routine_name);
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH,
           "fft_3d_%cw_c2c_Plocal_%i_%i_%i",
           direction == FFTW_FORWARD ? 'f' : 'b', fft_size[0], fft_size[1],
           fft_size[2]);
  const int handle2 = fft_start_timer(routine_name);
    fftw_plan_with_nthreads(number_of_threads);
    fftw_plan *plan = malloc(sizeof(fftw_plan));
    *plan = fftw_plan_dft_3d(key[4], key[5], key[6], grid_in,
                             grid_out, direction, fftw_planning_mode);
    add_plan_to_cache(key, plan);
    assert(plan != NULL);
  fft_stop_timer(handle2);
  fft_stop_timer(handle);
  return plan;
}

/*******************************************************************************
 * \brief Create plan of a local R2C/C2R 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
fftw_plan *fft_fftw_create_3d_plan_r2c(const int key[KEY_SIZE],
                                       double *grid_rs,
                                       double complex *grid_gs) {
  const int direction = key[3];
  const int *fft_size = key+4;
  const int number_of_threads = key[2];
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_%s_Plocal",
           direction == FFTW_FORWARD ? "fw_r2c" : "bw_c2r");
  const int handle = fft_start_timer(routine_name);
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_%s_Plocal_%i_%i_%i",
           direction == FFTW_FORWARD ? "fw_r2c" : "bw_c2r", fft_size[0],
           fft_size[1], fft_size[2]);
  const int handle2 = fft_start_timer(routine_name);
    fftw_plan_with_nthreads(number_of_threads);
    fftw_plan *plan = malloc(sizeof(fftw_plan));
    if (key[3] == FFTW_FORWARD) {
      *plan = fftw_plan_dft_r2c_3d(key[4], key[5], key[6],
                                   grid_rs, grid_gs,
                                   fftw_planning_mode);
    } else {
      // We use the buffers the other way around to prevent
      // out-of-bounds-accesses of the planner if the output array has only the
      // minimum size
      *plan =
          fftw_plan_dft_c2r_3d(key[4], key[5], key[6],
                               grid_gs, grid_rs, fftw_planning_mode);
    }
    add_plan_to_cache(key, plan);
    assert(plan != NULL);
  fft_stop_timer(handle2);
  fft_stop_timer(handle);
  return plan;
}

/*******************************************************************************
 * \brief Create plan of a local C2C 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
fftw_plan *fft_fftw_create_guru_plan(const int key[KEY_SIZE],
  double complex *grid_in,
                                     double complex *grid_out) {
  const int direction = key[3];
  const int rank = key[0]%4;
  const int howmany_rank = 3-rank-(key[6] == 0)-(key[5] == 0)-(key[4] == 0);
  fft_iodim dims[3];
  fft_iodim *howmany_dims = dims+rank;
  for (int r = 0; r < 3; r++) {
    dims[r].n = key[4+r];
    dims[r].is = key[7+r];
    dims[r].os = key[10+r];
  }
  const int number_of_threads = key[2];
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_guru_%cw_c2c_Plocal_%i_%i",
           direction == FFTW_FORWARD ? 'f' : 'b', rank, howmany_rank);
  const int handle = fft_start_timer(routine_name);

  assert(rank + howmany_rank <= 3 &&
         "Larger combined ranks than 3 are not implemented\n");
    fftw_plan_with_nthreads(number_of_threads);
    fftw_plan *plan = malloc(sizeof(fftw_plan));
    *plan = fftw_plan_guru_dft(rank, dims, howmany_rank, howmany_dims, grid_in,
                               grid_out, direction, fftw_planning_mode);
    add_plan_to_cache(key, plan);
    assert(plan != NULL);
  fft_stop_timer(handle);
  return plan;
}

/*******************************************************************************
 * \brief Create plan of a local R2C/C2R 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
fftw_plan *fft_fftw_create_guru_plan_r2c(
    const int key[KEY_SIZE], double *grid_rs, double complex *grid_gs) {
  const int direction = key[3];
  const int rank = key[0]%4;
  const int howmany_rank = 3-rank-(key[6] == 0)-(key[5] == 0)-(key[4] == 0);
  fft_iodim dims[3];
  for (int r = 0; r < 3; r++) {
    dims[r].n = key[4+r];
    dims[r].is = key[7+r];
    dims[r].os = key[10+r];
  }
  fft_iodim *howmany_dims = dims+rank;
  const int number_of_threads = key[2];
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_guru_%s_Plocal_%i_%i",
           direction == FFTW_FORWARD ? "fw_r2c" : "bw_c2r", rank, howmany_rank);
  const int handle = fft_start_timer(routine_name);

    fftw_plan_with_nthreads(number_of_threads);
    fftw_plan *plan = malloc(sizeof(fftw_plan));
    if (direction == FFTW_FORWARD) {
      *plan = fftw_plan_guru_dft_r2c(rank, dims, howmany_rank, howmany_dims,
                                     grid_rs, grid_gs,
                                     fftw_planning_mode);
    } else {
      // We use the buffers the other way around to prevent
      // out-of-bounds-accesses of the planner if the output array has only the
      // minimum size
      *plan =
          fftw_plan_guru_dft_c2r(rank, dims, howmany_rank, howmany_dims,
                                 grid_gs, grid_rs, fftw_planning_mode);
    }
    add_plan_to_cache(key, plan);
    assert(plan != NULL);
  fft_stop_timer(handle);
  return plan;
}

#if defined(__USE_FFTW3_MPI)
/*******************************************************************************
 * \brief Create plan of a distributed C2C 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
fftw_plan *fft_fftw_create_distributed_2d_plan(const int key[KEY_SIZE],
                                               double complex *grid_out) {
  const int direction = key[3];
  const int *fft_size = key+4;
  const int number_of_ffts = key[6];
  const int number_of_threads = key[2];
  cp_mpi_comm_t comm = cp_mpi_comm_f2c(key[1]);
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_2d_%cw_c2c_Pdistr",
           direction == FFTW_FORWARD ? 'f' : 'b');
  const int handle = fft_start_timer(routine_name);
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH,
           "fft_2d_%cw_c2c_Pdistr_%i_%i_%i_%i",
           direction == FFTW_FORWARD ? 'f' : 'b', cp_mpi_comm_size(comm),
           fft_size[0], fft_size[1], number_of_ffts);
  const int handle2 = fft_start_timer(routine_name);
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    fftw_plan_with_nthreads(number_of_threads);
    if (number_of_ffts == 0)
      return plan;
    const int block_size_0 =
        (fft_size[0] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
    const int block_size_1 =
        (fft_size[1] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
    ptrdiff_t local_n0, local_0_start;
    ptrdiff_t local_n1, local_1_start;
    const ptrdiff_t n[2] = {fft_size[0], fft_size[1]};
    const ptrdiff_t howmany = number_of_ffts;
    const int buffer_size = fftw_mpi_local_size_many_transposed(
        2, n, howmany, block_size_0, block_size_1, comm, &local_n0,
        &local_0_start, &local_n1, &local_1_start);
    double complex *buffer_1 = fftw_alloc_complex(buffer_size);
    double complex *buffer_2 = grid_out;
    plan = malloc(sizeof(fftw_plan));
    fflush(stderr);
    if (direction == FFTW_FORWARD) {
      *plan = fftw_mpi_plan_many_dft(
          2, n, howmany, block_size_0, block_size_1, buffer_1, buffer_2, comm,
          direction, fftw_planning_mode + FFTW_MPI_TRANSPOSED_OUT);
    } else {
      *plan = fftw_mpi_plan_many_dft(
          2, n, howmany, block_size_1, block_size_0, buffer_1, buffer_2, comm,
          direction, fftw_planning_mode + FFTW_MPI_TRANSPOSED_IN);
    }
    assert(plan != NULL);
    fftw_free(buffer_1);
    add_plan_to_cache(key, plan);
  }
  fft_stop_timer(handle2);
  fft_stop_timer(handle);
  return plan;
}
/*******************************************************************************
 * \brief Create plan of a distributed R2C/C2R 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
fftw_plan *fft_fftw_create_distributed_2d_plan_r2c(const int key[KEY_SIZE],
                                                   double complex *grid_out) {
  const int direction = key[3];
  const int *fft_size = key+4;
  const int number_of_ffts = key[6];
  const int number_of_threads = key[2];
  cp_mpi_comm_t comm = cp_mpi_comm_f2c(key[1]);
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_2d_%s_Pdistr",
           direction == FFTW_FORWARD ? "fw_r2c" : "bw_c2r");
  const int handle = fft_start_timer(routine_name);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_2d_%s_Pdistr_%i_%i_%i_%i",
           direction == FFTW_FORWARD ? "fw_r2c" : "bw_c2r",
           cp_mpi_comm_size(comm), fft_size[0], fft_size[1], number_of_ffts);
  const int handle2 = fft_start_timer(routine_name);
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    fftw_plan_with_nthreads(number_of_threads);
    if (number_of_ffts == 0)
      return plan;
    const int block_size_0 =
        (fft_size[0] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
    const int block_size_1 =
        (fft_size[1] / 2 + 1 + cp_mpi_comm_size(comm) - 1) /
        cp_mpi_comm_size(comm);
    ptrdiff_t local_n0, local_0_start;
    ptrdiff_t local_n1, local_1_start;
    const ptrdiff_t n[2] = {fft_size[0], fft_size[1]};
    const ptrdiff_t howmany = number_of_ffts;
    const int buffer_size = fftw_mpi_local_size_many_transposed(
        2, (const ptrdiff_t[2]){fft_size[0], fft_size[1] / 2 + 1}, howmany,
        block_size_0, block_size_1, comm, &local_n0, &local_0_start, &local_n1,
        &local_1_start);
    double *double_buffer = fftw_alloc_real(2 * buffer_size);
    double complex *complex_buffer = grid_out;
    plan = malloc(sizeof(fftw_plan));
    if (direction == FFTW_FORWARD) {
      *plan = fftw_mpi_plan_many_dft_r2c(
          2, n, howmany, block_size_0, block_size_1, double_buffer,
          complex_buffer, comm, fftw_planning_mode + FFTW_MPI_TRANSPOSED_OUT);
    } else {
      // We use the buffers the other way around to prevent
      // out-of-bounds-accesses of the planner if the output array has only the
      // minimum size
      *plan = fftw_mpi_plan_many_dft_c2r(
          2, n, howmany, block_size_1, block_size_0,
          (double complex *)double_buffer, (double *)complex_buffer, comm,
          fftw_planning_mode + FFTW_MPI_TRANSPOSED_IN);
    }
    assert(plan != NULL);
    fftw_free(double_buffer);
    add_plan_to_cache(key, plan);
  }
  fft_stop_timer(handle2);
  fft_stop_timer(handle);
  return plan;
}

/*******************************************************************************
 * \brief Create plan of a distributed C2C 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
fftw_plan *fft_fftw_create_distributed_3d_plan(const int key[KEY_SIZE],
                                               double complex *grid_out) {
  const int direction = key[3];
  const int *fft_size = key+4;
  const int number_of_threads = key[2];
  cp_mpi_comm_t comm = cp_mpi_comm_f2c(key[1]);
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_%s_Pdistr",
           direction == FFTW_FORWARD ? "fw_r2c" : "bw_c2r");
  const int handle = fft_start_timer(routine_name);
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_%s_Pdistr_%i_%i_%i_%i",
           direction == FFTW_FORWARD ? "fw_r2c" : "bw_c2r",
           cp_mpi_comm_size(comm), fft_size[0], fft_size[1], fft_size[2]);
  const int handle2 = fft_start_timer(routine_name);
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    fftw_plan_with_nthreads(number_of_threads);
    const int block_size_0 =
        (fft_size[0] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
    const int block_size_1 =
        (fft_size[1] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
    ptrdiff_t local_n0, local_0_start;
    ptrdiff_t local_n1, local_1_start;
    const ptrdiff_t n[3] = {fft_size[0], fft_size[1], fft_size[2]};
    const int buffer_size = fftw_mpi_local_size_many_transposed(
        3, n, 1, block_size_0, block_size_1, comm, &local_n0, &local_0_start,
        &local_n1, &local_1_start);
    double complex *buffer_1 = fftw_alloc_complex(buffer_size);
    double complex *buffer_2 = grid_out;
    plan = malloc(sizeof(fftw_plan));
    if (direction == FFTW_FORWARD) {
      *plan = fftw_mpi_plan_many_dft(
          3, n, 1, block_size_0, block_size_1, buffer_1, buffer_2, comm,
          direction, fftw_planning_mode + FFTW_MPI_TRANSPOSED_OUT);
    } else {
      *plan = fftw_mpi_plan_many_dft(
          3, n, 1, block_size_1, block_size_0, buffer_1, buffer_2, comm,
          direction, fftw_planning_mode + FFTW_MPI_TRANSPOSED_IN);
    }
    assert(plan != NULL);
    add_plan_to_cache(key, plan);
    fftw_free(buffer_1);
  }
  fft_stop_timer(handle2);
  fft_stop_timer(handle);
  return plan;
}

/*******************************************************************************
 * \brief Create plan of a distributed R2C/C2R 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
fftw_plan *fft_fftw_create_distributed_3d_plan_r2c(const int key[KEY_SIZE],
                                                   double complex *grid_out) {
  const int direction = key[3];
  const int *fft_size = key+4;
  const int number_of_threads = key[2];
  cp_mpi_comm_t comm = cp_mpi_comm_f2c(key[1]);
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_%s_Pdistr",
           direction == FFTW_FORWARD ? "fw_r2c" : "bw_c2r");
  const int handle = fft_start_timer(routine_name);
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_%s_Pdistr_%i_%i_%i_%i",
           direction == FFTW_FORWARD ? "fw_r2c" : "bw_c2r",
           cp_mpi_comm_size(comm), fft_size[0], fft_size[1], fft_size[2]);
  const int handle2 = fft_start_timer(routine_name);
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    fftw_plan_with_nthreads(number_of_threads);
    const int block_size_0 =
        (fft_size[0] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
    const int block_size_1 =
        (fft_size[1] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
    ptrdiff_t local_n0, local_0_start;
    ptrdiff_t local_n1, local_1_start;
    const ptrdiff_t n[3] = {fft_size[0], fft_size[1], fft_size[2]};
    const int buffer_size = fftw_mpi_local_size_many_transposed(
        3, (const ptrdiff_t[3]){fft_size[0], fft_size[1], fft_size[2] / 2 + 1},
        1, block_size_0, block_size_1, comm, &local_n0, &local_0_start,
        &local_n1, &local_1_start);
    double *buffer_1 = fftw_alloc_real(2 * buffer_size);
    double complex *buffer_2 = grid_out;
    plan = malloc(sizeof(fftw_plan));
    if (direction == FFTW_FORWARD) {
      *plan = fftw_mpi_plan_many_dft_r2c(
          3, n, 1, block_size_0, block_size_1, buffer_1, buffer_2, comm,
          fftw_planning_mode + FFTW_MPI_TRANSPOSED_OUT);
    } else {
      *plan = fftw_mpi_plan_many_dft_c2r(
          3, n, 1, block_size_1, block_size_0, buffer_2, buffer_1, comm,
          fftw_planning_mode + FFTW_MPI_TRANSPOSED_IN);
    }
    assert(plan != NULL);
    add_plan_to_cache(key, plan);
    fftw_free(buffer_1);
  }
  fft_stop_timer(handle2);
  fft_stop_timer(handle);
  return plan;
}
#endif
#endif

/*******************************************************************************
 * \brief Performs a local forward C2C 1D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_1d_fw_local(const int fft_size, const int number_of_ffts,
                          const bool transpose_rs, const bool transpose_gs,
                          const int leading_dimension_rs, const int leading_dimension_gs,
                          double complex *grid_in, double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  assert(is_initialized);
  if (fft_size == 0 || number_of_ffts == 0) return;
  const bool in_place = grid_in == grid_out;
  int number_of_threads = 1;
#pragma omp parallel default(none) shared(number_of_threads)
  {
#pragma omp single
    { number_of_threads = omp_get_num_threads(); }
  }
  if (fftw_planning_mode == FFTW_ESTIMATE) {
    fftw_plan *plan = NULL, *plan_last_thread = NULL;
    bool has_plan_for_last_thread = false;
    const int block_size =
        (number_of_ffts + number_of_threads - 1) / number_of_threads;
  int key[KEY_SIZE];
  get_key_1d(FFTW_FORWARD, fft_size, block_size,
                        transpose_rs, transpose_gs, 
                        leading_dimension_rs, leading_dimension_gs,
                        1, in_place, key);
  plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
  double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
    plan = fft_fftw_create_1d_plan(key, buffer, grid_out);
    fftw_free(buffer);
  }
    if (block_size * number_of_threads != number_of_ffts) {
      const int block_size_last_thread =
          number_of_ffts - (number_of_threads - 1) * block_size;
      int key[KEY_SIZE];
      get_key_1d(FFTW_FORWARD, fft_size, block_size_last_thread,
                            transpose_rs, transpose_gs, 
                            leading_dimension_rs, leading_dimension_gs,
                            1, in_place, key);
      plan_last_thread = lookup_plan_from_cache(key);
      if (plan == NULL) {
        double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
        plan_last_thread = fft_fftw_create_1d_plan(key, buffer, grid_out);
        fftw_free(buffer);
      }
      has_plan_for_last_thread = true;
    }
    const int offset_in = transpose_rs ? block_size : block_size * fft_size;
    const int offset_out = transpose_gs ? block_size : block_size * fft_size;
#pragma omp parallel default(none)                                             \
    shared(grid_in, grid_out, plan, plan_last_thread, number_of_threads,       \
               offset_in, offset_out, has_plan_for_last_thread)
    {
      const int thread_id = omp_get_thread_num();
      if (thread_id + 1 < number_of_threads || !has_plan_for_last_thread) {
        fftw_execute_dft(*plan, grid_in + thread_id * offset_in,
                         grid_out + thread_id * offset_out);
      } else {
        fftw_execute_dft(*plan_last_thread, grid_in + thread_id * offset_in,
                         grid_out + thread_id * offset_out);
      }
    }
  } else {
    int key[KEY_SIZE];
    get_key_1d(FFTW_FORWARD, fft_size, number_of_ffts,
                          transpose_rs, transpose_gs, 
                          leading_dimension_rs, leading_dimension_gs,
                          omp_get_max_threads(), in_place, key);
    fftw_plan *plan = lookup_plan_from_cache(key);
    if (plan == NULL) {
      double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
      plan = fft_fftw_create_1d_plan(key, buffer, grid_in == grid_out ? buffer : grid_out);
      fftw_free(buffer);
    }
    fftw_execute_dft(*plan, grid_in, grid_out);
  }
#else
  (void)fft_size;
  (void)number_of_ffts;
  (void)grid_in;
  (void)grid_out;
  (void)transpose_rs;
  (void)transpose_gs;
  (void)leading_dimension_rs;
  (void)leading_dimension_gs;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs a local forward R2C FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_1d_fw_local_r2c(const int fft_size, const int number_of_ffts,
                              const bool transpose_rs, const bool transpose_gs,
                     const int leading_dimension_rs, const int leading_dimension_gs,
                              double *grid_in, double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  assert(is_initialized);
  if (fft_size == 0 || number_of_ffts == 0) return;
  int key[KEY_SIZE];
  get_key_1d_r2c(
      FFTW_FORWARD, fft_size, number_of_ffts, transpose_rs, transpose_gs,
                                   leading_dimension_rs, leading_dimension_gs,
      omp_get_max_threads(), (double complex *)grid_in == grid_out, key);
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    double *buffer = fftw_alloc_real(2*get_buffer_size_from_key(key));
    plan = fft_fftw_create_1d_plan_r2c(key, buffer, grid_in == (double*)grid_out ? (double complex*)buffer : grid_out);
    fftw_free(buffer);
  }
  assert(plan != NULL);
  fftw_execute_dft_r2c(*plan, grid_in, grid_out);
#else
  (void)fft_size;
  (void)number_of_ffts;
  (void)grid_in;
  (void)grid_out;
  (void)transpose_rs;
  (void)transpose_gs;
  (void)leading_dimension_rs;
  (void)leading_dimension_gs;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs a local backwards C2C 1D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_1d_bw_local(const int fft_size, const int number_of_ffts,
                          const bool transpose_rs, const bool transpose_gs,
                     const int leading_dimension_rs, const int leading_dimension_gs,
                          double complex *grid_in, double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  assert(is_initialized);
  if (fft_size == 0 || number_of_ffts == 0) return;
  const bool in_place = grid_in == grid_out;
  int number_of_threads = 1;
#pragma omp parallel default(none) shared(number_of_threads)
  {
#pragma omp single
    { number_of_threads = omp_get_num_threads(); }
  }
  if (fftw_planning_mode == FFTW_ESTIMATE) {
    fftw_plan *plan = NULL, *plan_last_thread = NULL;
    bool has_plan_for_last_thread = false;
    const int block_size =
        (number_of_ffts + number_of_threads - 1) / number_of_threads;
    int key[KEY_SIZE];
    get_key_1d(FFTW_BACKWARD, fft_size, block_size,
                          transpose_rs, transpose_gs, 
                          leading_dimension_rs, leading_dimension_gs,
                          1, in_place, key);
    plan = lookup_plan_from_cache(key);
    if (plan == NULL) {
      double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
      plan = fft_fftw_create_1d_plan(key, buffer, grid_out);
      fftw_free(buffer);
    }
    if (block_size * number_of_threads != number_of_ffts) {
      const int block_size_last_thread =
          number_of_ffts - (number_of_threads - 1) * block_size;
      int key[KEY_SIZE];
      get_key_1d(FFTW_BACKWARD, fft_size, block_size_last_thread,
                        transpose_rs, transpose_gs, 
                        leading_dimension_rs, leading_dimension_gs,
                        1, in_place, key);
      plan_last_thread = lookup_plan_from_cache(key);
      if (plan == NULL) {
        double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
        plan_last_thread = fft_fftw_create_1d_plan(key, buffer, grid_out);
        fftw_free(buffer);
      }
      has_plan_for_last_thread = true;
    }
    const int offset_in = transpose_gs ? block_size : block_size * fft_size;
    const int offset_out = transpose_rs ? block_size : block_size * fft_size;
#pragma omp parallel default(none)                                             \
    shared(grid_in, grid_out, plan, plan_last_thread, number_of_threads,       \
               offset_in, offset_out, has_plan_for_last_thread)
    {
      const int thread_id = omp_get_thread_num();
      if (thread_id + 1 < number_of_threads || !has_plan_for_last_thread) {
        fftw_execute_dft(*plan, grid_in + thread_id * offset_in,
                         grid_out + thread_id * offset_out);
      } else {
        fftw_execute_dft(*plan_last_thread, grid_in + thread_id * offset_in,
                         grid_out + thread_id * offset_out);
      }
    }
  } else {
    int key[KEY_SIZE];
    get_key_1d(FFTW_BACKWARD, fft_size, number_of_ffts,
                          transpose_rs, transpose_gs, 
                          leading_dimension_rs, leading_dimension_gs,
                          omp_get_max_threads(), in_place, key);
    fftw_plan *plan = lookup_plan_from_cache(key);
    if (plan == NULL) {
      double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
      plan = fft_fftw_create_1d_plan(key, buffer, grid_in == grid_out ? buffer : grid_out);
      fftw_free(buffer);
    }
    fftw_execute_dft(*plan, grid_in, grid_out);
  }
#else
  (void)fft_size;
  (void)number_of_ffts;
  (void)grid_in;
  (void)grid_out;
  (void)transpose_rs;
  (void)transpose_gs;
  (void)leading_dimension_rs;
  (void)leading_dimension_gs;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs a local backwards C2R 1D FFT
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_1d_bw_local_c2r(const int fft_size, const int number_of_ffts,
                              const bool transpose_rs, const bool transpose_gs,
                     const int leading_dimension_rs, const int leading_dimension_gs,
                              double complex *grid_in, double *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  assert(is_initialized);
  if (fft_size == 0 || number_of_ffts == 0) return;
  int key[KEY_SIZE];
  get_key_1d_r2c(
      FFTW_BACKWARD, fft_size, number_of_ffts, transpose_rs, transpose_gs,
      leading_dimension_rs, leading_dimension_gs,
      omp_get_max_threads(), grid_in == (double complex *)grid_out, key);
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
    plan = fft_fftw_create_1d_plan_r2c(key, (double*)grid_in == grid_out ? (double*)buffer : grid_out, buffer);
    fftw_free(buffer);
  }
  fftw_execute_dft_c2r(*plan, grid_in, grid_out);
#else
  (void)fft_size;
  (void)number_of_ffts;
  (void)grid_in;
  (void)grid_out;
  (void)transpose_rs;
  (void)transpose_gs;
  (void)leading_dimension_rs;
  (void)leading_dimension_gs;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs a local forward C2C 2D FFT
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_2d_fw_local(const int fft_size[2], const int number_of_ffts,
                          const bool transpose_rs, const bool transpose_gs,
                          double complex *grid_in, double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  assert(is_initialized);
  if (fft_size[0] == 0 || fft_size[1] == 0 || number_of_ffts == 0) return;
  int key[KEY_SIZE];
  get_key_2d(FFTW_FORWARD, fft_size, number_of_ffts,
                        transpose_rs, transpose_gs, 
                        omp_get_max_threads(), grid_in == grid_out, key);
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
    plan = fft_fftw_create_2d_plan(key, buffer, grid_in == grid_out ? buffer : grid_out);
    fftw_free(buffer);
  }
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
 * \brief Performs a local forward R2C 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_2d_fw_local_r2c(const int fft_size[2], const int number_of_ffts,
                              const bool transpose_rs, const bool transpose_gs,
                              double *grid_in, double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  assert(is_initialized);
  if (fft_size[0] == 0 || fft_size[1] == 0 || number_of_ffts == 0) return;
  int key[KEY_SIZE];
  get_key_2d_r2c(FFTW_FORWARD, fft_size, number_of_ffts,
                        transpose_rs, transpose_gs, 
                        omp_get_max_threads(), grid_in == (double*)grid_out, key);
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    double *buffer = fftw_alloc_real(2*get_buffer_size_from_key(key));
    plan = fft_fftw_create_2d_plan_r2c(key, buffer, grid_in == (double*)grid_out ? (double complex*)buffer : grid_out);
    fftw_free(buffer);
  }
  fftw_execute_dft_r2c(*plan, grid_in, grid_out);
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
 * \brief Performs a local backwards C2C 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_2d_bw_local(const int fft_size[2], const int number_of_ffts,
                          const bool transpose_rs, const bool transpose_gs,
                          double complex *grid_in, double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  assert(is_initialized);
  if (fft_size[0] == 0 || fft_size[1] == 0 || number_of_ffts == 0) return;
  int key[KEY_SIZE];
  get_key_2d(FFTW_BACKWARD, fft_size, number_of_ffts,
                        transpose_rs, transpose_gs, 
                        omp_get_max_threads(), grid_in == grid_out, key);
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
    plan = fft_fftw_create_2d_plan(key, buffer, grid_in == grid_out ? buffer : grid_out);
    fftw_free(buffer);
  }
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
 * \brief Performs a local backwards C2R 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_2d_bw_local_c2r(const int fft_size[2], const int number_of_ffts,
                              const bool transpose_rs, const bool transpose_gs,
                              double complex *grid_in, double *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  assert(is_initialized);
  if (fft_size[0] == 0 || fft_size[1] == 0 || number_of_ffts == 0) return;
  int key[KEY_SIZE];
  get_key_2d_r2c(FFTW_BACKWARD, fft_size, number_of_ffts,
                        transpose_rs, transpose_gs, 
                        omp_get_max_threads(), grid_in == (double complex*)grid_out, key);
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
    plan = fft_fftw_create_2d_plan_r2c(key, (double*)grid_in == grid_out ? (double*)buffer : grid_out, buffer);
    fftw_free(buffer);
  }
  fftw_execute_dft_c2r(*plan, grid_in, grid_out);
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
 * \brief Performs a local C2C 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_fw_guru(int rank, const fft_iodim *dims, int howmany_rank,
                      const fft_iodim *howmany_dims,
                      const int number_of_threads, double complex *grid_in,
                      double complex *grid_out) {
#if defined(__FFTW3)
  assert(has_guru_interface);
  assert(is_initialized);
  for (int r = 0; r < rank; r++) {
    if (dims[r].n == 0) return;
  }
  for (int r = 0; r < howmany_rank; r++) {
    if (howmany_dims[r].n == 0) return;
  }
  int key[KEY_SIZE];
  get_key_guru(
      FFTW_FORWARD, rank, dims, howmany_rank, howmany_dims, number_of_threads, grid_in == grid_out, key);
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
    plan = fft_fftw_create_guru_plan(key, buffer, grid_in == grid_out ? buffer : grid_out);
    fftw_free(buffer);
  }
  fftw_execute_dft(*plan, grid_in, grid_out);
#else
  (void)rank;
  (void)dims;
  (void)howmany_rank;
  (void)howmany_dims;
  (void)number_of_threads;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs a local forward R2C 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_fw_guru_r2c(int rank, const fft_iodim *dims, int howmany_rank,
                          const fft_iodim *howmany_dims,
                          const int number_of_threads, double *grid_in,
                          double complex *grid_out) {
#if defined(__FFTW3)
  assert(has_guru_interface);
  assert(is_initialized);
  for (int r = 0; r < rank; r++) {
    if (dims[r].n == 0) return;
  }
  for (int r = 0; r < howmany_rank; r++) {
    if (howmany_dims[r].n == 0) return;
  }
  int key[KEY_SIZE];
  get_key_guru_r2c(FFTW_FORWARD, rank, dims, howmany_rank, howmany_dims, number_of_threads,
      grid_in == (double *)grid_out, key);
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    double *buffer = fftw_alloc_real(2*get_buffer_size_from_key(key));
    plan = fft_fftw_create_guru_plan_r2c(key, buffer, grid_in == (double*)grid_out ? (double complex*)buffer : grid_out);
    fftw_free(buffer);
  }
  fftw_execute_dft_r2c(*plan, grid_in, grid_out);
#else
  (void)rank;
  (void)dims;
  (void)howmany_rank;
  (void)howmany_dims;
  (void)number_of_threads;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs a local backwards C2C 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_bw_guru(int rank, const fft_iodim *dims, int howmany_rank,
                      const fft_iodim *howmany_dims,
                      const int number_of_threads, double complex *grid_in,
                      double complex *grid_out) {
#if defined(__FFTW3)
  assert(has_guru_interface);
  assert(is_initialized);
  for (int r = 0; r < rank; r++) {
    if (dims[r].n == 0) return;
  }
  for (int r = 0; r < howmany_rank; r++) {
    if (howmany_dims[r].n == 0) return;
  }
  int key[KEY_SIZE];
  get_key_guru(
      FFTW_BACKWARD, rank, dims, howmany_rank, howmany_dims, number_of_threads,
      grid_in == grid_out, key);
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
    plan = fft_fftw_create_guru_plan(key, buffer, grid_in == grid_out ? buffer : grid_out);
    fftw_free(buffer);
  }
  fftw_execute_dft(*plan, grid_in, grid_out);
#else
  (void)rank;
  (void)dims;
  (void)howmany_rank;
  (void)howmany_dims;
  (void)number_of_threads;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs a local backwards R2C 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_bw_guru_c2r(int rank, const fft_iodim *dims, int howmany_rank,
                          const fft_iodim *howmany_dims,
                          const int number_of_threads, double complex *grid_in,
                          double *grid_out) {
#if defined(__FFTW3)
  assert(has_guru_interface);
  assert(is_initialized);
  for (int r = 0; r < rank; r++) {
    if (dims[r].n == 0) return;
  }
  for (int r = 0; r < howmany_rank; r++) {
    if (howmany_dims[r].n == 0) return;
  }
  int key[KEY_SIZE];
  get_key_guru_r2c(FFTW_BACKWARD, rank, dims, howmany_rank, howmany_dims, number_of_threads,
      (double *)grid_in == grid_out, key);
  fftw_plan *plan = lookup_plan_from_cache(key);
  if (plan == NULL) {
    double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
    plan = fft_fftw_create_guru_plan_r2c(key, (double*)grid_in == grid_out ? (double*)buffer : grid_out, buffer);
    fftw_free(buffer);
  }
  fftw_execute_dft_c2r(*plan, grid_in, grid_out);
#else
  (void)rank;
  (void)dims;
  (void)howmany_rank;
  (void)howmany_dims;
  (void)number_of_threads;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs a local C2C 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_3d_fw_local(const int fft_size[3], double complex *grid_in,
                          double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  assert(is_initialized);
  if (fft_size[0] == 0 || fft_size[1] == 0 || fft_size[2] == 0) return;
  const bool in_place = grid_in == grid_out;
  if (has_guru_interface &&
      ((fft_size[0] >= 256 || fft_size[1] >= 256 || fft_size[2] >= 256 ||
       omp_get_max_threads() > 1)) && !in_place &&
      (fftw_planning_mode == FFTW_ESTIMATE)) {
    // The 3D FFT is not efficient with threading and estimate planning mode
    // So, we decompose it in a sequence of 1D FFTs
    int number_of_threads = 1;
#pragma omp parallel default(none) shared(number_of_threads)
    {
#pragma omp single
      { number_of_threads = omp_get_num_threads(); }
    }
    int block_sizes[3];
    fftw_plan *plans[3], *plans_last_thread[3];
    bool has_plan_for_last_thread[3] = {false, false, false};
    {
      const int number_of_ffts = fft_size[1] * fft_size[2];
      block_sizes[0] =
          (number_of_ffts + number_of_threads - 1) / number_of_threads;
      fft_iodim dim = {
          .n = fft_size[0], .is = number_of_ffts, .os = number_of_ffts};
      fft_iodim howmany_dim = {.n = block_sizes[0], .is = 1, .os = 1};
      int key[KEY_SIZE];
                get_key_guru(FFTW_FORWARD, 1, &dim, 1, &howmany_dim, 1, in_place, key);
            
      plans[0] = lookup_plan_from_cache(key);
      if (plans[0] == NULL) {
        double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
        plans[0] = fft_fftw_create_guru_plan(key, buffer, grid_out);
        fftw_free(buffer);
      }
      if (block_sizes[0] * number_of_threads != number_of_ffts) {
        const int block_size_last_thread =
            number_of_ffts - (number_of_threads - 1) * block_sizes[0];
        fft_iodim howmany_dim = {.n = block_size_last_thread, .is = 1, .os = 1};
        int key[KEY_SIZE];
        get_key_guru(FFTW_FORWARD, 1, &dim, 1, &howmany_dim, 1, in_place, key);
        plans_last_thread[0] = lookup_plan_from_cache(key);
        if (plans_last_thread[0] == NULL) {
          double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
          plans_last_thread[0] = fft_fftw_create_guru_plan(key, buffer, grid_out);
          fftw_free(buffer);
        }
        has_plan_for_last_thread[0] = true;
      }
    }
    {
      const int number_of_ffts = fft_size[0];
      block_sizes[1] =
          (number_of_ffts + number_of_threads - 1) / number_of_threads;
      fft_iodim dim = {.n = fft_size[1], .is = fft_size[2], .os = fft_size[2]};
      fft_iodim howmany_dims[2] = {{.n = block_sizes[1],
                                    .is = fft_size[1] * fft_size[2],
                                    .os = fft_size[1] * fft_size[2]},
                                   {.n = fft_size[2], .is = 1, .os = 1}};
      int key[KEY_SIZE];
      get_key_guru(FFTW_FORWARD, 1, &dim, 2, howmany_dims, 1, in_place, key);
      plans[1] = lookup_plan_from_cache(key);
      if (plans[1] == NULL) {
        double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
        plans[1] = fft_fftw_create_guru_plan(key, buffer, grid_out);
        fftw_free(buffer);
      }
      if (block_sizes[1] * number_of_threads != number_of_ffts) {
        const int block_size_last_thread =
            number_of_ffts - (number_of_threads - 1) * block_sizes[1];
        fft_iodim howmany_dims[2] = {{.n = block_size_last_thread,
                                      .is = fft_size[1] * fft_size[2],
                                      .os = fft_size[1] * fft_size[2]},
                                     {.n = fft_size[2], .is = 1, .os = 1}};
        int key[KEY_SIZE];
        get_key_guru(FFTW_FORWARD, 1, &dim, 2, howmany_dims, 1, in_place, key);
        plans_last_thread[1] = lookup_plan_from_cache(key);
        if (plans_last_thread[1] == NULL) {
          double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
          plans_last_thread[1] = fft_fftw_create_guru_plan(key, buffer, grid_out);
          fftw_free(buffer);
        }
        has_plan_for_last_thread[1] = true;
      }
    }
    {
      const int number_of_ffts = fft_size[0] * fft_size[1];
      block_sizes[2] =
          (number_of_ffts + number_of_threads - 1) / number_of_threads;
      fft_iodim dim = {.n = fft_size[2], .is = 1, .os = 1};
      fft_iodim howmany_dim = {
          .n = block_sizes[2], .is = fft_size[2], .os = fft_size[2]};
      int key[KEY_SIZE];
      get_key_guru(FFTW_FORWARD, 1, &dim, 1, &howmany_dim, 1, in_place, key);
      plans[2] = lookup_plan_from_cache(key);
      if (plans[2] == NULL) {
        double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
        plans[2] = fft_fftw_create_guru_plan(key, buffer, grid_out);
        fftw_free(buffer);
      }
      if (block_sizes[2] * number_of_threads != number_of_ffts) {
        const int block_size_last_thread =
            number_of_ffts - (number_of_threads - 1) * block_sizes[2];
        fft_iodim howmany_dim = {
            .n = block_size_last_thread, .is = fft_size[2], .os = fft_size[2]};
        int key[KEY_SIZE];
        get_key_guru(FFTW_FORWARD, 1, &dim, 1, &howmany_dim, 1, in_place, key);
        
        plans_last_thread[2] = lookup_plan_from_cache(key);
        if (plans_last_thread[2] == NULL) {
          double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
          plans_last_thread[2] = fft_fftw_create_guru_plan(key, buffer, grid_out);
          fftw_free(buffer);
        }
        has_plan_for_last_thread[2] = true;
      }
    }
#pragma omp parallel default(none)                                             \
    shared(number_of_threads, has_plan_for_last_thread, plans,                 \
               plans_last_thread, block_sizes, fft_size, grid_in, grid_out)
    {
      const int thread_id = omp_get_thread_num();
      if (thread_id < number_of_threads - 1 || !has_plan_for_last_thread[2]) {
        assert(*plans[2] != NULL);
        fftw_execute_dft(*plans[2],
                         grid_in + block_sizes[2] * fft_size[2] * thread_id,
                         grid_out + block_sizes[2] * fft_size[2] * thread_id);
      } else {
        assert(*plans_last_thread[2] != NULL);
        fftw_execute_dft(*plans_last_thread[2],
                         grid_in + block_sizes[2] * fft_size[2] * thread_id,
                         grid_out + block_sizes[2] * fft_size[2] * thread_id);
      }
#pragma omp barrier
      if (thread_id < number_of_threads - 1 || !has_plan_for_last_thread[1]) {
        assert(*plans[1] != NULL);
        fftw_execute_dft(
            *plans[1],
            grid_out + block_sizes[1] * fft_size[1] * fft_size[2] * thread_id,
            grid_in + block_sizes[1] * fft_size[1] * fft_size[2] * thread_id);
      } else {
        assert(*plans_last_thread[1] != NULL);
        fftw_execute_dft(
            *plans_last_thread[1],
            grid_out + block_sizes[1] * fft_size[1] * fft_size[2] * thread_id,
            grid_in + block_sizes[1] * fft_size[1] * fft_size[2] * thread_id);
      }
#pragma omp barrier
      if (thread_id < number_of_threads - 1 || !has_plan_for_last_thread[0]) {
        assert(*plans[0] != NULL);
        fftw_execute_dft(*plans[0], grid_in + block_sizes[0] * thread_id,
                         grid_out + block_sizes[0] * thread_id);
      } else {
        assert(*plans_last_thread[0] != NULL);
        fftw_execute_dft(*plans_last_thread[0],
                         grid_in + block_sizes[0] * thread_id,
                         grid_out + block_sizes[0] * thread_id);
      }
    }
  } else {
    int key[KEY_SIZE];
    get_key_3d(FFTW_FORWARD, fft_size, omp_get_max_threads(), in_place, key);
    fftw_plan *plan = lookup_plan_from_cache(key);
    if (plan == NULL) {
      double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
      plan = fft_fftw_create_3d_plan(key, buffer, in_place ? buffer : grid_out);
      fftw_free(buffer);
    }
    fftw_execute_dft(*plan, grid_in, grid_out);
  }
#else
  (void)fft_size;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs a local forward R2C 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_3d_fw_local_r2c(const int fft_size[3], double *grid_in,
                              double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  assert(is_initialized);
  if (fft_size[0] == 0 || fft_size[1] == 0 || fft_size[2] == 0) return;
    const bool in_place = grid_in == (double *)grid_out;
  if (((fft_size[0] >= 256 || fft_size[1] >= 256 || fft_size[2] >= 256 ||
       omp_get_max_threads() > 1)) && !in_place &&
      (fftw_planning_mode == FFTW_ESTIMATE)) {
    // The 3D FFT is not efficient with threading and estimate planning mode
    // So, we decompose it in a sequence of 1D FFTs
    int number_of_threads = 1;
#pragma omp parallel default(none) shared(number_of_threads)
    {
#pragma omp single
      { number_of_threads = omp_get_num_threads(); }
    }
    int block_sizes[3];
    fftw_plan *plans[3], *plans_last_thread[3];
    bool has_plan_for_last_thread[3] = {false, false, false};
    {
      const int number_of_ffts = fft_size[1] * (fft_size[2] / 2 + 1);
      block_sizes[0] =
          (number_of_ffts + number_of_threads - 1) / number_of_threads;
      fft_iodim dim = {
          .n = fft_size[0], .is = number_of_ffts, .os = number_of_ffts};
      fft_iodim howmany_dim = {.n = block_sizes[0], .is = 1, .os = 1};
      int key[KEY_SIZE];
      get_key_guru(FFTW_FORWARD, 1, &dim, 1, &howmany_dim, 1,
                                      in_place, key);
        
      plans[0] = lookup_plan_from_cache(key);
      if (plans[0] == NULL) {
        double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
        plans[0] = fft_fftw_create_guru_plan(key, buffer, grid_out);
        fftw_free(buffer);
      }
      if (block_sizes[0] * number_of_threads != number_of_ffts) {
        const int block_size_last_thread =
            number_of_ffts - (number_of_threads - 1) * block_sizes[0];
        fft_iodim howmany_dim = {.n = block_size_last_thread, .is = 1, .os = 1};
        int key[KEY_SIZE];
        get_key_guru(FFTW_FORWARD, 1, &dim, 1, &howmany_dim, 1,
                                      in_place, key);
        
        plans_last_thread[0] = lookup_plan_from_cache(key);
        if (plans_last_thread[0] == NULL) {
          double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
          plans_last_thread[0] = fft_fftw_create_guru_plan(key, buffer, grid_out);
          fftw_free(buffer);
        }
        has_plan_for_last_thread[0] = true;
      }
    }
    {
      const int number_of_ffts = fft_size[0];
      block_sizes[1] =
          (number_of_ffts + number_of_threads - 1) / number_of_threads;
      fft_iodim dim = {.n = fft_size[1],
                       .is = fft_size[2] / 2 + 1,
                       .os = fft_size[2] / 2 + 1};
      fft_iodim howmany_dims[2] = {
          {.n = block_sizes[1],
           .is = fft_size[1] * (fft_size[2] / 2 + 1),
           .os = fft_size[1] * (fft_size[2] / 2 + 1)},
          {.n = (fft_size[2] / 2 + 1), .is = 1, .os = 1}};
      int key[KEY_SIZE];
      get_key_guru(FFTW_FORWARD, 1, &dim, 2, howmany_dims, 1,
                                      in_place, key);
        
      plans[1] = lookup_plan_from_cache(key);
      if (plans[1] == NULL) {
        double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
        plans[1] = fft_fftw_create_guru_plan(key, buffer, grid_out);
        fftw_free(buffer);
      }
      if (block_sizes[1] * number_of_threads != number_of_ffts) {
        const int block_size_last_thread =
            number_of_ffts - (number_of_threads - 1) * block_sizes[1];
        fft_iodim howmany_dims[2] = {
            {.n = block_size_last_thread,
             .is = fft_size[1] * (fft_size[2] / 2 + 1),
             .os = fft_size[1] * (fft_size[2] / 2 + 1)},
            {.n = fft_size[2] / 2 + 1, .is = 1, .os = 1}};
        int key[KEY_SIZE];
        get_key_guru(FFTW_FORWARD, 1, &dim, 2, howmany_dims, 1,
                                      in_place, key);
        
        plans_last_thread[1] = lookup_plan_from_cache(key);
        if (plans_last_thread[1] == NULL) {
          double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
          plans_last_thread[1] = fft_fftw_create_guru_plan(key, buffer, grid_out);
          fftw_free(buffer);
        }
        has_plan_for_last_thread[1] = true;
      }
    }
    {
      const int number_of_ffts = fft_size[0] * fft_size[1];
      block_sizes[2] =
          (number_of_ffts + number_of_threads - 1) / number_of_threads;
      fft_iodim dim = {.n = fft_size[2], .is = 1, .os = 1};
      fft_iodim howmany_dim = {
          .n = block_sizes[2], .is = fft_size[2], .os = fft_size[2] / 2 + 1};
      int key[KEY_SIZE];
      get_key_guru_r2c(FFTW_FORWARD, 1, &dim, 1, &howmany_dim, 1, in_place, key);
      
      plans[2] = lookup_plan_from_cache(key);
      if (plans[2] == NULL) {
        double *buffer = fftw_alloc_real(2*get_buffer_size_from_key(key));
        plans[2] = fft_fftw_create_guru_plan_r2c(key, buffer, grid_out);
        fftw_free(buffer);
      }
      if (block_sizes[2] * number_of_threads != number_of_ffts) {
        const int block_size_last_thread =
            number_of_ffts - (number_of_threads - 1) * block_sizes[2];
        fft_iodim howmany_dim = {.n = block_size_last_thread,
                                 .is = fft_size[2],
                                 .os = fft_size[2] / 2 + 1};
        int key[KEY_SIZE];
        get_key_guru_r2c(FFTW_FORWARD, 1, &dim, 1, &howmany_dim, 1, in_place, key);
        
        plans_last_thread[2] = lookup_plan_from_cache(key);
        if (plans_last_thread[2] == NULL) {
          double *buffer = fftw_alloc_real(2*get_buffer_size_from_key(key));
          plans_last_thread[2] = fft_fftw_create_guru_plan_r2c(key, buffer, grid_out);
          fftw_free(buffer);
        }
        has_plan_for_last_thread[2] = true;
      }
    }
#pragma omp parallel default(none)                                             \
    shared(number_of_threads, has_plan_for_last_thread, plans,                 \
               plans_last_thread, block_sizes, fft_size, grid_in, grid_out)
    {
      const int thread_id = omp_get_thread_num();
      if (thread_id < number_of_threads - 1 || !has_plan_for_last_thread[2]) {
        assert(*plans[2] != NULL);
        fftw_execute_dft_r2c(
            *plans[2], grid_in + block_sizes[2] * fft_size[2] * thread_id,
            grid_out + block_sizes[2] * (fft_size[2] / 2 + 1) * thread_id);
      } else {
        assert(*plans_last_thread[2] != NULL);
        fftw_execute_dft_r2c(*plans_last_thread[2],
                             grid_in + block_sizes[2] * fft_size[2] * thread_id,
                             grid_out + block_sizes[2] * (fft_size[2] / 2 + 1) *
                                            thread_id);
      }
#pragma omp barrier
      if (thread_id < number_of_threads - 1 || !has_plan_for_last_thread[1]) {
        assert(*plans[1] != NULL);
        fftw_execute_dft(*plans[1],
                         grid_out + block_sizes[1] * fft_size[1] *
                                        (fft_size[2] / 2 + 1) * thread_id,
                         ((double complex *)grid_in) +
                             block_sizes[1] * fft_size[1] *
                                 (fft_size[2] / 2 + 1) * thread_id);
      } else {
        assert(*plans_last_thread[1] != NULL);
        fftw_execute_dft(*plans_last_thread[1],
                         grid_out + block_sizes[1] * fft_size[1] *
                                        (fft_size[2] / 2 + 1) * thread_id,
                         ((double complex *)grid_in) +
                             block_sizes[1] * fft_size[1] *
                                 (fft_size[2] / 2 + 1) * thread_id);
      }
#pragma omp barrier
      if (thread_id < number_of_threads - 1 || !has_plan_for_last_thread[0]) {
        assert(*plans[0] != NULL);
        fftw_execute_dft(*plans[0],
                         (double complex *)grid_in + block_sizes[0] * thread_id,
                         grid_out + block_sizes[0] * thread_id);
      } else {
        assert(*plans_last_thread[0] != NULL);
        fftw_execute_dft(*plans_last_thread[0],
                         (double complex *)grid_in + block_sizes[0] * thread_id,
                         grid_out + block_sizes[0] * thread_id);
      }
    }
  } else {
    int key[KEY_SIZE];
    get_key_3d_r2c(FFTW_FORWARD, fft_size, omp_get_max_threads(), in_place, key);
    fftw_plan *plan = lookup_plan_from_cache(key);
    if (plan == NULL) {
      double *buffer = fftw_alloc_real(2*get_buffer_size_from_key(key));
      plan = fft_fftw_create_3d_plan_r2c(key, buffer, in_place ? (double complex*)buffer : grid_out);
      fftw_free(buffer);
    }
    fftw_execute_dft_r2c(*plan, grid_in, grid_out);
  }
#else
  (void)fft_size;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs a local backwards C2C 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_3d_bw_local(const int fft_size[3], double complex *grid_in,
                          double complex *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  assert(is_initialized);
  if (fft_size[0] == 0 || fft_size[1] == 0 || fft_size[2] == 0) return;
  const bool in_place = grid_in == grid_out;
  if (((fft_size[0] >= 256 || fft_size[1] >= 256 || fft_size[2] >= 256 ||
       omp_get_max_threads() > 1)) && !in_place &&
      (fftw_planning_mode == FFTW_ESTIMATE)) {
    // The 3D FFT is not efficient with threading and estimate planning mode
    // So, we decompose it in a sequence of 1D FFTs
    int number_of_threads = 1;
#pragma omp parallel default(none) shared(number_of_threads)
    {
#pragma omp single
      { number_of_threads = omp_get_num_threads(); }
    }
    int block_sizes[3];
    fftw_plan *plans[3], *plans_last_thread[3];
    bool has_plan_for_last_thread[3] = {false, false, false};
    {
      const int number_of_ffts = fft_size[1] * fft_size[2];
      block_sizes[0] =
          (number_of_ffts + number_of_threads - 1) / number_of_threads;
      fft_iodim dim = {
          .n = fft_size[0], .is = number_of_ffts, .os = number_of_ffts};
      fft_iodim howmany_dim = {.n = block_sizes[0], .is = 1, .os = 1};
      int key[KEY_SIZE];
      get_key_guru(FFTW_BACKWARD, 1, &dim, 1, &howmany_dim, 1, in_place, key);
      
      plans[0] = lookup_plan_from_cache(key);
      if (plans[0] == NULL) {
        double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
        plans[0] = fft_fftw_create_guru_plan(key, buffer, grid_out);
        fftw_free(buffer);
      }
      if (block_sizes[0] * number_of_threads != number_of_ffts) {
        const int block_size_last_thread =
            number_of_ffts - (number_of_threads - 1) * block_sizes[0];
        fft_iodim howmany_dim = {.n = block_size_last_thread, .is = 1, .os = 1};
        int key[KEY_SIZE];
        get_key_guru(FFTW_BACKWARD, 1, &dim, 1, &howmany_dim, 1, in_place, key);
        
        plans_last_thread[0] = lookup_plan_from_cache(key);
        if (plans_last_thread[0] == NULL) {
          double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
          plans_last_thread[0] = fft_fftw_create_guru_plan(key, buffer, grid_out);
          fftw_free(buffer);
        }
        has_plan_for_last_thread[0] = true;
      }
    }
    {
      const int number_of_ffts = fft_size[0];
      block_sizes[1] =
          (number_of_ffts + number_of_threads - 1) / number_of_threads;
      fft_iodim dim = {.n = fft_size[1], .is = fft_size[2], .os = fft_size[2]};
      fft_iodim howmany_dims[2] = {{.n = block_sizes[1],
                                    .is = fft_size[1] * fft_size[2],
                                    .os = fft_size[1] * fft_size[2]},
                                   {.n = fft_size[2], .is = 1, .os = 1}};
      int key[KEY_SIZE];
      get_key_guru(FFTW_BACKWARD, 1, &dim, 2, howmany_dims, 1, in_place, key);
      plans[1] = lookup_plan_from_cache(key);
      if (plans[1] == NULL) {
        double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
        plans[1] = fft_fftw_create_guru_plan(key, buffer, grid_out);
        fftw_free(buffer);
      }
      if (block_sizes[1] * number_of_threads != number_of_ffts) {
        const int block_size_last_thread =
            number_of_ffts - (number_of_threads - 1) * block_sizes[1];
        fft_iodim howmany_dims[2] = {{.n = block_size_last_thread,
                                      .is = fft_size[1] * fft_size[2],
                                      .os = fft_size[1] * fft_size[2]},
                                     {.n = fft_size[2], .is = 1, .os = 1}};
      int key[KEY_SIZE];
      get_key_guru(FFTW_BACKWARD, 1, &dim, 2, howmany_dims, 1, in_place, key);
        
      plans_last_thread[1] = lookup_plan_from_cache(key);
      if (plans_last_thread[1] == NULL) {
        double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
        plans_last_thread[1] = fft_fftw_create_guru_plan(key, buffer, grid_out);
        fftw_free(buffer);
      }
        has_plan_for_last_thread[1] = true;
      }
    }
    {
      const int number_of_ffts = fft_size[0] * fft_size[1];
      block_sizes[2] =
          (number_of_ffts + number_of_threads - 1) / number_of_threads;
      fft_iodim dim = {.n = fft_size[2], .is = 1, .os = 1};
      fft_iodim howmany_dim = {
          .n = block_sizes[2], .is = fft_size[2], .os = fft_size[2]};
      int key[KEY_SIZE];
      get_key_guru(FFTW_BACKWARD, 1, &dim, 1, &howmany_dim, 1, in_place, key);
      
      plans[2] = lookup_plan_from_cache(key);
      if (plans[2] == NULL) {
        double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
        plans[2] = fft_fftw_create_guru_plan(key, buffer, grid_out);
        fftw_free(buffer);
      }
      if (block_sizes[2] * number_of_threads != number_of_ffts) {
        const int block_size_last_thread =
            number_of_ffts - (number_of_threads - 1) * block_sizes[2];
        fft_iodim howmany_dim = {
            .n = block_size_last_thread, .is = fft_size[2], .os = fft_size[2]};
      int key[KEY_SIZE];
      get_key_guru(FFTW_BACKWARD, 1, &dim, 1, &howmany_dim, 1, in_place, key);
        
      plans_last_thread[2] = lookup_plan_from_cache(key);
      if (plans_last_thread[2] == NULL) {
        double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
        plans_last_thread[2] = fft_fftw_create_guru_plan(key, buffer, grid_out);
        fftw_free(buffer);
      }
        has_plan_for_last_thread[2] = true;
      }
    }
#pragma omp parallel default(none)                                             \
    shared(number_of_threads, has_plan_for_last_thread, plans,                 \
               plans_last_thread, block_sizes, fft_size, grid_in, grid_out)
    {
      const int thread_id = omp_get_thread_num();
      if (thread_id < number_of_threads - 1 || !has_plan_for_last_thread[2]) {
        assert(*plans[2] != NULL);
        fftw_execute_dft(*plans[2],
                         grid_in + block_sizes[2] * fft_size[2] * thread_id,
                         grid_out + block_sizes[2] * fft_size[2] * thread_id);
      } else {
        assert(*plans_last_thread[2] != NULL);
        fftw_execute_dft(*plans_last_thread[2],
                         grid_in + block_sizes[2] * fft_size[2] * thread_id,
                         grid_out + block_sizes[2] * fft_size[2] * thread_id);
      }
#pragma omp barrier
      if (thread_id < number_of_threads - 1 || !has_plan_for_last_thread[1]) {
        assert(*plans[1] != NULL);
        fftw_execute_dft(
            *plans[1],
            grid_out + block_sizes[1] * fft_size[1] * fft_size[2] * thread_id,
            grid_in + block_sizes[1] * fft_size[1] * fft_size[2] * thread_id);
      } else {
        assert(*plans_last_thread[1] != NULL);
        fftw_execute_dft(
            *plans_last_thread[1],
            grid_out + block_sizes[1] * fft_size[1] * fft_size[2] * thread_id,
            grid_in + block_sizes[1] * fft_size[1] * fft_size[2] * thread_id);
      }
#pragma omp barrier
      if (thread_id < number_of_threads - 1 || !has_plan_for_last_thread[0]) {
        assert(*plans[0] != NULL);
        fftw_execute_dft(*plans[0], grid_in + block_sizes[0] * thread_id,
                         grid_out + block_sizes[0] * thread_id);
      } else {
        assert(*plans_last_thread[0] != NULL);
        fftw_execute_dft(*plans_last_thread[0],
                         grid_in + block_sizes[0] * thread_id,
                         grid_out + block_sizes[0] * thread_id);
      }
    }
  } else {
    int key[KEY_SIZE];
    get_key_3d(FFTW_BACKWARD, fft_size, omp_get_max_threads(), in_place, key);
    fftw_plan *plan = lookup_plan_from_cache(key);
    if (plan == NULL) {
      double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
      plan = fft_fftw_create_3d_plan(key, buffer, in_place ? buffer : grid_out);
      fftw_free(buffer);
    }
    fftw_execute_dft(*plan, grid_in, grid_out);
  }
#else
  (void)fft_size;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Performs a local backwards R2C 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_3d_bw_local_c2r(const int fft_size[3], double complex *grid_in,
                              double *grid_out) {
#if defined(__FFTW3)
  assert(omp_get_num_threads() == 1);
  assert(is_initialized);
  if (fft_size[0] == 0 || fft_size[1] == 0 || fft_size[2] == 0) return;
  const bool in_place = (double *)grid_in == grid_out;
  if (((fft_size[0] >= 256 || fft_size[1] >= 256 || fft_size[2] >= 256 ||
       omp_get_max_threads() > 1) && !in_place) &&
      (fftw_planning_mode == FFTW_ESTIMATE)) {
    // We cannot use the output buffer for planning because it may be too small
    // So, we allocate a new one
    double complex *buffer =
        fftw_alloc_complex(fft_size[0] * fft_size[1] * (fft_size[2] / 2 + 1));
    // The 3D FFT is not efficient with threading and estimate planning mode
    // So, we decompose it in a sequence of 1D FFTs
    int number_of_threads = 1;
#pragma omp parallel default(none) shared(number_of_threads)
    {
#pragma omp single
      { number_of_threads = omp_get_num_threads(); }
    }
    int block_sizes[3];
    fftw_plan *plans[3], *plans_last_thread[3];
    bool has_plan_for_last_thread[3] = {false, false, false};
    {
      const int number_of_ffts = fft_size[1] * (fft_size[2] / 2 + 1);
      block_sizes[0] =
          (number_of_ffts + number_of_threads - 1) / number_of_threads;
      fft_iodim dim = {
          .n = fft_size[0], .is = number_of_ffts, .os = number_of_ffts};
      fft_iodim howmany_dim = {.n = block_sizes[0], .is = 1, .os = 1};
      int key[KEY_SIZE];
      get_key_guru(FFTW_BACKWARD, 1, &dim, 1, &howmany_dim, 1, in_place, key);
      
      plans[0] = lookup_plan_from_cache(key);
      if (plans[0] == NULL) {
        plans[0] = fft_fftw_create_guru_plan(key, (double complex*)grid_out, buffer);
      }
      if (block_sizes[0] * number_of_threads != number_of_ffts) {
        const int block_size_last_thread =
            number_of_ffts - (number_of_threads - 1) * block_sizes[0];
        fft_iodim howmany_dim = {.n = block_size_last_thread, .is = 1, .os = 1};
      int key[KEY_SIZE];
      get_key_guru(FFTW_BACKWARD, 1, &dim, 1, &howmany_dim, 1, in_place, key);
        
      plans_last_thread[0] = lookup_plan_from_cache(key);
      if (plans_last_thread[0] == NULL) {
        double complex *buffer2 = fftw_alloc_complex(get_buffer_size_from_key(key));
        plans_last_thread[0] = fft_fftw_create_guru_plan(key, buffer2, buffer);
        fftw_free(buffer2);
      }
        has_plan_for_last_thread[0] = true;
      }
    }
    {
      const int number_of_ffts = fft_size[0];
      block_sizes[1] =
          (number_of_ffts + number_of_threads - 1) / number_of_threads;
      fft_iodim dim = {.n = fft_size[1],
                       .is = fft_size[2] / 2 + 1,
                       .os = fft_size[2] / 2 + 1};
      fft_iodim howmany_dims[2] = {
          {.n = block_sizes[1],
           .is = fft_size[1] * (fft_size[2] / 2 + 1),
           .os = fft_size[1] * (fft_size[2] / 2 + 1)},
          {.n = (fft_size[2] / 2 + 1), .is = 1, .os = 1}};
      int key[KEY_SIZE];
      get_key_guru(FFTW_BACKWARD, 1, &dim, 2, howmany_dims, 1, in_place, key);
      
      plans[1] = lookup_plan_from_cache(key);
      if (plans[1] == NULL) {
        double complex *buffer2 = fftw_alloc_complex(get_buffer_size_from_key(key));
        plans[1] = fft_fftw_create_guru_plan(key, buffer2, buffer);
        fftw_free(buffer2);
      }
      if (block_sizes[1] * number_of_threads != number_of_ffts) {
        const int block_size_last_thread =
            number_of_ffts - (number_of_threads - 1) * block_sizes[1];
        fft_iodim howmany_dims[2] = {
            {.n = block_size_last_thread,
             .is = fft_size[1] * (fft_size[2] / 2 + 1),
             .os = fft_size[1] * (fft_size[2] / 2 + 1)},
            {.n = fft_size[2] / 2 + 1, .is = 1, .os = 1}};
        int key[KEY_SIZE];
        get_key_guru(FFTW_BACKWARD, 1, &dim, 2, howmany_dims, 1, in_place, key);
          
        plans_last_thread[1] = lookup_plan_from_cache(key);
        if (plans_last_thread[1] == NULL) {
          double complex *buffer2 = fftw_alloc_complex(get_buffer_size_from_key(key));
          plans_last_thread[1] = fft_fftw_create_guru_plan(key, buffer2, buffer);
          fftw_free(buffer2);
        }
        has_plan_for_last_thread[1] = true;
      }
    }
    {
      const int number_of_ffts = fft_size[0] * fft_size[1];
      block_sizes[2] =
          (number_of_ffts + number_of_threads - 1) / number_of_threads;
      fft_iodim dim = {.n = fft_size[2], .is = 1, .os = 1};
      fft_iodim howmany_dim = {
          .n = block_sizes[2], .is = fft_size[2] / 2 + 1, .os = fft_size[2]};
      int key[KEY_SIZE];
      get_key_guru_r2c(FFTW_BACKWARD, 1, &dim, 1, &howmany_dim, 1, in_place, key);
      
      plans[2] = lookup_plan_from_cache(key);
      if (plans[2] == NULL) {
        double complex *buffer2 = fftw_alloc_complex(get_buffer_size_from_key(key));
        plans[2] = fft_fftw_create_guru_plan_r2c(key, (double *)buffer, buffer2);
        fftw_free(buffer2);
      }
      if (block_sizes[2] * number_of_threads != number_of_ffts) {
        const int block_size_last_thread =
            number_of_ffts - (number_of_threads - 1) * block_sizes[2];
        fft_iodim howmany_dim = {.n = block_size_last_thread,
                                 .is = fft_size[2] / 2 + 1,
                                 .os = fft_size[2]};
        int key[KEY_SIZE];
        get_key_guru_r2c(FFTW_BACKWARD, 1, &dim, 1, &howmany_dim, 1, in_place, key);
          
        plans_last_thread[2] = lookup_plan_from_cache(key);
        if (plans_last_thread[2] == NULL) {
          double complex *buffer2 = fftw_alloc_complex(get_buffer_size_from_key(key));
          plans_last_thread[2] = fft_fftw_create_guru_plan_r2c(key, (double*)buffer, buffer2);
          fftw_free(buffer2);
        }
        has_plan_for_last_thread[2] = true;
      }
    }
#pragma omp parallel default(none) shared(                                     \
        number_of_threads, has_plan_for_last_thread, plans, plans_last_thread, \
            block_sizes, fft_size, grid_in, grid_out, buffer)
    {
      const int thread_id = omp_get_thread_num();
      if (thread_id < number_of_threads - 1 || !has_plan_for_last_thread[0]) {
        assert(*plans[0] != NULL);
        fftw_execute_dft(*plans[0], grid_in + block_sizes[0] * thread_id,
                         buffer + block_sizes[0] * thread_id);
      } else {
        assert(*plans_last_thread[0] != NULL);
        fftw_execute_dft(*plans_last_thread[0],
                         grid_in + block_sizes[0] * thread_id,
                         buffer + block_sizes[0] * thread_id);
      }
#pragma omp barrier
      if (thread_id < number_of_threads - 1 || !has_plan_for_last_thread[1]) {
        assert(*plans[1] != NULL);
        fftw_execute_dft(*plans[1],
                         buffer + block_sizes[1] * fft_size[1] *
                                      (fft_size[2] / 2 + 1) * thread_id,
                         grid_in + block_sizes[1] * fft_size[1] *
                                       (fft_size[2] / 2 + 1) * thread_id);
      } else {
        assert(*plans_last_thread[1] != NULL);
        fftw_execute_dft(*plans_last_thread[1],
                         buffer + block_sizes[1] * fft_size[1] *
                                      (fft_size[2] / 2 + 1) * thread_id,
                         grid_in + block_sizes[1] * fft_size[1] *
                                       (fft_size[2] / 2 + 1) * thread_id);
      }
#pragma omp barrier
      if (thread_id < number_of_threads - 1 || !has_plan_for_last_thread[2]) {
        assert(*plans[2] != NULL);
        fftw_execute_dft_c2r(
            *plans[2],
            grid_in + block_sizes[2] * (fft_size[2] / 2 + 1) * thread_id,
            grid_out + block_sizes[2] * fft_size[2] * thread_id);
      } else {
        assert(*plans_last_thread[2] != NULL);
        fftw_execute_dft_c2r(
            *plans_last_thread[2],
            grid_in + block_sizes[2] * (fft_size[2] / 2 + 1) * thread_id,
            grid_out + block_sizes[2] * fft_size[2] * thread_id);
      }
    }
    fftw_free(buffer);
  } else {
    int key[KEY_SIZE];
    get_key_3d_r2c(FFTW_BACKWARD, fft_size, omp_get_max_threads(), in_place, key);
    fftw_plan *plan = lookup_plan_from_cache(key);
    if (plan == NULL) {
      double complex *buffer = fftw_alloc_complex(get_buffer_size_from_key(key));
      plan = fft_fftw_create_3d_plan_r2c(key, in_place ? (double*)buffer : grid_out, buffer);
      fftw_free(buffer);
    }
    fftw_execute_dft_c2r(*plan, grid_in, grid_out);
  }
#else
  (void)fft_size;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with FFTW support.");
#endif
}

/*******************************************************************************
 * \brief Returns sizes and starts of distributed C2C 2D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_fftw_2d_distributed_sizes(const int npts_global[2],
                                  const int number_of_ffts,
                                  const cp_mpi_comm_t comm, int *local_n0,
                                  int *local_n0_start, int *local_n1,
                                  int *local_n1_start) {
#if defined(__USE_FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(is_initialized);
  assert(use_fftw_mpi);
  if (npts_global[0] <= 0 || npts_global[1] <= 0 || number_of_ffts <= 0) {
    *local_n0_start = 0;
    *local_n1_start = 0;
    *local_n0 = 0;
    *local_n1 = 0;
    return 0;
  }
  const ptrdiff_t n[2] = {npts_global[0], npts_global[1]};
  const ptrdiff_t howmany = number_of_ffts;
  const ptrdiff_t block_size_0 =
      (npts_global[0] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
  const ptrdiff_t block_size_1 =
      (npts_global[1] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
  ptrdiff_t my_local_n0, my_local_n0_start, my_local_n1, my_local_n1_start;
  const ptrdiff_t buffer_size = fftw_mpi_local_size_many_transposed(
      2, n, howmany, block_size_0, block_size_1, comm, &my_local_n0,
      &my_local_n0_start, &my_local_n1, &my_local_n1_start);
  *local_n0 = my_local_n0;
  *local_n0_start = my_local_n0_start;
  *local_n1 = my_local_n1;
  *local_n1_start = my_local_n1_start;
  return buffer_size;
#else
  (void)npts_global;
  (void)number_of_ffts;
  (void)comm;
  (void)local_n0;
  (void)local_n0_start;
  (void)local_n1;
  (void)local_n1_start;
  assert(0 && "The grid library was not compiled with FFTW support.");
  return -1;
#endif
}

/*******************************************************************************
 * \brief Returns sizes and starts of distributed R2C/C2R 2D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_fftw_2d_distributed_sizes_r2c(const int npts_global[2],
                                      const int number_of_ffts,
                                      const cp_mpi_comm_t comm, int *local_n0,
                                      int *local_n0_start, int *local_n1,
                                      int *local_n1_start) {
#if defined(__USE_FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(is_initialized);
  assert(use_fftw_mpi);
  if (npts_global[0] <= 0 || npts_global[1] <= 0 || number_of_ffts <= 0) {
    *local_n0_start = 0;
    *local_n1_start = 0;
    *local_n0 = 0;
    *local_n1 = 0;
    return 0;
  }
  const ptrdiff_t n[2] = {npts_global[0], npts_global[1] / 2 + 1};
  const ptrdiff_t howmany = number_of_ffts;
  const ptrdiff_t block_size_0 =
      (npts_global[0] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
  const ptrdiff_t block_size_1 =
      (npts_global[1] / 2 + 1 + cp_mpi_comm_size(comm) - 1) /
      cp_mpi_comm_size(comm);
  ptrdiff_t my_local_n0, my_local_n0_start, my_local_n1, my_local_n1_start;
  const ptrdiff_t buffer_size = fftw_mpi_local_size_many_transposed(
      2, n, howmany, block_size_0, block_size_1, comm, &my_local_n0,
      &my_local_n0_start, &my_local_n1, &my_local_n1_start);
  *local_n0 = my_local_n0;
  *local_n0_start = my_local_n0_start;
  *local_n1 = my_local_n1;
  *local_n1_start = my_local_n1_start;
  return buffer_size;
#else
  (void)npts_global;
  (void)number_of_ffts;
  (void)comm;
  (void)local_n0;
  (void)local_n0_start;
  (void)local_n1;
  (void)local_n1_start;
  assert(0 && "The grid library was not compiled with FFTW support.");
  return -1;
#endif
}

/*******************************************************************************
 * \brief Returns sizes and starts of distributed C2C 3D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_fftw_3d_distributed_sizes(const int npts_global[3],
                                  const cp_mpi_comm_t comm, int *local_n0,
                                  int *local_n0_start, int *local_n1,
                                  int *local_n1_start) {
#if defined(__USE_FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(is_initialized);
  assert(use_fftw_mpi);
  if (npts_global[0] <= 0 || npts_global[1] <= 0 || npts_global[2] <= 0) {
    *local_n0_start = 0;
    *local_n1_start = 0;
    *local_n0 = 0;
    *local_n1 = 0;
    return 0;
  }
  const ptrdiff_t n[3] = {npts_global[0], npts_global[1], npts_global[2]};
  ptrdiff_t my_local_n0, my_local_n0_start;
  ptrdiff_t my_local_n1, my_local_n1_start;
  const ptrdiff_t block_size_0 =
      (npts_global[0] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
  const ptrdiff_t block_size_1 =
      (npts_global[1] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
  const ptrdiff_t my_buffer_size = fftw_mpi_local_size_many_transposed(
      3, n, 1, block_size_0, block_size_1, comm, &my_local_n0,
      &my_local_n0_start, &my_local_n1, &my_local_n1_start);
  *local_n0 = my_local_n0;
  *local_n0_start = my_local_n0_start;
  *local_n1 = my_local_n1;
  *local_n1_start = my_local_n1_start;
  return my_buffer_size;
#else
  (void)npts_global;
  (void)comm;
  (void)local_n0;
  (void)local_n0_start;
  (void)local_n1;
  (void)local_n1_start;
  assert(0 && "The grid library was not compiled with FFTW support.");
  return -1;
#endif
}

/*******************************************************************************
 * \brief Returns sizes and starts of distributed R2C/C2R 3D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_fftw_3d_distributed_sizes_r2c(const int npts_global[3],
                                      const cp_mpi_comm_t comm, int *local_n0,
                                      int *local_n0_start, int *local_n1,
                                      int *local_n1_start) {
#if defined(__USE_FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(is_initialized);
  assert(use_fftw_mpi);
  if (npts_global[0] <= 0 || npts_global[1] <= 0 || npts_global[2] <= 0) {
    *local_n0_start = 0;
    *local_n1_start = 0;
    *local_n0 = 0;
    *local_n1 = 0;
    return 0;
  }
  const ptrdiff_t n[3] = {npts_global[0], npts_global[1], npts_global[2]};
  ptrdiff_t my_local_n0, my_local_n0_start;
  ptrdiff_t my_local_n1, my_local_n1_start;
  const ptrdiff_t block_size_0 =
      (npts_global[0] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
  const ptrdiff_t block_size_1 =
      (npts_global[1] + cp_mpi_comm_size(comm) - 1) / cp_mpi_comm_size(comm);
  const ptrdiff_t my_buffer_size = fftw_mpi_local_size_many_transposed(
      3, n, 1, block_size_0, block_size_1, comm, &my_local_n0,
      &my_local_n0_start, &my_local_n1, &my_local_n1_start);
  *local_n0 = my_local_n0;
  *local_n0_start = my_local_n0_start;
  *local_n1 = my_local_n1;
  *local_n1_start = my_local_n1_start;
  return my_buffer_size;
#else
  (void)npts_global;
  (void)comm;
  (void)local_n0;
  (void)local_n0_start;
  (void)local_n1;
  (void)local_n1_start;
  assert(0 && "The grid library was not compiled with FFTW support.");
  return -1;
#endif
}

/*******************************************************************************
 * \brief Performs a distributed forward C2C 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_2d_fw_distributed(const int npts_global[2],
                                const int number_of_ffts,
                                const cp_mpi_comm_t comm,
                                double complex *grid_in,
                                double complex *grid_out) {
#if defined(__USE_FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(is_initialized);
  assert(use_fftw_mpi);
  if (npts_global[0] == 0 || npts_global[1] == 0 || number_of_ffts == 0) return;
  int key[KEY_SIZE];
  get_key_2d_distributed(FFTW_FORWARD, npts_global, number_of_ffts,
                          comm, omp_get_max_threads(), key);
  fftw_plan *plan = fft_fftw_create_distributed_2d_plan(key, grid_out);
  assert(plan != NULL);
  fftw_mpi_execute_dft(*plan, grid_in, grid_out);
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
 * \brief Performs a distributed forward R2C 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_2d_fw_distributed_r2c(const int npts_global[2],
                                    const int number_of_ffts,
                                    const cp_mpi_comm_t comm, double *grid_in,
                                    double complex *grid_out) {
#if defined(__USE_FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(is_initialized);
  assert(use_fftw_mpi);
  if (npts_global[0] == 0 || npts_global[1] == 0 || number_of_ffts == 0) return;
  int key[KEY_SIZE];
  get_key_2d_r2c_distributed(FFTW_FORWARD, npts_global, number_of_ffts,
                          comm, omp_get_max_threads(), key);
  fftw_plan *plan = fft_fftw_create_distributed_2d_plan_r2c(key, grid_out);
  assert(plan != NULL);
  fftw_mpi_execute_dft_r2c(*plan, grid_in, grid_out);
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
 * \brief Performs a distributed backwards C2C 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_2d_bw_distributed(const int npts_global[2],
                                const int number_of_ffts,
                                const cp_mpi_comm_t comm,
                                double complex *grid_in,
                                double complex *grid_out) {
#if defined(__USE_FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(is_initialized);
  assert(use_fftw_mpi);
  if (npts_global[0] == 0 || npts_global[1] == 0 || number_of_ffts == 0) return;
  int key[KEY_SIZE];
  get_key_2d_distributed(FFTW_BACKWARD, npts_global, number_of_ffts,
                          comm, omp_get_max_threads(), key);
  fftw_plan *plan = fft_fftw_create_distributed_2d_plan(key, grid_out);
  assert(plan != NULL);
  fftw_mpi_execute_dft(*plan, grid_in, grid_out);
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
 * \brief Performs a distributed backwards C2R 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_2d_bw_distributed_c2r(const int npts_global[2],
                                    const int number_of_ffts,
                                    const cp_mpi_comm_t comm,
                                    double complex *grid_in, double *grid_out) {
#if defined(__USE_FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(is_initialized);
  assert(use_fftw_mpi);
  if (npts_global[0] == 0 || npts_global[1] == 0 || number_of_ffts == 0) return;
  int key[KEY_SIZE];
  get_key_2d_r2c_distributed(FFTW_BACKWARD, npts_global, number_of_ffts,
                          comm, omp_get_max_threads(), key);
  fftw_plan *plan = fft_fftw_create_distributed_2d_plan_r2c(
      key, (double complex *)grid_out);
  assert(plan != NULL);
  fftw_mpi_execute_dft_c2r(*plan, grid_in, grid_out);
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
 * \brief Performs a distributed forwards C2C 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_3d_fw_distributed(const int npts_global[3],
                                const cp_mpi_comm_t comm,
                                double complex *grid_in,
                                double complex *grid_out) {
#if defined(__USE_FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(is_initialized);
  assert(use_fftw_mpi);
  if (npts_global[0] == 0 || npts_global[1] == 0 || npts_global[2] == 0) return;
  int key[KEY_SIZE];
  get_key_3d_distributed(FFTW_FORWARD, npts_global, comm, omp_get_max_threads(), key);
  fftw_plan *plan = fft_fftw_create_distributed_3d_plan(key, grid_out);
  assert(plan != NULL);
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
 * \brief Performs a distributed forward R2C 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_3d_fw_distributed_r2c(const int npts_global[3],
                                    const cp_mpi_comm_t comm, double *grid_in,
                                    double complex *grid_out) {
#if defined(__USE_FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(is_initialized);
  assert(use_fftw_mpi);
  if (npts_global[0] == 0 || npts_global[1] == 0 || npts_global[2] == 0) return;
  int key[KEY_SIZE];
  get_key_3d_r2c_distributed(FFTW_FORWARD, npts_global, comm, omp_get_max_threads(), key);
  fftw_plan *plan = fft_fftw_create_distributed_3d_plan_r2c(key, grid_out);
  assert(plan != NULL);
  fftw_mpi_execute_dft_r2c(*plan, grid_in, grid_out);
#else
  (void)npts_global;
  (void)comm;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with FFTW and MPI support.");
#endif
}

/*******************************************************************************
 * \brief Performs a distributed backwards C2C 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_3d_bw_distributed(const int npts_global[3],
                                const cp_mpi_comm_t comm,
                                double complex *grid_in,
                                double complex *grid_out) {
#if defined(__USE_FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(is_initialized);
  assert(use_fftw_mpi);
  if (npts_global[0] == 0 || npts_global[1] == 0 || npts_global[2] == 0) return;
  int key[KEY_SIZE];
  get_key_3d_distributed(FFTW_BACKWARD, npts_global, comm, omp_get_max_threads(), key);
  fftw_plan *plan = fft_fftw_create_distributed_3d_plan(key, grid_out);
  assert(plan != NULL);
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
 * \brief Performs a distributed backwards C2R 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_3d_bw_distributed_c2r(const int npts_global[3],
                                    const cp_mpi_comm_t comm,
                                    double complex *grid_in, double *grid_out) {
#if defined(__USE_FFTW3_MPI)
  assert(omp_get_num_threads() == 1);
  assert(is_initialized);
  assert(use_fftw_mpi);
  if (npts_global[0] == 0 || npts_global[1] == 0 || npts_global[2] == 0) return;
  int key[KEY_SIZE];
  get_key_3d_r2c_distributed(FFTW_BACKWARD, npts_global, comm, omp_get_max_threads(), key);
  fftw_plan *plan = fft_fftw_create_distributed_3d_plan_r2c(key, (double complex*)grid_out);
  assert(plan != NULL);
  fftw_mpi_execute_dft_c2r(*plan, grid_in, grid_out);
#else
  (void)npts_global;
  (void)comm;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with FFTW and MPI support.");
#endif
}

// EOF
