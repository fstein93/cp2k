/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "../offload/offload_runtime.h"
#include "grid_fft_lib_gpu.h"

#include "../offload/offload_fft.h"
#include "../offload/offload_library.h"
#include <assert.h>
#include <omp.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>


/*******************************************************************************
 * \brief Static variables for retaining objects that are expensive to create.
 * \author Ole Schuett
 ******************************************************************************/
typedef struct {
  int key[5];
  offload_fftHandle *plan;
} cache_entry;

#define PW_GPU_CACHE_SIZE 32
static cache_entry cache[PW_GPU_CACHE_SIZE];
static int cache_oldest_entry = 0; // used for LRU eviction

static double *buffer_dev_1, *buffer_dev_2;
static int *ghatmap_dev;
static size_t allocated_buffer_size, allocated_map_size;

static offloadStream_t stream;
static bool is_initialized = false;

/*******************************************************************************
 * \brief Initialize the FFT library (if not done externally).
 * \author Ole Schuett
 ******************************************************************************/
void fft_gpu_init_lib() {
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
assert(omp_get_num_threads() == 1);
  if (is_initialized) {
    // fprintf(stderr, "Error: pw_gpu was already initialized.\n");
    // TODO abort();
    return;
  }
  memset(cache, 0, sizeof(cache_entry) * PW_GPU_CACHE_SIZE);
  cache_oldest_entry = 0;

  allocated_buffer_size = 1; // start small
  allocated_map_size = 1;
  offload_activate_chosen_device();
  offloadMalloc((void **)&buffer_dev_1, allocated_buffer_size);
  offloadMalloc((void **)&buffer_dev_2, allocated_buffer_size);
  offloadMalloc((void **)&ghatmap_dev, allocated_map_size);

  offloadStreamCreate(&stream);
  is_initialized = true;
#endif
}

/*******************************************************************************
 * \brief Finalize the FFT library (if not done externally).
 * \author Ole Schuett
 ******************************************************************************/
void fft_gpu_finalize_lib() {
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
   assert(omp_get_num_threads() == 1);
  if (!is_initialized) {
    // fprintf(stderr, "Error: pw_gpu is not initialized.\n");
    // TODO abort();
    return;
  }
  for (int i = 0; i < PW_GPU_CACHE_SIZE; i++) {
    if (cache[i].plan != NULL) {
      offload_fftDestroy(*cache[i].plan);
      free(cache[i].plan);
    }
  }
  offloadFree(buffer_dev_1);
  offloadFree(buffer_dev_2);
  offloadFree(ghatmap_dev);
  offloadStreamDestroy(stream);
  is_initialized = false;
#endif
}

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_gpu_allocate_double(const int length, double **buffer) {
  assert(buffer != NULL);
  assert(*buffer == NULL);
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
  offload_host_malloc((void**)buffer, length);
#else
  (void)length;
  (void)buffer;
  assert(0 && "The grid library was not compiled with GPU support.");
#endif
}

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_gpu_allocate_complex(const int length, double complex **buffer) {
  assert(buffer != NULL);
  assert(*buffer == NULL);
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
  offload_host_malloc((void**)buffer, 2*length);
#else
  (void)length;
  (void)buffer;
  assert(0 && "The grid library was not compiled with GPU support.");
#endif
}

/*******************************************************************************
 * \brief Free buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_gpu_free_double(double *buffer) {
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
  offload_host_free(buffer);
#else
  (void)buffer;
  assert(0 && "The grid library was not compiled with GPU support.");
#endif
}

/*******************************************************************************
 * \brief Free buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_gpu_free_complex(double complex *buffer) {
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
  offload_host_free(buffer);
#else
  (void)buffer;
  assert(0 && "The grid library was not compiled with GPU support.");
#endif
}

/*******************************************************************************
 * \brief Checks size of device buffers and re-allocates them if necessary.
 * \author Ole Schuett
 ******************************************************************************/
static void ensure_memory_sizes(const size_t requested_buffer_size,
                                const size_t requested_map_size) {
  assert(is_initialized);
  if (requested_buffer_size > allocated_buffer_size) {
    offloadFree(buffer_dev_1);
    offloadFree(buffer_dev_2);
    offloadMalloc((void **)&buffer_dev_1, requested_buffer_size);
    offloadMalloc((void **)&buffer_dev_2, requested_buffer_size);
    allocated_buffer_size = requested_buffer_size;
  }
  if (requested_map_size > allocated_map_size) {
    offloadFree(ghatmap_dev);
    offloadMalloc((void **)&ghatmap_dev, requested_map_size);
    allocated_map_size = requested_map_size;
  }
}

/*******************************************************************************
 * \brief Fetches an fft plan from the cache. Returns NULL if not found.
 * \author Ole Schuett
 ******************************************************************************/
static offload_fftHandle *lookup_plan_from_cache(const int key[5]) {
  assert(is_initialized);
  for (int i = 0; i < PW_GPU_CACHE_SIZE; i++) {
    const int *x = cache[i].key;
    if (x[0] == key[0] && x[1] == key[1] && x[2] == key[2] && x[3] == key[3] && x[4] == key[4]) {
      return cache[i].plan;
    }
  }
  return NULL;
}

/*******************************************************************************
 * \brief Adds an fft plan to the cache. Assumes ownership of plan's memory.
 * \author Ole Schuett
 ******************************************************************************/
static void add_plan_to_cache(const int key[5], offload_fftHandle *plan) {
  const int i = cache_oldest_entry;
  cache_oldest_entry = (cache_oldest_entry + 1) % PW_GPU_CACHE_SIZE;
  if (cache[i].plan != NULL) {
    offload_fftDestroy(*cache[i].plan);
    free(cache[i].plan);
  }
  cache[i].key[0] = key[0];
  cache[i].key[1] = key[1];
  cache[i].key[2] = key[2];
  cache[i].key[3] = key[3];
  cache[i].key[4] = key[4];
  cache[i].plan = plan;
}


/*******************************************************************************
 * \brief   Performs a scaled double precision complex 1D-FFT many times on
 *          the GPU.
 *          Input/output are DEVICE pointers (data_in, date_out).
 * \author  Andreas Gloess, Ole Schuett
 ******************************************************************************/
void fft_gpu_1d_local_gpu(const int direction, const int n, const int m,
                   const double *data_in, double *data_out) {
  const int key[5] = {1, direction, n, m, 0}; // first key entry is dimensions
  offload_fftHandle *plan = lookup_plan_from_cache(key);

  if (plan == NULL) {
    int nsize[1] = {n};
    int inembed[1] = {0}; // Is ignored, but is not allowed to be NULL.
    int onembed[1] = {0}; // Is ignored, but is not allowed to be NULL.
    int batch = m;
    int istride, idist, ostride, odist;
    if (direction == OFFLOAD_FFT_FORWARD) {
      istride = m;
      idist = 1;
      ostride = 1;
      odist = n;
    } else {
      istride = 1;
      idist = n;
      ostride = m;
      odist = 1;
    }
    plan = malloc(sizeof(cache_entry));
    offload_fftPlanMany(plan, 1, nsize, inembed, istride, idist, onembed,
                        ostride, odist, OFFLOAD_FFT_Z2Z, batch);
    offload_fftSetStream(*plan, stream);
    add_plan_to_cache(key, plan);
  }

  offload_fftExecZ2Z(*plan, data_in, data_out, direction);
}

/*******************************************************************************
 * \brief   Performs a scaled double precision complex 1D-FFT many times on
 *          the GPU.
 *          Input/output are DEVICE pointers (data_in, date_out).
 * \author  Frederick Stein
 ******************************************************************************/
void fft_gpu_2d_local_gpu(const int direction, const int n1, const int n2, const int m,
                   const double *data_in, double *data_out) {
  const int key[5] = {2, direction, n1, n2, m}; // first key entry is dimensions
  offload_fftHandle *plan = lookup_plan_from_cache(key);

  if (plan == NULL) {
    int nsize[2] = {n2, n1};
    int inembed[2] = {0, 0}; // Is ignored, but is not allowed to be NULL.
    int onembed[2] = {0, 0}; // Is ignored, but is not allowed to be NULL.
    int batch = m;
    int istride, idist, ostride, odist;
    if (direction == OFFLOAD_FFT_FORWARD) {
      istride = m;
      idist = 1;
      ostride = 1;
      odist = n1*n2;
    } else {
      istride = 1;
      idist = n1*n2;
      ostride = m;
      odist = 1;
    }
    plan = malloc(sizeof(cache_entry));
    offload_fftPlanMany(plan, 1, nsize, inembed, istride, idist, onembed,
                        ostride, odist, OFFLOAD_FFT_Z2Z, batch);
    offload_fftSetStream(*plan, stream);
    add_plan_to_cache(key, plan);
  }

  offload_fftExecZ2Z(*plan, data_in, data_out, direction);
}

/*******************************************************************************
 * \brief   Performs a scaled double precision complex 3D-FFT on the GPU.
 *          Input/output is a DEVICE pointer (data).
 * \author  Andreas Gloess, Ole Schuett
 ******************************************************************************/
void fft_gpu_3d_local_gpu(const int direction, const int nx, const int ny,
                   const int nz, double *data) {
  const int key[5] = {3, direction, nx, ny, nz}; // first key entry is dimensions
  offload_fftHandle *plan = lookup_plan_from_cache(key);

  if (plan == NULL) {
    plan = malloc(sizeof(cache_entry));
    offload_fftPlan3d(plan, nx, ny, nz, OFFLOAD_FFT_Z2Z);
    offload_fftSetStream(*plan, stream);
    add_plan_to_cache(key, plan);
  }

  offload_fftExecZ2Z(*plan, data, data, direction);
}

/*******************************************************************************
 * \brief GPU implementation of FFT from transposed format.
 * \author Ole Schuett
 ******************************************************************************/
void fft_gpu_1d_local_low(const int direction, const int n, const int m,
                          double complex *grid_in, double complex *grid_out) {
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
	  // Check inputs.
  assert(omp_get_num_threads() == 1);
  const int nrpts = n * m;
  if (nrpts == 0) {
    return; // Nothing to do.
  }

  // Allocate device memory.
  offload_activate_chosen_device();
  const size_t buffer_size = 2 * sizeof(double) * nrpts;
  ensure_memory_sizes(buffer_size, 0);

  // Upload COMPLEX input to device.
  offloadMemcpyAsyncHtoD(buffer_dev_1, grid_in, buffer_size, stream);

  // Run FFT on the device.
    fft_gpu_1d_local_gpu(direction, n, m, buffer_dev_1, buffer_dev_2);

  // Download COMPLEX results from device.
  offloadMemcpyAsyncDtoH(grid_out, buffer_dev_2, buffer_size, stream);
  offloadStreamSynchronize(stream);
#else
  (void)direction;
  (void)n;
  (void)m;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with GPU support.");
#endif
}

void fft_gpu_1d_fw_local( const int n, const int m,
                          double complex *grid_in, double complex *grid_out) {
	fft_gpu_1d_local_low(OFFLOAD_FFT_FORWARD, n, m, grid_in, grid_out);
}

void fft_gpu_1d_bw_local( const int n, const int m,
                          double complex *grid_in, double complex *grid_out) {
        fft_gpu_1d_local_low(OFFLOAD_FFT_INVERSE, n, m, grid_in, grid_out);
}

/*******************************************************************************
 * \brief Local transposition.
 * \author Frederick Stein
 ******************************************************************************/
void fft_gpu_transpose_local(double complex *grid,
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
void fft_gpu_2d_local_low(const int direction, const int n1, const int n2, const int m,
                          double complex *grid_in, double complex *grid_out) {
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
	  assert(omp_get_num_threads() == 1);
  const int nrpts = n1 * n2 * m;
  if (nrpts == 0) {
    return; // Nothing to do.
  }

  // Allocate device memory.
  offload_activate_chosen_device();
  const size_t buffer_size = 2 * sizeof(double) * nrpts;
  ensure_memory_sizes(buffer_size, 0);

  // Upload COMPLEX input to device.
  offloadMemcpyAsyncHtoD(buffer_dev_1, grid_in, buffer_size, stream);

  // Run FFT on the device.
    fft_gpu_2d_local_gpu(direction, n1, n2, m, buffer_dev_1, buffer_dev_2);

  // Download COMPLEX results from device.
  offloadMemcpyAsyncDtoH(grid_out, buffer_dev_2, buffer_size, stream);
  offloadStreamSynchronize(stream);
#else
  (void)direction;
  (void)n1;
  (void)n2;
  (void)m;
  (void)grid_in;
  (void)grid_out;
  assert(0 && "The grid library was not compiled with GPU support.");
#endif
}

void fft_gpu_2d_fw_local( const int n1, const int n2, const int m,
                          double complex *grid_in, double complex *grid_out) {
        fft_gpu_2d_local_low(OFFLOAD_FFT_FORWARD, n1, n2, m, grid_in, grid_out);
}

void fft_gpu_2d_bw_local( const int n1, const int n2, const int m,
                          double complex *grid_in, double complex *grid_out) {
        fft_gpu_2d_local_low(OFFLOAD_FFT_INVERSE, n1, n2, m, grid_in, grid_out);
}

/*******************************************************************************
 * \brief Performs local 3D FFT.
 * \author Ole Schuett
 ******************************************************************************/
void fft_gpu_3d_local_low(const int direction, const int nx, const int ny, const int nz,
                          double complex *grid_in, double complex *grid_out) {
#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_PW)
  // Check inputs.
  assert(omp_get_num_threads() == 1);
  const int nrpts = nx*ny*nz;
  if (nrpts == 0) {
    return; // Nothing to do.
  }

  // Allocate device memory.
  offload_activate_chosen_device();
  const size_t buffer_size = sizeof(double) * nrpts;
  ensure_memory_sizes(buffer_size, 0);

  // Upload COMPLEX input to device.
  offloadMemcpyAsyncHtoD(buffer_dev_1, grid_in, buffer_size, stream);

  // Run FFT on the device.
    fft_gpu_3d_local_gpu(direction, nx, ny, nz, buffer_dev_1);

  // Download COMPLEX results from device.
  offloadMemcpyAsyncDtoH(grid_out, buffer_dev_1, buffer_size, stream);
  offloadStreamSynchronize(stream);
#else
  (void)plan_fw;
  assert(0 && "The grid library was not compiled with GPU support.");
#endif
}

void fft_gpu_3d_fw_local( const int nx, const int ny, const int nz,
                          double complex *grid_in, double complex *grid_out) {
        fft_gpu_3d_local_low(OFFLOAD_FFT_FORWARD, nx, ny, nz, grid_in, grid_out);
}

void fft_gpu_3d_bw_local( const int nx, const int ny, const int nz,
                          double complex *grid_in, double complex *grid_out) {
        fft_gpu_3d_local_low(OFFLOAD_FFT_INVERSE, nx, ny, nz, grid_in, grid_out);
}

// EOF
