/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

/*
 * cuFFTMp distributed FFT backend.
 *
 * The local FFTs (1D/2D/3D and the guru interface) are still served by the
 * FFTW backend.  This file only implements the distributed FFT interface
 * (fft_*_distributed and fft_*_distributed_sizes) on top of cuFFTMp.
 *
 * cuFFTMp works on device memory and supports arbitrary slab/pencil
 * decompositions through cufftXtSetDistribution().  We describe the same
 * transposed pencil decomposition that FFTW's MPI backend uses.  cuFFTMp,
 * however, stores a local box in row-major order of the global axes, whereas
 * CP2K/FFTW store the (transposed) output distribution with the distributed
 * axis as the outermost one and the batch of a 2D transform as the innermost
 * one.  The difference is bridged by explicitly packing/unpacking the host
 * buffer around each cuFFTMp call.
 */

#include "fft_lib_cufftmp_mpi.h"

#include "../base/cp_assert.h"
#include "../mpiwrap/cp_mpi.h"
#include "../offload/offload_library.h"
#include "fft_utils.h"

#include <complex.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__CUFFTMP) && defined(__parallel)

#include <cuda_runtime.h>

/* The cuFFTMp header uses a C++-style opaque struct typedef that is not valid
 * C.  Provide the tag typedef up front so that the header also works in C. */
#ifndef __cplusplus
typedef struct cufftReshapeHandle_st cufftReshapeHandle_st;
#endif

/* Including the header through its sub-directory makes the relative include
 * of the version-locked "cufft.h" work. */
#include <cufftmp/cufftMp.h>

#define CUFFTMP_CACHE_SIZE 64

typedef enum { CUFFTMP_C2C, CUFFTMP_R2C, CUFFTMP_C2R } cufftmp_type;

typedef struct {
  bool used;
  int rank;
  int npts_global[3];
  cufftmp_type type;
  bool direction;
  int comm_f;
  cp_mpi_comm_t comm;
  cufftHandle plan;
  cufftType fft_type;
  long long lower_in[3], upper_in[3], strides_in[3];
  long long lower_out[3], upper_out[3], strides_out[3];
  /* Sizes of the consumed (input) and produced (output) local buffers for a
   * single transform, and which descriptor format is consumed. */
  size_t consumed_elems;
  size_t produced_elems;
  bool consumed_is_output;
  /* Layout information needed to pack/unpack the FFTW-style host layout. */
  int n0, n1, l0, l1;
  int seg_in, seg_out;
  int host_last_in, host_last_out;
  bool is_r2c;
} cufftmp_plan_entry;

static cufftmp_plan_entry plan_cache[CUFFTMP_CACHE_SIZE];
static int cache_oldest = 0;

/*******************************************************************************
 * \brief Partition n elements over P ranks using FFTW's block distribution.
 ******************************************************************************/
static void block_dist(const int n, const int P, const int proc, int *start,
                       int *length) {
  const int block = (n + P - 1) / P;
  int s = proc * block;
  if (s > n)
    s = n;
  int e = s + block;
  if (e > n)
    e = n;
  *length = e - s;
  *start = (*length == 0) ? 0 : s;
}

/*******************************************************************************
 * \brief Check cuFFT(Mp) return codes.
 ******************************************************************************/
static void cufftmp_check(const cufftResult result, const char *what) {
  if (result != CUFFT_SUCCESS) {
    fprintf(stderr, "cuFFTMp error %d in %s\n", (int)result, what);
    cp_assert(0 && "cuFFTMp call failed.");
  }
}

/*******************************************************************************
 * \brief Fill the input/output boxes for the transposed pencil layout.
 ******************************************************************************/
static void cufftmp_compute_boxes(const int proc, const int P,
                                  const int npts_global[3],
                                  const cufftmp_type type,
                                  const bool direction,
                                  long long lower_in[3], long long upper_in[3],
                                  long long strides_in[3],
                                  long long lower_out[3],
                                  long long upper_out[3],
                                  long long strides_out[3], int *rank_out,
                                  int *local0_out, int *local1_out,
                                  size_t *consumed_elems, size_t *produced_elems,
                                  bool *consumed_is_output) {
  const int rank = (npts_global[2] > 0) ? 3 : 2;
  *rank_out = rank;

  int local0_start, local0, local1_start, local1;
  block_dist(npts_global[0], P, proc, &local0_start, &local0);

  size_t in_elems, out_elems;

  if (rank == 3) {
    const int n0 = npts_global[0];
    const int n1 = npts_global[1];
    const int n2 = npts_global[2];
    const int n2c = n2 / 2 + 1;

    block_dist(n1, P, proc, &local1_start, &local1);
    in_elems = (size_t)local0 * n1 * n2;
    out_elems = (size_t)n0 * local1 *
                ((type == CUFFTMP_C2C) ? (size_t)n2 : (size_t)n2c);

    lower_in[0] = local0_start;
    lower_in[1] = 0;
    lower_in[2] = 0;
    upper_in[0] = local0_start + local0;
    upper_in[1] = n1;
    upper_in[2] = n2;
    strides_in[0] = (long long)n1 * n2;
    strides_in[1] = n2;
    strides_in[2] = 1;

    lower_out[0] = 0;
    lower_out[1] = local1_start;
    lower_out[2] = 0;
    upper_out[0] = n0;
    upper_out[1] = local1_start + local1;
    upper_out[2] = (type == CUFFTMP_C2C) ? n2 : n2c;
    strides_out[0] = (long long)local1 * upper_out[2];
    strides_out[1] = upper_out[2];
    strides_out[2] = 1;
  } else {
    const int n0 = npts_global[0];
    const int n1 = npts_global[1];
    const int n1_out = (type == CUFFTMP_C2C) ? n1 : (n1 / 2 + 1);

    block_dist(n1_out, P, proc, &local1_start, &local1);
    in_elems = (size_t)local0 * n1;
    out_elems = (size_t)n0 * local1;

    lower_in[0] = local0_start;
    lower_in[1] = 0;
    lower_in[2] = 0;
    upper_in[0] = local0_start + local0;
    upper_in[1] = n1;
    upper_in[2] = 1;
    strides_in[0] = n1;
    strides_in[1] = 1;
    strides_in[2] = 1;

    lower_out[0] = 0;
    lower_out[1] = local1_start;
    lower_out[2] = 0;
    upper_out[0] = n0;
    upper_out[1] = local1_start + local1;
    upper_out[2] = 1;
    strides_out[0] = local1;
    strides_out[1] = 1;
    strides_out[2] = 1;
  }

  *local0_out = local0;
  *local1_out = local1;

  /* "consumed" is the buffer this transform reads, "produced" is what it
   * writes.  For R2C the input is real, for C2R the input is complex. */
  if (type == CUFFTMP_C2C) {
    if (direction) {
      *consumed_elems = in_elems;
      *produced_elems = out_elems;
      *consumed_is_output = false;
    } else {
      *consumed_elems = out_elems;
      *produced_elems = in_elems;
      *consumed_is_output = true;
    }
  } else if (type == CUFFTMP_R2C) {
    *consumed_elems = in_elems;  /* real */
    *produced_elems = out_elems; /* complex */
    *consumed_is_output = false;
  } else {
    *consumed_elems = out_elems; /* complex */
    *produced_elems = in_elems;  /* real */
    *consumed_is_output = true;
  }
}

/*******************************************************************************
 * \brief Find or create the cuFFTMp plan for the given transform.
 ******************************************************************************/
static cufftmp_plan_entry *
cufftmp_get_plan(const int rank, const int npts_global[3],
                 const cufftmp_type type, const bool direction,
                 const cp_mpi_comm_t comm) {
  const int P = cp_mpi_comm_size(comm);
  const int proc = cp_mpi_comm_rank(comm);
  const int comm_f = cp_mpi_comm_c2f(comm);

  for (int i = 0; i < CUFFTMP_CACHE_SIZE; i++) {
    cufftmp_plan_entry *e = &plan_cache[i];
    if (!e->used)
      continue;
    if (e->rank != rank || e->type != type || e->comm_f != comm_f)
      continue;
    if (type == CUFFTMP_C2C && e->direction != direction)
      continue;
    bool same = true;
    for (int d = 0; d < rank; d++) {
      if (e->npts_global[d] != npts_global[d])
        same = false;
    }
    if (same)
      return e;
  }

  cufftmp_plan_entry *e = &plan_cache[cache_oldest];
  cache_oldest = (cache_oldest + 1) % CUFFTMP_CACHE_SIZE;
  if (e->used)
    cufftDestroy(e->plan);
  memset(e, 0, sizeof(*e));
  e->used = true;
  e->rank = rank;
  for (int d = 0; d < rank; d++)
    e->npts_global[d] = npts_global[d];
  e->type = type;
  e->direction = direction;
  e->comm_f = comm_f;
  e->comm = comm;

  cufftmp_compute_boxes(proc, P, npts_global, type, direction, e->lower_in,
                        e->upper_in, e->strides_in, e->lower_out,
                        e->upper_out, e->strides_out, &e->rank, &e->l0,
                        &e->l1, &e->consumed_elems, &e->produced_elems,
                        &e->consumed_is_output);
  e->n0 = npts_global[0];
  e->n1 = npts_global[1];
  e->is_r2c = (type != CUFFTMP_C2C);
  if (e->rank == 3) {
    const int n2 = npts_global[2];
    const int n2c = n2 / 2 + 1;
    e->seg_in = n2;
    e->seg_out = e->is_r2c ? n2c : n2;
    e->host_last_in = e->is_r2c ? 2 * n2c : n2;
    e->host_last_out = e->seg_out;
  } else {
    e->seg_in = 1;
    e->seg_out = 1;
    e->host_last_in = e->is_r2c ? 2 * (npts_global[1] / 2 + 1) : npts_global[1];
    e->host_last_out = 1;
  }

  switch (type) {
  case CUFFTMP_C2C:
    e->fft_type = CUFFT_Z2Z;
    break;
  case CUFFTMP_R2C:
    e->fft_type = CUFFT_D2Z;
    break;
  case CUFFTMP_C2R:
    e->fft_type = CUFFT_Z2D;
    break;
  }

  if (offload_get_chosen_device() >= 0)
    offload_activate_chosen_device();
  cufftmp_check(cufftCreate(&e->plan), "cufftCreate");
  cufftmp_check(cufftMpAttachComm(e->plan, CUFFT_COMM_MPI, (void *)&e->comm),
                "cufftMpAttachComm");
  cufftmp_check(cufftXtSetDistribution(
                    e->plan, e->rank, e->lower_in, e->upper_in, e->lower_out,
                    e->upper_out, e->strides_in, e->strides_out),
                "cufftXtSetDistribution");
  size_t workspace = 0;
  if (e->rank == 3) {
    cufftmp_check(cufftMakePlan3d(e->plan, npts_global[0], npts_global[1],
                                  npts_global[2], e->fft_type, &workspace),
                  "cufftMakePlan3d");
  } else {
    cufftmp_check(cufftMakePlan2d(e->plan, npts_global[0], npts_global[1],
                                  e->fft_type, &workspace),
                  "cufftMakePlan2d");
  }
  return e;
}

/*******************************************************************************
 * \brief Copy one batch between the host (FFTW transposed layout) and the
 *        contiguous row-major layout expected by cuFFTMp (dim 0 distributed).
 ******************************************************************************/
static void pack_dim0(char *restrict ro, const char *restrict host,
                      const int l0, const int n1, const int seg,
                      const int host_last, const int H, const bool dim3,
                      const size_t esize, const int b, const bool to_ro) {
  for (int i0 = 0; i0 < l0; i0++) {
    for (int i1 = 0; i1 < n1; i1++) {
      if (dim3) {
        for (int k = 0; k < seg; k++) {
          const size_t roff = ((size_t)i0 * n1 + i1) * seg + k;
          const size_t hoff = ((size_t)i0 * n1 + i1) * host_last + k;
          if (to_ro)
            memcpy(ro + roff * esize, host + hoff * esize, esize);
          else
            memcpy((char *)host + hoff * esize, ro + roff * esize, esize);
        }
      } else {
        const size_t roff = (size_t)i0 * n1 + i1;
        const size_t hoff = (size_t)b + ((size_t)i0 * host_last + i1) * H;
        if (to_ro)
          memcpy(ro + roff * esize, host + hoff * esize, esize);
        else
          memcpy((char *)host + hoff * esize, ro + roff * esize, esize);
      }
    }
  }
}

/*******************************************************************************
 * \brief Copy one batch between the host (FFTW transposed layout) and the
 *        contiguous row-major layout expected by cuFFTMp (dim 1 distributed).
 ******************************************************************************/
static void pack_dim1(char *restrict ro, const char *restrict host,
                      const int n0, const int l1, const int seg,
                      const int host_last, const int H, const bool dim3,
                      const size_t esize, const int b, const bool to_ro) {
  for (int i0 = 0; i0 < n0; i0++) {
    for (int i1 = 0; i1 < l1; i1++) {
      if (dim3) {
        for (int k = 0; k < seg; k++) {
          const size_t roff = ((size_t)i0 * l1 + i1) * seg + k;
          const size_t hoff = ((size_t)i1 * n0 + i0) * host_last + k;
          if (to_ro)
            memcpy(ro + roff * esize, host + hoff * esize, esize);
          else
            memcpy((char *)host + hoff * esize, ro + roff * esize, esize);
        }
      } else {
        const size_t roff = (size_t)i0 * l1 + i1;
        const size_t hoff = (size_t)b + ((size_t)i1 * n0 + i0) * H;
        if (to_ro)
          memcpy(ro + roff * esize, host + hoff * esize, esize);
        else
          memcpy((char *)host + hoff * esize, ro + roff * esize, esize);
      }
    }
  }
}

/*******************************************************************************
 * \brief Execute a (possibly batched) distributed FFT through cuFFTMp.
 ******************************************************************************/
static void cufftmp_execute(const int rank, const int npts_global[3],
                            const cufftmp_type type, const bool direction,
                            const int number_of_ffts,
                            const cp_mpi_comm_t comm, void *grid_in,
                            void *grid_out) {
  if (npts_global[0] == 0 || npts_global[1] == 0 ||
      (rank == 3 && npts_global[2] == 0) || number_of_ffts == 0)
    return;

  cufftmp_plan_entry *e =
      cufftmp_get_plan(rank, npts_global, type, direction, comm);

  const cufftXtSubFormat consumed_format =
      e->consumed_is_output ? CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT
                            : CUFFT_XT_FORMAT_DISTRIBUTED_INPUT;
  const int fft_direction = direction ? CUFFT_FORWARD : CUFFT_INVERSE;

  const size_t consumed_esize =
      (type == CUFFTMP_R2C) ? sizeof(double) : sizeof(double complex);
  const size_t produced_esize =
      (type == CUFFTMP_C2R) ? sizeof(double) : sizeof(double complex);

  const bool consumed_dim1 = e->consumed_is_output;
  const bool produced_dim1 = !e->consumed_is_output;
  const bool dim3 = (e->rank == 3);

  const size_t max_elems = (e->consumed_elems > e->produced_elems)
                               ? e->consumed_elems
                               : e->produced_elems;
  char *ro = (char *)malloc(max_elems * sizeof(double complex));
  cp_assert(ro != NULL);

  for (int b = 0; b < number_of_ffts; b++) {
    if (consumed_dim1) {
      pack_dim1(ro, (const char *)grid_in, e->n0, e->l1, e->seg_out,
                e->host_last_out, number_of_ffts, dim3, consumed_esize, b,
                true);
    } else {
      pack_dim0(ro, (const char *)grid_in, e->l0, e->n1, e->seg_in,
                e->host_last_in, number_of_ffts, dim3, consumed_esize, b,
                true);
    }

    cudaLibXtDesc *desc = NULL;
    cufftmp_check(cufftXtMalloc(e->plan, &desc, consumed_format),
                  "cufftXtMalloc");
    cufftmp_check(cufftXtMemcpy(e->plan, (void *)desc, ro,
                                CUFFT_COPY_HOST_TO_DEVICE),
                  "cufftXtMemcpy (H2D)");
    cufftmp_check(cufftXtExecDescriptor(e->plan, desc, desc, fft_direction),
                  "cufftXtExecDescriptor");
    cp_assert(cudaDeviceSynchronize() == cudaSuccess);
    cufftmp_check(cufftXtMemcpy(e->plan, ro, (void *)desc,
                                CUFFT_COPY_DEVICE_TO_HOST),
                  "cufftXtMemcpy (D2H)");
    cufftmp_check(cufftXtFree(desc), "cufftXtFree");

    if (produced_dim1) {
      pack_dim1(ro, (char *)grid_out, e->n0, e->l1, e->seg_out,
                e->host_last_out, number_of_ffts, dim3, produced_esize, b,
                false);
    } else {
      pack_dim0(ro, (char *)grid_out, e->l0, e->n1, e->seg_in,
                e->host_last_in, number_of_ffts, dim3, produced_esize, b,
                false);
    }
  }
  free(ro);
}

/*******************************************************************************
 * \brief Common implementation of the distributed sizes routines.
 ******************************************************************************/
static int cufftmp_sizes_impl(const int rank, const int npts_global[3],
                              const int number_of_ffts, const bool r2c,
                              const cp_mpi_comm_t comm, int *local_n0,
                              int *local_n0_start, int *local_n1,
                              int *local_n1_start) {
  const int P = cp_mpi_comm_size(comm);
  const int proc = cp_mpi_comm_rank(comm);

  int l0_start, l0, l1_start, l1;
  block_dist(npts_global[0], P, proc, &l0_start, &l0);
  if (rank == 3) {
    block_dist(npts_global[1], P, proc, &l1_start, &l1);
  } else if (r2c) {
    block_dist(npts_global[1] / 2 + 1, P, proc, &l1_start, &l1);
  } else {
    block_dist(npts_global[1], P, proc, &l1_start, &l1);
  }

  *local_n0 = l0;
  *local_n0_start = l0_start;
  *local_n1 = l1;
  *local_n1_start = l1_start;

  /* The real input of an R2C transform is padded like an FFTW in-place
   * transform (leading dimension 2*(n/2+1)).  The returned size is in units of
   * complex elements. */
  size_t in_elems, out_elems;
  if (rank == 3) {
    const int n2c = npts_global[2] / 2 + 1;
    if (r2c) {
      in_elems = (size_t)l0 * npts_global[1] * n2c;
      out_elems = (size_t)npts_global[0] * l1 * n2c;
    } else {
      in_elems = (size_t)l0 * npts_global[1] * npts_global[2];
      out_elems = (size_t)npts_global[0] * l1 * npts_global[2];
    }
  } else {
    if (r2c) {
      in_elems = (size_t)l0 * (npts_global[1] / 2 + 1);
      out_elems = (size_t)npts_global[0] * l1;
    } else {
      in_elems = (size_t)l0 * npts_global[1];
      out_elems = (size_t)npts_global[0] * l1;
    }
  }
  const size_t per_fft = (in_elems > out_elems) ? in_elems : out_elems;
  return (int)((size_t)number_of_ffts * per_fft);
}

/* ------------------------------------------------------------------------- */
/* Public interface                                                          */
/* ------------------------------------------------------------------------- */

int fft_cufftmp_mpi_init_lib(void) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1)
    return 1;
  /* cuFFTMp/NVSHMEM require an explicitly selected CUDA device. */
  if (offload_get_chosen_device() >= 0)
    offload_activate_chosen_device();
  memset(plan_cache, 0, sizeof(plan_cache));
  cache_oldest = 0;
  return 0;
}

void fft_cufftmp_mpi_finalize_lib(void) {
  for (int i = 0; i < CUFFTMP_CACHE_SIZE; i++) {
    if (plan_cache[i].used) {
      cufftDestroy(plan_cache[i].plan);
      plan_cache[i].used = false;
    }
  }
  cache_oldest = 0;
}

void fft_cufftmp_mpi_register_2d_distributed(
    const bool direction, const int npts_global[2], const int number_of_ffts,
    const cp_mpi_comm_t comm, double complex *restrict grid_in,
    double complex *restrict grid_out) {
  (void)number_of_ffts;
  (void)grid_in;
  (void)grid_out;
  cufftmp_get_plan(2, (const int[3]){npts_global[0], npts_global[1], 0},
                   CUFFTMP_C2C, direction, comm);
}

void fft_cufftmp_mpi_register_2d_r2c_distributed(
    const int npts_global[2], const int number_of_ffts,
    const cp_mpi_comm_t comm, double *restrict grid_in,
    double complex *restrict grid_out) {
  (void)number_of_ffts;
  (void)grid_in;
  (void)grid_out;
  cufftmp_get_plan(2, (const int[3]){npts_global[0], npts_global[1], 0},
                   CUFFTMP_R2C, true, comm);
}

void fft_cufftmp_mpi_register_2d_c2r_distributed(
    const int npts_global[2], const int number_of_ffts,
    const cp_mpi_comm_t comm, double complex *restrict grid_in,
    double *restrict grid_out) {
  (void)number_of_ffts;
  (void)grid_in;
  (void)grid_out;
  cufftmp_get_plan(2, (const int[3]){npts_global[0], npts_global[1], 0},
                   CUFFTMP_C2R, false, comm);
}

void fft_cufftmp_mpi_register_3d_distributed(
    const bool direction, const int npts_global[3], const cp_mpi_comm_t comm,
    double complex *restrict grid_in, double complex *restrict grid_out) {
  (void)grid_in;
  (void)grid_out;
  cufftmp_get_plan(3, npts_global, CUFFTMP_C2C, direction, comm);
}

void fft_cufftmp_mpi_register_3d_r2c_distributed(
    const int npts_global[3], const cp_mpi_comm_t comm,
    double *restrict grid_in, double complex *restrict grid_out) {
  (void)grid_in;
  (void)grid_out;
  cufftmp_get_plan(3, npts_global, CUFFTMP_R2C, true, comm);
}

void fft_cufftmp_mpi_register_3d_c2r_distributed(
    const int npts_global[3], const cp_mpi_comm_t comm,
    double complex *restrict grid_in, double *restrict grid_out) {
  (void)grid_in;
  (void)grid_out;
  cufftmp_get_plan(3, npts_global, CUFFTMP_C2R, false, comm);
}

void fft_cufftmp_mpi_2d_distributed(const bool direction,
                                const int npts_global[2],
                                const int number_of_ffts,
                                const cp_mpi_comm_t comm,
                                double complex *restrict grid_in,
                                double complex *restrict grid_out) {
  cufftmp_execute(2, (const int[3]){npts_global[0], npts_global[1], 0},
                  CUFFTMP_C2C, direction, number_of_ffts, comm, grid_in,
                  grid_out);
}

void fft_cufftmp_mpi_2d_r2c_distributed(const int npts_global[2],
                                    const int number_of_ffts,
                                    const cp_mpi_comm_t comm,
                                    double *restrict grid_in,
                                    double complex *restrict grid_out) {
  cufftmp_execute(2, (const int[3]){npts_global[0], npts_global[1], 0},
                  CUFFTMP_R2C, true, number_of_ffts, comm, grid_in, grid_out);
}

void fft_cufftmp_mpi_2d_c2r_distributed(const int npts_global[2],
                                    const int number_of_ffts,
                                    const cp_mpi_comm_t comm,
                                    double complex *restrict grid_in,
                                    double *restrict grid_out) {
  cufftmp_execute(2, (const int[3]){npts_global[0], npts_global[1], 0},
                  CUFFTMP_C2R, false, number_of_ffts, comm, grid_in, grid_out);
}

void fft_cufftmp_mpi_3d_distributed(const bool direction,
                                const int npts_global[3],
                                const cp_mpi_comm_t comm,
                                double complex *restrict grid_in,
                                double complex *restrict grid_out) {
  cufftmp_execute(3, npts_global, CUFFTMP_C2C, direction, 1, comm, grid_in,
                  grid_out);
}

void fft_cufftmp_mpi_3d_r2c_distributed(const int npts_global[3],
                                    const cp_mpi_comm_t comm,
                                    double *restrict grid_in,
                                    double complex *restrict grid_out) {
  cufftmp_execute(3, npts_global, CUFFTMP_R2C, true, 1, comm, grid_in,
                  grid_out);
}

void fft_cufftmp_mpi_3d_c2r_distributed(const int npts_global[3],
                                    const cp_mpi_comm_t comm,
                                    double complex *restrict grid_in,
                                    double *restrict grid_out) {
  cufftmp_execute(3, npts_global, CUFFTMP_C2R, false, 1, comm, grid_in,
                  grid_out);
}

int fft_cufftmp_mpi_2d_distributed_sizes(const int npts_global[2],
                                     const int number_of_ffts,
                                     const cp_mpi_comm_t comm, int *local_n0,
                                     int *local_n0_start, int *local_n1,
                                     int *local_n1_start) {
  return cufftmp_sizes_impl(2, (const int[3]){npts_global[0], npts_global[1], 0},
                            number_of_ffts, false, comm, local_n0,
                            local_n0_start, local_n1, local_n1_start);
}

int fft_cufftmp_mpi_2d_distributed_sizes_r2c(const int npts_global[2],
                                         const int number_of_ffts,
                                         const cp_mpi_comm_t comm,
                                         int *local_n0, int *local_n0_start,
                                         int *local_n1, int *local_n1_start) {
  return cufftmp_sizes_impl(2, (const int[3]){npts_global[0], npts_global[1], 0},
                            number_of_ffts, true, comm, local_n0,
                            local_n0_start, local_n1, local_n1_start);
}

int fft_cufftmp_mpi_3d_distributed_sizes(const int npts_global[3],
                                     const cp_mpi_comm_t comm, int *local_n0,
                                     int *local_n0_start, int *local_n1,
                                     int *local_n1_start) {
  return cufftmp_sizes_impl(3, npts_global, 1, false, comm, local_n0,
                            local_n0_start, local_n1, local_n1_start);
}

int fft_cufftmp_mpi_3d_distributed_sizes_r2c(const int npts_global[3],
                                         const cp_mpi_comm_t comm,
                                         int *local_n0, int *local_n0_start,
                                         int *local_n1, int *local_n1_start) {
  return cufftmp_sizes_impl(3, npts_global, 1, true, comm, local_n0,
                            local_n0_start, local_n1, local_n1_start);
}

#else /* !(__CUFFTMP && __parallel) */

/* Stubs so that fft_lib.c can reference the cuFFTMp dispatch targets in builds
 * that do not enable the cuFFTMp backend.  They are never called in that
 * case. */

int fft_cufftmp_mpi_init_lib(void) { return 1; }
void fft_cufftmp_mpi_finalize_lib(void) {}

#define CUFFTMP_UNUSED(x) (void)(x)

void fft_cufftmp_mpi_register_2d_distributed(const bool direction,
                                         const int npts_global[2],
                                         const int number_of_ffts,
                                         const cp_mpi_comm_t comm,
                                         double complex *restrict grid_in,
                                         double complex *restrict grid_out) {
  CUFFTMP_UNUSED(direction);
  CUFFTMP_UNUSED(npts_global);
  CUFFTMP_UNUSED(number_of_ffts);
  CUFFTMP_UNUSED(comm);
  CUFFTMP_UNUSED(grid_in);
  CUFFTMP_UNUSED(grid_out);
  cp_assert(0 && "cuFFTMp backend not compiled in.");
}

void fft_cufftmp_mpi_register_2d_r2c_distributed(
    const int npts_global[2], const int number_of_ffts,
    const cp_mpi_comm_t comm, double *restrict grid_in,
    double complex *restrict grid_out) {
  CUFFTMP_UNUSED(npts_global);
  CUFFTMP_UNUSED(number_of_ffts);
  CUFFTMP_UNUSED(comm);
  CUFFTMP_UNUSED(grid_in);
  CUFFTMP_UNUSED(grid_out);
  cp_assert(0 && "cuFFTMp backend not compiled in.");
}

void fft_cufftmp_mpi_register_2d_c2r_distributed(
    const int npts_global[2], const int number_of_ffts,
    const cp_mpi_comm_t comm, double complex *restrict grid_in,
    double *restrict grid_out) {
  CUFFTMP_UNUSED(npts_global);
  CUFFTMP_UNUSED(number_of_ffts);
  CUFFTMP_UNUSED(comm);
  CUFFTMP_UNUSED(grid_in);
  CUFFTMP_UNUSED(grid_out);
  cp_assert(0 && "cuFFTMp backend not compiled in.");
}

void fft_cufftmp_mpi_register_3d_distributed(
    const bool direction, const int npts_global[3], const cp_mpi_comm_t comm,
    double complex *restrict grid_in, double complex *restrict grid_out) {
  CUFFTMP_UNUSED(direction);
  CUFFTMP_UNUSED(npts_global);
  CUFFTMP_UNUSED(comm);
  CUFFTMP_UNUSED(grid_in);
  CUFFTMP_UNUSED(grid_out);
  cp_assert(0 && "cuFFTMp backend not compiled in.");
}

void fft_cufftmp_mpi_register_3d_r2c_distributed(
    const int npts_global[3], const cp_mpi_comm_t comm,
    double *restrict grid_in, double complex *restrict grid_out) {
  CUFFTMP_UNUSED(npts_global);
  CUFFTMP_UNUSED(comm);
  CUFFTMP_UNUSED(grid_in);
  CUFFTMP_UNUSED(grid_out);
  cp_assert(0 && "cuFFTMp backend not compiled in.");
}

void fft_cufftmp_mpi_register_3d_c2r_distributed(
    const int npts_global[3], const cp_mpi_comm_t comm,
    double complex *restrict grid_in, double *restrict grid_out) {
  CUFFTMP_UNUSED(npts_global);
  CUFFTMP_UNUSED(comm);
  CUFFTMP_UNUSED(grid_in);
  CUFFTMP_UNUSED(grid_out);
  cp_assert(0 && "cuFFTMp backend not compiled in.");
}

void fft_cufftmp_mpi_2d_distributed(const bool direction,
                                const int npts_global[2],
                                const int number_of_ffts,
                                const cp_mpi_comm_t comm,
                                double complex *restrict grid_in,
                                double complex *restrict grid_out) {
  CUFFTMP_UNUSED(direction);
  CUFFTMP_UNUSED(npts_global);
  CUFFTMP_UNUSED(number_of_ffts);
  CUFFTMP_UNUSED(comm);
  CUFFTMP_UNUSED(grid_in);
  CUFFTMP_UNUSED(grid_out);
  cp_assert(0 && "cuFFTMp backend not compiled in.");
}

void fft_cufftmp_mpi_2d_r2c_distributed(const int npts_global[2],
                                    const int number_of_ffts,
                                    const cp_mpi_comm_t comm,
                                    double *restrict grid_in,
                                    double complex *restrict grid_out) {
  CUFFTMP_UNUSED(npts_global);
  CUFFTMP_UNUSED(number_of_ffts);
  CUFFTMP_UNUSED(comm);
  CUFFTMP_UNUSED(grid_in);
  CUFFTMP_UNUSED(grid_out);
  cp_assert(0 && "cuFFTMp backend not compiled in.");
}

void fft_cufftmp_mpi_2d_c2r_distributed(const int npts_global[2],
                                    const int number_of_ffts,
                                    const cp_mpi_comm_t comm,
                                    double complex *restrict grid_in,
                                    double *restrict grid_out) {
  CUFFTMP_UNUSED(npts_global);
  CUFFTMP_UNUSED(number_of_ffts);
  CUFFTMP_UNUSED(comm);
  CUFFTMP_UNUSED(grid_in);
  CUFFTMP_UNUSED(grid_out);
  cp_assert(0 && "cuFFTMp backend not compiled in.");
}

void fft_cufftmp_mpi_3d_distributed(const bool direction,
                                const int npts_global[3],
                                const cp_mpi_comm_t comm,
                                double complex *restrict grid_in,
                                double complex *restrict grid_out) {
  CUFFTMP_UNUSED(direction);
  CUFFTMP_UNUSED(npts_global);
  CUFFTMP_UNUSED(comm);
  CUFFTMP_UNUSED(grid_in);
  CUFFTMP_UNUSED(grid_out);
  cp_assert(0 && "cuFFTMp backend not compiled in.");
}

void fft_cufftmp_mpi_3d_r2c_distributed(const int npts_global[3],
                                    const cp_mpi_comm_t comm,
                                    double *restrict grid_in,
                                    double complex *restrict grid_out) {
  CUFFTMP_UNUSED(npts_global);
  CUFFTMP_UNUSED(comm);
  CUFFTMP_UNUSED(grid_in);
  CUFFTMP_UNUSED(grid_out);
  cp_assert(0 && "cuFFTMp backend not compiled in.");
}

void fft_cufftmp_mpi_3d_c2r_distributed(const int npts_global[3],
                                    const cp_mpi_comm_t comm,
                                    double complex *restrict grid_in,
                                    double *restrict grid_out) {
  CUFFTMP_UNUSED(npts_global);
  CUFFTMP_UNUSED(comm);
  CUFFTMP_UNUSED(grid_in);
  CUFFTMP_UNUSED(grid_out);
  cp_assert(0 && "cuFFTMp backend not compiled in.");
}

int fft_cufftmp_mpi_2d_distributed_sizes(const int npts_global[2],
                                     const int number_of_ffts,
                                     const cp_mpi_comm_t comm, int *local_n0,
                                     int *local_n0_start, int *local_n1,
                                     int *local_n1_start) {
  CUFFTMP_UNUSED(npts_global);
  CUFFTMP_UNUSED(number_of_ffts);
  CUFFTMP_UNUSED(comm);
  CUFFTMP_UNUSED(local_n0);
  CUFFTMP_UNUSED(local_n0_start);
  CUFFTMP_UNUSED(local_n1);
  CUFFTMP_UNUSED(local_n1_start);
  cp_assert(0 && "cuFFTMp backend not compiled in.");
  return -1;
}

int fft_cufftmp_mpi_2d_distributed_sizes_r2c(const int npts_global[2],
                                         const int number_of_ffts,
                                         const cp_mpi_comm_t comm,
                                         int *local_n0, int *local_n0_start,
                                         int *local_n1, int *local_n1_start) {
  return fft_cufftmp_mpi_2d_distributed_sizes(npts_global, number_of_ffts, comm,
                                          local_n0, local_n0_start, local_n1,
                                          local_n1_start);
}

int fft_cufftmp_mpi_3d_distributed_sizes(const int npts_global[3],
                                     const cp_mpi_comm_t comm, int *local_n0,
                                     int *local_n0_start, int *local_n1,
                                     int *local_n1_start) {
  CUFFTMP_UNUSED(npts_global);
  CUFFTMP_UNUSED(comm);
  CUFFTMP_UNUSED(local_n0);
  CUFFTMP_UNUSED(local_n0_start);
  CUFFTMP_UNUSED(local_n1);
  CUFFTMP_UNUSED(local_n1_start);
  cp_assert(0 && "cuFFTMp backend not compiled in.");
  return -1;
}

int fft_cufftmp_mpi_3d_distributed_sizes_r2c(const int npts_global[3],
                                         const cp_mpi_comm_t comm,
                                         int *local_n0, int *local_n0_start,
                                         int *local_n1, int *local_n1_start) {
  return fft_cufftmp_mpi_3d_distributed_sizes(npts_global, comm, local_n0,
                                          local_n0_start, local_n1,
                                          local_n1_start);
}

#endif /* __CUFFTMP && __parallel */

// EOF
