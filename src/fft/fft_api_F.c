/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "fft_api_F.h"
#include "fft_grid_layout.h"
#include "fft_lib.h"
#include "fft_timer.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

int fft_library_init_F(const int backend_F, const int fftw_plan,
                       const bool use_fft_mpi, const bool use_guru_interface,
                       const char *wisdom_file) {
  fft_lib backend;
  switch (backend_F) {
  case FFT_LIBRARY_BACKEND_DEFAULT:
    backend = FFT_LIB_DEFAULT;
    break;
  case FFT_LIBRARY_BACKEND_FFTW:
    backend = FFT_LIB_FFTW;
    break;
  case FFT_LIBRARY_BACKEND_GPU:
    backend = FFT_LIB_GPU;
    break;
  default:
    return 1;
  }
  fftw_plan_type plan_type;
  switch (fftw_plan) {
  case FFT_LIBRARY_FFTW_MODE_ESTIMATE:
    plan_type = FFT_ESTIMATE;
    break;
  case FFT_LIBRARY_FFTW_MODE_MEASURE:
    plan_type = FFT_MEASURE;
    break;
  case FFT_LIBRARY_FFTW_MODE_PATIENT:
    plan_type = FFT_PATIENT;
    break;
  case FFT_LIBRARY_FFTW_MODE_EXHAUSTIVE:
    plan_type = FFT_EXHAUSTIVE;
    break;
  default:
    return 1;
  }
  const int status = fft_init_lib(backend, plan_type, use_fft_mpi,
                                  use_guru_interface, wisdom_file);
  fft_init_timer(false);
  return status;
}

void fft_library_finalize_F(const char *wisdom_file) {
  fft_finalize_lib(wisdom_file);
}

void fft_print_timing_report_F(const double threshold) {
  fft_print_timing_report(threshold);
}

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_allocate_double_F(const int length, double **buffer) {
  fft_allocate_double(length, buffer);
}

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_allocate_complex_F(const int length, double complex **buffer) {
  fft_allocate_complex(length, buffer);
}

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_free_double_F(double *buffer) { fft_free_double(buffer); }

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_free_complex_F(double complex *buffer) { fft_free_complex(buffer); }

int fft_backend_in_use_F() { return fft_lib_backend_in_use(); }

bool fft_has_guru_interface_F() { return fft_lib_has_guru_interface(); }

void fft_register_1d_local_F(const bool dir, const int fft_size,
                             const int number_of_ffts, const bool transpose_in,
                             const bool transpose_out,
                             const int leading_dimension_in,
                             const int leading_dimension_out,
                             double complex *grid_in,
                             double complex *grid_out) {
  fft_register_1d_local(dir, fft_size, number_of_ffts, transpose_in,
                        transpose_out, leading_dimension_in,
                        leading_dimension_out, grid_in, grid_out);
}

void fft_register_2d_local_F(const bool dir, const int fft_size[2],
                             const int number_of_ffts, const bool transpose_in,
                             const bool transpose_out, double complex *grid_in,
                             double complex *grid_out) {
  fft_register_2d_local(dir, (const int[]){fft_size[1], fft_size[0]},
                        number_of_ffts, transpose_in, transpose_out, grid_in,
                        grid_out);
}

void fft_register_3d_local_F(const bool dir, const int fft_size[3],
                             double complex *grid_in,
                             double complex *grid_out) {
  fft_register_3d_local(dir,
                        (const int[]){fft_size[2], fft_size[1], fft_size[0]},
                        grid_in, grid_out);
}

void fft_register_1d_local_inplace_F(const bool dir, const int fft_size,
                                     const int number_of_ffts,
                                     const bool transpose,
                                     const int leading_dimension,
                                     double complex *grid) {
  fft_register_1d_local(dir, fft_size, number_of_ffts, transpose, transpose,
                        leading_dimension, leading_dimension, grid, grid);
}

void fft_register_2d_local_inplace_F(const bool dir, const int fft_size[2],
                                     const int number_of_ffts,
                                     const bool transpose,
                                     double complex *grid) {
  fft_register_2d_local(dir, (const int[]){fft_size[1], fft_size[0]},
                        number_of_ffts, transpose, transpose, grid, grid);
}

void fft_register_3d_local_inplace_F(const bool dir, const int fft_size[3],
                                     double complex *grid) {
  fft_register_3d_local(
      dir, (const int[]){fft_size[2], fft_size[1], fft_size[0]}, grid, grid);
}

void fft_1d_local_F(const bool dir, const int fft_size,
                    const int number_of_ffts, const bool transpose_in,
                    const bool transpose_out, const int leading_dimension_in,
                    const int leading_dimension_out, double complex *grid_in,
                    double complex *grid_out) {
  fft_1d_local(dir, fft_size, number_of_ffts, transpose_in, transpose_out,
               leading_dimension_in, leading_dimension_out, grid_in, grid_out);
}

void fft_2d_local_F(const bool dir, const int fft_size[2],
                    const int number_of_ffts, const bool transpose_in,
                    const bool transpose_out, double complex *grid_in,
                    double complex *grid_out) {
  fft_2d_local(dir, (const int[]){fft_size[1], fft_size[0]}, number_of_ffts,
               transpose_in, transpose_out, grid_in, grid_out);
}

void fft_3d_local_F(const bool dir, const int fft_size[3],
                    double complex *grid_in, double complex *grid_out) {
  fft_3d_local(dir, (const int[]){fft_size[2], fft_size[1], fft_size[0]},
               grid_in, grid_out);
}

void fft_1d_local_inplace_F(const bool dir, const int fft_size,
                            const int number_of_ffts, const bool transpose,
                            const int leading_dimension, double complex *grid) {
  fft_1d_local(dir, fft_size, number_of_ffts, transpose, transpose,
               leading_dimension, leading_dimension, grid, grid);
}

void fft_2d_local_inplace_F(const bool dir, const int fft_size[2],
                            const int number_of_ffts, const bool transpose,
                            double complex *grid) {
  fft_2d_local(dir, (const int[]){fft_size[1], fft_size[0]}, number_of_ffts,
               transpose, transpose, grid, grid);
}

void fft_3d_local_inplace_F(const bool dir, const int fft_size[3],
                            double complex *grid) {
  fft_3d_local(dir, (const int[]){fft_size[2], fft_size[1], fft_size[0]}, grid,
               grid);
}

/*******************************************************************************
 * \brief Create a FFT grid layout.
 * \author Frederick Stein
 ******************************************************************************/
void fft_create_grid_F(fft_grid_layout **fft_grid, const int comm_F,
                       const int npts_global[3], const double dh_inv[3][3],
                       const bool use_halfspace, const double cutoff,
                       const int *pgrid_guess,
                       const int *external_local_bounds) {
  int local_bounds[6];
  int *local_bounds_ptr = NULL;
  if (external_local_bounds != NULL) {
    local_bounds_ptr = local_bounds;
    local_bounds[0] = external_local_bounds[4];
    local_bounds[1] = external_local_bounds[5];
    local_bounds[2] = external_local_bounds[2];
    local_bounds[3] = external_local_bounds[3];
    local_bounds[4] = external_local_bounds[0];
    local_bounds[5] = external_local_bounds[1];
  }
  grid_create_fft_grid_layout(
      fft_grid, cp_mpi_comm_f2c(comm_F),
      (const int[3]){npts_global[2], npts_global[1], npts_global[0]},
      (const double[3][3]){{dh_inv[2][2], dh_inv[2][1], dh_inv[2][0]},
                           {dh_inv[1][2], dh_inv[1][1], dh_inv[1][0]},
                           {dh_inv[0][2], dh_inv[0][1], dh_inv[0][0]}},
      use_halfspace, cutoff, pgrid_guess, local_bounds_ptr);
}

/*******************************************************************************
 * \brief Create a FFT grid layout from a reference grid layout.
 * \author Frederick Stein
 ******************************************************************************/
void fft_create_grid_from_reference_F(
    fft_grid_layout **fft_grid, const int npts_global[3], const double cutoff,
    const int *external_local_bounds, const fft_grid_layout *fft_grid_ref) {
  assert(fft_grid_ref != NULL);
  int local_bounds[6];
  int *local_bounds_ptr = NULL;
  if (external_local_bounds != NULL) {
    local_bounds_ptr = local_bounds;
    local_bounds[0] = external_local_bounds[4];
    local_bounds[1] = external_local_bounds[5];
    local_bounds[2] = external_local_bounds[2];
    local_bounds[3] = external_local_bounds[3];
    local_bounds[4] = external_local_bounds[0];
    local_bounds[5] = external_local_bounds[1];
  }
  grid_create_fft_grid_layout_from_reference(
      fft_grid, (const int[3]){npts_global[2], npts_global[1], npts_global[0]},
      cutoff, local_bounds_ptr, fft_grid_ref);
}

/*******************************************************************************
 * \brief Get the global number of points in each direction.
 * \author Frederick Stein
 ******************************************************************************/
void fft_grid_get_npts_global_F(const fft_grid_layout *fft_grid,
                                int *npts_global) {
  assert(fft_grid != NULL);
  memcpy(npts_global, fft_grid->npts_global, 3 * sizeof(int));
}

/*******************************************************************************
 * \brief Get the cartesian communicator of the FFT grid.
 * \author Frederick Stein
 ******************************************************************************/
void fft_grid_get_comm_F(const fft_grid_layout *fft_grid, int *comm_handle) {
  assert(fft_grid != NULL);
  *comm_handle = cp_mpi_comm_c2f(fft_grid->comm);
}

/*******************************************************************************
 * \brief Get the process-to-local-bounds mapping in real space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_grid_get_proc2local_rs_F(const fft_grid_layout *fft_grid,
                                  int *proc2local_rs) {
  assert(fft_grid != NULL);
  memcpy(proc2local_rs, fft_grid->proc2local_rs,
         6 * cp_mpi_comm_size(fft_grid->comm) * sizeof(int));
}

/*******************************************************************************
 * \brief Get the process-to-local-bounds mapping in mixed space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_grid_get_proc2local_ms_F(const fft_grid_layout *fft_grid,
                                  int *proc2local_ms) {
  assert(fft_grid != NULL);
  memcpy(proc2local_ms, fft_grid->proc2local_ms,
         6 * cp_mpi_comm_size(fft_grid->comm) * sizeof(int));
}

/*******************************************************************************
 * \brief Get the process-to-local-bounds mapping in reciprocal space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_grid_get_proc2local_gs_F(const fft_grid_layout *fft_grid,
                                  int *proc2local_gs) {
  assert(fft_grid != NULL);
  memcpy(proc2local_gs, fft_grid->proc2local_gs,
         6 * cp_mpi_comm_size(fft_grid->comm) * sizeof(int));
}

/*******************************************************************************
 * \brief Free a FFT grid layout.
 * \author Frederick Stein
 ******************************************************************************/
void fft_free_grid_F(fft_grid_layout *fft_grid) {
  grid_free_fft_grid_layout(fft_grid);
}

// EOF
