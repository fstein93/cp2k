/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef GRID_FFT_LIB_FFTW_H
#define GRID_FFT_LIB_FFTW_H

#include "common/grid_mpi.h"

// We include this first to require FFTW to use C complex numbers
#include <complex.h>

#if defined(__FFTW3)
#include <fftw3.h>
#endif
#include <stdbool.h>

typedef enum {
  FFT_ESTIMATE,
  FFT_MEASURE,
  FFT_PATIENT,
  FFT_EXHAUSTIVE
} fftw_plan_type;

/*******************************************************************************
 * \brief Initialize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_init_lib(const fftw_plan_type fftw_planning_flag,
                       const bool use_fft_mpi);

/*******************************************************************************
 * \brief Finalize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_finalize_lib();

/*******************************************************************************
 * \brief Whether a compound MPI implementation of FFT is available.
 * \author Frederick Stein
 ******************************************************************************/
bool fft_fftw_lib_use_mpi();

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_allocate_double(const int length, double **buffer);

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_allocate_complex(const int length, double complex **buffer);

/*******************************************************************************
 * \brief Free buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_free_double(double *buffer);

/*******************************************************************************
 * \brief Free buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_free_complex(double complex *buffer);

/*******************************************************************************
 * \brief 1D Forward FFT from transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_1d_fw_local(const int fft_size, const int number_of_ffts,
                          const bool transpose_rs, const bool transpose_gs,
                          double complex *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief 1D Forward FFT from transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_1d_fw_local_r2c(const int fft_size, const int number_of_ffts,
                              const bool transpose_rs, const bool transpose_gs,
                              double *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief 1D Backward FFT to transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_1d_bw_local(const int fft_size, const int number_of_ffts,
                          const bool transpose_rs, const bool transpose_gs,
                          double complex *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief 1D Backward FFT to transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_1d_bw_local_c2r(const int fft_size, const int number_of_ffts,
                              const bool transpose_rs, const bool transpose_gs,
                              double complex *grid_in, double *grid_out);

/*******************************************************************************
 * \brief Local transposition.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_transpose_local(double complex *grid,
                              double complex *grid_transposed,
                              const int number_of_columns_grid,
                              const int number_of_rows_grid);

/*******************************************************************************
 * \brief Naive implementation of 2D FFT (transposed format, no normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_2d_fw_local(const int fft_size[2], const int number_of_ffts,
                          const bool transpose_rs, const bool transpose_gs,
                          double complex *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief Performs local 2D FFT (reverse to fw routine, no normalization).
 * \note fft_2d_bw_local(grid_gs, grid_rs, n1, n2, m) is the reverse to
 * fft_2d_rw_local(grid_rs, grid_gs, n1, n2, m) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_2d_bw_local(const int fft_size[2], const int number_of_ffts,
                          const bool transpose_rs, const bool transpose_gs,
                          double complex *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief Performs local 3D FFT (no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_3d_fw_local(const int fft_size[3], double complex *grid_in,
                          double complex *grid_out);

/*******************************************************************************
 * \brief Performs local 3D FFT (reverse to fw routine, no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_3d_bw_local(const int fft_size[3], double complex *grid_in,
                          double complex *grid_out);

/*******************************************************************************
 * \brief Return buffer size and local sizes and start for distributed 2D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_fftw_2d_distributed_sizes(const int npts_global[2],
                                  const int number_of_ffts,
                                  const grid_mpi_comm comm, int *local_n0,
                                  int *local_n0_start, int *local_n1,
                                  int *local_n1_start);

/*******************************************************************************
 * \brief Return buffer size and local sizes and start for distributed 3D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_fftw_3d_distributed_sizes(const int npts_global[3],
                                  const grid_mpi_comm comm, int *local_n2,
                                  int *local_n2_start, int *local_n1,
                                  int *local_n1_start);

/*******************************************************************************
 * \brief Performs a distributed 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_2d_fw_distributed(const int npts_global[2],
                                const int number_of_ffts,
                                const grid_mpi_comm comm,
                                double complex *grid_in,
                                double complex *grid_out);

/*******************************************************************************
 * \brief Performs a distributed 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_2d_bw_distributed(const int npts_global[2],
                                const int number_of_ffts,
                                const grid_mpi_comm comm,
                                double complex *grid_in,
                                double complex *grid_out);

/*******************************************************************************
 * \brief Performs a distributed 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_3d_fw_distributed(const int npts_global[3],
                                const grid_mpi_comm comm,
                                double complex *grid_in,
                                double complex *grid_out);

/*******************************************************************************
 * \brief Performs a distributed 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_3d_bw_distributed(const int npts_global[3],
                                const grid_mpi_comm comm,
                                double complex *grid_in,
                                double complex *grid_out);

#endif /* GRID_FFT_LIB_FFTW_H */

// EOF
