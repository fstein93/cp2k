/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef GRID_FFT_LIB_H
#define GRID_FFT_LIB_H

#include "common/grid_mpi.h"
#include "grid_fft_lib_fftw.h"

#include <complex.h>
#include <stdbool.h>

typedef enum { GRID_FFT_LIB_REF, GRID_FFT_LIB_FFTW } grid_fft_lib;

#if defined(__FFTW3)
static const grid_fft_lib GRID_FFT_LIB_DEFAULT = GRID_FFT_LIB_FFTW;
#else
static const grid_fft_lib GRID_FFT_LIB_DEFAULT = GRID_FFT_LIB_REF;
#endif

/*******************************************************************************
 * \brief Initialize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_init_lib(const grid_fft_lib lib, const int fftw_planning_flag,
                  const bool use_fft_mpi);

/*******************************************************************************
 * \brief Finalize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_finalize_lib();

/*******************************************************************************
 * \brief Whether compound MPI implementations are available.
 * \author Frederick Stein
 ******************************************************************************/
bool fft_lib_use_mpi();

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_allocate_double(const int length, double **buffer);

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_allocate_complex(const int length, double complex **buffer);

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_free_double(double *buffer);

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_free_complex(double complex *buffer);

/*******************************************************************************
 * \brief 1D Forward FFT from transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_1d_fw_local(const int fft_size, const int number_of_ffts,
                     const bool transpose_rs, const bool transpose_gs,
                     double complex *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief 1D Forward FFT from transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_1d_fw_local_r2c(const int fft_size, const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs,
                         double *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief 1D Backward FFT to transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_1d_bw_local(const int fft_size, const int number_of_ffts,
                     const bool transpose_rs, const bool transpose_gs,
                     double complex *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief 1D Backward FFT to transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_1d_bw_local_c2r(const int fft_size, const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs,
                         double complex *grid_in, double *grid_out);

/*******************************************************************************
 * \brief Local transposition.
 * \author Frederick Stein
 ******************************************************************************/
void transpose_local(double complex *grid, double complex *grid_transposed,
                     const int number_of_columns_grid,
                     const int number_of_rows_grid);

/*******************************************************************************
 * \brief Naive implementation of 2D FFT (transposed format, no normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_fw_local(const int fft_size[2], const int number_of_ffts,
                     const bool transpose_rs, const bool transpose_gs,
                     double complex *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief Naive implementation of 2D FFT (transposed format, no normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_fw_local_r2c(const int fft_size[2], const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs,
                         double *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief Performs local 2D FFT (reverse to fw routine, no normalization).
 * \note fft_2d_bw_local(grid_gs, grid_rs, n1, n2, m) is the reverse to
 * fft_2d_rw_local(grid_rs, grid_gs, n1, n2, m) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_bw_local(const int fft_size[2], const int number_of_ffts,
                     const bool transpose_rs, const bool transpose_gs,
                     double complex *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief Performs local 2D FFT (reverse to fw routine, no normalization).
 * \note fft_2d_bw_local(grid_gs, grid_rs, n1, n2, m) is the reverse to
 * fft_2d_rw_local(grid_rs, grid_gs, n1, n2, m) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_bw_local_c2r(const int fft_size[2], const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs,
                         double complex *grid_in, double *grid_out);

/*******************************************************************************
 * \brief Performs local 3D FFT (no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_local(const int fft_size[3], double complex *grid_in,
                     double complex *grid_out);

/*******************************************************************************
 * \brief Performs local 3D FFT (reverse to fw routine, no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_local(const int fft_size[3], double complex *grid_in,
                     double complex *grid_out);

/*******************************************************************************
 * \brief Return buffer size and local sizes and start for distributed 2D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_2d_distributed_sizes(const int npts_global[2], const int number_of_ffts,
                             const grid_mpi_comm comm, int *local_n0,
                             int *local_n0_start, int *local_n1,
                             int *local_n1_start);

/*******************************************************************************
 * \brief Return buffer size and local sizes and start for distributed 3D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_3d_distributed_sizes(const int npts_global[3], const grid_mpi_comm comm,
                             int *local_n2, int *local_n2_start, int *local_n1,
                             int *local_n1_start);

/*******************************************************************************
 * \brief Performs a distributed 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_fw_distributed(const int npts_global[2], const int number_of_ffts,
                           const grid_mpi_comm comm, double complex *grid_in,
                           double complex *grid_out);

/*******************************************************************************
 * \brief Performs a distributed 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_bw_distributed(const int npts_global[2], const int number_of_ffts,
                           const grid_mpi_comm comm, double complex *grid_in,
                           double complex *grid_out);

/*******************************************************************************
 * \brief Performs a distributed 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_distributed(const int npts_global[3], const grid_mpi_comm comm,
                           double complex *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief Performs a distributed 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_distributed(const int npts_global[3], const grid_mpi_comm comm,
                           double complex *grid_in, double complex *grid_out);

#endif /* GRID_FFT_LIB_H */

// EOF
