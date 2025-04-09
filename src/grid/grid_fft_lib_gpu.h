/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef GRID_FFT_LIB_GPU_H
#define GRID_FFT_LIB_GPU_H

#include <complex.h>

#if defined(__OFFLOAD)
#include "../offload/offload_library.h"
#include "../offload/offload_fft.h"
typedef offload_fftHandle grid_fft_gpu_plan;
#else
typedef void *grid_fft_gpu_plan;
#endif

/*******************************************************************************
 * \brief Initialize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_gpu_init_lib();

/*******************************************************************************
 * \brief Finalize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_gpu_finalize_lib();

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_gpu_allocate_double(const int length, double **buffer);

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_gpu_allocate_complex(const int length, double complex **buffer);

/*******************************************************************************
 * \brief Free buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_gpu_free_double(double *buffer);

/*******************************************************************************
 * \brief Free buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_gpu_free_complex(double complex *buffer);

/*******************************************************************************
 * \brief 1D Forward FFT from transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_gpu_1d_fw_local(const int n, const int m,
                          double complex *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief 1D Backward FFT to transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_gpu_1d_bw_local(const int n, const int m,
                          double complex *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief Local transposition.
 * \author Frederick Stein
 ******************************************************************************/
void fft_gpu_transpose_local(double complex *grid,
                              double complex *grid_transposed,
                              const int number_of_columns_grid,
                              const int number_of_rows_grid);

/*******************************************************************************
 * \brief Naive implementation of 2D FFT (transposed format, no normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_gpu_2d_fw_local(const int n1, const int n2, const int m,
                          double complex *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief Performs local 2D FFT (reverse to fw routine, no normalization).
 * \note fft_2d_bw_local(grid_gs, grid_rs, n1, n2, m) is the reverse to
 * fft_2d_rw_local(grid_rs, grid_gs, n1, n2, m) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_gpu_2d_bw_local(const int n1, const int n2, const int m,
                          double complex *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief Performs local 3D FFT (no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_gpu_3d_fw_local(const int nx, const int ny, const int nz,
                          double complex *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief Performs local 3D FFT (reverse to fw routine, no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_gpu_3d_bw_local(const int nx, const int ny, const int nz,
                          double complex *grid_in, double complex *grid_out);

#endif /* GRID_FFT_LIB_GPU_H */

// EOF
