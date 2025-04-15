/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef GRID_FFT_LIB_FFTW_H
#define GRID_FFT_LIB_FFTW_H

#include <complex.h>

#if defined(__FFTW3)
#include <fftw3.h>
#endif

/*******************************************************************************
 * \brief Initialize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_init_lib();

/*******************************************************************************
 * \brief Finalize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_finalize_lib();

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
                          double complex *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief 1D Backward FFT to transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_1d_bw_local(const int fft_size, const int number_of_ffts,
                          double complex *grid_in, double complex *grid_out);

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
                          double complex *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief Performs local 2D FFT (reverse to fw routine, no normalization).
 * \note fft_2d_bw_local(grid_gs, grid_rs, n1, n2, m) is the reverse to
 * fft_2d_rw_local(grid_rs, grid_gs, n1, n2, m) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_fftw_2d_bw_local(const int fft_size[2], const int number_of_ffts,
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

#endif /* GRID_FFT_LIB_FFTW_H */

// EOF
