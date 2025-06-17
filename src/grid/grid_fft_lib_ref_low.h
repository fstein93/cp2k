/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef GRID_FFT_LIB_REF_LOW_H
#define GRID_FFT_LIB_REF_LOW_H

#include <complex.h>
#include <stdbool.h>

/*******************************************************************************
 * \brief 1D Forward FFT from transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_fw_local_low(double complex *grid_in, double complex *grid_out,
                             const int fft_size, const int number_of_ffts,
                             const int stride_in, const int stride_out,
                             const int distance_in, const int distance_out);

/*******************************************************************************
 * \brief 1D Forward FFT from transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_fw_local_r2c_low(double *grid_in, double complex *grid_out,
                                 const int fft_size, const int number_of_ffts,
                                 const int stride_in, const int stride_out,
                                 const int distance_in, const int distance_out);

/*******************************************************************************
 * \brief 1D Backward FFT to transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_bw_local_low(double complex *grid_in, double complex *grid_out,
                             const int fft_size, const int number_of_ffts,
                             const int stride_in, const int stride_out,
                             const int distance_in, const int distance_out);

/*******************************************************************************
 * \brief 1D Backward FFT to transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_1d_bw_local_c2r_low(double complex *grid_in, double *grid_out,
                                 const int fft_size, const int number_of_ffts,
                                 const int stride_in, const int stride_out,
                                 const int distance_in, const int distance_out);

/*******************************************************************************
 * \brief 2D Forward FFT from transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_2d_fw_local_low(double complex *grid_in, double complex *grid_out,
                             const int fft_size[2], const int number_of_ffts,
                             const int stride_in, const int stride_out,
                             const int distance_in, const int distance_out);

/*******************************************************************************
 * \brief 2D Forward FFT from transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_2d_fw_local_r2c_low(double *grid_in, double complex *grid_out,
                                 const int fft_size[2],
                                 const int number_of_ffts, const int stride_in,
                                 const int stride_out, const int distance_in,
                                 const int distance_out);

/*******************************************************************************
 * \brief 2D Backward FFT to transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_2d_bw_local_low(double complex *grid_in, double complex *grid_out,
                             const int fft_size[2], const int number_of_ffts,
                             const int stride_in, const int stride_out,
                             const int distance_in, const int distance_out);

/*******************************************************************************
 * \brief 2D Backward FFT to transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_2d_bw_local_c2r_low(double complex *grid_in, double *grid_out,
                                 const int fft_size[2],
                                 const int number_of_ffts, const int stride_in,
                                 const int stride_out, const int distance_in,
                                 const int distance_out);

/*******************************************************************************
 * \brief 3D Forward FFT from transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_3d_fw_local_low(double complex *grid_in, double complex *grid_out,
                             const int fft_size[3]);

/*******************************************************************************
 * \brief 3D Forward FFT from transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_3d_fw_local_r2c_low(double *grid_in, double complex *grid_out,
                                 const int fft_size[3]);

/*******************************************************************************
 * \brief 3D Backward FFT to transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_3d_bw_local_low(double complex *grid_in, double complex *grid_out,
                             const int fft_size[3]);

/*******************************************************************************
 * \brief 3D Backward FFT to transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_ref_3d_bw_local_c2r_low(double complex *grid_in, double *grid_out,
                                 const int fft_size[3]);

#endif /* GRID_FFT_LIB_REF_LOW_H */

// EOF
