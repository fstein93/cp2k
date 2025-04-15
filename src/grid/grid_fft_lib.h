/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef GRID_FFT_LIB_H
#define GRID_FFT_LIB_H

#include "grid_fft_lib_fftw.h"

#include <complex.h>

typedef enum { GRID_FFT_LIB_REF, GRID_FFT_LIB_FFTW } grid_fft_lib;

typedef struct {
  int fft_size[3];
  int number_of_ffts;
  grid_fft_fftw_plan fftw_plan_fw;
  grid_fft_fftw_plan fftw_plan_bw;
} grid_fft_plan;

#if defined(__FFTW3)
static const grid_fft_lib GRID_FFT_LIB_DEFAULT = GRID_FFT_LIB_FFTW;
#else
static const grid_fft_lib GRID_FFT_LIB_DEFAULT = GRID_FFT_LIB_REF;
#endif

/*******************************************************************************
 * \brief Initialize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_init_lib(const grid_fft_lib lib);

/*******************************************************************************
 * \brief Finalize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_finalize_lib();

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
 * \brief Create a plan for a 1D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_create_1d_plan(double complex *grid_rs, double complex *grid_gs,
                        const int fft_size, const int number_of_ffts,
                        grid_fft_plan *plan);

/*******************************************************************************
 * \brief Create a plan for a 1D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_create_2d_plan(const int fft_size[2], const int number_of_ffts,
                        grid_fft_plan *plan);

/*******************************************************************************
 * \brief Create a plan for a 1D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_create_3d_plan(const int fft_size[3], grid_fft_plan *plan);

/*******************************************************************************
 * \brief Frees FFT plans.
 * \author Frederick Stein
 ******************************************************************************/
void fft_free_plan(grid_fft_plan *plan);

/*******************************************************************************
 * \brief 1D Forward FFT from transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_1d_fw_local(const grid_fft_plan *plan, double complex *grid_in,
                     double complex *grid_out);

/*******************************************************************************
 * \brief 1D Backward FFT to transposed format.
 * \author Frederick Stein
 ******************************************************************************/
void fft_1d_bw_local(const grid_fft_plan *plan, double complex *grid_in,
                     double complex *grid_out);

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
                     double complex *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief Performs local 2D FFT (reverse to fw routine, no normalization).
 * \note fft_2d_bw_local(grid_gs, grid_rs, n1, n2, m) is the reverse to
 * fft_2d_rw_local(grid_rs, grid_gs, n1, n2, m) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_bw_local(const int fft_size[2], const int number_of_ffts,
                     double complex *grid_in, double complex *grid_out);

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

#endif /* GRID_FFT_LIB_H */

// EOF
