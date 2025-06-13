/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef GRID_FFT_GRID_H
#define GRID_FFT_GRID_H

#include "grid_fft_grid_layout.h"

#include <complex.h>
#include <stdbool.h>

typedef struct {
  grid_fft_grid_layout *fft_grid_layout;
  double *data;
} grid_fft_real_rs_grid;

typedef struct {
  grid_fft_grid_layout *fft_grid_layout;
  double complex *data;
} grid_fft_complex_rs_grid;

typedef struct {
  grid_fft_grid_layout *fft_grid_layout;
  double complex *data;
} grid_fft_complex_gs_grid;

/*******************************************************************************
 * \brief Create a real-valued real-space grid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_create_real_rs_grid(grid_fft_real_rs_grid *grid,
                              grid_fft_grid_layout *grid_layout);

/*******************************************************************************
 * \brief Create a complex-valued real-space grid.
 * \note grid_layout has not had been setup with use_halfspace=true
 * \author Frederick Stein
 ******************************************************************************/
void grid_create_complex_rs_grid(grid_fft_complex_rs_grid *grid,
                                 grid_fft_grid_layout *grid_layout);

/*******************************************************************************
 * \brief Create a complex-valued reciprocal-space grid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_create_complex_gs_grid(grid_fft_complex_gs_grid *grid,
                                 grid_fft_grid_layout *grid_layout);

/*******************************************************************************
 * \brief Frees a real-valued real-space grid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_free_real_rs_grid(grid_fft_real_rs_grid *grid);

/*******************************************************************************
 * \brief Frees a complex-valued real-space grid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_free_complex_rs_grid(grid_fft_complex_rs_grid *grid);

/*******************************************************************************
 * \brief Frees a complex-valued reciprocal-space grid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_free_complex_gs_grid(grid_fft_complex_gs_grid *grid);

/*******************************************************************************
 * \brief Add one grid to another one in reciprocal space.
 * \author Frederick Stein
 ******************************************************************************/
void grid_add_to_fine_grid(const grid_fft_complex_gs_grid *coarse_grid,
                           const grid_fft_complex_gs_grid *fine_grid);

/*******************************************************************************
 * \brief Copy fine grid to coarse grid in reciprocal space
 * \author Frederick Stein
 ******************************************************************************/
void grid_copy_to_coarse_grid(const grid_fft_complex_gs_grid *fine_grid,
                              const grid_fft_complex_gs_grid *coarse_grid);

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a high-level FFT grid.
 * \param grid_rs complex-valued data in real space
 * \param grid_gs complex data in reciprocal space
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw(const grid_fft_complex_rs_grid *grid_rs,
               const grid_fft_complex_gs_grid *grid_gs);

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a high-level FFT grid.
 * \param grid_rs real-valued data in real space
 * \param grid_gs complex data in reciprocal space,
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_r2c(const grid_fft_real_rs_grid *grid_rs,
                   const grid_fft_complex_gs_grid *grid_gs);

/*******************************************************************************
 * \brief Performs a backward 3D-FFT using a high-level FFT grid.
 * \param fft_grid FFT grid object
 * \param grid_gs complex data in reciprocal space
 * \param grid_rs complex-valued data in real space
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw(const grid_fft_complex_gs_grid *grid_gs,
               const grid_fft_complex_rs_grid *grid_rs);

/*******************************************************************************
 * \brief Performs a backward 3D-FFT using a high-level FFT grid.
 * \param fft_grid FFT grid object
 * \param grid_gs complex data in reciprocal space, ordered according to
 *fft_grid->index_to_g \param grid_rs real-valued data in real space, ordered
 *according to fft_grid->proc2local_rs \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_c2r(const grid_fft_complex_gs_grid *grid_gs,
                   const grid_fft_real_rs_grid *grid_rs);

#endif

// EOF
