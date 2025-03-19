/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef GRID_FFT_GRID_H
#define GRID_FFT_GRID_H

#include "common/grid_mpi.h"

#include <complex.h>
#include <stdbool.h>

/*******************************************************************************
 * \brief Container to represent fft grids.
 * \author Frederick Stein
 ******************************************************************************/
typedef struct {
  // ID for comparison and referencing grids
  int grid_id;
  // ID of the reference grid
  int ref_grid_id;
  // Reference counter
  int ref_counter;
  // Global number of points
  int npts_global[3];
  // Grid spacing in reciprocal space
  double h_inv[3][3];
  // Number of local points in g-space (relevant with ray-distribution)
  int npts_gs_local;
  bool ray_distribution;
  int (*ray_to_yz)[2];
  int *yz_to_process;
  int *rays_per_process;
  // maps of index in g-space to g-space vectors
  int (*index_to_g)[3];
  int *local_index_to_ref_grid;
  // New communicator
  grid_mpi_comm comm;
  int proc_grid[2];
  int periodic[2];
  int proc_coords[2];
  // distributions for each FFT step (real space/mixed-space 1 (rs), mixed space
  // 1/mixed space 2 (ms), mixed-space 2/g-space (gs)) first index is for the
  // process, the second for the coordinate, the third for start (0) / end(1)
  // proc2local_rs is also used for the distribution of the data in realspace
  // (that's why it is called "rs") proc2local_gs is also used for the
  // distribution of the data in reciprocal (g)-space (that's why it is called
  // "gs") in blocked mode (usually the finest grid)
  int (*proc2local_rs)[3][2]; // Order: (x, y, z)
  int (*proc2local_ms)[3][2]; // Order: (z, x, y)
  int (*proc2local_gs)[3][2]; // Order: (y, z, x)
  // Buffers for FFTs
  double complex *buffer_1;
  double complex *buffer_2;
  // buffers for different purposes
} grid_fft_grid_layout;

typedef struct {
  grid_fft_grid_layout *fft_grid_layout;
  double *data;
} grid_fft_real_rs_grid;

typedef struct {
  grid_fft_grid_layout *fft_grid_layout;
  double complex *data;
} grid_fft_complex_gs_grid;

/*******************************************************************************
 * \brief Frees a FFT grid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_free_fft_grid_layout(grid_fft_grid_layout *fft_grid);

/*******************************************************************************
 * \brief Create a FFT grid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_create_fft_grid_layout(grid_fft_grid_layout **fft_grid,
                                 const grid_mpi_comm comm,
                                 const int npts_global[3],
                                 const double dh_inv[3][3]);

/*******************************************************************************
 * \brief Create a FFT grid using a reference grid to interact with this grid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_create_fft_grid_layout_from_reference(
    grid_fft_grid_layout **fft_grid, const int npts_global[3],
    const grid_fft_grid_layout *fft_grid_ref);

/*******************************************************************************
 * \brief Retains a grid layout.
 * \author Frederick Stein
 ******************************************************************************/
void grid_retain_fft_grid_layout(grid_fft_grid_layout *fft_grid);

/*******************************************************************************
 * \brief Create a real-valued real-space grid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_create_real_rs_grid(grid_fft_real_rs_grid *grid,
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
 * \brief Frees a complex-valued reciprocal-space grid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_free_complex_gs_grid(grid_fft_complex_gs_grid *grid);

/*******************************************************************************
 * \brief Convert between C indices (0...n-1) and shifted indices (-n/2...n/2).
 * \author Frederick Stein
 ******************************************************************************/
inline int convert_c_index_to_shifted_index(const int c_index, const int npts) {
  return (c_index > npts / 2 ? c_index - npts : c_index);
}

/*******************************************************************************
 * \brief Convert between shifted indices (-n/2...n/2) and C indices (0...n-1).
 * \author Frederick Stein
 ******************************************************************************/
inline int convert_shifted_index_to_c_index(const int shifted_index,
                                            const int npts) {
  return (shifted_index < 0 ? npts + shifted_index : shifted_index);
}

/*******************************************************************************
 * \brief Check whether a shifted index is on the grid.
 * \author Frederick Stein
 ******************************************************************************/
inline bool is_on_grid(const int shifted_index, const int npts) {
  return (shifted_index >= -(npts - 1) / 2 && shifted_index <= npts / 2);
}

#endif

// EOF
