/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "grid_fft_grid.h"
#include "common/grid_common.h"
#include "grid_fft_grid_layout.h"
#include "grid_fft_lib.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*******************************************************************************
 * \brief Add one grid to another one in reciprocal space.
 * \author Frederick Stein
 ******************************************************************************/
void grid_add_to_fine_grid(const grid_fft_complex_gs_grid *coarse_grid,
                           const grid_fft_complex_gs_grid *fine_grid) {
  assert(coarse_grid != NULL);
  assert(fine_grid != NULL);
  for (int index = 0; index < coarse_grid->fft_grid_layout->npts_gs_local;
       index++) {
    const int ref_index =
        coarse_grid->fft_grid_layout->local_index_to_ref_grid[index];
    for (int dir = 0; dir < 3; dir++)
      assert(convert_c_index_to_shifted_index(
                 coarse_grid->fft_grid_layout->index_to_g[index][dir],
                 coarse_grid->fft_grid_layout->npts_global[dir]) ==
             convert_c_index_to_shifted_index(
                 fine_grid->fft_grid_layout->index_to_g[ref_index][dir],
                 fine_grid->fft_grid_layout->npts_global[dir]));
    fine_grid->data[ref_index] += coarse_grid->data[index];
  }
}

/*******************************************************************************
 * \brief Copy fine grid to coarse grid in reciprocal space
 * \author Frederick Stein
 ******************************************************************************/
void grid_copy_to_coarse_grid(const grid_fft_complex_gs_grid *fine_grid,
                              const grid_fft_complex_gs_grid *coarse_grid) {
  assert(fine_grid != NULL);
  assert(coarse_grid != NULL);
  for (int index = 0; index < coarse_grid->fft_grid_layout->npts_gs_local;
       index++) {
    const int ref_index =
        coarse_grid->fft_grid_layout->local_index_to_ref_grid[index];
    for (int dir = 0; dir < 3; dir++) {
      assert(convert_c_index_to_shifted_index(
                 coarse_grid->fft_grid_layout->index_to_g[index][dir],
                 coarse_grid->fft_grid_layout->npts_global[dir]) ==
             convert_c_index_to_shifted_index(
                 fine_grid->fft_grid_layout->index_to_g[ref_index][dir],
                 fine_grid->fft_grid_layout->npts_global[dir]));
    }
    coarse_grid->data[index] = fine_grid->data[ref_index];
  }
}

/*******************************************************************************
 * \brief Create a real-valued real-space grid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_create_real_rs_grid(grid_fft_real_rs_grid *grid,
                              grid_fft_grid_layout *grid_layout) {
  assert(grid != NULL);
  assert(grid_layout->ref_counter > 0);
  grid->fft_grid_layout = grid_layout;
  grid_retain_fft_grid_layout(grid->fft_grid_layout);
  const int(*my_bounds)[2] =
      grid_layout->proc2local_rs[grid_mpi_comm_rank(grid_layout->comm)];
  int number_of_elements = 1;
  for (int dir = 0; dir < 3; dir++) {
    number_of_elements *= imax(0, my_bounds[dir][1] - my_bounds[dir][0] + 1);
  }
  grid->data = NULL;
  fft_allocate_double(number_of_elements, &grid->data);
}

/*******************************************************************************
 * \brief Create a complex-valued reciprocal-space grid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_create_complex_gs_grid(grid_fft_complex_gs_grid *grid,
                                 grid_fft_grid_layout *grid_layout) {
  assert(grid != NULL);
  grid->fft_grid_layout = grid_layout;
  grid_retain_fft_grid_layout(grid->fft_grid_layout);
  grid->data = NULL;
  fft_allocate_complex(grid_layout->npts_gs_local, &grid->data);
}

/*******************************************************************************
 * \brief Frees a real-valued real-space grid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_free_real_rs_grid(grid_fft_real_rs_grid *grid) {
  if (grid != NULL) {
    fft_free_double(grid->data);
    grid->data = NULL;
    grid_free_fft_grid_layout(grid->fft_grid_layout);
    grid->fft_grid_layout = NULL;
  }
}

/*******************************************************************************
 * \brief Frees a complex-valued reciprocal-space grid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_free_complex_gs_grid(grid_fft_complex_gs_grid *grid) {
  if (grid != NULL) {
    fft_free_complex(grid->data);
    grid->data = NULL;
    grid_free_fft_grid_layout(grid->fft_grid_layout);
    grid->fft_grid_layout = NULL;
  }
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT.
 * \param grid_rs real-valued data in real space.
 * \param grid_gs complex data in reciprocal space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw(const grid_fft_real_rs_grid *grid_rs,
               const grid_fft_complex_gs_grid *grid_gs) {
  assert(grid_rs != NULL);
  assert(grid_gs != NULL);
  assert(grid_rs->fft_grid_layout->grid_id ==
         grid_gs->fft_grid_layout->grid_id);
  const grid_fft_grid_layout *grid_layout = grid_gs->fft_grid_layout;
  fft_3d_fw_sorted(grid_rs->data, grid_gs->data, grid_layout);
  const double scale =
      1.0 / (((double)grid_layout->npts_global[0]) *
             grid_layout->npts_global[1] * grid_layout->npts_global[2]);
#pragma omp parallel for default(none) shared(grid_gs, grid_layout, scale)
  for (int index = 0; index < grid_layout->npts_gs_local; index++) {
    grid_gs->data[index] *= scale;
  }
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT.
 * \param grid_gs complex data in reciprocal space.
 * \param grid_rs real-valued data in real space.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw(const grid_fft_complex_gs_grid *grid_gs,
               const grid_fft_real_rs_grid *grid_rs) {
  assert(grid_rs->fft_grid_layout->grid_id ==
         grid_gs->fft_grid_layout->grid_id);
  const grid_fft_grid_layout *grid_layout = grid_rs->fft_grid_layout;
  fft_3d_bw_sorted(grid_gs->data, grid_rs->data, grid_layout);
}

// EOF
