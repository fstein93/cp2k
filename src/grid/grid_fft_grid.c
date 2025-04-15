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
 * \brief Performs a forward 3D-FFT using a high-level FFT grid.
 * \param grid_rs real-valued data in real space, ordered according to
 *fft_grid->proc2local_rs \param grid_gs complex data in reciprocal space,
 *ordered according to fft_grid->index_to_g \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw(const grid_fft_real_rs_grid *grid_rs,
               const grid_fft_complex_gs_grid *grid_gs) {
  assert(grid_rs->fft_grid_layout->grid_id ==
         grid_gs->fft_grid_layout->grid_id);
  const grid_fft_grid_layout *grid_layout = grid_rs->fft_grid_layout;
  const int my_process = grid_mpi_comm_rank(grid_layout->comm);
  int local_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++) {
    local_sizes_rs[dir] = grid_layout->proc2local_rs[my_process][dir][1] -
                          grid_layout->proc2local_rs[my_process][dir][0] + 1;
  }
  double *buffer_1_real = (double *)grid_layout->buffer_1;
  memcpy(buffer_1_real, grid_rs->data,
         local_sizes_rs[0] * local_sizes_rs[1] * local_sizes_rs[2] *
             sizeof(double));
  if (grid_layout->ray_distribution) {
    fft_3d_fw_ray(buffer_1_real, grid_layout->buffer_2,
                  grid_layout->npts_global, grid_layout->proc2local_rs,
                  grid_layout->proc2local_ms, grid_layout->yz_to_process,
                  grid_layout->rays_per_process, grid_layout->ray_to_yz,
                  grid_layout->comm, grid_layout->sub_comm);
    int(*my_ray_to_yz)[2] = grid_layout->ray_to_yz;
    for (int process = 0; process < my_process; process++) {
      my_ray_to_yz += grid_layout->rays_per_process[process];
    }
    for (int index = 0; index < grid_layout->npts_gs_local; index++) {
      int *index_g = grid_layout->index_to_g[index];
      for (int yz_ray = 0; yz_ray < grid_layout->rays_per_process[my_process];
           yz_ray++) {
        if (my_ray_to_yz[yz_ray][0] == index_g[1] &&
            my_ray_to_yz[yz_ray][1] == index_g[2]) {
          grid_gs->data[index] =
              grid_layout
                  ->buffer_2[yz_ray * grid_layout->npts_global[0] + index_g[0]];
          break;
        }
      }
    }
  } else {
    fft_3d_fw_blocked(buffer_1_real, grid_layout->buffer_2,
                      grid_layout->npts_global, grid_layout->proc2local_rs,
                      grid_layout->proc2local_ms, grid_layout->proc2local_gs,
                      grid_layout->comm, grid_layout->sub_comm);
    int local_sizes_gs[3];
    for (int dir = 0; dir < 3; dir++) {
      local_sizes_gs[dir] = grid_layout->proc2local_gs[my_process][dir][1] -
                            grid_layout->proc2local_gs[my_process][dir][0] + 1;
    }
    for (int index = 0; index < grid_layout->npts_gs_local; index++) {
      int *index_g = grid_layout->index_to_g[index];
      grid_gs->data[index] =
          grid_layout->buffer_2
              [(index_g[2] - grid_layout->proc2local_gs[my_process][2][0]) *
                   local_sizes_gs[0] * local_sizes_gs[1] +
               (index_g[1] - grid_layout->proc2local_gs[my_process][1][0]) *
                   local_sizes_gs[0] +
               (index_g[0] - grid_layout->proc2local_gs[my_process][0][0])];
    }
  }
  const double scale =
      1.0 / (((double)grid_layout->npts_global[0]) *
             grid_layout->npts_global[1] * grid_layout->npts_global[2]);
  for (int index = 0; index < grid_layout->npts_gs_local; index++) {
    grid_gs->data[index] *= scale;
  }
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT using a high-level FFT grid.
 * \param fft_grid FFT grid object
 * \param grid_gs complex data in reciprocal space, ordered according to
 *fft_grid->index_to_g \param grid_rs real-valued data in real space, ordered
 *according to fft_grid->proc2local_rs \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw(const grid_fft_complex_gs_grid *grid_gs,
               const grid_fft_real_rs_grid *grid_rs) {
  assert(grid_rs->fft_grid_layout->grid_id ==
         grid_gs->fft_grid_layout->grid_id);
  const grid_fft_grid_layout *grid_layout = grid_rs->fft_grid_layout;
  const int my_process = grid_mpi_comm_rank(grid_layout->comm);
  int local_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++) {
    local_sizes_rs[dir] = grid_layout->proc2local_rs[my_process][dir][1] -
                          grid_layout->proc2local_rs[my_process][dir][0] + 1;
  }
  double *buffer_1_real = (double *)grid_layout->buffer_1;
  if (grid_layout->ray_distribution) {
    int(*my_ray_to_yz)[2] = grid_layout->ray_to_yz;
    for (int process = 0; process < my_process; process++) {
      my_ray_to_yz += grid_layout->rays_per_process[process];
    }
    for (int index = 0; index < grid_layout->npts_gs_local; index++) {
      int *index_g = grid_layout->index_to_g[index];
      for (int yz_ray = 0; yz_ray < grid_layout->rays_per_process[my_process];
           yz_ray++) {
        if (my_ray_to_yz[yz_ray][0] == index_g[1] &&
            my_ray_to_yz[yz_ray][1] == index_g[2]) {
          grid_layout
              ->buffer_2[yz_ray * grid_layout->npts_global[0] + index_g[0]] =
              grid_gs->data[index];
          break;
        }
      }
    }
    fft_3d_bw_ray(grid_layout->buffer_2, buffer_1_real,
                  grid_layout->npts_global, grid_layout->proc2local_rs,
                  grid_layout->proc2local_ms, grid_layout->yz_to_process,
                  grid_layout->rays_per_process, grid_layout->ray_to_yz,
                  grid_layout->comm, grid_layout->sub_comm);
  } else {
    int local_sizes_gs[3];
    for (int dir = 0; dir < 3; dir++) {
      local_sizes_gs[dir] = grid_layout->proc2local_gs[my_process][dir][1] -
                            grid_layout->proc2local_gs[my_process][dir][0] + 1;
    }
    for (int index = 0; index < grid_layout->npts_gs_local; index++) {
      int *index_g = grid_layout->index_to_g[index];
      grid_layout->buffer_2
          [(index_g[2] - grid_layout->proc2local_gs[my_process][2][0]) *
               local_sizes_gs[0] * local_sizes_gs[1] +
           (index_g[1] - grid_layout->proc2local_gs[my_process][1][0]) *
               local_sizes_gs[0] +
           (index_g[0] - grid_layout->proc2local_gs[my_process][0][0])] =
          grid_gs->data[index];
    }
    fft_3d_bw_blocked(grid_layout->buffer_2, buffer_1_real,
                      grid_layout->npts_global, grid_layout->proc2local_rs,
                      grid_layout->proc2local_ms, grid_layout->proc2local_gs,
                      grid_layout->comm, grid_layout->sub_comm);
  }
  memcpy(grid_rs->data, buffer_1_real,
         local_sizes_rs[0] * local_sizes_rs[1] * local_sizes_rs[2] *
             sizeof(double));
}

// EOF
