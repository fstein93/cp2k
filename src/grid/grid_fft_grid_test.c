/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "grid_fft_grid_test.h"

#include "common/grid_common.h"
#include "common/grid_mpi.h"
#include "grid_fft_grid.h"
#include "grid_fft_grid_layout.h"
#include "grid_fft_lib.h"
#include "grid_fft_reorder.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*******************************************************************************
 * \brief Function to test the parallel FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_3d_blocked(const int npts_global[3]) {
  const grid_mpi_comm comm = grid_mpi_comm_world;
  const int my_process = grid_mpi_comm_rank(comm);

  int errors = 0;

  const double pi = acos(-1);
  const double dh_inv[3][3] = {
      {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  grid_fft_grid_layout *fft_grid_layout = NULL;
  grid_create_fft_grid_layout(&fft_grid_layout, comm, npts_global, dh_inv);

  const int(*my_bounds_rs)[2] = fft_grid_layout->proc2local_rs[my_process];
  int my_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_rs[dir] = my_bounds_rs[dir][1] - my_bounds_rs[dir][0] + 1;
  const int my_number_of_elements_rs = product3(my_sizes_rs);

  const int(*my_bounds_gs)[2] = fft_grid_layout->proc2local_gs[my_process];
  int my_sizes_gs[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_gs[dir] = my_bounds_gs[dir][1] - my_bounds_gs[dir][0] + 1;
  const int my_number_of_elements_gs = product3(my_sizes_gs);

  const double scale = 1.0 / ((double)npts_global[0]) /
                       ((double)npts_global[1]) / ((double)npts_global[2]);

  grid_fft_real_rs_grid grid_rs;
  grid_create_real_rs_grid(&grid_rs, fft_grid_layout);
  grid_fft_complex_gs_grid grid_gs;
  grid_create_complex_gs_grid(&grid_gs, fft_grid_layout);

  // Check forward 3D FFTs
  double max_error = 0.0;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int ny = 0; ny < npts_global[1]; ny++) {
      for (int nz = 0; nz < npts_global[2]; nz++) {
        double *buffer_1_real = (double *)fft_grid_layout->buffer_1;
        memset(buffer_1_real, 0, my_number_of_elements_rs * sizeof(double));

        if (nx >= my_bounds_rs[0][0] && nx <= my_bounds_rs[0][1] &&
            ny >= my_bounds_rs[1][0] && ny <= my_bounds_rs[1][1] &&
            nz >= my_bounds_rs[2][0] && nz <= my_bounds_rs[2][1])
          buffer_1_real[(nz - my_bounds_rs[2][0]) * my_sizes_rs[0] *
                            my_sizes_rs[1] +
                        (ny - my_bounds_rs[1][0]) * my_sizes_rs[0] +
                        (nx - my_bounds_rs[0][0])] = 1.0;

        fft_3d_fw_blocked(
            buffer_1_real, fft_grid_layout->buffer_2,
            fft_grid_layout->npts_global, fft_grid_layout->proc2local_rs,
            fft_grid_layout->proc2local_ms, fft_grid_layout->proc2local_gs,
            fft_grid_layout->comm, fft_grid_layout->sub_comm);

        for (int mx = 0; mx < my_sizes_gs[0]; mx++) {
          for (int my = 0; my < my_sizes_gs[1]; my++) {
            for (int mz = 0; mz < my_sizes_gs[2]; mz++) {
              const double complex my_value =
                  fft_grid_layout
                      ->buffer_2[mz * my_sizes_gs[0] * my_sizes_gs[1] +
                                 my * my_sizes_gs[0] + mx];
              const double complex ref_value = cexp(
                  -2.0 * I * pi *
                  (((double)mx + my_bounds_gs[0][0]) * nx / npts_global[0] +
                   ((double)my + my_bounds_gs[1][0]) * ny / npts_global[1] +
                   ((double)mz + my_bounds_gs[2][0]) * nz / npts_global[2]));
              double current_error = cabs(my_value - ref_value);
              max_error = fmax(max_error, current_error);
            }
          }
        }
      }
    }
  }
  grid_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf(
          "The 3D forward FFT with blocked layout does not work correctly (%i "
          "%i %i): %f!\n",
          npts_global[0], npts_global[1], npts_global[2], max_error);
    errors++;
  }

  // Check forward 3D FFTs
  max_error = 0.0;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int ny = 0; ny < npts_global[1]; ny++) {
      for (int nz = 0; nz < npts_global[2]; nz++) {

        memset(grid_rs.data, 0, my_number_of_elements_rs * sizeof(double));

        if (nx >= my_bounds_rs[0][0] && nx <= my_bounds_rs[0][1] &&
            ny >= my_bounds_rs[1][0] && ny <= my_bounds_rs[1][1] &&
            nz >= my_bounds_rs[2][0] && nz <= my_bounds_rs[2][1])
          grid_rs.data[(nz - my_bounds_rs[2][0]) * my_sizes_rs[0] *
                           my_sizes_rs[1] +
                       (ny - my_bounds_rs[1][0]) * my_sizes_rs[0] +
                       (nx - my_bounds_rs[0][0])] = 1.0;

        fft_3d_fw(&grid_rs, &grid_gs);

        for (int index = 0; index < my_number_of_elements_gs; index++) {
          const int mx = fft_grid_layout->index_to_g[index][0];
          const int my = fft_grid_layout->index_to_g[index][1];
          const int mz = fft_grid_layout->index_to_g[index][2];
          const double complex my_value = grid_gs.data[index];
          const double complex ref_value =
              scale * cexp(-2.0 * I * pi *
                           (((double)mx) * nx / npts_global[0] +
                            ((double)my) * ny / npts_global[1] +
                            ((double)mz) * nz / npts_global[2]));
          double current_error = cabs(my_value - ref_value);
          if (current_error > 1e-12)
            printf("%i ERROR %i %i %i/%i %i %i: (%f %f) (%f %f)\n", my_process,
                   nx, ny, nz, mx, my, mz, creal(my_value), cimag(my_value),
                   creal(ref_value), cimag(ref_value));
          max_error = fmax(max_error, current_error);
        }
      }
    }
  }
  grid_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The 3D forward FFT (blocked) to ordered layout does not work "
             "correctly (%i "
             "%i %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], max_error);
    errors++;
  }

  double *buffer_1_real = (double *)fft_grid_layout->buffer_1;

  // Check backwards 3D FFTs
  max_error = 0.0;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int ny = 0; ny < npts_global[1]; ny++) {
      for (int nz = 0; nz < npts_global[2]; nz++) {
        memset(fft_grid_layout->buffer_2, 0,
               my_number_of_elements_gs * sizeof(double complex));

        if (nx >= my_bounds_gs[0][0] && nx <= my_bounds_gs[0][1] &&
            ny >= my_bounds_gs[1][0] && ny <= my_bounds_gs[1][1] &&
            nz >= my_bounds_gs[2][0] && nz <= my_bounds_gs[2][1])
          fft_grid_layout->buffer_2[(nz - my_bounds_gs[2][0]) * my_sizes_gs[0] *
                                        my_sizes_gs[1] +
                                    (ny - my_bounds_gs[1][0]) * my_sizes_gs[0] +
                                    (nx - my_bounds_gs[0][0])] = 1.0;

        fft_3d_bw_blocked(
            fft_grid_layout->buffer_2, buffer_1_real,
            fft_grid_layout->npts_global, fft_grid_layout->proc2local_rs,
            fft_grid_layout->proc2local_ms, fft_grid_layout->proc2local_gs,
            fft_grid_layout->comm, fft_grid_layout->sub_comm);

        for (int mx = 0; mx < my_sizes_rs[0]; mx++) {
          for (int my = 0; my < my_sizes_rs[1]; my++) {
            for (int mz = 0; mz < my_sizes_rs[2]; mz++) {
              const double my_value =
                  buffer_1_real[mz * my_sizes_rs[0] * my_sizes_rs[1] +
                                my * my_sizes_rs[0] + mx];
              const double ref_value = cos(
                  2.0 * pi *
                  (((double)mx + my_bounds_rs[0][0]) * nx / npts_global[0] +
                   ((double)my + my_bounds_rs[1][0]) * ny / npts_global[1] +
                   ((double)mz + my_bounds_rs[2][0]) * nz / npts_global[2]));
              double current_error = fabs(my_value - ref_value);
              max_error = fmax(max_error, current_error);
            }
          }
        }
      }
    }
  }
  grid_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The backwards 3D FFT with blocked layout does not work correctly "
             "(%i %i %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], max_error);
    errors++;
  }

  // Check backwards 3D FFTs
  max_error = 0.0;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int ny = 0; ny < npts_global[1]; ny++) {
      for (int nz = 0; nz < npts_global[2]; nz++) {
        memset(grid_gs.data, 0,
               my_number_of_elements_gs * sizeof(double complex));

        for (int index = 0; index < my_number_of_elements_gs; index++) {
          if (nx == fft_grid_layout->index_to_g[index][0] &&
              ny == fft_grid_layout->index_to_g[index][1] &&
              nz == fft_grid_layout->index_to_g[index][2]) {
            grid_gs.data[index] = 1.0;
            break;
          }
        }

        fft_3d_bw(&grid_gs, &grid_rs);

        for (int mx = 0; mx < my_sizes_rs[0]; mx++) {
          for (int my = 0; my < my_sizes_rs[1]; my++) {
            for (int mz = 0; mz < my_sizes_rs[2]; mz++) {
              const double my_value =
                  grid_rs.data[mz * my_sizes_rs[0] * my_sizes_rs[1] +
                               my * my_sizes_rs[0] + mx];
              const double ref_value = cos(
                  2.0 * pi *
                  (((double)mx + my_bounds_rs[0][0]) * nx / npts_global[0] +
                   ((double)my + my_bounds_rs[1][0]) * ny / npts_global[1] +
                   ((double)mz + my_bounds_rs[2][0]) * nz / npts_global[2]));
              double current_error = fabs(my_value - ref_value);
              max_error = fmax(max_error, current_error);
            }
          }
        }
      }
    }
  }
  grid_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The backwards 3D FFT (blocked) to ordered layout does not work "
             "correctly "
             "(%i %i %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], max_error);
    errors++;
  }

  grid_free_real_rs_grid(&grid_rs);
  grid_free_complex_gs_grid(&grid_gs);
  grid_free_fft_grid_layout(fft_grid_layout);

  if (errors == 0 && my_process == 0)
    printf("The 3D FFT with blocked layout does work correctly (sizes %i %i "
           "%i)!\n",
           npts_global[0], npts_global[1], npts_global[2]);
  return errors;
}

/*******************************************************************************
 * \brief Function to test the parallel FFT backend.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_3d_ray(const int npts_global[3], const int npts_global_ref[3]) {
  const grid_mpi_comm comm = grid_mpi_comm_world;
  const int my_process = grid_mpi_comm_rank(comm);

  int errors = 0;

  const double pi = acos(-1);
  const double dh_inv[3][3] = {
      {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  grid_fft_grid_layout *ref_grid_layout = NULL;
  grid_create_fft_grid_layout(&ref_grid_layout, comm, npts_global_ref, dh_inv);

  grid_fft_grid_layout *fft_grid_layout = NULL;
  grid_create_fft_grid_layout_from_reference(&fft_grid_layout, npts_global,
                                             ref_grid_layout);

  const int(*my_bounds_rs)[2] = fft_grid_layout->proc2local_rs[my_process];
  int my_sizes_rs[3];
  for (int dir = 0; dir < 3; dir++)
    my_sizes_rs[dir] = my_bounds_rs[dir][1] - my_bounds_rs[dir][0] + 1;
  const int my_number_of_elements_rs = product3(my_sizes_rs);

  const int my_number_of_elements_gs =
      fft_grid_layout->rays_per_process[my_process] * npts_global[0];

  int my_ray_offset = 0;
  for (int process = 0; process < my_process; process++)
    my_ray_offset += fft_grid_layout->rays_per_process[process];

  double *buffer_1_real = (double *)fft_grid_layout->buffer_1;

  // Check forward 3D FFTs
  double max_error = 0.0;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int ny = 0; ny < npts_global[1]; ny++) {
      for (int nz = 0; nz < npts_global[2]; nz++) {
        memset(buffer_1_real, 0, my_number_of_elements_rs * sizeof(double));

        if (nx >= my_bounds_rs[0][0] && nx <= my_bounds_rs[0][1] &&
            ny >= my_bounds_rs[1][0] && ny <= my_bounds_rs[1][1] &&
            nz >= my_bounds_rs[2][0] && nz <= my_bounds_rs[2][1])
          buffer_1_real[(nz - my_bounds_rs[2][0]) * my_sizes_rs[0] *
                            my_sizes_rs[1] +
                        (ny - my_bounds_rs[1][0]) * my_sizes_rs[0] +
                        (nx - my_bounds_rs[0][0])] = 1.0;

        fft_3d_fw_ray(
            buffer_1_real, fft_grid_layout->buffer_2,
            fft_grid_layout->npts_global, fft_grid_layout->proc2local_rs,
            fft_grid_layout->proc2local_ms, fft_grid_layout->yz_to_process,
            fft_grid_layout->rays_per_process, fft_grid_layout->ray_to_yz,
            fft_grid_layout->comm, fft_grid_layout->sub_comm);

        for (int index_x = 0; index_x < npts_global[0]; index_x++) {
          for (int yz_ray = 0;
               yz_ray < fft_grid_layout->rays_per_process[my_process];
               yz_ray++) {
            const int index_y =
                fft_grid_layout->ray_to_yz[my_ray_offset + yz_ray][0];
            const int index_z =
                fft_grid_layout->ray_to_yz[my_ray_offset + yz_ray][1];
            const double complex my_value =
                fft_grid_layout->buffer_2[yz_ray * npts_global[0] + index_x];
            const double complex ref_value =
                cexp(-2.0 * I * pi *
                     (((double)index_x) * nx / npts_global[0] +
                      ((double)index_y) * ny / npts_global[1] +
                      ((double)index_z) * nz / npts_global[2]));
            double current_error = cabs(my_value - ref_value);
            max_error = fmax(max_error, current_error);
          }
        }
      }
    }
  }
  grid_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf(
          "The 3D forward FFT with ray layout does not work correctly (%i %i "
          "%i)/(%i %i %i): %f!\n",
          npts_global[0], npts_global[1], npts_global[2], npts_global_ref[0],
          npts_global_ref[1], npts_global_ref[2], max_error);
    errors++;
  }

  // Check backwards 3D FFTs
  int total_number_of_gs_elements = 0;
  for (int process = 0; process < grid_mpi_comm_size(comm); process++)
    total_number_of_gs_elements += fft_grid_layout->rays_per_process[process];
  max_error = 0.0;
  for (int nx = 0; nx < npts_global[0]; nx++) {
    for (int nyz = 0; nyz < total_number_of_gs_elements; nyz++) {
      const int ny = fft_grid_layout->ray_to_yz[nyz][0];
      const int nz = fft_grid_layout->ray_to_yz[nyz][1];
      memset(fft_grid_layout->buffer_2, 0,
             my_number_of_elements_gs * sizeof(double complex));

      if (nyz >= my_ray_offset &&
          nyz < my_ray_offset + fft_grid_layout->rays_per_process[my_process]) {
        fft_grid_layout->buffer_2[(nyz - my_ray_offset) * npts_global[0] + nx] =
            1.0;
      }

      fft_3d_bw_ray(
          fft_grid_layout->buffer_2, buffer_1_real,
          fft_grid_layout->npts_global, fft_grid_layout->proc2local_rs,
          fft_grid_layout->proc2local_ms, fft_grid_layout->yz_to_process,
          fft_grid_layout->rays_per_process, fft_grid_layout->ray_to_yz,
          fft_grid_layout->comm, fft_grid_layout->sub_comm);

      for (int mx = 0; mx < my_sizes_rs[0]; mx++) {
        for (int my = 0; my < my_sizes_rs[1]; my++) {
          for (int mz = 0; mz < my_sizes_rs[2]; mz++) {
            const double my_value =
                buffer_1_real[mz * my_sizes_rs[0] * my_sizes_rs[1] +
                              my * my_sizes_rs[0] + mx];
            const double ref_value =
                cos(2.0 * pi *
                    (((double)mx + my_bounds_rs[0][0]) * nx / npts_global[0] +
                     ((double)my + my_bounds_rs[1][0]) * ny / npts_global[1] +
                     ((double)mz + my_bounds_rs[2][0]) * nz / npts_global[2]));
            double current_error = fabs(my_value - ref_value);
            max_error = fmax(max_error, current_error);
          }
        }
      }
    }
  }
  grid_mpi_max_double(&max_error, 1, comm);

  grid_free_fft_grid_layout(fft_grid_layout);
  grid_free_fft_grid_layout(ref_grid_layout);

  if (max_error > 1e-12) {
    if (my_process == 0)
      printf("The backwards 3D FFT with ray layout does not work correctly (%i "
             "%i %i)/(%i %i %i): %f!\n",
             npts_global[0], npts_global[1], npts_global[2], npts_global_ref[0],
             npts_global_ref[1], npts_global_ref[2], max_error);
    errors++;
  }

  if (errors == 0 && my_process == 0)
    printf("The 3D FFT with ray layout does work correctly (%i %i %i)/(%i %i "
           "%i)!\n",
           npts_global[0], npts_global[1], npts_global[2], npts_global_ref[0],
           npts_global_ref[1], npts_global_ref[2]);
  return errors;
}

int fft_test_3d() {
  const int my_process = grid_mpi_comm_rank(grid_mpi_comm_world);

  int errors = 0;

  // Grid sizes to be checked
  const int npts_global_x[3] = {8, 1, 1};
  const int npts_global_y[3] = {1, 8, 1};
  const int npts_global_z[3] = {1, 1, 8};
  const int npts_global_yz[3] = {1, 8, 8};
  const int npts_global_xz[3] = {8, 1, 8};
  const int npts_global_xy[3] = {8, 8, 1};
  const int npts_global[3] = {2, 4, 8};
  const int npts_global_small[3] = {2, 3, 5};
  const int npts_global_reverse[3] = {8, 4, 2};
  const int npts_global_small_reverse[3] = {5, 3, 2};

  // Check the blocked layout
  errors += fft_test_3d_blocked(npts_global);
  errors += fft_test_3d_blocked(npts_global_small);
  errors += fft_test_3d_blocked(npts_global_reverse);
  errors += fft_test_3d_blocked(npts_global_small_reverse);
  errors += fft_test_3d_blocked(npts_global_x);
  errors += fft_test_3d_blocked(npts_global_y);
  errors += fft_test_3d_blocked(npts_global_z);
  errors += fft_test_3d_blocked(npts_global_yz);
  errors += fft_test_3d_blocked(npts_global_xz);
  errors += fft_test_3d_blocked(npts_global_xy);

  // Check the ray layout with the same grid sizes
  errors += fft_test_3d_ray(npts_global_y, npts_global_yz);
  errors += fft_test_3d_ray(npts_global_z, npts_global_yz);
  errors += fft_test_3d_ray(npts_global_x, npts_global_xz);
  errors += fft_test_3d_ray(npts_global_z, npts_global_xz);
  errors += fft_test_3d_ray(npts_global_x, npts_global_xy);
  errors += fft_test_3d_ray(npts_global_y, npts_global_xy);
  errors += fft_test_3d_ray(npts_global, npts_global);
  errors += fft_test_3d_ray(npts_global_small, npts_global_small);
  errors += fft_test_3d_ray(npts_global_reverse, npts_global_reverse);
  errors +=
      fft_test_3d_ray(npts_global_small_reverse, npts_global_small_reverse);

  if (errors == 0 && my_process == 0)
    fprintf(stdout, "\n The 3D FFT routines work correctly!\n");
  return errors;
}

int fft_test_add_copy_low(const int npts_global_fine[3],
                          const int npts_global_coarse[3]) {
  const grid_mpi_comm comm = grid_mpi_comm_world;
  const double dh_inv[3][3] = {
      {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  int errors = 0;

  grid_fft_grid_layout *fft_grid_fine_layout = NULL;
  grid_create_fft_grid_layout(&fft_grid_fine_layout, comm, npts_global_fine,
                              dh_inv);

  grid_fft_grid_layout *fft_grid_coarse_layout = NULL;
  grid_create_fft_grid_layout_from_reference(
      &fft_grid_coarse_layout, npts_global_coarse, fft_grid_fine_layout);

  grid_fft_complex_gs_grid grid_fine;
  grid_create_complex_gs_grid(&grid_fine, fft_grid_fine_layout);

  grid_fft_complex_gs_grid grid_coarse;
  grid_create_complex_gs_grid(&grid_coarse, fft_grid_coarse_layout);

  for (int index = 0; index < fft_grid_fine_layout->npts_gs_local; index++) {
    const int index_g[3] = {fft_grid_fine_layout->index_to_g[index][0],
                            fft_grid_fine_layout->index_to_g[index][1],
                            fft_grid_fine_layout->index_to_g[index][2]};
    const int shifted_index_g[3] = {
        convert_c_index_to_shifted_index(index_g[0], npts_global_fine[0]),
        convert_c_index_to_shifted_index(index_g[1], npts_global_fine[1]),
        convert_c_index_to_shifted_index(index_g[2], npts_global_fine[2])};
    grid_fine.data[index] = fabs((double)shifted_index_g[0]) +
                            fabs((double)shifted_index_g[1]) +
                            fabs((double)shifted_index_g[2]);
  }
  memset(grid_coarse.data, 0,
         fft_grid_coarse_layout->npts_gs_local * sizeof(double complex));

  grid_copy_to_coarse_grid(&grid_fine, &grid_coarse);

  double max_error = 0.0;
  for (int index = 0; index < fft_grid_coarse_layout->npts_gs_local; index++) {
    const int index_g[3] = {fft_grid_coarse_layout->index_to_g[index][0],
                            fft_grid_coarse_layout->index_to_g[index][1],
                            fft_grid_coarse_layout->index_to_g[index][2]};
    const int shifted_index_g[3] = {
        convert_c_index_to_shifted_index(index_g[0], npts_global_coarse[0]),
        convert_c_index_to_shifted_index(index_g[1], npts_global_coarse[1]),
        convert_c_index_to_shifted_index(index_g[2], npts_global_coarse[2])};
    const double complex my_value = grid_coarse.data[index];
    for (int index_fine = 0; index_fine < fft_grid_fine_layout->npts_gs_local;
         index_fine++) {
      const int index_g_fine[3] = {
          fft_grid_fine_layout->index_to_g[index_fine][0],
          fft_grid_fine_layout->index_to_g[index_fine][1],
          fft_grid_fine_layout->index_to_g[index_fine][2]};
      const int shifted_index_g_fine[3] = {
          convert_c_index_to_shifted_index(index_g_fine[0],
                                           npts_global_fine[0]),
          convert_c_index_to_shifted_index(index_g_fine[1],
                                           npts_global_fine[1]),
          convert_c_index_to_shifted_index(index_g_fine[2],
                                           npts_global_fine[2])};
      if (shifted_index_g_fine[0] == shifted_index_g[0] &&
          shifted_index_g_fine[1] == shifted_index_g[1] &&
          shifted_index_g_fine[2] == shifted_index_g[2]) {
        const double complex ref_value = fabs((double)shifted_index_g_fine[0]) +
                                         fabs((double)shifted_index_g_fine[1]) +
                                         fabs((double)shifted_index_g_fine[2]);
        double current_error = cabs(my_value - ref_value);
        max_error = fmax(max_error, current_error);
        break;
      }
    }
  }
  grid_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1.0e-12) {
    if (grid_mpi_comm_rank(comm) == 0)
      fprintf(stderr,
              "The copy between different grids does not work correctly (%i "
              "%i %i)/(%i %i %i): %f!\n",
              npts_global_fine[0], npts_global_fine[1], npts_global_fine[2],
              npts_global_coarse[0], npts_global_coarse[1],
              npts_global_coarse[2], max_error);
    errors++;
  }

  for (int index = 0; index < fft_grid_coarse_layout->npts_gs_local; index++) {
    const int index_g[3] = {fft_grid_coarse_layout->index_to_g[index][0],
                            fft_grid_coarse_layout->index_to_g[index][1],
                            fft_grid_coarse_layout->index_to_g[index][2]};
    const int shifted_index_g[3] = {
        convert_c_index_to_shifted_index(index_g[0], npts_global_coarse[0]),
        convert_c_index_to_shifted_index(index_g[1], npts_global_coarse[1]),
        convert_c_index_to_shifted_index(index_g[2], npts_global_coarse[2])};
    grid_coarse.data[index] = (double)shifted_index_g[0] +
                              (double)shifted_index_g[1] +
                              (double)shifted_index_g[2];
  }

  grid_add_to_fine_grid(&grid_coarse, &grid_fine);

  max_error = 0.0;
  for (int index = 0; index < fft_grid_fine_layout->npts_gs_local; index++) {
    const int index_g[3] = {fft_grid_fine_layout->index_to_g[index][0],
                            fft_grid_fine_layout->index_to_g[index][1],
                            fft_grid_fine_layout->index_to_g[index][2]};
    const int shifted_index_g[3] = {
        convert_c_index_to_shifted_index(index_g[0], npts_global_fine[0]),
        convert_c_index_to_shifted_index(index_g[1], npts_global_fine[1]),
        convert_c_index_to_shifted_index(index_g[2], npts_global_fine[2])};
    const double complex my_value = grid_fine.data[index];
    bool found = false;
    for (int index = 0; index < fft_grid_coarse_layout->npts_gs_local;
         index++) {
      const int index_g_coarse[3] = {
          fft_grid_coarse_layout->index_to_g[index][0],
          fft_grid_coarse_layout->index_to_g[index][1],
          fft_grid_coarse_layout->index_to_g[index][2]};
      const int shifted_index_g_coarse[3] = {
          convert_c_index_to_shifted_index(index_g_coarse[0],
                                           npts_global_coarse[0]),
          convert_c_index_to_shifted_index(index_g_coarse[1],
                                           npts_global_coarse[1]),
          convert_c_index_to_shifted_index(index_g_coarse[2],
                                           npts_global_coarse[2])};
      if (shifted_index_g_coarse[0] == shifted_index_g[0] &&
          shifted_index_g_coarse[1] == shifted_index_g[1] &&
          shifted_index_g_coarse[2] == shifted_index_g[2]) {
        const double complex ref_value = fabs((double)shifted_index_g[0]) +
                                         fabs((double)shifted_index_g[1]) +
                                         fabs((double)shifted_index_g[2]) +
                                         (double)shifted_index_g_coarse[0] +
                                         (double)shifted_index_g_coarse[1] +
                                         (double)shifted_index_g_coarse[2];
        double current_error = cabs(my_value - ref_value);
        max_error = fmax(max_error, current_error);
        found = true;
        break;
      }
    }
    if (!found) {
      const double complex ref_value = fabs((double)shifted_index_g[0]) +
                                       fabs((double)shifted_index_g[1]) +
                                       fabs((double)shifted_index_g[2]);
      double current_error = cabs(my_value - ref_value);
      max_error = fmax(max_error, current_error);
    }
  }
  grid_mpi_max_double(&max_error, 1, comm);

  if (max_error > 1.0e-12) {
    if (grid_mpi_comm_rank(comm) == 0)
      fprintf(
          stderr,
          "The addition between different grids does not work correctly (%i "
          "%i %i)/(%i %i %i): %f!\n",
          npts_global_fine[0], npts_global_fine[1], npts_global_fine[2],
          npts_global_coarse[0], npts_global_coarse[1], npts_global_coarse[2],
          max_error);
    errors++;
  }

  grid_free_complex_gs_grid(&grid_fine);
  grid_free_complex_gs_grid(&grid_coarse);

  grid_free_fft_grid_layout(fft_grid_fine_layout);
  grid_free_fft_grid_layout(fft_grid_coarse_layout);

  if (errors == 0 && grid_mpi_comm_rank(comm) == 0) {
    fprintf(stdout,
            "The addition between different grids works correctly (%i "
            "%i %i)/(%i %i %i)\n",
            npts_global_fine[0], npts_global_fine[1], npts_global_fine[2],
            npts_global_coarse[0], npts_global_coarse[1],
            npts_global_coarse[2]);
  }

  return errors;
}

/*******************************************************************************
 * \brief Function to test the addition/copy between different grids.
 * \author Frederick Stein
 ******************************************************************************/
int fft_test_add_copy() {
  int errors = 0;

  errors +=
      fft_test_add_copy_low((const int[3]){2, 4, 8}, (const int[3]){2, 4, 8});
  errors +=
      fft_test_add_copy_low((const int[3]){8, 4, 2}, (const int[3]){7, 3, 2});
  errors +=
      fft_test_add_copy_low((const int[3]){2, 4, 8}, (const int[3]){1, 2, 4});
  errors +=
      fft_test_add_copy_low((const int[3]){11, 7, 5}, (const int[3]){5, 3, 2});

  return errors;
}

// EOF
