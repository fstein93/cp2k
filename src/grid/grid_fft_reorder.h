/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef GRID_FFT_REORDER_H
#define GRID_FFT_REORDER_H

#include "common/grid_mpi.h"

#include <complex.h>

void collect_y_and_distribute_z_blocked(
    const double complex *grid, double complex *transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int (*proc2local_transposed)[3][2], const grid_mpi_comm comm,
    const grid_mpi_comm sub_comm[2]);

void collect_z_and_distribute_y_blocked(
    const double complex *grid, double complex *transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int (*proc2local_transposed)[3][2], const grid_mpi_comm comm,
    const grid_mpi_comm sub_comm[2]);

void collect_x_and_distribute_z_blocked_transpose(
    const double complex *grid, double complex *transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int (*proc2local_transposed)[3][2], const grid_mpi_comm comm,
    const grid_mpi_comm sub_comm[2]);

void collect_z_and_distribute_x_blocked_transpose(
    const double complex *grid, double complex *transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int (*proc2local_transposed)[3][2], const grid_mpi_comm comm,
    const grid_mpi_comm sub_comm[2]);

void collect_x_and_distribute_y_blocked(
    const double complex *grid, double complex *transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int (*proc2local_transposed)[3][2], const grid_mpi_comm comm,
    const grid_mpi_comm sub_comm[2]);

void collect_y_and_distribute_x_blocked(
    const double complex *grid, double complex *transposed,
    const int npts_global[3], const int (*proc2local)[3][2],
    const int (*proc2local_transposed)[3][2], const grid_mpi_comm comm,
    const grid_mpi_comm sub_comm[2]);

void collect_z_and_distribute_y_ray(const double complex *grid,
                                    double complex *transposed,
                                    const int npts_global[3],
                                    const int (*proc2local)[3][2],
                                    const int *number_of_rays,
                                    const int (*ray_to_xy)[2],
                                    const grid_mpi_comm comm);

void collect_y_and_distribute_z_ray(const double complex *grid,
                                    double complex *transposed,
                                    const int npts_global[3],
                                    const int (*proc2local_transposed)[3][2],
                                    const int *number_of_rays,
                                    const int (*ray_to_xy)[2],
                                    const grid_mpi_comm comm);

#endif /* GRID_FFT_REORDER_H */

// EOF
