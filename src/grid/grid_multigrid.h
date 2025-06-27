/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef GRID_MULTIGRID_H
#define GRID_MULTIGRID_H

#include "../fft/fft_grid.h"
#include "../fft/fft_grid_layout.h"
#include "../mpiwrap/mp_mpi.h"
#include "../offload/offload_buffer.h"
#include "cpu/grid_cpu_multigrid.h"
#include "ref/grid_ref_multigrid.h"

#include <complex.h>
#include <stdbool.h>

/*******************************************************************************
 * \brief Container to precalculate data for redistribution of grids.
 *        Arrays were abstracted from the direction RS2PW.
 * \author Frederick Stein
 ******************************************************************************/
typedef struct {
  int number_of_processes;
  // Ranges of local grid points in each direction in each step
  int local_ranges[4][3][3];
  // Size of the communication buffers
  int size_of_buffer_to_halo;
  int size_of_buffer_to_inner;
  // Sizes of the actual send/recv data
  int *sizes_to_inner;
  int *sizes_to_halo;
  // Offsets for send/recv buffers
  int *buffer_offsets_to_inner;
  int *buffer_offsets_to_halo;
  // Local coordinates for each process
  int **index2local_to_inner;
  int **index2local_to_halo;
  // Offsets for sending and receiving arrays
  // Used to adress all arrays
  int offset_to_inner[3];
  int offset_to_halo[3];
  // Number of communication processes in each direction
  int number_of_processes_to_inner[3];
  int number_of_processes_to_halo[3];
  int total_number_of_processes_to_inner;
  int total_number_of_processes_to_halo;
  int max_number_of_processes_to_inner;
  int max_number_of_processes_to_halo;
  // MPI processes to send and receive from
  int *processes_to_inner;
  int *processes_to_halo;
} grid_redistribute;

/*******************************************************************************
 * \brief Internal representation of a multigrid, abstracting various backends.
 * \author Frederick Stein
 ******************************************************************************/
typedef struct {
  int backend;
  bool orthorhombic;
  int nlevels;
  int (*npts_global)[3];
  int (*npts_local)[3];
  int (*shift_local)[3];
  int (*border_width)[3];
  double (*dh)[3][3];
  double (*dh_inv)[3][3];
  offload_buffer **grids;
  mp_mpi_comm comm;
  int (*pgrid_dims)[3];
  int *proc2local;
  grid_redistribute *redistribute;
  fft_grid_layout **fft_grid_layouts;
  fft_real_rs_grid *fft_rs_grids;
  fft_complex_gs_grid *fft_gs_grids;
  grid_ref_multigrid *ref;
  grid_cpu_multigrid *cpu;
  // more backends to be added here
} grid_multigrid;

/*******************************************************************************
 * \brief Wrapper around grid_create_multigrid to deal with calls from Fortran.
 *        Compare grid_create_multigrid for information on the other arguments.
 *
 * \param fortran_comm     The integer Fortran handle
 *
 * \author Frederick Stein
 ******************************************************************************/
void grid_create_multigrid_f(
    const bool orthorhombic, const int nlevels,
    const int npts_global[nlevels][3], const int npts_local[nlevels][3],
    const int shift_local[nlevels][3], const int border_width[nlevels][3],
    const double dh[nlevels][3][3], const double dh_inv[nlevels][3][3],
    const int pgrid_dims[nlevels][3], const mp_mpi_fint fortran_comm,
    grid_multigrid **multigrid_out);

bool grid_get_multigrid_orthorhombic(const grid_multigrid *multigrid);

int grid_get_multigrid_nlevels(const grid_multigrid *multigrid);

void grid_get_multigrid_npts_global(const grid_multigrid *multigrid,
                                    int *nlevel, int **npts_global);

void grid_get_multigrid_npts_local(const grid_multigrid *multigrid, int *nlevel,
                                   int **npts_local);

void grid_get_multigrid_shift_local(const grid_multigrid *multigrid,
                                    int *nlevel, int **shift_local);

void grid_get_multigrid_border_width(const grid_multigrid *multigrid,
                                     int *nlevel, int **border_width);

void grid_get_multigrid_dh(const grid_multigrid *multigrid, int *nlevel,
                           double **dh);

void grid_get_multigrid_dh_inv(const grid_multigrid *multigrid, int *nlevel,
                               double **dh_inv);

mp_mpi_fint grid_get_multigrid_fortran_comm(const grid_multigrid *multigrid);

mp_mpi_comm grid_get_multigrid_comm(const grid_multigrid *multigrid);

void grid_copy_to_multigrid_local(const grid_multigrid *multigrid,
                                  const offload_buffer **grids);

void grid_copy_from_multigrid_local(const grid_multigrid *multigrid,
                                    offload_buffer **grids);

void grid_copy_to_multigrid_local_single(const grid_multigrid *multigrid,
                                         const double *grid, const int level);

void grid_copy_from_multigrid_local_single(const grid_multigrid *multigrid,
                                           double *grid, const int level);

void grid_copy_to_multigrid_single(const grid_multigrid *multigrid,
                                   const double *grid, const mp_mpi_comm comm,
                                   const int (*proc2local)[3][2]);

void grid_copy_from_multigrid_single(const grid_multigrid *multigrid,
                                     double *grid, const mp_mpi_comm comm,
                                     const int (*proc2local)[3][2]);

void grid_copy_to_multigrid_single_f(const grid_multigrid *multigrid,
                                     const double *grid, const mp_mpi_fint comm,
                                     const int (*proc2local)[3][2]);

void grid_copy_from_multigrid_single_f(const grid_multigrid *multigrid,
                                       double *grid, const mp_mpi_fint comm,
                                       const int (*proc2local)[3][2]);

void grid_copy_to_multigrid_local_single_f(const grid_multigrid *multigrid,
                                           const double *grid, const int level);

void grid_copy_from_multigrid_local_single_f(const grid_multigrid *multigrid,
                                             double *grid, const int level);

void grid_copy_to_multigrid_general(const grid_multigrid *multigrid,
                                    const double *grids[multigrid->nlevels],
                                    const mp_mpi_comm comm[multigrid->nlevels],
                                    const int *proc2local);

void grid_copy_to_multigrid_general_f(
    const grid_multigrid *multigrid, const double *grids[multigrid->nlevels],
    const mp_mpi_fint fortran_comm[multigrid->nlevels], const int *proc2local);

void grid_copy_from_multigrid_general(
    const grid_multigrid *multigrid, double *grids[multigrid->nlevels],
    const mp_mpi_comm comm[multigrid->nlevels], const int *proc2local);

void grid_copy_from_multigrid_general_f(
    const grid_multigrid *multigrid, double *grids[multigrid->nlevels],
    const mp_mpi_fint fortran_comm[multigrid->nlevels], const int *proc2local);

void grid_copy_to_multigrid_general_single(const grid_multigrid *multigrid,
                                           const int level, const double *grids,
                                           const mp_mpi_comm comm,
                                           const int *proc2local);

void grid_copy_to_multigrid_general_single_f(const grid_multigrid *multigrid,
                                             const int level,
                                             const double *grid,
                                             const mp_mpi_fint fortran_comm,
                                             const int *proc2local);

void grid_copy_from_multigrid_general_single(const grid_multigrid *multigrid,
                                             const int level, double *grid,
                                             const mp_mpi_comm comm,
                                             const int *proc2local);

void grid_copy_from_multigrid_general_f_single(const grid_multigrid *multigrid,
                                               const int level, double *grid,
                                               const mp_mpi_fint fortran_comm,
                                               const int *proc2local);

/*******************************************************************************
 * \brief Allocates a multigrid which is passed to task list-based and
 *        pgf_product-based routines.
 *
 * \param orthorhombic     Whether simulation box is orthorhombic.
 * \param nlevels          Number of grid levels.
 * \param npts_global      Number of global grid points in each direction.
 * \param npts_local       Number of local grid points in each direction.
 * \param shift_local      Number of points local grid is shifted wrt global
 *                         grid
 * \param border_width     Width of halo region in grid points in each
 *                         direction.
 * \param dh               Incremental grid matrix.
 * \param dh_inv           Inverse incremental grid matrix.
 * \param comm             The C-communicator
 *
 * \param multigrid        Handle to the created multigrid.
 *
 * \author Frederick Stein
 ******************************************************************************/
void grid_create_multigrid(
    const bool orthorhombic, const int nlevels,
    const int npts_global[nlevels][3], const int npts_local[nlevels][3],
    const int shift_local[nlevels][3], const int border_width[nlevels][3],
    const double dh[nlevels][3][3], const double dh_inv[nlevels][3][3],
    const int pgrid_dims[nlevels][3], const mp_mpi_comm comm,
    grid_multigrid **multigrid_out);

/*******************************************************************************
 * \brief Deallocates given multigrid.
 * \author Frederick Stein
 ******************************************************************************/
void grid_free_multigrid(grid_multigrid *multigrid);

/*******************************************************************************
 * \brief Redistributes data between two grids with a given distribution and the
 *same order of global indices. \author Frederick Stein
 ******************************************************************************/
void redistribute_grids(
    const double *grid_in, double *grid_out, const mp_mpi_comm comm_in,
    const mp_mpi_comm comm_out, const int npts_global[3],
    const int proc2local_in[mp_mpi_comm_size(comm_in)][3][2],
    const int proc2local_out[mp_mpi_comm_size(comm_out)][3][2]);

#endif

// EOF
