/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "fft_driver_gpu.h"
#include "../offload/offload_runtime.h"
#include "fft_lib.h"
#include "fft_redistribution.h"
#include "fft_timer.h"
#include "fft_utils.h"
#include "gpu/fft_gpu.h"
#include "gpu/fft_gpu_kernels.h"

#include "../base/cp_assert.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FFT_STREAM ((offloadStream_t)fft_stream)
#define FFT_H2D(dst, src, size)                                                \
  offloadMemcpyAsyncHtoD((dst), (src), (size), FFT_STREAM)
#define FFT_D2H(dst, src, size)                                                \
  do {                                                                         \
    offloadMemcpyAsyncDtoH((dst), (src), (size), FFT_STREAM);                  \
    offloadStreamSynchronize(FFT_STREAM);                                      \
  } while (0)

/*******************************************************************************
 * \brief Registers all FFTs using a blocked distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_register_3d_gpu_blocked(const int npts_global[3],
                                 int (*proc2local_rs)[3][2],
                                 int (*proc2local_ms)[3][2],
                                 int (*proc2local_gs)[3][2], cp_mpi_comm_t comm,
                                 const cp_mpi_comm_t sub_comm[2]) {
  (void)sub_comm;

  const int my_process = cp_mpi_comm_rank(comm);

  double complex *grid_buffer_1 = get_buffer_1();
  double complex *grid_buffer_2 = get_buffer_2();
  double complex *grid_buffer_dev_1 = fft_gpu_get_buffer_1();
  double complex *grid_buffer_dev_2 = fft_gpu_get_buffer_2();
  void *fft_stream = fft_gpu_get_stream();

  int (*my_bounds_rs)[2] = proc2local_rs[my_process];
  int (*my_bounds_ms)[2] = proc2local_ms[my_process];
  int (*my_bounds_gs)[2] = proc2local_gs[my_process];

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {my_bounds_rs[0][1], my_bounds_rs[1][1],
                         my_bounds_rs[2][1]};
  int fft_sizes_ms[3] = {my_bounds_ms[0][1], my_bounds_ms[1][1],
                         my_bounds_ms[2][1]};
  int fft_sizes_gs[3] = {my_bounds_gs[0][1], my_bounds_gs[1][1],
                         my_bounds_gs[2][1]};

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  cp_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  // We use different data distribution schemes depending on the availability of
  // a distributed FFT library because FFTW requires the data to the different
  // FFTs to be consecutively stored in memory. This is not possible without a
  // distributed FFT library because this would require the implementation of
  // the Guru interface which is not available with all implementations of the
  // FFTW interface
  if (proc_grid[0] > 1 && proc_grid[1] > 1) {
    if (fft_sizes_rs[2] > 0) {
      // Perform the first FFT

      fft_gpu_register_1d_local(
          true, npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], true, false,
          fft_sizes_rs[1] * fft_sizes_rs[2], npts_global[0], grid_buffer_dev_1,
          grid_buffer_dev_2);
      fft_gpu_register_1d_local(
          false, npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], false, true,
          npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], grid_buffer_dev_1,
          grid_buffer_dev_2);

      // Perform the second FFT
      fft_gpu_register_1d_local(
          true, npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2], true, false,
          fft_sizes_ms[0] * fft_sizes_ms[2], npts_global[1], grid_buffer_dev_2,
          grid_buffer_dev_1);
      fft_gpu_register_1d_local(
          false, npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2], false, true,
          npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2], grid_buffer_dev_2,
          grid_buffer_dev_1);
    }

    fft_gpu_register_1d_local(true, npts_global[2],
                              fft_sizes_gs[0] * fft_sizes_gs[1], true, false,
                              fft_sizes_gs[0] * fft_sizes_gs[1], npts_global[2],
                              grid_buffer_dev_1, grid_buffer_dev_2);
    fft_gpu_register_1d_local(false, npts_global[2],
                              fft_sizes_gs[0] * fft_sizes_gs[1], false, true,
                              npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1],
                              grid_buffer_dev_1, grid_buffer_dev_2);
  } else if (proc_grid[0] > 1) {
    cp_assert(fft_sizes_rs[1] == npts_global[1]);

    fft_gpu_register_2d_local(
        true, (const int[2]){npts_global[0], npts_global[1]}, fft_sizes_rs[2],
        true, false, grid_buffer_dev_2, grid_buffer_dev_1);
    fft_gpu_register_2d_local(
        false, (const int[2]){npts_global[0], npts_global[1]}, fft_sizes_ms[2],
        false, true, grid_buffer_dev_2, grid_buffer_dev_1);

    fft_gpu_register_1d_local(true, npts_global[2],
                              fft_sizes_gs[0] * fft_sizes_gs[1], true, false,
                              fft_sizes_gs[0] * fft_sizes_gs[1], npts_global[2],
                              grid_buffer_dev_1, grid_buffer_dev_2);
    fft_gpu_register_1d_local(false, npts_global[2],
                              fft_sizes_gs[0] * fft_sizes_gs[1], false, true,
                              npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1],
                              grid_buffer_dev_1, grid_buffer_dev_2);

  } else if (proc_grid[1] > 1) {
    if (fft_sizes_rs[2] > 0) {
      // Perform the first FFT

      fft_gpu_register_1d_local(
          true, npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], true, false,
          fft_sizes_rs[1] * fft_sizes_rs[2], npts_global[0], grid_buffer_dev_1,
          grid_buffer_dev_2);
      fft_gpu_register_1d_local(
          false, npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], false, true,
          npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], grid_buffer_dev_2,
          grid_buffer_dev_1);

      // Perform the second FFT
      fft_gpu_register_2d_local(
          true, (const int[2]){npts_global[1], npts_global[2]}, fft_sizes_ms[0],
          true, false, grid_buffer_dev_2, grid_buffer_dev_1);
      fft_gpu_register_2d_local(
          false, (const int[2]){npts_global[1], npts_global[2]},
          fft_sizes_gs[0], false, true, grid_buffer_dev_1, grid_buffer_dev_2);
    }
  } else {
    fft_gpu_register_3d_local(true, npts_global, 1, false, false,
                              grid_buffer_dev_1, grid_buffer_dev_2);
    fft_gpu_register_3d_local(false, npts_global, 1, false, false,
                              grid_buffer_dev_1, grid_buffer_dev_2);
  }
}

/*******************************************************************************
 * \brief Registers all FFTs using a blocked distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_register_3d_gpu_r2c_blocked(const int npts_global[3],
                                     const int npts_global_gspace[3],
                                     int (*proc2local_rs)[3][2],
                                     int (*proc2local_ms)[3][2],
                                     int (*proc2local_gs)[3][2],
                                     cp_mpi_comm_t comm,
                                     const cp_mpi_comm_t sub_comm[2]) {
  (void)sub_comm;

  const int my_process = cp_mpi_comm_rank(comm);

  double complex *grid_buffer_1 = get_buffer_1();
  double complex *grid_buffer_2 = get_buffer_2();
  double complex *grid_buffer_dev_1 = fft_gpu_get_buffer_1();
  double complex *grid_buffer_dev_2 = fft_gpu_get_buffer_2();
  void *fft_stream = fft_gpu_get_stream();

  int (*my_bounds_rs)[2] = proc2local_rs[my_process];
  int (*my_bounds_ms)[2] = proc2local_ms[my_process];
  int (*my_bounds_gs)[2] = proc2local_gs[my_process];

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {my_bounds_rs[0][1], my_bounds_rs[1][1],
                         my_bounds_rs[2][1]};
  int fft_sizes_ms[3] = {my_bounds_ms[0][1], my_bounds_ms[1][1],
                         my_bounds_ms[2][1]};
  int fft_sizes_gs[3] = {my_bounds_gs[0][1], my_bounds_gs[1][1],
                         my_bounds_gs[2][1]};

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  cp_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  // We use different data distribution schemes depending on the availability of
  // a distributed FFT library because FFTW requires the data to the different
  // FFTs to be consecutively stored in memory. This is not possible without a
  // distributed FFT library because this would require the implementation of
  // the Guru interface which is not available with all implementations of the
  // FFTW interface
  if (proc_grid[0] > 1 && proc_grid[1] > 1) {
    if (fft_sizes_rs[2] > 0) {
      // Perform the first FFT

      fft_gpu_register_1d_r2c_local(
          npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], true, false,
          fft_sizes_rs[1] * fft_sizes_rs[2], npts_global[0] / 2 + 1,
          (double *)grid_buffer_dev_1, grid_buffer_dev_2);
      fft_gpu_register_1d_c2r_local(
          npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], false, true,
          npts_global[0] / 2 + 1, fft_sizes_rs[1] * fft_sizes_rs[2],
          grid_buffer_dev_1, (double *)grid_buffer_dev_2);

      // Perform the second FFT
      fft_gpu_register_1d_local(
          true, npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2], true, false,
          fft_sizes_ms[0] * fft_sizes_ms[2], npts_global[1], grid_buffer_dev_2,
          grid_buffer_dev_1);
      fft_gpu_register_1d_local(
          false, npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2], false, true,
          npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2], grid_buffer_dev_2,
          grid_buffer_dev_1);
    }
    fft_gpu_register_1d_local(true, npts_global[2],
                              fft_sizes_gs[0] * fft_sizes_gs[1], true, false,
                              fft_sizes_gs[0] * fft_sizes_gs[1], npts_global[2],
                              grid_buffer_dev_1, grid_buffer_dev_2);
    fft_gpu_register_1d_local(false, npts_global[2],
                              fft_sizes_gs[0] * fft_sizes_gs[1], false, true,
                              npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1],
                              grid_buffer_dev_1, grid_buffer_dev_2);
  } else if (proc_grid[0] > 1) {
    cp_assert(fft_sizes_rs[1] == npts_global[1]);

    // Perform the first FFT
    fft_gpu_register_1d_r2c_local(
        npts_global[0], npts_global[1] * fft_sizes_rs[2], true, false,
        npts_global[1] * fft_sizes_rs[2], npts_global[0] / 2 + 1,
        (double *)grid_buffer_dev_1, grid_buffer_dev_2);
    fft_gpu_register_1d_c2r_local(
        npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], false, true,
        npts_global[0] / 2 + 1, fft_sizes_rs[1] * fft_sizes_rs[2],
        grid_buffer_dev_2, (double *)grid_buffer_dev_1);
    // Perform the second FFT and one transposition (y,z_d,x) -> (x,y,z_d)
    fft_gpu_register_1d_local(
        true, npts_global[1], npts_global_gspace[0] * fft_sizes_rs[2], true,
        false, npts_global_gspace[0] * fft_sizes_rs[2], npts_global[1],
        grid_buffer_dev_2, grid_buffer_dev_1);
    fft_gpu_register_1d_local(false, npts_global[1],
                              fft_sizes_ms[0] * fft_sizes_ms[2], false, true,
                              npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2],
                              grid_buffer_dev_2, grid_buffer_dev_1);

    fft_gpu_register_1d_local(true, npts_global[2],
                              fft_sizes_gs[0] * fft_sizes_gs[1], true, false,
                              fft_sizes_gs[0] * fft_sizes_gs[1], npts_global[2],
                              grid_buffer_dev_1, grid_buffer_dev_2);
    fft_gpu_register_1d_local(false, npts_global[2],
                              fft_sizes_gs[0] * fft_sizes_gs[1], false, true,
                              npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1],
                              grid_buffer_dev_1, grid_buffer_dev_2);

  } else if (proc_grid[1] > 1) {
    if (fft_sizes_rs[2] > 0) {

      // Perform the first FFT in z-direction (x_d,y_d,z)->(z,x_d,y_d)
      fft_gpu_register_2d_local(
          true, (const int[2]){npts_global[1], npts_global[2]}, fft_sizes_gs[0],
          true, false, grid_buffer_dev_2, grid_buffer_dev_1);
      fft_gpu_register_2d_local(
          false, (const int[2]){npts_global[1], npts_global[2]},
          fft_sizes_gs[0], false, true, grid_buffer_dev_1, grid_buffer_dev_2);

      // Perform the third FFT and one transposition (y,x,z)->(z,y,x)
      fft_gpu_register_1d_r2c_local(
          npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], true, false,
          fft_sizes_rs[1] * fft_sizes_rs[2], npts_global[0] / 2 + 1,
          (double *)grid_buffer_dev_1, grid_buffer_dev_2);
      fft_gpu_register_1d_c2r_local(
          npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], false, true,
          npts_global[0] / 2 + 1, fft_sizes_rs[1] * fft_sizes_rs[2],
          grid_buffer_dev_2, (double *)grid_buffer_dev_1);
    }
  } else {

    // first FFT (x,y,z) -> (x,y,z)
    fft_gpu_register_1d_r2c_local(
        npts_global[0], npts_global[1] * npts_global[2], true, true,
        npts_global[1] * npts_global[2], npts_global[0] / 2 + 1,
        (double *)grid_buffer_dev_2, grid_buffer_dev_1);
    fft_gpu_register_1d_c2r_local(
        npts_global[0], npts_global[1] * npts_global[2], true, true,
        npts_global[1] * npts_global[2], npts_global[1] * npts_global[2],
        grid_buffer_dev_2, (double *)grid_buffer_dev_1);
    // second FFT (x,y,z) -> (x,y,z)
    fft_gpu_register_2d_local(true,
                              (const int[2]){npts_global[1], npts_global[2]},
                              npts_global_gspace[0], false, false,
                              grid_buffer_dev_1, grid_buffer_dev_2);
    fft_gpu_register_2d_local(false,
                              (const int[2]){npts_global[1], npts_global[2]},
                              npts_global_gspace[0], false, false,
                              grid_buffer_dev_1, grid_buffer_dev_2);
  }
}

/*******************************************************************************
 * \brief Registers all FFTs using a ray distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_register_3d_gpu_ray(const int npts_global[3],
                             int (*proc2local_rs)[3][2],
                             int (*proc2local_ms)[3][2], int *rays_per_process,
                             cp_mpi_comm_t comm,
                             const cp_mpi_comm_t sub_comm[2]) {
  (void)sub_comm;

  const int my_process = cp_mpi_comm_rank(comm);

  double complex *grid_buffer_1 = get_buffer_1();
  double complex *grid_buffer_2 = get_buffer_2();
  double complex *grid_buffer_dev_1 = fft_gpu_get_buffer_1();
  double complex *grid_buffer_dev_2 = fft_gpu_get_buffer_2();
  void *fft_stream = fft_gpu_get_stream();

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {proc2local_rs[my_process][0][1],
                         proc2local_rs[my_process][1][1],
                         proc2local_rs[my_process][2][1]};
  int fft_sizes_ms[3] = {proc2local_ms[my_process][0][1],
                         proc2local_ms[my_process][1][1],
                         proc2local_ms[my_process][2][1]};
  int number_of_local_xy_rays = rays_per_process[my_process];

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  cp_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  // We use different data distribution schemes depending on the
  // availability of a distributed FFT library because FFTW requires the
  // data to the different FFTs to be consecutively stored in memory. This
  // is not possible without a distributed FFT library because this would
  // require the implementation of the Guru interface which is not available
  // with all implementations of the FFTW interface
  if (proc_grid[0] > 1 && proc_grid[1] > 1) {

    // Perform the first FFT (x,y_d,z_d) -> (y_d,z_d,x)
    fft_gpu_register_1d_local(true, npts_global[0],
                              fft_sizes_rs[1] * fft_sizes_rs[2], true, false,
                              fft_sizes_rs[1] * fft_sizes_rs[2], npts_global[0],
                              grid_buffer_dev_2, grid_buffer_dev_1);
    fft_gpu_register_1d_local(false, npts_global[0],
                              fft_sizes_rs[1] * fft_sizes_rs[2], false, true,
                              npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2],
                              grid_buffer_dev_2, grid_buffer_dev_1);

    // Perform the second FFT (y,z_d,x_d) -> (z_d,x_d,y)
    fft_gpu_register_1d_local(true, npts_global[1],
                              fft_sizes_ms[0] * fft_sizes_ms[2], true, false,
                              fft_sizes_ms[0] * fft_sizes_ms[2], npts_global[1],
                              grid_buffer_dev_1, grid_buffer_dev_2);
    fft_gpu_register_1d_local(false, npts_global[1],
                              fft_sizes_ms[0] * fft_sizes_ms[2], false, true,
                              npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2],
                              grid_buffer_dev_1, grid_buffer_dev_2);

    // Perform the third FFT (z,xy_d) -> (xy_d,z)
    fft_gpu_register_1d_local(true, npts_global[2], number_of_local_xy_rays,
                              false, false, npts_global[2], npts_global[2],
                              grid_buffer_dev_1, grid_buffer_dev_2);
    fft_gpu_register_1d_local(false, npts_global[2], number_of_local_xy_rays,
                              false, false, npts_global[2], npts_global[2],
                              grid_buffer_dev_1, grid_buffer_dev_2);
  } else if (proc_grid[0] > 1) {
    // Now, we only need a local 2D FFT (x,y,z_d) -> (z_d,x,y)
    fft_gpu_register_2d_local(
        true, (const int[2]){npts_global[0], npts_global[1]}, fft_sizes_ms[2],
        true, false, grid_buffer_dev_1, grid_buffer_dev_2);
    fft_gpu_register_2d_local(
        false, (const int[2]){npts_global[0], npts_global[1]}, fft_sizes_rs[2],
        false, true, grid_buffer_dev_2, grid_buffer_dev_1);

    // Perform the third FFT (z,xy_d) -> (xy_d,z)
    fft_gpu_register_1d_local(true, npts_global[2], number_of_local_xy_rays,
                              false, false, npts_global[2], npts_global[2],
                              grid_buffer_dev_1, grid_buffer_dev_2);
    fft_gpu_register_1d_local(false, npts_global[2], number_of_local_xy_rays,
                              false, false, npts_global[2], npts_global[2],
                              grid_buffer_dev_1, grid_buffer_dev_2);
  } else if (proc_grid[1] > 1) {

    // Perform the first FFT (x,y_d,z_d) -> (y_d,z_d,x)
    fft_gpu_register_1d_local(true, npts_global[0],
                              fft_sizes_rs[1] * fft_sizes_rs[2], true, false,
                              fft_sizes_rs[1] * fft_sizes_rs[2], npts_global[0],
                              grid_buffer_dev_2, grid_buffer_dev_1);
    fft_gpu_register_1d_local(false, npts_global[0],
                              fft_sizes_rs[1] * fft_sizes_rs[2], false, true,
                              npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2],
                              grid_buffer_dev_2, grid_buffer_dev_1);

    // Perform the second FFT (y,z_d,x_d) -> (z_d,x_d,y)
    fft_gpu_register_1d_local(true, npts_global[1],
                              fft_sizes_ms[0] * fft_sizes_ms[2], true, false,
                              fft_sizes_ms[0] * fft_sizes_ms[2], npts_global[1],
                              grid_buffer_dev_1, grid_buffer_dev_2);
    fft_gpu_register_1d_local(false, npts_global[1],
                              fft_sizes_ms[0] * fft_sizes_ms[2], false, true,
                              npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2],
                              grid_buffer_dev_1, grid_buffer_dev_2);

    // Perform the third FFT (z,xy_d) -> (xy_d,z)
    fft_gpu_register_1d_local(true, npts_global[2], number_of_local_xy_rays,
                              false, false, npts_global[2], npts_global[2],
                              grid_buffer_dev_1, grid_buffer_dev_2);
    fft_gpu_register_1d_local(false, npts_global[2], number_of_local_xy_rays,
                              false, false, npts_global[2], npts_global[2],
                              grid_buffer_dev_1, grid_buffer_dev_2);
  } else {
    fft_gpu_register_3d_local(true, npts_global, 1, false, false,
                              grid_buffer_dev_2, grid_buffer_dev_1);
    fft_gpu_register_3d_local(false, npts_global, 1, false, false,
                              grid_buffer_dev_2, grid_buffer_dev_1);
  }
}

/*******************************************************************************
 * \brief Registers all FFTs using a ray distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_register_3d_gpu_r2c_ray(const int npts_global[3],
                                 const int npts_global_gspace[3],
                                 int (*proc2local_rs)[3][2],
                                 int (*proc2local_ms)[3][2],
                                 int *rays_per_process, cp_mpi_comm_t comm,
                                 const cp_mpi_comm_t sub_comm[2]) {
  (void)sub_comm;

  const int my_process = cp_mpi_comm_rank(comm);

  double complex *grid_buffer_1 = get_buffer_1();
  double complex *grid_buffer_2 = get_buffer_2();
  double complex *grid_buffer_dev_1 = fft_gpu_get_buffer_1();
  double complex *grid_buffer_dev_2 = fft_gpu_get_buffer_2();
  void *fft_stream = fft_gpu_get_stream();

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {proc2local_rs[my_process][0][1],
                         proc2local_rs[my_process][1][1],
                         proc2local_rs[my_process][2][1]};
  int fft_sizes_ms[3] = {proc2local_ms[my_process][0][1],
                         proc2local_ms[my_process][1][1],
                         proc2local_ms[my_process][2][1]};
  int number_of_local_xy_rays = rays_per_process[my_process];

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  cp_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  // We use different data distribution schemes depending on the
  // availability of a distributed FFT library because FFTW requires the
  // data to the different FFTs to be consecutively stored in memory. This
  // is not possible without a distributed FFT library because this would
  // require the implementation of the Guru interface which is not available
  // with all implementations of the FFTW interface
  if (proc_grid[0] > 1 && proc_grid[1] > 1) {

    // Perform the first FFT (x,y_d,z_d) -> (y_d,z_d,x)
    fft_gpu_register_1d_r2c_local(
        npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], true, false,
        fft_sizes_rs[1] * fft_sizes_rs[2], npts_global[0] / 2 + 1,
        (double *)grid_buffer_dev_2, grid_buffer_dev_1);
    fft_gpu_register_1d_c2r_local(
        npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], false, true,
        npts_global[0] / 2 + 1, fft_sizes_rs[1] * fft_sizes_rs[2],
        grid_buffer_dev_2, (double *)grid_buffer_dev_1);

    // Perform the second FFT (y,z_d,x_d) -> (z_d,x_d,y)
    fft_gpu_register_1d_local(true, npts_global[1],
                              fft_sizes_ms[0] * fft_sizes_ms[2], true, false,
                              fft_sizes_ms[0] * fft_sizes_ms[2], npts_global[1],
                              grid_buffer_dev_1, grid_buffer_dev_2);
    fft_gpu_register_1d_local(false, npts_global[1],
                              fft_sizes_ms[0] * fft_sizes_ms[2], false, true,
                              npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2],
                              grid_buffer_dev_1, grid_buffer_dev_2);

    // Perform the third FFT (z,xy_d) -> (xy_d,z)
    fft_gpu_register_1d_local(true, npts_global[2], number_of_local_xy_rays,
                              false, false, npts_global[2], npts_global[2],
                              grid_buffer_dev_1, grid_buffer_dev_2);
    // Perform the first FFT (xy_d,z) -> (z,xy_d)
    fft_gpu_register_1d_local(false, npts_global[2], number_of_local_xy_rays,
                              false, false, npts_global[2], npts_global[2],
                              grid_buffer_dev_1, grid_buffer_dev_2);

  } else if (proc_grid[0] > 1) {

    // The first two FFTs can be performed locally
    // FFTs (x,y,z_d) -> (y,z_d,x)
    fft_gpu_register_1d_r2c_local(
        npts_global[0], npts_global[1] * fft_sizes_ms[2], true, false,
        npts_global[1] * fft_sizes_ms[2], npts_global[0] / 2 + 1,
        (double *)grid_buffer_dev_2, grid_buffer_dev_1);
    fft_gpu_register_1d_c2r_local(
        npts_global[0], npts_global[1] * fft_sizes_ms[2], false, true,
        npts_global[0] / 2 + 1, npts_global[1] * fft_sizes_ms[2],
        grid_buffer_dev_2, (double *)grid_buffer_dev_1);
    // second FFT (y,z_d,x) -> (z_d,x,y)
    fft_gpu_register_1d_local(
        true, npts_global[1], npts_global_gspace[0] * fft_sizes_ms[2], true,
        false, npts_global_gspace[0] * fft_sizes_ms[2], npts_global[1],
        grid_buffer_dev_1, grid_buffer_dev_2);
    fft_gpu_register_1d_local(
        false, npts_global[1], npts_global_gspace[0] * fft_sizes_ms[2], false,
        true, npts_global[1], npts_global_gspace[0] * fft_sizes_ms[2],
        grid_buffer_dev_1, grid_buffer_dev_2);

    // Perform the third FFT (z,xy_d) -> (xy_d,z)
    fft_gpu_register_1d_local(true, npts_global[2], number_of_local_xy_rays,
                              false, false, npts_global[2], npts_global[2],
                              grid_buffer_dev_1, grid_buffer_dev_2);
    fft_gpu_register_1d_local(false, npts_global[2], number_of_local_xy_rays,
                              false, false, npts_global[2], npts_global[2],
                              grid_buffer_dev_1, grid_buffer_dev_2);
  } else if (proc_grid[1] > 1) {

    // Perform the first FFT (x,y_d,z_d) -> (y_d,z_d,x)
    fft_gpu_register_1d_r2c_local(
        npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], true, false,
        fft_sizes_rs[1] * fft_sizes_rs[2], npts_global[0] / 2 + 1,
        (double *)grid_buffer_dev_2, grid_buffer_dev_1);
    fft_gpu_register_1d_c2r_local(
        npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], false, true,
        npts_global[0] / 2 + 1, fft_sizes_rs[1] * fft_sizes_rs[2],
        grid_buffer_dev_2, (double *)grid_buffer_dev_1);

    // Perform the second FFT (y,z_d,x_d) -> (z_d,x_d,y)
    fft_gpu_register_1d_local(true, npts_global[1],
                              fft_sizes_ms[0] * fft_sizes_ms[2], true, false,
                              fft_sizes_ms[0] * fft_sizes_ms[2], npts_global[1],
                              grid_buffer_dev_1, grid_buffer_dev_2);
    fft_gpu_register_1d_local(false, npts_global[1],
                              fft_sizes_ms[0] * fft_sizes_ms[2], false, true,
                              npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2],
                              grid_buffer_dev_1, grid_buffer_dev_2);

    // Perform the third FFT (y_d,z_d,x) -> (x,y_d,z_d)

    // Perform the third FFT (z,xy_d) -> (xy_d,z)
    fft_gpu_register_1d_local(true, npts_global[2], number_of_local_xy_rays,
                              false, false, npts_global[2], npts_global[2],
                              grid_buffer_dev_1, grid_buffer_dev_2);
    fft_gpu_register_1d_local(false, npts_global[2], number_of_local_xy_rays,
                              false, false, npts_global[2], npts_global[2],
                              grid_buffer_dev_1, grid_buffer_dev_2);

  } else {

    // FFTs (x,y,z_d) -> (y,z_d,x)
    fft_gpu_register_1d_r2c_local(
        npts_global[0], npts_global[1] * npts_global[2], true, true,
        npts_global[1] * npts_global[2], npts_global[1] * npts_global[2],
        (double *)grid_buffer_dev_1, grid_buffer_dev_2);
    fft_gpu_register_1d_c2r_local(
        npts_global[0], npts_global[1] * npts_global[2], true, true,
        npts_global[1] * npts_global[2], npts_global[1] * npts_global[2],
        grid_buffer_dev_1, (double *)grid_buffer_dev_2);
    // second FFT (y,z_d,x) -> (z_d,x,y)
    fft_gpu_register_2d_local(true,
                              (const int[2]){npts_global[1], npts_global[2]},
                              npts_global_gspace[0], false, false,
                              grid_buffer_dev_2, grid_buffer_dev_1);
    fft_gpu_register_2d_local(false,
                              (const int[2]){npts_global[1], npts_global[2]},
                              npts_global_gspace[0], false, false,
                              grid_buffer_dev_2, grid_buffer_dev_1);
  }
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a blocked distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_gpu_fw_blocked(const double complex *restrict grid_rs,
                           bool is_complex, double complex *restrict grid_gs,
                           int *index_to_cart, const int npts_gs_local,
                           const int npts_global[3], int (*proc2local_rs)[3][2],
                           int (*proc2local_ms)[3][2],
                           int (*proc2local_gs)[3][2],
                           int (*proc2local_x_gs)[2], int (*proc2local_y_gs)[2],
                           fft_redistribution_t *redistribution,
                           cp_mpi_comm_t comm,
                           const cp_mpi_comm_t sub_comm[2]) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "3d_fw_b");
  const int handle = fft_start_timer(routine_name);
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "3d_fw_b_%i_%i_%i_%i",
           npts_global[0], npts_global[1], npts_global[2],
           cp_mpi_comm_size(comm));
  const int handle2 = fft_start_timer(routine_name);

  const int my_process = cp_mpi_comm_rank(comm);

  double complex *grid_buffer_1 = get_buffer_1();
  double complex *grid_buffer_2 = get_buffer_2();
  double complex *grid_buffer_dev_1 = fft_gpu_get_buffer_1();
  double complex *grid_buffer_dev_2 = fft_gpu_get_buffer_2();
  void *fft_stream = fft_gpu_get_stream();

  int (*my_bounds_rs)[2] = proc2local_rs[my_process];
  int (*my_bounds_ms)[2] = proc2local_ms[my_process];
  int (*my_bounds_gs)[2] = proc2local_gs[my_process];

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {my_bounds_rs[0][1], my_bounds_rs[1][1],
                         my_bounds_rs[2][1]};
  int fft_sizes_ms[3] = {my_bounds_ms[0][1], my_bounds_ms[1][1],
                         my_bounds_ms[2][1]};
  int fft_sizes_gs[3] = {my_bounds_gs[0][1], my_bounds_gs[1][1],
                         my_bounds_gs[2][1]};

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  cp_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  // The total number of points may be larger than the largest representable
  // number of type int (large semiempirical systems) So, we need to convert the
  // lengths to long int first
  const long int number_of_points = ((long int)npts_global[0]) *
                                    ((long int)npts_global[1]) *
                                    ((long int)npts_global[2]);
  const double scaling_factor = 1.0 / ((double)number_of_points);
  const int number_of_points_to_scale =
      index_to_cart != NULL ? npts_gs_local : product3(fft_sizes_gs);
  const int stride_size = 1;

  // We use different data distribution schemes depending on the availability of
  // a distributed FFT library because FFTW requires the data to the different
  // FFTs to be consecutively stored in memory. This is not possible without a
  // distributed FFT library because this would require the implementation of
  // the Guru interface which is not available with all implementations of the
  // FFTW interface
  if (proc_grid[0] > 1 && proc_grid[1] > 1) {
    if (fft_sizes_rs[2] > 0) {
      // Perform the first FFT

      if (is_complex) {
        FFT_H2D(grid_buffer_dev_1, grid_rs,
                product3(fft_sizes_rs) * sizeof(double complex));
      } else {
        const int npts_real = product3(fft_sizes_rs);
        FFT_H2D(grid_buffer_dev_2, grid_rs, npts_real * sizeof(double));
        fft_gpu_launch_real_to_complex((const double *)grid_buffer_dev_2,
                                       (double *)grid_buffer_dev_1, npts_real,
                                       FFT_STREAM);
      }

      fft_gpu_1d_local_device(true, npts_global[0],
                              fft_sizes_rs[1] * fft_sizes_rs[2], true, false,
                              fft_sizes_rs[1] * fft_sizes_rs[2], npts_global[0],
                              grid_buffer_dev_1, grid_buffer_dev_2);

      // Pack buffers
      collect_y_and_distribute_x_blocked_pack_device(
          grid_buffer_dev_2, grid_buffer_dev_1, redistribution, proc2local_x_gs,
          fft_stream);

      // Communicate buffers
      offloadStreamSynchronize((offloadStream_t)fft_stream);
      collect_y_and_distribute_x_blocked_comm_device(
          grid_buffer_dev_1, grid_buffer_dev_2, redistribution, sub_comm[1]);

      // Perform the second FFT
      fft_gpu_2d_local_device(
          true, (const int[2]){npts_global[1], npts_global[2]}, fft_sizes_ms[0],
          true, false, grid_buffer_dev_2, grid_buffer_dev_1);
    }

    // Perform the third FFT
    if (index_to_cart != NULL) {
      FFT_D2H(grid_buffer_1, grid_buffer_dev_1,
              (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none) shared(                                 \
        npts_gs_local, index_to_cart, grid_gs, grid_buffer_1, scaling_factor)
      for (int index = 0; index < npts_gs_local; index++) {
        grid_gs[index] = scaling_factor * grid_buffer_1[index_to_cart[index]];
      }
    } else {
      FFT_D2H(grid_buffer_1, grid_buffer_dev_1,
              (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(number_of_points_to_scale, grid_gs, grid_buffer_1, scaling_factor)
      for (int index = 0; index < number_of_points_to_scale; index++) {
        grid_gs[index] = scaling_factor * grid_buffer_1[index];
      }
    }
  } else if (proc_grid[0] > 1) {
    cp_assert(fft_sizes_rs[1] == npts_global[1]);

    if (is_complex) {
      FFT_H2D(grid_buffer_dev_2, grid_rs,
              product3(fft_sizes_rs) * sizeof(double complex));
    } else {
      const int npts_real = product3(fft_sizes_rs);
      FFT_H2D(grid_buffer_dev_1, grid_rs, npts_real * sizeof(double));
      fft_gpu_launch_real_to_complex((const double *)grid_buffer_dev_1,
                                     (double *)grid_buffer_dev_2, npts_real,
                                     FFT_STREAM);
    }

    fft_gpu_2d_local_device(
        true, (const int[2]){npts_global[0], npts_global[1]}, fft_sizes_rs[2],
        true, false, grid_buffer_dev_2, grid_buffer_dev_1);

    // Perform second transpose
    collect_z_and_distribute_y_blocked_pack_device(
        grid_buffer_dev_1, grid_buffer_dev_2, redistribution, proc2local_y_gs,
        fft_stream);
    offloadStreamSynchronize((offloadStream_t)fft_stream);
    collect_z_and_distribute_y_blocked_comm_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, sub_comm[0]);

    // Perform the third FFT
    if (index_to_cart != NULL) {
      fft_gpu_1d_local_device(true, npts_global[2],
                              fft_sizes_gs[0] * fft_sizes_gs[1], true, false,
                              fft_sizes_gs[0] * fft_sizes_gs[1], npts_global[2],
                              grid_buffer_dev_1, grid_buffer_dev_2);

      FFT_D2H(grid_buffer_2, grid_buffer_dev_2,
              (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none) shared(                                 \
        npts_gs_local, index_to_cart, grid_gs, grid_buffer_2, scaling_factor)
      for (int index = 0; index < npts_gs_local; index++) {
        grid_gs[index] = scaling_factor * grid_buffer_2[index_to_cart[index]];
      }
    } else {
      fft_gpu_1d_local_device(true, npts_global[2],
                              fft_sizes_gs[0] * fft_sizes_gs[1], true, false,
                              fft_sizes_gs[0] * fft_sizes_gs[1], npts_global[2],
                              grid_buffer_dev_1, grid_buffer_dev_2);
      fft_gpu_launch_scale((double *)grid_buffer_dev_2, scaling_factor,
                           number_of_points_to_scale, FFT_STREAM);
      FFT_D2H(grid_gs, grid_buffer_dev_2,
              number_of_points_to_scale * sizeof(double complex));
    }

  } else if (proc_grid[1] > 1) {
    if (fft_sizes_rs[2] > 0) {
      // Perform the first FFT

      if (is_complex) {
        FFT_H2D(grid_buffer_dev_1, grid_rs,
                product3(fft_sizes_rs) * sizeof(double complex));
      } else {
        const int npts_real = product3(fft_sizes_rs);
        FFT_H2D(grid_buffer_dev_2, grid_rs, npts_real * sizeof(double));
        fft_gpu_launch_real_to_complex((const double *)grid_buffer_dev_2,
                                       (double *)grid_buffer_dev_1, npts_real,
                                       FFT_STREAM);
      }

      fft_gpu_1d_local_device(true, npts_global[0],
                              fft_sizes_rs[1] * fft_sizes_rs[2], true, false,
                              fft_sizes_rs[1] * fft_sizes_rs[2], npts_global[0],
                              grid_buffer_1, grid_buffer_2);

      // Pack buffers
      collect_y_and_distribute_x_blocked_pack_device(
          grid_buffer_dev_2, grid_buffer_dev_1, redistribution, proc2local_x_gs,
          fft_stream);

      // Communicate buffers
      offloadStreamSynchronize((offloadStream_t)fft_stream);
      collect_y_and_distribute_x_blocked_comm_device(
          grid_buffer_dev_1, grid_buffer_dev_2, redistribution, sub_comm[1]);

      // Perform the second FFT
      fft_gpu_2d_local_device(
          true, (const int[2]){npts_global[1], npts_global[2]}, fft_sizes_ms[0],
          true, false, grid_buffer_2, grid_buffer_1);
    }

    // Perform the third FFT
    if (index_to_cart != NULL) {
      FFT_D2H(grid_buffer_1, grid_buffer_dev_1,
              (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none) shared(                                 \
        npts_gs_local, index_to_cart, grid_gs, grid_buffer_1, scaling_factor)
      for (int index = 0; index < npts_gs_local; index++) {
        grid_gs[index] = scaling_factor * grid_buffer_1[index_to_cart[index]];
      }
    } else {
      FFT_D2H(grid_buffer_1, grid_buffer_dev_1,
              (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(number_of_points_to_scale, grid_gs, grid_buffer_1, scaling_factor)
      for (int index = 0; index < number_of_points_to_scale; index++) {
        grid_gs[index] = scaling_factor * grid_buffer_1[index];
      }
    }
  } else {
    if (is_complex) {
      FFT_H2D(grid_buffer_dev_1, grid_rs,
              product3(fft_sizes_rs) * sizeof(double complex));
    } else {
      const int npts_real = product3(fft_sizes_rs);
      FFT_H2D(grid_buffer_dev_2, grid_rs, npts_real * sizeof(double));
      fft_gpu_launch_real_to_complex((const double *)grid_buffer_dev_2,
                                     (double *)grid_buffer_dev_1, npts_real,
                                     FFT_STREAM);
    }

    if (index_to_cart != NULL) {
      fft_gpu_3d_local_device(true, npts_global, 1, false, false,
                              grid_buffer_dev_1, grid_buffer_dev_2);

      FFT_D2H(grid_buffer_2, grid_buffer_dev_2,
              (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none) shared(                                 \
        npts_gs_local, index_to_cart, grid_gs, grid_buffer_2, scaling_factor)
      for (int index = 0; index < npts_gs_local; index++) {
        grid_gs[index] = scaling_factor * grid_buffer_2[index_to_cart[index]];
      }
    } else {
      fft_gpu_3d_local_device(true, npts_global, 1, false, false,
                              grid_buffer_dev_1, grid_buffer_dev_2);
      fft_gpu_launch_scale((double *)grid_buffer_dev_2, scaling_factor,
                           number_of_points_to_scale, FFT_STREAM);
      FFT_D2H(grid_gs, grid_buffer_dev_2,
              number_of_points_to_scale * sizeof(double complex));
    }
  }

  fft_stop_timer(handle2);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a blocked distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_gpu_fw_r2c_blocked(
    const double *restrict grid_rs, double complex *restrict grid_gs,
    int (*index_to_cart_pos)[2], int (*index_to_cart_neg)[2],
    int number_of_positive_points, int number_of_negative_points,
    const int npts_global[3], const int npts_global_gspace[3],
    int (*proc2local_rs)[3][2], int (*proc2local_ms)[3][2],
    int (*proc2local_gs)[3][2], int (*proc2local_x_gs)[2],
    int (*proc2local_y_gs)[2], fft_redistribution_t *redistribution,
    cp_mpi_comm_t comm, const cp_mpi_comm_t sub_comm[2]) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "3d_fw_r2c_b");
  const int handle = fft_start_timer(routine_name);
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "3d_fw_r2c_b_%i_%i_%i_%i",
           npts_global[0], npts_global[1], npts_global[2],
           cp_mpi_comm_size(comm));
  const int handle2 = fft_start_timer(routine_name);

  const int my_process = cp_mpi_comm_rank(comm);

  double complex *grid_buffer_1 = get_buffer_1();
  double complex *grid_buffer_2 = get_buffer_2();
  double complex *grid_buffer_dev_1 = fft_gpu_get_buffer_1();
  double complex *grid_buffer_dev_2 = fft_gpu_get_buffer_2();
  void *fft_stream = fft_gpu_get_stream();

  int (*my_bounds_rs)[2] = proc2local_rs[my_process];
  int (*my_bounds_ms)[2] = proc2local_ms[my_process];
  int (*my_bounds_gs)[2] = proc2local_gs[my_process];

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {my_bounds_rs[0][1], my_bounds_rs[1][1],
                         my_bounds_rs[2][1]};
  int fft_sizes_ms[3] = {my_bounds_ms[0][1], my_bounds_ms[1][1],
                         my_bounds_ms[2][1]};
  int fft_sizes_gs[3] = {my_bounds_gs[0][1], my_bounds_gs[1][1],
                         my_bounds_gs[2][1]};

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  cp_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  // The total number of points may be larger than the largest representable
  // number of type int (large semiempirical systems) So, we need to convert the
  // lengths to long int first
  const long int number_of_points = ((long int)npts_global[0]) *
                                    ((long int)npts_global[1]) *
                                    ((long int)npts_global[2]);
  const double scaling_factor = 1.0 / ((double)number_of_points);
  const int number_of_points_to_scale =
      index_to_cart_pos != NULL
          ? number_of_negative_points + number_of_positive_points
          : product3(fft_sizes_gs);
  const int stride_size = 1;

  // We use different data distribution schemes depending on the availability of
  // a distributed FFT library because FFTW requires the data to the different
  // FFTs to be consecutively stored in memory. This is not possible without a
  // distributed FFT library because this would require the implementation of
  // the Guru interface which is not available with all implementations of the
  // FFTW interface
  if (proc_grid[0] > 1 && proc_grid[1] > 1) {
    if (fft_sizes_rs[2] > 0) {
      // Perform the first FFT
      FFT_H2D(grid_buffer_dev_1, grid_rs,
              product3(fft_sizes_rs) * sizeof(double));

      fft_gpu_1d_r2c_local_device(
          npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], true, false,
          fft_sizes_rs[1] * fft_sizes_rs[2], npts_global[0] / 2 + 1,
          (double *)grid_buffer_dev_1, grid_buffer_dev_2);

      // Pack buffer
      collect_y_and_distribute_x_blocked_pack_device(
          grid_buffer_dev_2, grid_buffer_dev_1, redistribution, proc2local_x_gs,
          fft_stream);

      // Communicate buffer
      offloadStreamSynchronize((offloadStream_t)fft_stream);
      collect_y_and_distribute_x_blocked_comm_device(
          grid_buffer_dev_1, grid_buffer_dev_2, redistribution, sub_comm[1]);

      // Perform the second FFT
      fft_gpu_1d_local_device(true, npts_global[1],
                              fft_sizes_ms[0] * fft_sizes_ms[2], true, false,
                              fft_sizes_ms[0] * fft_sizes_ms[2], npts_global[1],
                              grid_buffer_dev_2, grid_buffer_dev_1);

      // Pack the buffer
      collect_z_and_distribute_y_blocked_pack_device(
          grid_buffer_dev_1, grid_buffer_dev_2, redistribution, proc2local_y_gs,
          fft_stream);
    }

    // Exchange data
    offloadStreamSynchronize((offloadStream_t)fft_stream);
    collect_z_and_distribute_y_blocked_comm_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, sub_comm[0]);

    // Perform the third FFT
    if (index_to_cart_neg != NULL && index_to_cart_pos != NULL) {
      fft_gpu_1d_local_device(true, npts_global[2],
                              fft_sizes_gs[0] * fft_sizes_gs[1], true, false,
                              fft_sizes_gs[0] * fft_sizes_gs[1], npts_global[2],
                              grid_buffer_dev_1, grid_buffer_dev_2);

      FFT_D2H(grid_buffer_2, grid_buffer_dev_2,
              (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(number_of_positive_points, index_to_cart_pos, grid_gs,              \
               grid_buffer_2, scaling_factor)
      for (int index = 0; index < number_of_positive_points; index++) {
        grid_gs[index_to_cart_pos[index][0]] =
            scaling_factor * grid_buffer_2[index_to_cart_pos[index][1]];
      }
      FFT_D2H(grid_buffer_2, grid_buffer_dev_2,
              (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(number_of_negative_points, index_to_cart_neg, grid_gs,              \
               grid_buffer_2, scaling_factor)
      for (int index = 0; index < number_of_negative_points; index++) {
        grid_gs[index_to_cart_neg[index][0]] =
            scaling_factor * conj(grid_buffer_2[index_to_cart_neg[index][1]]);
      }
    } else {
      fft_gpu_1d_local_device(true, npts_global[2],
                              fft_sizes_gs[0] * fft_sizes_gs[1], true, false,
                              fft_sizes_gs[0] * fft_sizes_gs[1], npts_global[2],
                              grid_buffer_dev_1, grid_buffer_dev_2);
      fft_gpu_launch_scale((double *)grid_buffer_dev_2, scaling_factor,
                           number_of_points_to_scale, FFT_STREAM);
      FFT_D2H(grid_gs, grid_buffer_dev_2,
              number_of_points_to_scale * sizeof(double complex));
    }
  } else if (proc_grid[0] > 1) {
    cp_assert(fft_sizes_rs[1] == npts_global[1]);

    FFT_H2D(grid_buffer_dev_1, grid_rs,
            product3(fft_sizes_rs) * sizeof(double));

    // Perform the first FFT
    fft_gpu_1d_r2c_local_device(
        npts_global[0], npts_global[1] * fft_sizes_rs[2], true, false,
        npts_global[1] * fft_sizes_rs[2], npts_global[0] / 2 + 1,
        (double *)grid_buffer_dev_1, grid_buffer_dev_2);
    fft_gpu_1d_local_device(
        true, npts_global[1], npts_global_gspace[0] * fft_sizes_rs[2], true,
        false, npts_global_gspace[0] * fft_sizes_rs[2], npts_global[1],
        grid_buffer_dev_2, grid_buffer_dev_1);

    // Perform second transpose
    collect_z_and_distribute_y_blocked_pack_device(
        grid_buffer_dev_1, grid_buffer_dev_2, redistribution, proc2local_y_gs,
        fft_stream);
    offloadStreamSynchronize((offloadStream_t)fft_stream);
    collect_z_and_distribute_y_blocked_comm_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, sub_comm[0]);

    // Perform the third FFT
    if (index_to_cart_neg != NULL) {
      fft_gpu_1d_local_device(true, npts_global[2],
                              fft_sizes_gs[0] * fft_sizes_gs[1], true, false,
                              fft_sizes_gs[0] * fft_sizes_gs[1], npts_global[2],
                              grid_buffer_dev_1, grid_buffer_dev_2);

      FFT_D2H(grid_buffer_2, grid_buffer_dev_2,
              (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(number_of_positive_points, index_to_cart_pos, grid_gs,              \
               grid_buffer_2, scaling_factor)
      for (int index = 0; index < number_of_positive_points; index++) {
        grid_gs[index_to_cart_pos[index][0]] =
            scaling_factor * grid_buffer_2[index_to_cart_pos[index][1]];
      }
      FFT_D2H(grid_buffer_2, grid_buffer_dev_2,
              (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(number_of_negative_points, index_to_cart_neg, grid_gs,              \
               grid_buffer_2, scaling_factor)
      for (int index = 0; index < number_of_negative_points; index++) {
        grid_gs[index_to_cart_neg[index][0]] =
            scaling_factor * conj(grid_buffer_2[index_to_cart_neg[index][1]]);
      }
    } else {
      fft_gpu_1d_local_device(true, npts_global[2],
                              fft_sizes_gs[0] * fft_sizes_gs[1], true, false,
                              fft_sizes_gs[0] * fft_sizes_gs[1], npts_global[2],
                              grid_buffer_dev_1, grid_buffer_dev_2);
      fft_gpu_launch_scale((double *)grid_buffer_dev_2, scaling_factor,
                           number_of_points_to_scale, FFT_STREAM);
      FFT_D2H(grid_gs, grid_buffer_dev_2,
              number_of_points_to_scale * sizeof(double complex));
    }

  } else if (proc_grid[1] > 1) {
    double complex *grid_result = NULL;
    if (fft_sizes_rs[2] > 0) {
      // Perform the first FFT

      cp_assert(grid_rs != NULL);
      FFT_H2D(grid_buffer_dev_1, grid_rs,
              product3(fft_sizes_rs) * sizeof(double));

      fft_gpu_1d_r2c_local_device(
          npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], true, false,
          fft_sizes_rs[1] * fft_sizes_rs[2], npts_global[0] / 2 + 1,
          (double *)grid_buffer_dev_1, grid_buffer_dev_2);

      // Pack buffer
      collect_y_and_distribute_x_blocked_pack_device(
          grid_buffer_dev_2, grid_buffer_dev_1, redistribution, proc2local_x_gs,
          fft_stream);

      // Communicate buffer
      offloadStreamSynchronize((offloadStream_t)fft_stream);
      collect_y_and_distribute_x_blocked_comm_device(
          grid_buffer_dev_1, grid_buffer_dev_2, redistribution, sub_comm[1]);

      // Perform the second FFT
      fft_gpu_2d_local_device(
          true, (const int[2]){npts_global[1], npts_global[2]}, fft_sizes_gs[0],
          true, false, grid_buffer_dev_2, grid_buffer_dev_1);
      FFT_D2H(grid_buffer_1, grid_buffer_dev_1,
              (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
      grid_result = grid_buffer_1;
    }

    // Perform the third FFT
    if (index_to_cart_neg != NULL && index_to_cart_pos != NULL) {
#pragma omp parallel for default(none)                                         \
    shared(number_of_positive_points, index_to_cart_pos, grid_gs, grid_result, \
               scaling_factor)
      for (int index = 0; index < number_of_positive_points; index++) {
        grid_gs[index_to_cart_pos[index][0]] =
            scaling_factor * grid_result[index_to_cart_pos[index][1]];
      }
#pragma omp parallel for default(none)                                         \
    shared(number_of_negative_points, index_to_cart_neg, grid_gs, grid_result, \
               scaling_factor)
      for (int index = 0; index < number_of_negative_points; index++) {
        grid_gs[index_to_cart_neg[index][0]] =
            scaling_factor * conj(grid_result[index_to_cart_neg[index][1]]);
      }
    } else {
#pragma omp parallel for default(none)                                         \
    shared(number_of_points_to_scale, grid_gs, grid_result, scaling_factor)
      for (int index = 0; index < number_of_points_to_scale; index++) {
        grid_gs[index] = scaling_factor * grid_result[index];
      }
    }
  } else {
    double complex *grid_result = NULL;
    if (fft_sizes_rs[2] > 0) {
      // Perform the first FFT

      FFT_H2D(grid_buffer_dev_1, grid_rs,
              product3(fft_sizes_rs) * sizeof(double));

      fft_gpu_1d_r2c_local_device(
          npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], true, true,
          fft_sizes_rs[1] * fft_sizes_rs[2], fft_sizes_rs[1] * fft_sizes_rs[2],
          (double *)grid_buffer_dev_1, grid_buffer_dev_2);

      // Perform the second FFT
      fft_gpu_2d_local_device(
          true, (const int[2]){npts_global[1], npts_global[2]}, fft_sizes_gs[0],
          false, false, grid_buffer_dev_2, grid_buffer_dev_1);
      FFT_D2H(grid_buffer_1, grid_buffer_dev_1,
              (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
      grid_result = grid_buffer_1;
    }

    // Perform the third FFT
    if (index_to_cart_neg != NULL && index_to_cart_pos != NULL) {
#pragma omp parallel for default(none)                                         \
    shared(number_of_positive_points, index_to_cart_pos, grid_gs, grid_result, \
               scaling_factor)
      for (int index = 0; index < number_of_positive_points; index++) {
        grid_gs[index_to_cart_pos[index][0]] =
            scaling_factor * grid_result[index_to_cart_pos[index][1]];
      }
#pragma omp parallel for default(none)                                         \
    shared(number_of_negative_points, index_to_cart_neg, grid_gs, grid_result, \
               scaling_factor)
      for (int index = 0; index < number_of_negative_points; index++) {
        grid_gs[index_to_cart_neg[index][0]] =
            scaling_factor * conj(grid_result[index_to_cart_neg[index][1]]);
      }
    } else {
#pragma omp parallel for default(none)                                         \
    shared(number_of_points_to_scale, grid_gs, grid_result, scaling_factor)
      for (int index = 0; index < number_of_points_to_scale; index++) {
        grid_gs[index] = scaling_factor * grid_result[index];
      }
    }
  }

  fft_stop_timer(handle2);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT using a blocked distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_gpu_bw_blocked(const double complex *restrict grid_gs,
                           int *index_to_cart, int number_of_local_gpts,
                           double complex *restrict grid_rs, bool is_complex,
                           const int npts_global[3], int (*proc2local_rs)[3][2],
                           int (*proc2local_ms)[3][2],
                           int (*proc2local_gs)[3][2],
                           int (*proc2local_x_gs)[2], int (*proc2local_y_gs)[2],
                           fft_redistribution_t *redistribution,
                           cp_mpi_comm_t comm,
                           const cp_mpi_comm_t sub_comm[2]) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "3d_bw_b");
  const int handle = fft_start_timer(routine_name);
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "3d_bw_b_%i_%i_%i_%i",
           npts_global[0], npts_global[1], npts_global[2],
           cp_mpi_comm_size(comm));
  const int handle2 = fft_start_timer(routine_name);

  const int my_process = cp_mpi_comm_rank(comm);

  double complex *grid_buffer_1 = get_buffer_1();
  double complex *grid_buffer_2 = get_buffer_2();
  double complex *grid_buffer_dev_1 = fft_gpu_get_buffer_1();
  double complex *grid_buffer_dev_2 = fft_gpu_get_buffer_2();
  void *fft_stream = fft_gpu_get_stream();

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {proc2local_rs[my_process][0][1],
                         proc2local_rs[my_process][1][1],
                         proc2local_rs[my_process][2][1]};
  int fft_sizes_ms[3] = {proc2local_ms[my_process][0][1],
                         proc2local_ms[my_process][1][1],
                         proc2local_ms[my_process][2][1]};
  int fft_sizes_gs[3] = {proc2local_gs[my_process][0][1],
                         proc2local_gs[my_process][1][1],
                         proc2local_gs[my_process][2][1]};

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  cp_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  // We use different data distribution schemes depending on the availability of
  // a distributed FFT library because FFTW requires the data to the different
  // FFTs to be consecutively stored in memory. This is not possible without a
  // distributed FFT library because this would require the implementation of
  // the Guru interface which is not available with all implementations of the
  // FFTW interface
  if (proc_grid[0] > 1 && proc_grid[1] > 1) {
    if (index_to_cart != NULL) {
      memset(grid_buffer_1, 0, product3(fft_sizes_gs) * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(number_of_local_gpts, index_to_cart, grid_buffer_1, grid_gs)
      for (int i = 0; i < number_of_local_gpts; i++) {
        grid_buffer_1[index_to_cart[i]] = grid_gs[i];
      }
    } else {
      memcpy(grid_buffer_1, grid_gs,
             product3(fft_sizes_gs) * sizeof(double complex));
    }
    FFT_H2D(grid_buffer_dev_1, grid_buffer_1,
            product3(fft_sizes_gs) * sizeof(double complex));

    // Perform the first FFT in z-direction
    fft_gpu_1d_local_device(false, npts_global[2],
                            fft_sizes_gs[0] * fft_sizes_gs[1], false, true,
                            npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1],
                            grid_buffer_dev_1, grid_buffer_dev_2);

    // Perform second redistribution and transpose
    // (z,x_d,y_d) -> (x_d,y,z_d)
    offloadStreamSynchronize((offloadStream_t)fft_stream);
    collect_y_and_distribute_z_blocked_comm_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, sub_comm[0]);

    if (fft_sizes_rs[2] > 0) {

      collect_y_and_distribute_z_blocked_unpack_device(
          grid_buffer_dev_1, grid_buffer_dev_2, redistribution, proc2local_y_gs,
          fft_stream);

      // Perform the second FFT and one transposition (z_D,x_D,y)->(y,z_D,x_D)
      fft_gpu_1d_local_device(false, npts_global[1],
                              fft_sizes_ms[0] * fft_sizes_ms[2], false, true,
                              npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2],
                              grid_buffer_dev_2, grid_buffer_dev_1);

      // Collect data in z-direction and distribute y-direction
      offloadStreamSynchronize((offloadStream_t)fft_stream);
      collect_x_and_distribute_y_blocked_comm_device(
          grid_buffer_dev_1, grid_buffer_dev_2, redistribution, sub_comm[1]);

      // Collect data in z-direction and distribute y-direction
      collect_x_and_distribute_y_blocked_unpack_device(
          grid_buffer_dev_2, grid_buffer_dev_1, redistribution, proc2local_x_gs,
          fft_stream);

      // Perform the third FFT and one transposition (y_D,z_D,x)->(x,y_D,z_D)
      fft_gpu_1d_local_device(false, npts_global[0],
                              fft_sizes_rs[1] * fft_sizes_rs[2], false, true,
                              npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2],
                              grid_buffer_dev_1, grid_buffer_dev_2);
      if (is_complex) {
        FFT_D2H(grid_rs, grid_buffer_dev_2,
                product3(fft_sizes_rs) * sizeof(double complex));
      }

      if (!is_complex) {
        double *grid_rs_double = (double *)grid_rs;
        FFT_D2H(grid_buffer_2, grid_buffer_dev_2,
                (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(fft_sizes_rs, grid_rs_double, grid_buffer_2)
        for (int i = 0; i < product3(fft_sizes_rs); i++)
          grid_rs_double[i] = creal(grid_buffer_2[i]);
      }
    }
  } else if (proc_grid[0] > 1) {

    if (index_to_cart != NULL) {
      memset(grid_buffer_1, 0, product3(fft_sizes_gs) * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(number_of_local_gpts, index_to_cart, grid_buffer_1, grid_gs)
      for (int i = 0; i < number_of_local_gpts; i++) {
        grid_buffer_1[index_to_cart[i]] = grid_gs[i];
      }
    } else {
      memcpy(grid_buffer_1, grid_gs,
             product3(fft_sizes_gs) * sizeof(double complex));
    }
    FFT_H2D(grid_buffer_dev_1, grid_buffer_1,
            product3(fft_sizes_gs) * sizeof(double complex));

    // Perform the first FFT and one transposition (x,y_d,z)->(z,x,y_d)
    fft_gpu_1d_local_device(false, npts_global[2],
                            fft_sizes_gs[0] * fft_sizes_gs[1], false, true,
                            npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1],
                            grid_buffer_dev_1, grid_buffer_dev_2);

    // Collect data in y-direction and distribute x-direction
    // (z,x,y_d)->(z_d,x,y)
    offloadStreamSynchronize((offloadStream_t)fft_stream);
    collect_y_and_distribute_z_blocked_comm_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, sub_comm[0]);
    collect_y_and_distribute_z_blocked_unpack_device(
        grid_buffer_dev_1, grid_buffer_dev_2, redistribution, proc2local_y_gs,
        fft_stream);

    // Perform the second FFT and one transposition (z_d,x,y)->(x,y,z_d)

    fft_gpu_2d_local_device(
        false, (const int[2]){npts_global[0], npts_global[1]}, fft_sizes_ms[2],
        false, true, grid_buffer_dev_2, grid_buffer_dev_1);
    if (is_complex) {
      FFT_D2H(grid_rs, grid_buffer_dev_1,
              product3(fft_sizes_rs) * sizeof(double complex));
    }

    if (!is_complex) {
      double *grid_rs_double = (double *)grid_rs;
      FFT_D2H(grid_buffer_1, grid_buffer_dev_1,
              (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(fft_sizes_rs, grid_rs_double, grid_buffer_1)
      for (int i = 0; i < product3(fft_sizes_rs); i++)
        grid_rs_double[i] = creal(grid_buffer_1[i]);
    }

  } else if (proc_grid[1] > 1) {
    if (index_to_cart != NULL) {
      memset(grid_buffer_1, 0, product3(fft_sizes_gs) * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(number_of_local_gpts, index_to_cart, grid_buffer_1, grid_gs)
      for (int i = 0; i < number_of_local_gpts; i++) {
        grid_buffer_1[index_to_cart[i]] = grid_gs[i];
      }
    } else {
      memcpy(grid_buffer_1, grid_gs,
             product3(fft_sizes_gs) * sizeof(double complex));
    }
    FFT_H2D(grid_buffer_dev_1, grid_buffer_1,
            product3(fft_sizes_gs) * sizeof(double complex));

    if (fft_sizes_rs[2] > 0) {

      // Perform the first FFT in z-direction
      fft_gpu_2d_local_device(
          false, (const int[2]){npts_global[1], npts_global[2]},
          fft_sizes_gs[0], false, true, grid_buffer_dev_1, grid_buffer_dev_2);

      // Collect data in z-direction and distribute y-direction
      offloadStreamSynchronize((offloadStream_t)fft_stream);
      collect_x_and_distribute_y_blocked_comm_device(
          grid_buffer_dev_2, grid_buffer_dev_1, redistribution, sub_comm[1]);

      // Collect data in z-direction and distribute y-direction
      collect_x_and_distribute_y_blocked_unpack_device(
          grid_buffer_dev_1, grid_buffer_dev_2, redistribution, proc2local_x_gs,
          fft_stream);

      // Perform the third FFT and one transposition (y_D,z_D,x)->(x,y_D,z_D)

      fft_gpu_1d_local_device(false, npts_global[0],
                              fft_sizes_rs[1] * fft_sizes_rs[2], false, true,
                              npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2],
                              grid_buffer_dev_2, grid_buffer_dev_1);
      if (is_complex) {
        FFT_D2H(grid_rs, grid_buffer_dev_1,
                product3(fft_sizes_rs) * sizeof(double complex));
      }

      if (!is_complex) {
        double *grid_rs_double = (double *)grid_rs;
        FFT_D2H(grid_buffer_1, grid_buffer_dev_1,
                (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(fft_sizes_rs, grid_rs_double, grid_buffer_1)
        for (int i = 0; i < product3(fft_sizes_rs); i++)
          grid_rs_double[i] = creal(grid_buffer_1[i]);
      }
    }
  } else {
    if (index_to_cart != NULL) {
      memset(grid_buffer_1, 0, product3(fft_sizes_gs) * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(number_of_local_gpts, index_to_cart, grid_buffer_1, grid_gs)
      for (int i = 0; i < number_of_local_gpts; i++) {
        grid_buffer_1[index_to_cart[i]] = grid_gs[i];
      }
    } else {
      memcpy(grid_buffer_1, grid_gs,
             product3(fft_sizes_gs) * sizeof(double complex));
    }
    FFT_H2D(grid_buffer_dev_1, grid_buffer_1,
            product3(fft_sizes_gs) * sizeof(double complex));

    fft_gpu_3d_local_device(false, npts_global, 1, false, false,
                            grid_buffer_dev_1, grid_buffer_dev_2);
    if (is_complex) {
      FFT_D2H(grid_rs, grid_buffer_dev_2,
              product3(fft_sizes_rs) * sizeof(double complex));
    }

    if (!is_complex) {
      double *grid_rs_double = (double *)grid_rs;
      FFT_D2H(grid_buffer_2, grid_buffer_dev_2,
              (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(fft_sizes_rs, grid_rs_double, grid_buffer_2)
      for (int i = 0; i < product3(fft_sizes_rs); i++)
        grid_rs_double[i] = creal(grid_buffer_2[i]);
    }
  }

  fft_stop_timer(handle2);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT using a blocked distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_gpu_bw_c2r_blocked(
    const double complex *restrict grid_gs, int (*index_to_cart_pos)[2],
    int number_of_positive_points, double *restrict grid_rs,
    const int npts_global[3], const int npts_global_gspace[3],
    int (*proc2local_rs)[3][2], int (*proc2local_ms)[3][2],
    int (*proc2local_gs)[3][2], int (*proc2local_x_gs)[2],
    int (*proc2local_y_gs)[2], fft_redistribution_t *redistribution,
    cp_mpi_comm_t comm, const cp_mpi_comm_t sub_comm[2]) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "3d_bw_c2r_b");
  const int handle = fft_start_timer(routine_name);
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "3d_bw_c2r_b_%i_%i_%i_%i",
           npts_global[0], npts_global[1], npts_global[2],
           cp_mpi_comm_size(comm));
  const int handle2 = fft_start_timer(routine_name);

  const int my_process = cp_mpi_comm_rank(comm);

  double complex *grid_buffer_1 = get_buffer_1();
  double complex *grid_buffer_2 = get_buffer_2();
  double complex *grid_buffer_dev_1 = fft_gpu_get_buffer_1();
  double complex *grid_buffer_dev_2 = fft_gpu_get_buffer_2();
  void *fft_stream = fft_gpu_get_stream();

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {proc2local_rs[my_process][0][1],
                         proc2local_rs[my_process][1][1],
                         proc2local_rs[my_process][2][1]};
  int fft_sizes_ms[3] = {proc2local_ms[my_process][0][1],
                         proc2local_ms[my_process][1][1],
                         proc2local_ms[my_process][2][1]};
  int fft_sizes_gs[3] = {proc2local_gs[my_process][0][1],
                         proc2local_gs[my_process][1][1],
                         proc2local_gs[my_process][2][1]};

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  cp_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  if (index_to_cart_pos != NULL) {
    memset(grid_buffer_1, 0, product3(fft_sizes_gs) * sizeof(double complex));
#pragma omp parallel for default(none) shared(                                 \
        number_of_positive_points, index_to_cart_pos, grid_gs, grid_buffer_1)
    for (int i = 0; i < number_of_positive_points; i++) {
      grid_buffer_1[index_to_cart_pos[i][1]] = grid_gs[index_to_cart_pos[i][0]];
    }
  } else {
    memcpy(grid_buffer_1, grid_gs,
           product3(fft_sizes_gs) * sizeof(double complex));
  }
  FFT_H2D(grid_buffer_dev_1, grid_buffer_1,
          product3(fft_sizes_gs) * sizeof(double complex));

  // We use different data distribution schemes depending on the availability of
  // a distributed FFT library because FFTW requires the data to the different
  // FFTs to be consecutively stored in memory. This is not possible without a
  // distributed FFT library because this would require the implementation of
  // the Guru interface which is not available with all implementations of the
  // FFTW interface
  if (proc_grid[0] > 1 && proc_grid[1] > 1) {
    if (index_to_cart_pos != NULL) {
      memset(grid_buffer_1, 0, product3(fft_sizes_gs) * sizeof(double complex));
#pragma omp parallel for default(none) shared(                                 \
        number_of_positive_points, index_to_cart_pos, grid_gs, grid_buffer_1)
      for (int i = 0; i < number_of_positive_points; i++) {
        grid_buffer_1[index_to_cart_pos[i][1]] =
            grid_gs[index_to_cart_pos[i][0]];
      }
    } else {
      memcpy(grid_buffer_1, grid_gs,
             product3(fft_sizes_gs) * sizeof(double complex));
    }
    FFT_H2D(grid_buffer_dev_1, grid_buffer_1,
            product3(fft_sizes_gs) * sizeof(double complex));

    // Perform the first FFT in z-direction (x_d,y_d,z)->(z,x_d,y_d)
    fft_gpu_1d_local_device(false, npts_global[2],
                            fft_sizes_gs[0] * fft_sizes_gs[1], false, true,
                            npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1],
                            grid_buffer_dev_1, grid_buffer_dev_2);

    // Perform second redistribution and transpose
    // (z,x_d,y_d) -> (x_d,y,z_d)
    offloadStreamSynchronize((offloadStream_t)fft_stream);
    collect_y_and_distribute_z_blocked_comm_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, sub_comm[0]);

    if (fft_sizes_rs[2] > 0) {

      collect_y_and_distribute_z_blocked_unpack_device(
          grid_buffer_dev_1, grid_buffer_dev_2, redistribution, proc2local_y_gs,
          fft_stream);

      // Perform the second FFT and one transposition (x,z,y)->(y,x,z)
      fft_gpu_1d_local_device(false, npts_global[1],
                              fft_sizes_ms[0] * fft_sizes_ms[2], false, true,
                              npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2],
                              grid_buffer_dev_2, grid_buffer_dev_1);

      // Collect data in z-direction and distribute y-direction
      offloadStreamSynchronize((offloadStream_t)fft_stream);
      collect_x_and_distribute_y_blocked_comm_device(
          grid_buffer_dev_1, grid_buffer_dev_2, redistribution, sub_comm[1]);
      collect_x_and_distribute_y_blocked_unpack_device(
          grid_buffer_dev_2, grid_buffer_dev_1, redistribution, proc2local_x_gs,
          fft_stream);

      // Perform the third FFT and one transposition (y,x,z)->(z,y,x)
      FFT_D2H(grid_buffer_1, grid_buffer_dev_1,
              (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
      fft_gpu_1d_c2r_local(npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2],
                           false, true, npts_global[0] / 2 + 1,
                           fft_sizes_rs[1] * fft_sizes_rs[2], grid_buffer_1,
                           grid_rs);
    }
  } else if (proc_grid[0] > 1) {

    if (index_to_cart_pos != NULL) {
      memset(grid_buffer_1, 0, product3(fft_sizes_gs) * sizeof(double complex));
#pragma omp parallel for default(none) shared(                                 \
        number_of_positive_points, index_to_cart_pos, grid_gs, grid_buffer_1)
      for (int i = 0; i < number_of_positive_points; i++) {
        grid_buffer_1[index_to_cart_pos[i][1]] =
            grid_gs[index_to_cart_pos[i][0]];
      }
    } else {
      memcpy(grid_buffer_1, grid_gs,
             product3(fft_sizes_gs) * sizeof(double complex));
    }
    FFT_H2D(grid_buffer_dev_1, grid_buffer_1,
            product3(fft_sizes_gs) * sizeof(double complex));

    // Perform the first FFT and one transposition (x,y_d,z)->(z,x,y_d)
    fft_gpu_1d_local_device(false, npts_global[2],
                            fft_sizes_gs[0] * fft_sizes_gs[1], false, true,
                            npts_global[2], fft_sizes_gs[0] * fft_sizes_gs[1],
                            grid_buffer_dev_1, grid_buffer_dev_2);

    // Collect data in y-direction and distribute x-direction (z,x,y_d) ->
    // (z_d,x,y)
    offloadStreamSynchronize((offloadStream_t)fft_stream);
    collect_y_and_distribute_z_blocked_comm_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, sub_comm[0]);
    collect_y_and_distribute_z_blocked_unpack_device(
        grid_buffer_dev_1, grid_buffer_dev_2, redistribution, proc2local_y_gs,
        fft_stream);

    // Perform the second FFT and one transposition (z_d,x,y)->(y,z_d,x)
    fft_gpu_1d_local_device(false, npts_global[1],
                            fft_sizes_ms[0] * fft_sizes_ms[2], false, true,
                            npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2],
                            grid_buffer_dev_2, grid_buffer_dev_1);
    // Perform the second FFT and one transposition (y,z_d,x) -> (x,y,z_d)
    FFT_D2H(grid_buffer_1, grid_buffer_dev_1,
            (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
    fft_gpu_1d_c2r_local(npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2],
                         false, true, npts_global[0] / 2 + 1,
                         fft_sizes_rs[1] * fft_sizes_rs[2], grid_buffer_1,
                         grid_rs);

  } else if (proc_grid[1] > 1) {
    if (index_to_cart_pos != NULL) {
      memset(grid_buffer_1, 0, product3(fft_sizes_gs) * sizeof(double complex));
#pragma omp parallel for default(none) shared(                                 \
        number_of_positive_points, index_to_cart_pos, grid_gs, grid_buffer_1)
      for (int i = 0; i < number_of_positive_points; i++) {
        grid_buffer_1[index_to_cart_pos[i][1]] =
            grid_gs[index_to_cart_pos[i][0]];
      }
    } else {
      memcpy(grid_buffer_1, grid_gs,
             product3(fft_sizes_gs) * sizeof(double complex));
    }
    FFT_H2D(grid_buffer_dev_1, grid_buffer_1,
            product3(fft_sizes_gs) * sizeof(double complex));

    if (fft_sizes_rs[2] > 0) {

      // Perform the first FFT in z-direction (x_d,y_d,z)->(z,x_d,y_d)
      fft_gpu_2d_local_device(
          false, (const int[2]){npts_global[1], npts_global[2]},
          fft_sizes_gs[0], false, true, grid_buffer_dev_1, grid_buffer_dev_2);

      // Collect data in z-direction and distribute y-direction
      offloadStreamSynchronize((offloadStream_t)fft_stream);
      collect_x_and_distribute_y_blocked_comm_device(
          grid_buffer_dev_2, grid_buffer_dev_1, redistribution, sub_comm[1]);
      collect_x_and_distribute_y_blocked_unpack_device(
          grid_buffer_dev_1, grid_buffer_dev_2, redistribution, proc2local_x_gs,
          fft_stream);

      // Perform the third FFT and one transposition (y,x,z)->(z,y,x)
      FFT_D2H(grid_buffer_2, grid_buffer_dev_2,
              (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
      fft_gpu_1d_c2r_local(npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2],
                           false, true, npts_global[0] / 2 + 1,
                           fft_sizes_rs[1] * fft_sizes_rs[2], grid_buffer_2,
                           grid_rs);
    }
  } else {
    if (index_to_cart_pos != NULL) {
      memset(grid_buffer_1, 0, product3(fft_sizes_gs) * sizeof(double complex));
#pragma omp parallel for default(none) shared(                                 \
        number_of_positive_points, index_to_cart_pos, grid_gs, grid_buffer_1)
      for (int i = 0; i < number_of_positive_points; i++) {
        grid_buffer_1[index_to_cart_pos[i][1]] =
            grid_gs[index_to_cart_pos[i][0]];
      }
    } else {
      memcpy(grid_buffer_1, grid_gs,
             product3(fft_sizes_gs) * sizeof(double complex));
    }
    FFT_H2D(grid_buffer_dev_1, grid_buffer_1,
            product3(fft_sizes_gs) * sizeof(double complex));

    // Perform the first two FFTs
    fft_gpu_2d_local_device(false,
                            (const int[2]){npts_global[1], npts_global[2]},
                            npts_global_gspace[0], false, false,
                            grid_buffer_dev_1, grid_buffer_dev_2);
    // And the R2C FFT separately to get rid of additional transposition steps
    FFT_D2H(grid_buffer_2, grid_buffer_dev_2,
            (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
    fft_gpu_1d_c2r_local(npts_global[0], npts_global[1] * npts_global[2], true,
                         true, npts_global[1] * npts_global[2],
                         npts_global[1] * npts_global[2], grid_buffer_2,
                         grid_rs);
  }

  fft_stop_timer(handle2);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a ray distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_gpu_fw_ray(const double complex *restrict grid_rs, bool is_complex,
                       double complex *restrict grid_gs, int npts_gs_local,
                       const int npts_global[3], int *index_to_ray,
                       int (*proc2local_rs)[3][2], int (*proc2local_ms)[3][2],
                       int (*proc2local_x_gs)[2], int *rays_per_process,
                       fft_redistribution_t *redistribution, cp_mpi_comm_t comm,
                       const cp_mpi_comm_t sub_comm[2]) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "3d_fw_r");
  const int handle = fft_start_timer(routine_name);
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "3d_fw_r_%i_%i_%i_%i",
           npts_global[0], npts_global[1], npts_global[2],
           cp_mpi_comm_size(comm));
  const int handle2 = fft_start_timer(routine_name);

  const int my_process = cp_mpi_comm_rank(comm);

  double complex *grid_buffer_1 = get_buffer_1();
  double complex *grid_buffer_2 = get_buffer_2();
  double complex *grid_buffer_dev_1 = fft_gpu_get_buffer_1();
  double complex *grid_buffer_dev_2 = fft_gpu_get_buffer_2();
  void *fft_stream = fft_gpu_get_stream();

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {proc2local_rs[my_process][0][1],
                         proc2local_rs[my_process][1][1],
                         proc2local_rs[my_process][2][1]};
  int fft_sizes_ms[3] = {proc2local_ms[my_process][0][1],
                         proc2local_ms[my_process][1][1],
                         proc2local_ms[my_process][2][1]};
  int number_of_local_xy_rays = rays_per_process[my_process];

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  cp_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  // The total number of points may be larger than the largest representable
  // number of type int (large semiempirical systems) So, we need to convert the
  // lengths to long int first
  const long int number_of_points = ((long int)npts_global[0]) *
                                    ((long int)npts_global[1]) *
                                    ((long int)npts_global[2]);
  const double scaling_factor = 1.0 / ((double)number_of_points);

  // We use different data distribution schemes depending on the
  // availability of a distributed FFT library because FFTW requires the
  // data to the different FFTs to be consecutively stored in memory. This
  // is not possible without a distributed FFT library because this would
  // require the implementation of the Guru interface which is not available
  // with all implementations of the FFTW interface
  if (proc_grid[0] > 1 && proc_grid[1] > 1) {

    if (is_complex) {
      FFT_H2D(grid_buffer_dev_2, grid_rs,
              product3(fft_sizes_rs) * sizeof(double complex));
    } else {
      const int npts_real = product3(fft_sizes_rs);
      FFT_H2D(grid_buffer_dev_1, grid_rs, npts_real * sizeof(double));
      fft_gpu_launch_real_to_complex((const double *)grid_buffer_dev_1,
                                     (double *)grid_buffer_dev_2, npts_real,
                                     FFT_STREAM);
    }

    // Perform the first FFT (x,y_d,z_d) -> (y_d,z_d,x)
    fft_gpu_1d_local_device(true, npts_global[0],
                            fft_sizes_rs[1] * fft_sizes_rs[2], true, false,
                            fft_sizes_rs[1] * fft_sizes_rs[2], npts_global[0],
                            grid_buffer_dev_2, grid_buffer_dev_1);

    // Perform redistribution (y_d,z_d,x) -> (y,z_d,x_d)
    collect_y_and_distribute_x_blocked_pack_device(
        grid_buffer_dev_1, grid_buffer_dev_2, redistribution, proc2local_x_gs,
        fft_stream);

    // Perform communication (y_d,z_d,x) -> (y,z_d,x_d)
    offloadStreamSynchronize((offloadStream_t)fft_stream);
    collect_y_and_distribute_x_blocked_comm_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, sub_comm[1]);

    // Perform the second FFT (y,z_d,x_d) -> (z_d,x_d,y)
    fft_gpu_1d_local_device(true, npts_global[1],
                            fft_sizes_ms[0] * fft_sizes_ms[2], true, false,
                            fft_sizes_ms[0] * fft_sizes_ms[2], npts_global[1],
                            grid_buffer_dev_1, grid_buffer_dev_2);

    // Perform second redistribution (z_d,x_d,y) -> (z,xy_d)
    collect_z_and_distribute_xy_ray_pack_device(grid_buffer_2, grid_buffer_1,
                                                redistribution, fft_stream);
    offloadStreamSynchronize((offloadStream_t)fft_stream);
    collect_z_and_distribute_xy_ray_comm_device(
        grid_buffer_dev_1, grid_buffer_dev_2, redistribution, comm);
    collect_z_and_distribute_xy_ray_unpack_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, fft_stream);

    // Perform the third FFT (z,xy_d) -> (xy_d,z)
    fft_gpu_1d_local_device(true, npts_global[2], number_of_local_xy_rays,
                            false, false, npts_global[2], npts_global[2],
                            grid_buffer_dev_1, grid_buffer_dev_2);

    FFT_D2H(grid_buffer_2, grid_buffer_dev_2,
            (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none) shared(                                 \
        npts_gs_local, grid_gs, grid_buffer_2, scaling_factor, index_to_ray)
    for (int index = 0; index < npts_gs_local; index++) {
      grid_gs[index] = scaling_factor * grid_buffer_2[index_to_ray[index]];
    }
  } else if (proc_grid[0] > 1) {
    if (is_complex) {
      FFT_H2D(grid_buffer_dev_1, grid_rs,
              product3(fft_sizes_rs) * sizeof(double complex));
    } else {
      const int npts_real = product3(fft_sizes_rs);
      FFT_H2D(grid_buffer_dev_2, grid_rs, npts_real * sizeof(double));
      fft_gpu_launch_real_to_complex((const double *)grid_buffer_dev_2,
                                     (double *)grid_buffer_dev_1, npts_real,
                                     FFT_STREAM);
    }

    // Now, we only need a local 2D FFT (x,y,z_d) -> (z_d,x,y)
    fft_gpu_2d_local_device(
        true, (const int[2]){npts_global[0], npts_global[1]}, fft_sizes_ms[2],
        true, false, grid_buffer_dev_1, grid_buffer_dev_2);

    // Perform second transpose (z_d,x,y) -> (z,xy_d)
    collect_z_and_distribute_xy_ray_pack_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, fft_stream);
    offloadStreamSynchronize((offloadStream_t)fft_stream);
    collect_z_and_distribute_xy_ray_comm_device(
        grid_buffer_dev_1, grid_buffer_dev_2, redistribution, comm);
    collect_z_and_distribute_xy_ray_unpack_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, fft_stream);

    // Perform the third FFT (z,xy_d) -> (xy_d,z)
    fft_gpu_1d_local_device(true, npts_global[2], number_of_local_xy_rays,
                            false, false, npts_global[2], npts_global[2],
                            grid_buffer_dev_1, grid_buffer_dev_2);

    FFT_D2H(grid_buffer_2, grid_buffer_dev_2,
            (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(npts_gs_local, grid_gs, grid_buffer_2, scaling_factor,              \
               index_to_ray, my_process)
    for (int index = 0; index < npts_gs_local; index++) {
      grid_gs[index] = scaling_factor * grid_buffer_2[index_to_ray[index]];
    }
  } else if (proc_grid[1] > 1) {

    if (is_complex) {
      FFT_H2D(grid_buffer_dev_2, grid_rs,
              product3(fft_sizes_rs) * sizeof(double complex));
    } else {
      const int npts_real = product3(fft_sizes_rs);
      FFT_H2D(grid_buffer_dev_1, grid_rs, npts_real * sizeof(double));
      fft_gpu_launch_real_to_complex((const double *)grid_buffer_dev_1,
                                     (double *)grid_buffer_dev_2, npts_real,
                                     FFT_STREAM);
    }

    // Perform the first FFT (x,y_d,z_d) -> (y_d,z_d,x)
    fft_gpu_1d_local_device(true, npts_global[0],
                            fft_sizes_rs[1] * fft_sizes_rs[2], true, false,
                            fft_sizes_rs[1] * fft_sizes_rs[2], npts_global[0],
                            grid_buffer_dev_2, grid_buffer_dev_1);

    // Perform redistribution (y_d,z_d,x) -> (y,z_d,x_d)
    collect_y_and_distribute_x_blocked_pack_device(
        grid_buffer_dev_1, grid_buffer_dev_2, redistribution, proc2local_x_gs,
        fft_stream);

    // Perform communication (y_d,z_d,x) -> (y,z_d,x_d)
    offloadStreamSynchronize((offloadStream_t)fft_stream);
    collect_y_and_distribute_x_blocked_comm_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, sub_comm[1]);

    // Perform the second FFT (y,z_d,x_d) -> (z_d,x_d,y)
    fft_gpu_1d_local_device(true, npts_global[1],
                            fft_sizes_ms[0] * fft_sizes_ms[2], true, false,
                            fft_sizes_ms[0] * fft_sizes_ms[2], npts_global[1],
                            grid_buffer_dev_1, grid_buffer_dev_2);

    // Perform second redistribution (z_d,x_d,y) -> (z,xy_d)
    collect_z_and_distribute_xy_ray_pack_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, fft_stream);
    offloadStreamSynchronize((offloadStream_t)fft_stream);
    collect_z_and_distribute_xy_ray_comm_device(
        grid_buffer_dev_1, grid_buffer_dev_2, redistribution, comm);
    collect_z_and_distribute_xy_ray_unpack_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, fft_stream);

    // Perform the third FFT (z,xy_d) -> (xy_d,z)
    fft_gpu_1d_local_device(true, npts_global[2], number_of_local_xy_rays,
                            false, false, npts_global[2], npts_global[2],
                            grid_buffer_dev_1, grid_buffer_dev_2);

    FFT_D2H(grid_buffer_2, grid_buffer_dev_2,
            (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none) shared(                                 \
        npts_gs_local, grid_gs, grid_buffer_2, scaling_factor, index_to_ray)
    for (int index = 0; index < npts_gs_local; index++) {
      grid_gs[index] = scaling_factor * grid_buffer_2[index_to_ray[index]];
    }
  } else {
    if (is_complex) {
      FFT_H2D(grid_buffer_dev_2, grid_rs,
              product3(fft_sizes_rs) * sizeof(double complex));
    } else {
      const int npts_real = product3(fft_sizes_rs);
      FFT_H2D(grid_buffer_dev_1, grid_rs, npts_real * sizeof(double));
      fft_gpu_launch_real_to_complex((const double *)grid_buffer_dev_1,
                                     (double *)grid_buffer_dev_2, npts_real,
                                     FFT_STREAM);
    }

    fft_gpu_3d_local_device(true, npts_global, 1, false, false,
                            grid_buffer_dev_2, grid_buffer_dev_1);
    FFT_D2H(grid_buffer_1, grid_buffer_dev_1,
            (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none) shared(                                 \
        npts_gs_local, grid_gs, grid_buffer_1, scaling_factor, index_to_ray)
    for (int index = 0; index < npts_gs_local; index++) {
      grid_gs[index] = scaling_factor * grid_buffer_1[index_to_ray[index]];
    }
  }

  fft_stop_timer(handle2);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a forward 3D-FFT using a ray distribution.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_gpu_fw_r2c_ray(
    const double *restrict grid_rs, double complex *restrict grid_gs,
    int (*index_to_ray_pos)[2], int (*index_to_ray_neg)[2],
    int number_of_positive_points, int number_of_negative_points,
    const int npts_global[3], const int npts_global_gspace[3],
    int (*proc2local_rs)[3][2], int (*proc2local_ms)[3][2],
    int (*proc2local_x_gs)[2], int *rays_per_process,
    fft_redistribution_t *redistribution, cp_mpi_comm_t comm,
    const cp_mpi_comm_t sub_comm[2]) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "3d_fw_r2c_r");
  const int handle = fft_start_timer(routine_name);
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "3d_fw_r2c_r_%i_%i_%i_%i",
           npts_global[0], npts_global[1], npts_global[2],
           cp_mpi_comm_size(comm));
  const int handle2 = fft_start_timer(routine_name);

  const int my_process = cp_mpi_comm_rank(comm);

  double complex *grid_buffer_1 = get_buffer_1();
  double complex *grid_buffer_2 = get_buffer_2();
  double complex *grid_buffer_dev_1 = fft_gpu_get_buffer_1();
  double complex *grid_buffer_dev_2 = fft_gpu_get_buffer_2();
  void *fft_stream = fft_gpu_get_stream();

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {proc2local_rs[my_process][0][1],
                         proc2local_rs[my_process][1][1],
                         proc2local_rs[my_process][2][1]};
  int fft_sizes_ms[3] = {proc2local_ms[my_process][0][1],
                         proc2local_ms[my_process][1][1],
                         proc2local_ms[my_process][2][1]};
  int number_of_local_xy_rays = rays_per_process[my_process];

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  cp_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  // The total number of points may be larger than the largest representable
  // number of type int (large semiempirical systems) So, we need to convert the
  // lengths to long int first
  const long int number_of_points = ((long int)npts_global[0]) *
                                    ((long int)npts_global[1]) *
                                    ((long int)npts_global[2]);
  const double scaling_factor = 1.0 / ((double)number_of_points);

  // We use different data distribution schemes depending on the
  // availability of a distributed FFT library because FFTW requires the
  // data to the different FFTs to be consecutively stored in memory. This
  // is not possible without a distributed FFT library because this would
  // require the implementation of the Guru interface which is not available
  // with all implementations of the FFTW interface
  if (proc_grid[0] > 1 && proc_grid[1] > 1) {

    FFT_H2D(grid_buffer_dev_2, grid_rs,
            product3(fft_sizes_rs) * sizeof(double));

    // Perform the first FFT (x,y_d,z_d) -> (y_d,z_d,x)
    fft_gpu_1d_r2c_local_device(
        npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], true, false,
        fft_sizes_rs[1] * fft_sizes_rs[2], npts_global[0] / 2 + 1,
        (double *)grid_buffer_dev_2, grid_buffer_dev_1);

    // Pack buffer for (y_d,z_d,x) -> (y,z_d,x_d)
    collect_y_and_distribute_x_blocked_pack_device(
        grid_buffer_dev_1, grid_buffer_dev_2, redistribution, proc2local_x_gs,
        fft_stream);

    // Perform communication for (y_d,z_d,x) -> (y,z_d,x_d)
    offloadStreamSynchronize((offloadStream_t)fft_stream);
    collect_y_and_distribute_x_blocked_comm_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, sub_comm[1]);

    // Perform the second FFT (y,z_d,x_d) -> (z_d,x_d,y)
    fft_gpu_1d_local_device(true, npts_global[1],
                            fft_sizes_ms[0] * fft_sizes_ms[2], true, false,
                            fft_sizes_ms[0] * fft_sizes_ms[2], npts_global[1],
                            grid_buffer_dev_1, grid_buffer_dev_2);

    // Perform second redistribution (z_d,x_d,y) -> (z,xy_d)
    collect_z_and_distribute_xy_ray_pack_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, fft_stream);
    offloadStreamSynchronize((offloadStream_t)fft_stream);
    collect_z_and_distribute_xy_ray_comm_device(
        grid_buffer_dev_1, grid_buffer_dev_2, redistribution, comm);
    collect_z_and_distribute_xy_ray_unpack_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, fft_stream);

    // Perform the third FFT (z,xy_d) -> (xy_d,z)
    fft_gpu_1d_local_device(true, npts_global[2], number_of_local_xy_rays,
                            false, false, npts_global[2], npts_global[2],
                            grid_buffer_dev_1, grid_buffer_dev_2);

    FFT_D2H(grid_buffer_2, grid_buffer_dev_2,
            (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(number_of_positive_points, index_to_ray_pos, grid_gs,               \
               grid_buffer_2, scaling_factor)
    for (int index = 0; index < number_of_positive_points; index++) {
      grid_gs[index_to_ray_pos[index][0]] =
          scaling_factor * grid_buffer_2[index_to_ray_pos[index][1]];
    }
    FFT_D2H(grid_buffer_2, grid_buffer_dev_2,
            (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(number_of_negative_points, index_to_ray_neg, grid_gs,               \
               grid_buffer_2, scaling_factor)
    for (int index = 0; index < number_of_negative_points; index++) {
      grid_gs[index_to_ray_neg[index][0]] =
          scaling_factor * conj(grid_buffer_2[index_to_ray_neg[index][1]]);
    }
  } else if (proc_grid[0] > 1) {

    FFT_H2D(grid_buffer_dev_2, grid_rs,
            product3(fft_sizes_rs) * sizeof(double));

    // The first two FFTs can be performed locally
    // FFTs (x,y,z_d) -> (y,z_d,x)
    fft_gpu_1d_r2c_local_device(
        npts_global[0], npts_global[1] * fft_sizes_ms[2], true, false,
        npts_global[1] * fft_sizes_ms[2], npts_global[0] / 2 + 1,
        (double *)grid_buffer_dev_2, grid_buffer_dev_1);
    // second FFT (y,z_d,x) -> (z_d,x,y)
    fft_gpu_1d_local_device(
        true, npts_global[1], npts_global_gspace[0] * fft_sizes_ms[2], true,
        false, npts_global_gspace[0] * fft_sizes_ms[2], npts_global[1],
        grid_buffer_dev_1, grid_buffer_dev_2);

    // but we need to redistribute to rays (z_d,x,y) -> (z,xy_d)
    collect_z_and_distribute_xy_ray_pack_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, fft_stream);
    offloadStreamSynchronize((offloadStream_t)fft_stream);
    collect_z_and_distribute_xy_ray_comm_device(
        grid_buffer_dev_1, grid_buffer_dev_2, redistribution, comm);
    collect_z_and_distribute_xy_ray_unpack_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, fft_stream);

    // Perform the third FFT (z,xy_d) -> (xy_d,z)
    fft_gpu_1d_local_device(true, npts_global[2], number_of_local_xy_rays,
                            false, false, npts_global[2], npts_global[2],
                            grid_buffer_dev_1, grid_buffer_dev_2);

    FFT_D2H(grid_buffer_2, grid_buffer_dev_2,
            (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(number_of_positive_points, index_to_ray_pos, grid_gs,               \
               grid_buffer_2, scaling_factor)
    for (int index = 0; index < number_of_positive_points; index++) {
      grid_gs[index_to_ray_pos[index][0]] =
          scaling_factor * grid_buffer_2[index_to_ray_pos[index][1]];
    }
    FFT_D2H(grid_buffer_2, grid_buffer_dev_2,
            (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(number_of_negative_points, index_to_ray_neg, grid_gs,               \
               grid_buffer_2, scaling_factor)
    for (int index = 0; index < number_of_negative_points; index++) {
      grid_gs[index_to_ray_neg[index][0]] =
          scaling_factor * conj(grid_buffer_2[index_to_ray_neg[index][1]]);
    }
  } else if (proc_grid[1] > 1) {

    FFT_H2D(grid_buffer_dev_2, grid_rs,
            product3(fft_sizes_rs) * sizeof(double));

    // Perform the first FFT (x,y_d,z_d) -> (y_d,z_d,x)
    fft_gpu_1d_r2c_local_device(
        npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2], true, false,
        fft_sizes_rs[1] * fft_sizes_rs[2], npts_global[0] / 2 + 1,
        (double *)grid_buffer_dev_2, grid_buffer_dev_1);

    // Pack buffer for (y_d,z_d,x) -> (y,z_d,x_d)
    collect_y_and_distribute_x_blocked_pack_device(
        grid_buffer_dev_1, grid_buffer_dev_2, redistribution, proc2local_x_gs,
        fft_stream);

    // Perform communication for (y_d,z_d,x) -> (y,z_d,x_d)
    offloadStreamSynchronize((offloadStream_t)fft_stream);
    collect_y_and_distribute_x_blocked_comm_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, sub_comm[1]);

    // Perform the second FFT (y,z_d,x_d) -> (z_d,x_d,y)
    fft_gpu_1d_local_device(true, npts_global[1],
                            fft_sizes_ms[0] * fft_sizes_ms[2], true, false,
                            fft_sizes_ms[0] * fft_sizes_ms[2], npts_global[1],
                            grid_buffer_dev_1, grid_buffer_dev_2);

    // Perform second redistribution (z_d,x_d,y) -> (z,xy_d)
    collect_z_and_distribute_xy_ray_pack_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, fft_stream);
    offloadStreamSynchronize((offloadStream_t)fft_stream);
    collect_z_and_distribute_xy_ray_comm_device(
        grid_buffer_dev_1, grid_buffer_dev_2, redistribution, comm);
    collect_z_and_distribute_xy_ray_unpack_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, fft_stream);

    // Perform the third FFT (z,xy_d) -> (xy_d,z)
    fft_gpu_1d_local_device(true, npts_global[2], number_of_local_xy_rays,
                            false, false, npts_global[2], npts_global[2],
                            grid_buffer_dev_1, grid_buffer_dev_2);

    FFT_D2H(grid_buffer_2, grid_buffer_dev_2,
            (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(number_of_positive_points, index_to_ray_pos, grid_gs,               \
               grid_buffer_2, scaling_factor)
    for (int index = 0; index < number_of_positive_points; index++) {
      grid_gs[index_to_ray_pos[index][0]] =
          scaling_factor * grid_buffer_2[index_to_ray_pos[index][1]];
    }
    FFT_D2H(grid_buffer_2, grid_buffer_dev_2,
            (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(number_of_negative_points, index_to_ray_neg, grid_gs,               \
               grid_buffer_2, scaling_factor)
    for (int index = 0; index < number_of_negative_points; index++) {
      grid_gs[index_to_ray_neg[index][0]] =
          scaling_factor * conj(grid_buffer_2[index_to_ray_neg[index][1]]);
    }
  } else {

    FFT_H2D(grid_buffer_dev_1, grid_rs,
            product3(fft_sizes_rs) * sizeof(double));

    // FFTs (x,y,z_d) -> (y,z_d,x)
    fft_gpu_1d_r2c_local_device(npts_global[0], npts_global[1] * npts_global[2],
                                true, true, npts_global[1] * npts_global[2],
                                npts_global[1] * npts_global[2],
                                (double *)grid_buffer_dev_1, grid_buffer_dev_2);
    // second FFT (y,z_d,x) -> (z_d,x,y)
    fft_gpu_2d_local_device(true,
                            (const int[2]){npts_global[1], npts_global[2]},
                            npts_global_gspace[0], false, false,
                            grid_buffer_dev_2, grid_buffer_dev_1);

    // Copy to the ray format
    FFT_D2H(grid_buffer_1, grid_buffer_dev_1,
            (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(number_of_positive_points, index_to_ray_pos, grid_gs,               \
               grid_buffer_1, scaling_factor)
    for (int index = 0; index < number_of_positive_points; index++) {
      grid_gs[index_to_ray_pos[index][0]] =
          scaling_factor * grid_buffer_1[index_to_ray_pos[index][1]];
    }
    FFT_D2H(grid_buffer_1, grid_buffer_dev_1,
            (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(number_of_negative_points, index_to_ray_neg, grid_gs,               \
               grid_buffer_1, scaling_factor)
    for (int index = 0; index < number_of_negative_points; index++) {
      grid_gs[index_to_ray_neg[index][0]] =
          scaling_factor * conj(grid_buffer_1[index_to_ray_neg[index][1]]);
    }
  }

  fft_stop_timer(handle2);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT overwriting the buffers.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_gpu_bw_ray(const double complex *restrict grid_gs,
                       int *index_to_ray, int number_of_local_gpts,
                       double complex *restrict grid_rs, bool is_complex,
                       const int npts_global[3], int (*proc2local_rs)[3][2],
                       int (*proc2local_ms)[3][2], int (*proc2local_x_gs)[2],
                       int *rays_per_process,
                       fft_redistribution_t *redistribution, cp_mpi_comm_t comm,
                       const cp_mpi_comm_t sub_comm[2]) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "3d_bw_r");
  const int handle = fft_start_timer(routine_name);
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "3d_bw_r_%i_%i_%i_%i",
           npts_global[0], npts_global[1], npts_global[2],
           cp_mpi_comm_size(comm));
  const int handle2 = fft_start_timer(routine_name);

  const int my_process = cp_mpi_comm_rank(comm);

  double complex *grid_buffer_1 = get_buffer_1();
  double complex *grid_buffer_2 = get_buffer_2();
  double complex *grid_buffer_dev_1 = fft_gpu_get_buffer_1();
  double complex *grid_buffer_dev_2 = fft_gpu_get_buffer_2();
  void *fft_stream = fft_gpu_get_stream();

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {proc2local_rs[my_process][0][1],
                         proc2local_rs[my_process][1][1],
                         proc2local_rs[my_process][2][1]};
  int fft_sizes_ms[3] = {proc2local_ms[my_process][0][1],
                         proc2local_ms[my_process][1][1],
                         proc2local_ms[my_process][2][1]};
  int number_of_local_xy_rays = rays_per_process[my_process];

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  cp_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  // We use different data distribution schemes depending on the
  // availability of a distributed FFT library because FFTW requires the
  // data to the different FFTs to be consecutively stored in memory. This
  // is not possible without a distributed FFT library because this would
  // require the implementation of the Guru interface which is not available
  // with all implementations of the FFTW interface
  if (proc_grid[0] > 1 && proc_grid[1] > 1) {
    memset(grid_buffer_1, 0,
           number_of_local_xy_rays * npts_global[2] * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(number_of_local_gpts, index_to_ray, grid_buffer_1, grid_gs)
    for (int i = 0; i < number_of_local_gpts; i++) {
      grid_buffer_1[index_to_ray[i]] = grid_gs[i];
    }
    FFT_H2D(grid_buffer_dev_1, grid_buffer_1,
            number_of_local_xy_rays * npts_global[2] * sizeof(double complex));

    // Perform the first FFT (xy_d,z) -> (z,xy_d)
    fft_gpu_1d_local_device(false, npts_global[2], number_of_local_xy_rays,
                            false, false, npts_global[2], npts_global[2],
                            grid_buffer_dev_1, grid_buffer_dev_2);

    // Perform transpose (z,xy_d) -> (z_d,x_d,y)
    collect_xy_and_distribute_z_ray_pack_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, fft_stream);
    offloadStreamSynchronize((offloadStream_t)fft_stream);
    collect_xy_and_distribute_z_ray_comm_device(
        grid_buffer_dev_1, grid_buffer_dev_2, redistribution, comm);
    collect_xy_and_distribute_z_ray_unpack_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, fft_stream);

    // Perform the second FFT (z_d,x_d,y) -> (y,z_d,x_d)
    fft_gpu_1d_local_device(false, npts_global[1],
                            fft_sizes_ms[0] * fft_sizes_ms[2], false, true,
                            npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2],
                            grid_buffer_dev_1, grid_buffer_dev_2);

    // Perform second transpose (y,z_d,x_d) -> (y_d,z_d,x)
    offloadStreamSynchronize((offloadStream_t)fft_stream);
    collect_x_and_distribute_y_blocked_comm_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, sub_comm[1]);

    // Perform second transpose (y,z_d,x_d) -> (y_d,z_d,x)
    collect_x_and_distribute_y_blocked_unpack_device(
        grid_buffer_dev_1, grid_buffer_dev_2, redistribution, proc2local_x_gs,
        fft_stream);

    // Perform the third FFT (y_d,z_d,x) -> (x,y_d,z_d)
    fft_gpu_1d_local_device(false, npts_global[0],
                            fft_sizes_rs[1] * fft_sizes_rs[2], false, true,
                            npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2],
                            grid_buffer_dev_2, grid_buffer_dev_1);
    if (is_complex) {
      FFT_D2H(grid_rs, grid_buffer_dev_1,
              product3(fft_sizes_rs) * sizeof(double complex));
    }

    if (!is_complex) {
      double *grid_rs_double = (double *)grid_rs;
      FFT_D2H(grid_buffer_1, grid_buffer_dev_1,
              (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(fft_sizes_rs, grid_rs_double, grid_buffer_1)
      for (int i = 0; i < product3(fft_sizes_rs); i++)
        grid_rs_double[i] = creal(grid_buffer_1[i]);
    }

  } else if (proc_grid[0] > 1) {
    memset(grid_buffer_1, 0,
           number_of_local_xy_rays * npts_global[2] * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(number_of_local_gpts, index_to_ray, grid_buffer_1, grid_gs)
    for (int i = 0; i < number_of_local_gpts; i++) {
      grid_buffer_1[index_to_ray[i]] = grid_gs[i];
    }
    FFT_H2D(grid_buffer_dev_1, grid_buffer_1,
            number_of_local_xy_rays * npts_global[2] * sizeof(double complex));
    // Perform the first FFT (xy_d,z) -> (z,xy_d)
    fft_gpu_1d_local_device(false, npts_global[2], number_of_local_xy_rays,
                            false, false, npts_global[2], npts_global[2],
                            grid_buffer_dev_1, grid_buffer_dev_2);

    // Perform transpose (z,xy_d) -> (z_d,x,y)
    collect_xy_and_distribute_z_ray_pack_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, fft_stream);
    offloadStreamSynchronize((offloadStream_t)fft_stream);
    collect_xy_and_distribute_z_ray_comm_device(
        grid_buffer_dev_1, grid_buffer_dev_2, redistribution, comm);
    collect_xy_and_distribute_z_ray_unpack_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, fft_stream);

    // Perform the second FFT
    fft_gpu_2d_local_device(
        false, (const int[2]){npts_global[0], npts_global[1]}, fft_sizes_rs[2],
        false, true, grid_buffer_dev_1, grid_buffer_dev_2);
    if (is_complex) {
      FFT_D2H(grid_rs, grid_buffer_dev_2,
              product3(fft_sizes_rs) * sizeof(double complex));
    }
    if (!is_complex) {
      double *grid_rs_double = (double *)grid_rs;
      FFT_D2H(grid_buffer_2, grid_buffer_dev_2,
              (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(fft_sizes_rs, grid_rs_double, grid_buffer_2)
      for (int i = 0; i < product3(fft_sizes_rs); i++)
        grid_rs_double[i] = creal(grid_buffer_2[i]);
    }
  } else if (proc_grid[1] > 1) {
    memset(grid_buffer_1, 0,
           number_of_local_xy_rays * npts_global[2] * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(number_of_local_gpts, index_to_ray, grid_buffer_1, grid_gs)
    for (int i = 0; i < number_of_local_gpts; i++) {
      grid_buffer_1[index_to_ray[i]] = grid_gs[i];
    }
    FFT_H2D(grid_buffer_dev_1, grid_buffer_1,
            number_of_local_xy_rays * npts_global[2] * sizeof(double complex));

    // Perform the first FFT (xy_d,z) -> (z,xy_d)
    fft_gpu_1d_local_device(false, npts_global[2], number_of_local_xy_rays,
                            false, false, npts_global[2], npts_global[2],
                            grid_buffer_dev_1, grid_buffer_dev_2);

    // Perform transpose (z,xy_d) -> (z_d,x_d,y)
    collect_xy_and_distribute_z_ray_pack_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, fft_stream);
    offloadStreamSynchronize((offloadStream_t)fft_stream);
    collect_xy_and_distribute_z_ray_comm_device(
        grid_buffer_dev_1, grid_buffer_dev_2, redistribution, comm);
    collect_xy_and_distribute_z_ray_unpack_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, fft_stream);

    // Perform the second FFT (z_d,x_d,y) -> (y,z_d,x_d)
    fft_gpu_1d_local_device(false, npts_global[1],
                            fft_sizes_ms[0] * fft_sizes_ms[2], false, true,
                            npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2],
                            grid_buffer_dev_1, grid_buffer_dev_2);

    // Perform second transpose (y,z_d,x_d) -> (y_d,z_d,x)
    offloadStreamSynchronize((offloadStream_t)fft_stream);
    collect_x_and_distribute_y_blocked_comm_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, sub_comm[1]);

    // Perform second transpose (y,z_d,x_d) -> (y_d,z_d,x)
    collect_x_and_distribute_y_blocked_unpack_device(
        grid_buffer_dev_1, grid_buffer_dev_2, redistribution, proc2local_x_gs,
        fft_stream);

    // Perform the third FFT (y_d,z_d,x) -> (x,y_d,z_d)
    fft_gpu_1d_local_device(false, npts_global[0],
                            fft_sizes_rs[1] * fft_sizes_rs[2], false, true,
                            npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2],
                            grid_buffer_dev_2, grid_buffer_dev_1);
    if (is_complex) {
      FFT_D2H(grid_rs, grid_buffer_dev_1,
              product3(fft_sizes_rs) * sizeof(double complex));
    }
    if (!is_complex) {
      double *grid_rs_double = (double *)grid_rs;
      FFT_D2H(grid_buffer_1, grid_buffer_dev_1,
              (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(fft_sizes_rs, grid_rs_double, grid_buffer_1)
      for (int i = 0; i < product3(fft_sizes_rs); i++)
        grid_rs_double[i] = creal(grid_buffer_1[i]);
    }

  } else {
    // Ray distribution
    memset(grid_buffer_2, 0, product3(npts_global) * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(number_of_local_gpts, index_to_ray, grid_buffer_2, grid_gs)
    for (int i = 0; i < number_of_local_gpts; i++) {
      grid_buffer_2[index_to_ray[i]] = grid_gs[i];
    }
    FFT_H2D(grid_buffer_dev_2, grid_buffer_2,
            product3(npts_global) * sizeof(double complex));

    if (is_complex) {
      fft_gpu_3d_local_device(false, npts_global, 1, false, false,
                              grid_buffer_dev_2, grid_buffer_dev_1);
      FFT_D2H(grid_rs, grid_buffer_dev_1,
              product3(fft_sizes_rs) * sizeof(double complex));
    } else {
      fft_gpu_3d_local_device(false, npts_global, 1, false, false,
                              grid_buffer_dev_2, grid_buffer_dev_1);

      double *grid_rs_double = (double *)grid_rs;
      FFT_D2H(grid_buffer_1, grid_buffer_dev_1,
              (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
#pragma omp parallel for default(none)                                         \
    shared(fft_sizes_rs, grid_rs_double, grid_buffer_1)
      for (int i = 0; i < product3(fft_sizes_rs); i++)
        grid_rs_double[i] = creal(grid_buffer_1[i]);
    }
  }

  fft_stop_timer(handle2);
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a backward 3D-FFT overwriting the buffers.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_gpu_bw_c2r_ray(
    const double complex *restrict grid_gs, int (*index_to_ray_pos)[2],
    int number_of_positive_points, double *restrict grid_rs,
    const int npts_global[3], const int npts_global_gspace[3],
    int (*proc2local_rs)[3][2], int (*proc2local_ms)[3][2],
    int (*proc2local_x_gs)[2], int *rays_per_process,
    fft_redistribution_t *redistribution, cp_mpi_comm_t comm,
    const cp_mpi_comm_t sub_comm[2]) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "3d_bw_c2r_r");
  const int handle = fft_start_timer(routine_name);
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "3d_bw_c2r_r_%i_%i_%i_%i",
           npts_global[0], npts_global[1], npts_global[2],
           cp_mpi_comm_size(comm));
  const int handle2 = fft_start_timer(routine_name);

  const int my_process = cp_mpi_comm_rank(comm);

  double complex *grid_buffer_1 = get_buffer_1();
  double complex *grid_buffer_2 = get_buffer_2();
  double complex *grid_buffer_dev_1 = fft_gpu_get_buffer_1();
  double complex *grid_buffer_dev_2 = fft_gpu_get_buffer_2();
  void *fft_stream = fft_gpu_get_stream();

  // Collect the local sizes (for buffer sizes and FFT dimensions)
  int fft_sizes_rs[3] = {proc2local_rs[my_process][0][1],
                         proc2local_rs[my_process][1][1],
                         proc2local_rs[my_process][2][1]};
  int fft_sizes_ms[3] = {proc2local_ms[my_process][0][1],
                         proc2local_ms[my_process][1][1],
                         proc2local_ms[my_process][2][1]};
  int number_of_local_xy_rays = rays_per_process[my_process];

  int proc_grid[2];
  int periods[2];
  int my_coord[2];
  cp_mpi_cart_get(comm, 2, proc_grid, periods, my_coord);

  // We use different data distribution schemes depending on the
  // availability of a distributed FFT library because FFTW requires the
  // data to the different FFTs to be consecutively stored in memory. This
  // is not possible without a distributed FFT library because this would
  // require the implementation of the Guru interface which is not available
  // with all implementations of the FFTW interface
  if (proc_grid[0] > 1 && proc_grid[1] > 1) {
    memset(grid_buffer_1, 0,
           number_of_local_xy_rays * npts_global[2] * sizeof(double complex));
#pragma omp parallel for default(none) shared(                                 \
        number_of_positive_points, index_to_ray_pos, grid_buffer_1, grid_gs)
    for (int i = 0; i < number_of_positive_points; i++) {
      grid_buffer_1[index_to_ray_pos[i][1]] = grid_gs[index_to_ray_pos[i][0]];
    }
    FFT_H2D(grid_buffer_dev_1, grid_buffer_1,
            number_of_local_xy_rays * npts_global[2] * sizeof(double complex));

    // Perform the first FFT (xy_d,z) -> (z,xy_d)
    fft_gpu_1d_local_device(false, npts_global[2], number_of_local_xy_rays,
                            false, false, npts_global[2], npts_global[2],
                            grid_buffer_dev_1, grid_buffer_dev_2);

    // Perform transpose (z,xy_d) -> (z_d,x_d,y)
    collect_xy_and_distribute_z_ray_pack_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, fft_stream);
    offloadStreamSynchronize((offloadStream_t)fft_stream);
    collect_xy_and_distribute_z_ray_comm_device(
        grid_buffer_dev_1, grid_buffer_dev_2, redistribution, comm);
    collect_xy_and_distribute_z_ray_unpack_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, fft_stream);

    // Perform the second FFT (z_d,x_d,y) -> (y,z_d,x_d)
    fft_gpu_1d_local_device(false, npts_global[1],
                            fft_sizes_ms[0] * fft_sizes_ms[2], false, true,
                            npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2],
                            grid_buffer_dev_1, grid_buffer_dev_2);

    // Perform second transpose (y,z_d,x_d) -> (y_d,z_d,x)
    offloadStreamSynchronize((offloadStream_t)fft_stream);
    collect_x_and_distribute_y_blocked_comm_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, sub_comm[1]);
    // Unpack the buffer
    collect_x_and_distribute_y_blocked_unpack_device(
        grid_buffer_dev_1, grid_buffer_dev_2, redistribution, proc2local_x_gs,
        fft_stream);

    // Perform the third FFT (y_d,z_d,x) -> (x,y_d,z_d)
    FFT_D2H(grid_buffer_2, grid_buffer_dev_2,
            (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
    fft_gpu_1d_c2r_local(npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2],
                         false, true, npts_global[0] / 2 + 1,
                         fft_sizes_rs[1] * fft_sizes_rs[2], grid_buffer_2,
                         grid_rs);

  } else if (proc_grid[0] > 1) {

    memset(grid_buffer_1, 0,
           number_of_local_xy_rays * npts_global[2] * sizeof(double complex));
#pragma omp parallel for default(none) shared(                                 \
        number_of_positive_points, index_to_ray_pos, grid_buffer_1, grid_gs)
    for (int i = 0; i < number_of_positive_points; i++) {
      grid_buffer_1[index_to_ray_pos[i][1]] = grid_gs[index_to_ray_pos[i][0]];
    }
    FFT_H2D(grid_buffer_dev_1, grid_buffer_1,
            number_of_local_xy_rays * npts_global[2] * sizeof(double complex));
    // Perform the first FFT (xy_d,z) -> (z,xy_d)
    fft_gpu_1d_local_device(false, npts_global[2], number_of_local_xy_rays,
                            false, false, npts_global[2], npts_global[2],
                            grid_buffer_dev_1, grid_buffer_dev_2);

    // Perform transpose (z,xy_d) -> (z_d,x,y)
    collect_xy_and_distribute_z_ray_pack_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, fft_stream);
    offloadStreamSynchronize((offloadStream_t)fft_stream);
    collect_xy_and_distribute_z_ray_comm_device(
        grid_buffer_dev_1, grid_buffer_dev_2, redistribution, comm);
    collect_xy_and_distribute_z_ray_unpack_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, fft_stream);

    // second FFT (z_d,x,y) -> (y,z_d,x)
    fft_gpu_1d_local_device(
        false, npts_global[1], npts_global_gspace[0] * fft_sizes_ms[2], false,
        true, npts_global[1], npts_global_gspace[0] * fft_sizes_ms[2],
        grid_buffer_dev_1, grid_buffer_dev_2);
    // third FFT (y,z_d,x) -> (x,y,z_d)
    FFT_D2H(grid_buffer_2, grid_buffer_dev_2,
            (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
    fft_gpu_1d_c2r_local(npts_global[0], npts_global[1] * fft_sizes_ms[2],
                         false, true, npts_global[0] / 2 + 1,
                         npts_global[1] * fft_sizes_ms[2], grid_buffer_2,
                         grid_rs);

  } else if (proc_grid[1] > 1) {
    memset(grid_buffer_1, 0,
           number_of_local_xy_rays * npts_global[2] * sizeof(double complex));
#pragma omp parallel for default(none) shared(                                 \
        number_of_positive_points, index_to_ray_pos, grid_buffer_1, grid_gs)
    for (int i = 0; i < number_of_positive_points; i++) {
      grid_buffer_1[index_to_ray_pos[i][1]] = grid_gs[index_to_ray_pos[i][0]];
    }
    FFT_H2D(grid_buffer_dev_1, grid_buffer_1,
            number_of_local_xy_rays * npts_global[2] * sizeof(double complex));

    // Perform the first FFT (xy_d,z) -> (z,xy_d)
    fft_gpu_1d_local_device(false, npts_global[2], number_of_local_xy_rays,
                            false, false, npts_global[2], npts_global[2],
                            grid_buffer_dev_1, grid_buffer_dev_2);

    // Perform transpose (z,xy_d) -> (z_d,x_d,y)
    collect_xy_and_distribute_z_ray_pack_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, fft_stream);
    offloadStreamSynchronize((offloadStream_t)fft_stream);
    collect_xy_and_distribute_z_ray_comm_device(
        grid_buffer_dev_1, grid_buffer_dev_2, redistribution, comm);
    collect_xy_and_distribute_z_ray_unpack_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, fft_stream);

    // Perform the second FFT (z_d,x_d,y) -> (y,z_d,x_d)
    fft_gpu_1d_local_device(false, npts_global[1],
                            fft_sizes_ms[0] * fft_sizes_ms[2], false, true,
                            npts_global[1], fft_sizes_ms[0] * fft_sizes_ms[2],
                            grid_buffer_dev_1, grid_buffer_dev_2);

    // Perform second transpose (y,z_d,x_d) -> (y_d,z_d,x)
    offloadStreamSynchronize((offloadStream_t)fft_stream);
    collect_x_and_distribute_y_blocked_comm_device(
        grid_buffer_dev_2, grid_buffer_dev_1, redistribution, sub_comm[1]);
    // Unpack the buffer
    collect_x_and_distribute_y_blocked_unpack_device(
        grid_buffer_dev_1, grid_buffer_dev_2, redistribution, proc2local_x_gs,
        fft_stream);

    // Perform the third FFT (y_d,z_d,x) -> (x,y_d,z_d)
    FFT_D2H(grid_buffer_2, grid_buffer_dev_2,
            (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
    fft_gpu_1d_c2r_local(npts_global[0], fft_sizes_rs[1] * fft_sizes_rs[2],
                         false, true, npts_global[0] / 2 + 1,
                         fft_sizes_rs[1] * fft_sizes_rs[2], grid_buffer_2,
                         grid_rs);

  } else {
    memset(grid_buffer_2, 0,
           product3(npts_global_gspace) * sizeof(double complex));
#pragma omp parallel for default(none) shared(                                 \
        number_of_positive_points, index_to_ray_pos, grid_buffer_2, grid_gs)
    for (int i = 0; i < number_of_positive_points; i++) {
      grid_buffer_2[index_to_ray_pos[i][1]] = grid_gs[index_to_ray_pos[i][0]];
    }
    FFT_H2D(grid_buffer_dev_2, grid_buffer_2,
            product3(npts_global_gspace) * sizeof(double complex));

    // second FFT (z_d,x,y) -> (y,z_d,x)
    fft_gpu_2d_local_device(false,
                            (const int[2]){npts_global[1], npts_global[2]},
                            npts_global_gspace[0], false, false,
                            grid_buffer_dev_2, grid_buffer_dev_1);
    // third FFT (y,z_d,x) -> (x,y,z_d)
    FFT_D2H(grid_buffer_1, grid_buffer_dev_1,
            (size_t)fft_gpu_get_buffer_size() * sizeof(double complex));
    fft_gpu_1d_c2r_local(npts_global[0], npts_global[1] * npts_global[2], true,
                         true, npts_global[1] * npts_global[2],
                         npts_global[1] * npts_global[2], grid_buffer_1,
                         grid_rs);
  }

  fft_stop_timer(handle2);
  fft_stop_timer(handle);
}

// EOF
