/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "../mpiwrap/mp_mpi.h"
#include "fft_grid.h"
#include "fft_grid_layout.h"
#include "fft_timer.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*******************************************************************************
 * \brief Performance test for the FFT code.
 * \author Frederick Stein
 ******************************************************************************/
static void run_test(const int fft_size[3], const int number_of_runs,
                     const bool use_halfspace) {
  const double dh_inv[3][3] = {
      {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  fft_grid_layout *fft_grid_layout = NULL;
  grid_create_fft_grid_layout(&fft_grid_layout, mp_mpi_comm_world, fft_size,
                              dh_inv, use_halfspace);

  fft_real_rs_grid grid_rs;
  grid_create_real_rs_grid(&grid_rs, fft_grid_layout);
  fft_complex_gs_grid grid_gs;
  grid_create_complex_gs_grid(&grid_gs, fft_grid_layout);

  const int(*my_bound)[2] =
      fft_grid_layout->proc2local_rs[mp_mpi_comm_rank(mp_mpi_comm_world)];
  memset(grid_rs.data, 0,
         (my_bound[0][1] - my_bound[0][0] + 1) *
             (my_bound[1][1] - my_bound[1][0] + 1) *
             (my_bound[2][1] - my_bound[2][0] + 1) * sizeof(double));
  mp_mpi_barrier(mp_mpi_comm_world);

  clock_t begin = clock();
  fft_3d_fw_r2c(&grid_rs, &grid_gs);
  fft_3d_bw_c2r(&grid_gs, &grid_rs);
  mp_mpi_barrier(mp_mpi_comm_world);
  clock_t end = clock();

  if (mp_mpi_comm_rank(mp_mpi_comm_world) == 0) {
    printf("Planning time for FW and BW %s FFTs of size %i %i %i : %f\n",
           use_halfspace ? "R2C/C2R" : "C2C", fft_size[0], fft_size[1],
           fft_size[2], (double)(end - begin) / CLOCKS_PER_SEC);
    fflush(stdout);
  }

  double min_time = -1.0;
  double max_time = -1.0;
  double sum_time = 0.0;
  for (int run = 0; run < number_of_runs; run++) {
    mp_mpi_barrier(mp_mpi_comm_world);
    begin = clock();
    fft_3d_fw_r2c(&grid_rs, &grid_gs);
    fft_3d_bw_c2r(&grid_gs, &grid_rs);
    mp_mpi_barrier(mp_mpi_comm_world);
    end = clock();
    const double current_time = (double)(end - begin) / CLOCKS_PER_SEC;
    min_time = min_time < 0.0 ? current_time : fmin(min_time, current_time);
    max_time = fmax(max_time, current_time);
    sum_time += current_time;
  }

  grid_free_real_rs_grid(&grid_rs);
  grid_free_complex_gs_grid(&grid_gs);
  grid_free_fft_grid_layout(fft_grid_layout);

  if (mp_mpi_comm_rank(mp_mpi_comm_world) == 0) {
    printf("Time for %i FW and BW %s FFTs of size %i %i %i : min %f, max %f, "
           "avg %f\n",
           number_of_runs, use_halfspace ? "R2C/C2R" : "C2C", fft_size[0],
           fft_size[1], fft_size[2], min_time, max_time,
           sum_time / number_of_runs);
  }
}

int main(int argc, char *argv[]) {
  mp_mpi_init(&argc, &argv);

  if (mp_mpi_comm_rank(mp_mpi_comm_world) == 0) {
    printf("Number of processes: %i\n",
           mp_mpi_comm_size(mp_mpi_comm_world));
    printf("Number of threads per process: %i\n", omp_get_max_threads());
    fflush(stdout);
  }

  fft_init_timer(true);
  fft_init_lib(FFT_LIB_REF, FFT_PATIENT, true, NULL);

  // These are approximate grid sizes of the finest grid level for the standard
  // benchmark systems in benchmarks/QS
  run_test((const int[3]){100, 100, 100}, 10, false);
  run_test((const int[3]){125, 125, 125}, 10, false);
  run_test((const int[3]){160, 160, 160}, 10, false);
  run_test((const int[3]){200, 200, 200}, 10, false);
  run_test((const int[3]){250, 250, 250}, 10, false);
  // run_test((const int[3]){315, 315, 315}, 10, false);
  // run_test((const int[3]){400, 400, 400}, 10, false);
  // run_test((const int[3]){500, 500, 500}, 10, false);
  // run_test((const int[3]){630, 630, 630}, 10, false);
  //  QS_low_scaling_GW
  run_test((const int[3]){600, 180, 120}, 10, false);

  // Repeat using the half-space formalism (R2C/C2R FFTs)
  run_test((const int[3]){100, 100, 100}, 10, true);
  run_test((const int[3]){125, 125, 125}, 10, true);
  run_test((const int[3]){160, 160, 160}, 10, true);
  run_test((const int[3]){200, 200, 200}, 10, true);
  run_test((const int[3]){250, 250, 250}, 10, true);
  // run_test((const int[3]){315, 315, 315}, 10, true);
  // run_test((const int[3]){400, 400, 400}, 10, true);
  // run_test((const int[3]){500, 500, 500}, 10, true);
  // run_test((const int[3]){630, 630, 630}, 10, true);
  //  QS_low_scaling_GW
  run_test((const int[3]){600, 180, 120}, 10, true);

  fft_print_timing_report(mp_mpi_comm_world);

  // Test also the reference backend and without distributed FFTs from the
  // library
  if (fft_lib_use_mpi()) {
    fft_finalize_timer();
    fft_finalize_lib(NULL);
    fft_init_timer(true);
    fft_init_lib(FFT_LIB_DEFAULT, FFT_PATIENT, false, NULL);

    // These are approximate grid sizes of the finest grid level for the
    // standard benchmark systems in benchmarks/QS
    run_test((const int[3]){100, 100, 100}, 10, false);
    run_test((const int[3]){125, 125, 125}, 10, false);
    run_test((const int[3]){160, 160, 160}, 10, false);
    run_test((const int[3]){200, 200, 200}, 10, false);
    run_test((const int[3]){250, 250, 250}, 10, false);
    // run_test((const int[3]){315, 315, 315}, 10, false);
    // run_test((const int[3]){400, 400, 400}, 10, false);
    // run_test((const int[3]){500, 500, 500}, 10, false);
    // run_test((const int[3]){630, 630, 630}, 10, false);
    //  QS_low_scaling_GW
    run_test((const int[3]){600, 180, 120}, 10, false);

    // Repeat using the half-space formalism (R2C/C2R FFTs)
    run_test((const int[3]){100, 100, 100}, 10, true);
    run_test((const int[3]){125, 125, 125}, 10, true);
    run_test((const int[3]){160, 160, 160}, 10, true);
    run_test((const int[3]){200, 200, 200}, 10, true);
    run_test((const int[3]){250, 250, 250}, 10, true);
    // run_test((const int[3]){315, 315, 315}, 10, true);
    // run_test((const int[3]){400, 400, 400}, 10, true);
    // run_test((const int[3]){500, 500, 500}, 10, true);
    // run_test((const int[3]){630, 630, 630}, 10, true);
    //  QS_low_scaling_GW
    run_test((const int[3]){600, 180, 120}, 10, true);

    fft_print_timing_report(mp_mpi_comm_world);
  }

  fft_finalize_lib(NULL);
  fft_finalize_timer();

  mp_mpi_finalize();

  return 0;
}

// EOF
