/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "fft_lib.h"
#include "fft_lib_fftw.h"
#include "fft_lib_ref.h"
#include "fft_timer.h"

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__FFTW3)
fft_lib fft_lib_choice = FFT_LIB_FFTW;
#else
fft_lib fft_lib_choice = FFT_LIB_REF;
#endif
bool fft_lib_initialized = false;

/*******************************************************************************
 * \brief Initialize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_init_lib(const fft_lib lib, const int fftw_planning_flag,
                  const bool use_fft_mpi, const char *wisdom_file) {
  if (fft_lib_initialized) {
    return;
  }
  fft_lib_initialized = true;
  fft_lib_choice = lib;
  fft_ref_init_lib();
  fft_fftw_init_lib(fftw_planning_flag, use_fft_mpi, wisdom_file);
  switch (fft_lib_choice) {
  case FFT_LIB_REF:
    printf("Using reference FFT library.\n");
    break;
  case FFT_LIB_FFTW:
    printf("Using FFTW library.\n");
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
}

/*******************************************************************************
 * \brief Finalize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_finalize_lib(const char *wisdom_file) {
  fft_ref_finalize_lib();
  fft_fftw_finalize_lib(wisdom_file);
  fft_lib_initialized = false;
}

/*******************************************************************************
 * \brief Whether a compound MPI implementation is available.
 * \author Frederick Stein
 ******************************************************************************/
bool fft_lib_use_mpi() {
  switch (fft_lib_choice) {
  case FFT_LIB_REF:
    return fft_ref_lib_use_mpi();
  case FFT_LIB_FFTW:
    return fft_fftw_lib_use_mpi();
  default:
    assert(0 && "Unknown FFT library.");
    return false;
  }
}

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_allocate_double(const int length, double **buffer) {
  if (fft_lib_choice == FFT_LIB_REF) {
    fft_ref_allocate_double(length, buffer);
  } else if (fft_lib_choice == FFT_LIB_FFTW) {
    fft_fftw_allocate_double(length, buffer);
  } else {
    assert(0 && "Unknown FFT library.");
  }
}

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_allocate_complex(const int length, double complex **buffer) {
  if (fft_lib_choice == FFT_LIB_REF) {
    fft_ref_allocate_complex(length, buffer);
  } else if (fft_lib_choice == FFT_LIB_FFTW) {
    fft_fftw_allocate_complex(length, buffer);
  } else {
    assert(0 && "Unknown FFT library.");
  }
}

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_free_double(double *buffer) {
  if (fft_lib_choice == FFT_LIB_REF) {
    fft_ref_free_double(buffer);
  } else if (fft_lib_choice == FFT_LIB_FFTW) {
    fft_fftw_free_double(buffer);
  } else {
    assert(0 && "Unknown FFT library.");
  }
}

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_free_complex(double complex *buffer) {
  if (fft_lib_choice == FFT_LIB_REF) {
    fft_ref_free_complex(buffer);
  } else if (fft_lib_choice == FFT_LIB_FFTW) {
    fft_fftw_free_complex(buffer);
  } else {
    assert(0 && "Unknown FFT library.");
  }
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_1d_fw_local(const int fft_size, const int number_of_ffts,
                     const bool transpose_rs, const bool transpose_gs,
                     double complex *restrict grid_in, double complex *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_1d_fw_c2c_local_%i_%i",
           fft_size, number_of_ffts);
  const int handle = fft_start_timer(routine_name);
  switch (fft_lib_choice) {
  case FFT_LIB_REF:
    fft_ref_1d_fw_local(grid_in, grid_out, fft_size, number_of_ffts,
                        transpose_rs, transpose_gs);
    break;
  case FFT_LIB_FFTW:
    fft_fftw_1d_fw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                         grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_1d_fw_local_r2c(const int fft_size, const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs,
                         double *restrict grid_in, double complex *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_1d_fw_r2c_local_%i_%i",
           fft_size, number_of_ffts);
  const int handle = fft_start_timer(routine_name);
  switch (fft_lib_choice) {
  case FFT_LIB_REF:
    fft_ref_1d_fw_local_r2c(grid_in, grid_out, fft_size, number_of_ffts,
                            transpose_rs, transpose_gs);
    break;
  case FFT_LIB_FFTW:
    fft_fftw_1d_fw_local_r2c(fft_size, number_of_ffts, transpose_rs,
                             transpose_gs, grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Naive implementation of backwards FFT to transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_1d_bw_local(const int fft_size, const int number_of_ffts,
                     const bool transpose_rs, const bool transpose_gs,
                     double complex *restrict grid_in, double complex *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_1d_bw_c2c_local_%i_%i",
           fft_size, number_of_ffts);
  const int handle = fft_start_timer(routine_name);
  switch (fft_lib_choice) {
  case FFT_LIB_REF:
    fft_ref_1d_bw_local(grid_in, grid_out, fft_size, number_of_ffts,
                        transpose_rs, transpose_gs);
    break;
  case FFT_LIB_FFTW:
    fft_fftw_1d_bw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                         grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Naive implementation of backwards FFT to transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_1d_bw_local_c2r(const int fft_size, const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs,
                         double complex *restrict grid_in, double *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_1d_bw_c2r_local_%i_%i",
           fft_size, number_of_ffts);
  const int handle = fft_start_timer(routine_name);
  switch (fft_lib_choice) {
  case FFT_LIB_REF:
    fft_ref_1d_bw_local_c2r(grid_in, grid_out, fft_size, number_of_ffts,
                            transpose_rs, transpose_gs);
    break;
  case FFT_LIB_FFTW:
    fft_fftw_1d_bw_local_c2r(fft_size, number_of_ffts, transpose_rs,
                             transpose_gs, grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Naive implementation of 2D FFT (transposed format, no normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_fw_local(const int fft_size[2], const int number_of_ffts,
                     const bool transpose_rs, const bool transpose_gs,
                     double complex *restrict grid_in, double complex *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_2d_fw_c2c_local_%i_%i_%i",
           fft_size[0], fft_size[1], number_of_ffts);
  const int handle = fft_start_timer(routine_name);
  switch (fft_lib_choice) {
  case FFT_LIB_REF:
    fft_ref_2d_fw_local(grid_in, grid_out, fft_size, number_of_ffts,
                        transpose_rs, transpose_gs);
    break;
  case FFT_LIB_FFTW:
    fft_fftw_2d_fw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                         grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Naive implementation of 2D FFT (transposed format, no normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_fw_local_r2c(const int fft_size[2], const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs,
                         double *restrict grid_in, double complex *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_2d_fw_r2c_local_%i_%i_%i",
           fft_size[0], fft_size[1], number_of_ffts);
  const int handle = fft_start_timer(routine_name);
  switch (fft_lib_choice) {
  case FFT_LIB_REF:
    fft_ref_2d_fw_local_r2c(grid_in, grid_out, fft_size, number_of_ffts,
                            transpose_rs, transpose_gs);
    break;
  case FFT_LIB_FFTW:
    fft_fftw_2d_fw_local_r2c(fft_size, number_of_ffts, transpose_rs,
                             transpose_gs, grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs local 2D FFT (reverse to fw routine, no normalization).
 * \note fft_2d_bw_local(grid_gs, grid_rs, n1, n2, m) is the reverse to
 * fft_2d_rw_local(grid_rs, grid_gs, n1, n2, m) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_bw_local(const int fft_size[2], const int number_of_ffts,
                     const bool transpose_rs, const bool transpose_gs,
                     double complex *restrict grid_in, double complex *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_2d_bw_c2c_local_%i_%i_%i",
           fft_size[0], fft_size[1], number_of_ffts);
  const int handle = fft_start_timer(routine_name);
  switch (fft_lib_choice) {
  case FFT_LIB_REF:
    fft_ref_2d_bw_local(grid_in, grid_out, fft_size, number_of_ffts,
                        transpose_rs, transpose_gs);
    break;
  case FFT_LIB_FFTW:
    fft_fftw_2d_bw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                         grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs local 2D FFT (reverse to fw routine, no normalization).
 * \note fft_2d_bw_local(grid_gs, grid_rs, n1, n2, m) is the reverse to
 * fft_2d_rw_local(grid_rs, grid_gs, n1, n2, m) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_bw_local_c2r(const int fft_size[2], const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs,
                         double complex *restrict grid_in, double *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_2d_bw_c2r_local_%i_%i_%i",
           fft_size[0], fft_size[1], number_of_ffts);
  const int handle = fft_start_timer(routine_name);
  switch (fft_lib_choice) {
  case FFT_LIB_REF:
    fft_ref_2d_bw_local_c2r(grid_in, grid_out, fft_size, number_of_ffts,
                            transpose_rs, transpose_gs);
    break;
  case FFT_LIB_FFTW:
    fft_fftw_2d_bw_local_c2r(fft_size, number_of_ffts, transpose_rs,
                             transpose_gs, grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs local 3D FFT (no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_local(const int fft_size[3], double complex *restrict grid_in,
                     double complex *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_fw_c2c_local_%i_%i_%i",
           fft_size[0], fft_size[1], fft_size[2]);
  const int handle = fft_start_timer(routine_name);
  switch (fft_lib_choice) {
  case FFT_LIB_REF:
    fft_ref_3d_fw_local(grid_in, grid_out, fft_size);
    break;
  case FFT_LIB_FFTW:
    fft_fftw_3d_fw_local(fft_size, grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs local 3D FFT (no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_local_r2c(const int fft_size[3], double *restrict grid_in,
                         double complex *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_fw_r2c_local_%i_%i_%i",
           fft_size[0], fft_size[1], fft_size[2]);
  const int handle = fft_start_timer(routine_name);
  switch (fft_lib_choice) {
  case FFT_LIB_REF:
    fft_ref_3d_fw_local_r2c(grid_in, grid_out, fft_size);
    break;
  case FFT_LIB_FFTW:
    fft_fftw_3d_fw_local_r2c(fft_size, grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs local 3D FFT (reverse to fw routine, no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_local(const int fft_size[3], double complex *restrict grid_in,
                     double complex *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_bw_c2c_local_%i_%i_%i",
           fft_size[0], fft_size[1], fft_size[2]);
  const int handle = fft_start_timer(routine_name);
  switch (fft_lib_choice) {
  case FFT_LIB_REF:
    fft_ref_3d_bw_local(grid_in, grid_out, fft_size);
    break;
  case FFT_LIB_FFTW:
    fft_fftw_3d_bw_local(fft_size, grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs local 3D FFT (reverse to fw routine, no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_local_c2r(const int fft_size[3], double complex *restrict grid_in,
                         double *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH, "fft_3d_bw_c2r_local_%i_%i_%i",
           fft_size[0], fft_size[1], fft_size[2]);
  const int handle = fft_start_timer(routine_name);
  switch (fft_lib_choice) {
  case FFT_LIB_REF:
    fft_ref_3d_bw_local_c2r(grid_in, grid_out, fft_size);
    break;
  case FFT_LIB_FFTW:
    fft_fftw_3d_bw_local_c2r(fft_size, grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Return buffer size and local sizes and start for distributed 2D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_2d_distributed_sizes(const int npts_global[2], const int number_of_ffts,
                             const mp_mpi_comm comm, int *local_n0,
                             int *local_n0_start, int *local_n1,
                             int *local_n1_start) {
  if (fft_lib_choice == FFT_LIB_REF) {
    assert(0 && "Distributed 2D FFT not available.");
    return -1;
  } else if (fft_lib_choice == FFT_LIB_FFTW) {
    return fft_fftw_2d_distributed_sizes(npts_global, number_of_ffts, comm,
                                         local_n0, local_n0_start, local_n1,
                                         local_n1_start);
  } else {
    assert(0 && "Unknown FFT library.");
    return -1;
  }
}

/*******************************************************************************
 * \brief Return buffer size and local sizes and start for distributed 2D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_2d_distributed_sizes_r2c(const int npts_global[2],
                                 const int number_of_ffts,
                                 const mp_mpi_comm comm, int *local_n0,
                                 int *local_n0_start, int *local_n1,
                                 int *local_n1_start) {
  if (fft_lib_choice == FFT_LIB_REF) {
    assert(0 && "Distributed 2D FFT not available.");
    return -1;
  } else if (fft_lib_choice == FFT_LIB_FFTW) {
    return fft_fftw_2d_distributed_sizes_r2c(npts_global, number_of_ffts, comm,
                                             local_n0, local_n0_start, local_n1,
                                             local_n1_start);
  } else {
    assert(0 && "Unknown FFT library.");
    return -1;
  }
}

/*******************************************************************************
 * \brief Return buffer size and local sizes and start for distributed 3D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_3d_distributed_sizes(const int npts_global[3], const mp_mpi_comm comm,
                             int *local_n2, int *local_n2_start, int *local_n1,
                             int *local_n1_start) {
  if (fft_lib_choice == FFT_LIB_REF) {
    assert(0 && "Distributed 3D FFT not available.");
    return -1;
  } else if (fft_lib_choice == FFT_LIB_FFTW) {
    return fft_fftw_3d_distributed_sizes(
        npts_global, comm, local_n2, local_n2_start, local_n1, local_n1_start);
  } else {
    assert(0 && "Unknown FFT library.");
    return -1;
  }
}

/*******************************************************************************
 * \brief Return buffer size and local sizes and start for distributed 3D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_3d_distributed_sizes_r2c(const int npts_global[3],
                                 const mp_mpi_comm comm, int *local_n0,
                                 int *local_n0_start, int *local_n1,
                                 int *local_n1_start) {
  if (fft_lib_choice == FFT_LIB_REF) {
    assert(0 && "Distributed 3D FFT not available.");
    return -1;
  } else if (fft_lib_choice == FFT_LIB_FFTW) {
    return fft_fftw_3d_distributed_sizes_r2c(
        npts_global, comm, local_n0, local_n0_start, local_n1, local_n1_start);
  } else {
    assert(0 && "Unknown FFT library.");
    return -1;
  }
}

/*******************************************************************************
 * \brief Performs a distributed 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_fw_distributed(const int npts_global[2], const int number_of_ffts,
                           const mp_mpi_comm comm, double complex *restrict grid_in,
                           double complex *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH,
           "fft_2d_fw_c2c_distr_%i_%i_%i_%i", mp_mpi_comm_size(comm),
           npts_global[0], npts_global[1], number_of_ffts);
  const int handle = fft_start_timer(routine_name);
  assert(fft_lib_use_mpi());
  switch (fft_lib_choice) {
  case FFT_LIB_FFTW:
    fft_fftw_2d_fw_distributed(npts_global, number_of_ffts, comm, grid_in,
                               grid_out);
    break;
  default:
    assert(0 && "Distributed 2D FFT not available.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a distributed 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_fw_distributed_r2c(const int npts_global[2],
                               const int number_of_ffts, const mp_mpi_comm comm,
                               double *restrict grid_in, double complex *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH,
           "fft_2d_fw_r2c_distr_%i_%i_%i_%i", mp_mpi_comm_size(comm),
           npts_global[0], npts_global[1], number_of_ffts);
  const int handle = fft_start_timer(routine_name);
  assert(fft_lib_use_mpi());
  switch (fft_lib_choice) {
  case FFT_LIB_FFTW:
    fft_fftw_2d_fw_distributed_r2c(npts_global, number_of_ffts, comm, grid_in,
                                   grid_out);
    break;
  default:
    assert(0 && "Distributed 2D FFT not available.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a distributed 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_bw_distributed(const int npts_global[2], const int number_of_ffts,
                           const mp_mpi_comm comm, double complex *restrict grid_in,
                           double complex *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH,
           "fft_2d_bw_c2c_distr_%i_%i_%i_%i", mp_mpi_comm_size(comm),
           npts_global[0], npts_global[1], number_of_ffts);
  const int handle = fft_start_timer(routine_name);
  assert(fft_lib_use_mpi());
  switch (fft_lib_choice) {
  case FFT_LIB_FFTW:
    fft_fftw_2d_bw_distributed(npts_global, number_of_ffts, comm, grid_in,
                               grid_out);
    break;
  default:
    assert(0 && "Distributed 2D FFT not available.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a distributed 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_bw_distributed_c2r(const int npts_global[2],
                               const int number_of_ffts, const mp_mpi_comm comm,
                               double complex *restrict grid_in, double *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH,
           "fft_2d_bw_c2r_distr_%i_%i_%i_%i", mp_mpi_comm_size(comm),
           npts_global[0], npts_global[1], number_of_ffts);
  const int handle = fft_start_timer(routine_name);
  assert(fft_lib_use_mpi());
  switch (fft_lib_choice) {
  case FFT_LIB_FFTW:
    fft_fftw_2d_bw_distributed_c2r(npts_global, number_of_ffts, comm, grid_in,
                                   grid_out);
    break;
  default:
    assert(0 && "Distributed 2D FFT not available.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a distributed 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_distributed(const int npts_global[3], const mp_mpi_comm comm,
                           double complex *restrict grid_in, double complex *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH,
           "fft_3d_fw_c2c_distr_%i_%i_%i_%i", mp_mpi_comm_size(comm),
           npts_global[0], npts_global[1], npts_global[2]);
  const int handle = fft_start_timer(routine_name);
  assert(fft_lib_use_mpi());
  switch (fft_lib_choice) {
  case FFT_LIB_FFTW:
    fft_fftw_3d_fw_distributed(npts_global, comm, grid_in, grid_out);
    break;
  default:
    assert(0 && "Distributed 3D FFT not available.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a distributed 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_distributed_r2c(const int npts_global[3], const mp_mpi_comm comm,
                               double *restrict grid_in, double complex *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH,
           "fft_3d_fw_r2c_distr_%i_%i_%i_%i", mp_mpi_comm_size(comm),
           npts_global[0], npts_global[1], npts_global[2]);
  const int handle = fft_start_timer(routine_name);
  assert(fft_lib_use_mpi());
  switch (fft_lib_choice) {
  case FFT_LIB_FFTW:
    fft_fftw_3d_fw_distributed_r2c(npts_global, comm, grid_in, grid_out);
    break;
  default:
    assert(0 && "Distributed 3D FFT not available.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a distributed 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_distributed(const int npts_global[3], const mp_mpi_comm comm,
                           double complex *restrict grid_in, double complex *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH,
           "fft_3d_bw_c2c_distr_%i_%i_%i_%i", mp_mpi_comm_size(comm),
           npts_global[0], npts_global[1], npts_global[2]);
  const int handle = fft_start_timer(routine_name);
  assert(fft_lib_use_mpi());
  switch (fft_lib_choice) {
  case FFT_LIB_FFTW:
    fft_fftw_3d_bw_distributed(npts_global, comm, grid_in, grid_out);
    break;
  default:
    assert(0 && "Distributed 3D FFT not available.");
  }
  fft_stop_timer(handle);
}

/*******************************************************************************
 * \brief Performs a distributed 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_distributed_c2r(const int npts_global[3], const mp_mpi_comm comm,
                               double complex *restrict grid_in, double *restrict grid_out) {
  char routine_name[FFT_MAX_STRING_LENGTH + 1];
  memset(routine_name, '\0', FFT_MAX_STRING_LENGTH + 1);
  snprintf(routine_name, FFT_MAX_STRING_LENGTH,
           "fft_3d_bw_c2r_distr_%i_%i_%i_%i", mp_mpi_comm_size(comm),
           npts_global[0], npts_global[1], npts_global[2]);
  const int handle = fft_start_timer(routine_name);
  assert(fft_lib_use_mpi());
  switch (fft_lib_choice) {
  case FFT_LIB_FFTW:
    fft_fftw_3d_bw_distributed_c2r(npts_global, comm, grid_in, grid_out);
    break;
  default:
    assert(0 && "Distributed 3D FFT not available.");
  }
  fft_stop_timer(handle);
}

// EOF
