/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/

#include "grid_fft_lib.h"
#include "grid_fft_lib_fftw.h"
#include "grid_fft_lib_ref.h"

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__FFTW3)
grid_fft_lib grid_fft_lib_choice = GRID_FFT_LIB_FFTW;
#else
grid_fft_lib grid_fft_lib_choice = GRID_FFT_LIB_REF;
#endif
bool grid_fft_lib_initialized = false;

/*******************************************************************************
 * \brief Initialize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_init_lib(const grid_fft_lib lib, const int fftw_planning_flag,
                  const bool use_fft_mpi) {
  if (grid_fft_lib_initialized) {
    return;
  }
  grid_fft_lib_initialized = true;
  grid_fft_lib_choice = lib;
  fft_ref_init_lib();
  fft_fftw_init_lib(fftw_planning_flag, use_fft_mpi);
  switch (grid_fft_lib_choice) {
  case GRID_FFT_LIB_REF:
    printf("Using reference FFT library.\n");
    break;
  case GRID_FFT_LIB_FFTW:
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
void fft_finalize_lib() {
  fft_ref_finalize_lib();
  fft_fftw_finalize_lib();
  grid_fft_lib_initialized = false;
}

/*******************************************************************************
 * \brief Whether a compound MPI implementation is available.
 * \author Frederick Stein
 ******************************************************************************/
bool fft_lib_use_mpi() {
  switch (grid_fft_lib_choice) {
  case GRID_FFT_LIB_REF:
    return fft_ref_lib_use_mpi();
  case GRID_FFT_LIB_FFTW:
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
  if (grid_fft_lib_choice == GRID_FFT_LIB_REF) {
    fft_ref_allocate_double(length, buffer);
  } else if (grid_fft_lib_choice == GRID_FFT_LIB_FFTW) {
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
  if (grid_fft_lib_choice == GRID_FFT_LIB_REF) {
    fft_ref_allocate_complex(length, buffer);
  } else if (grid_fft_lib_choice == GRID_FFT_LIB_FFTW) {
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
  if (grid_fft_lib_choice == GRID_FFT_LIB_REF) {
    fft_ref_free_double(buffer);
  } else if (grid_fft_lib_choice == GRID_FFT_LIB_FFTW) {
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
  if (grid_fft_lib_choice == GRID_FFT_LIB_REF) {
    fft_ref_free_complex(buffer);
  } else if (grid_fft_lib_choice == GRID_FFT_LIB_FFTW) {
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
                     double complex *grid_in, double complex *grid_out) {
  switch (grid_fft_lib_choice) {
  case GRID_FFT_LIB_REF:
    fft_ref_1d_fw_local(grid_in, grid_out, fft_size, number_of_ffts,
                        transpose_rs, transpose_gs);
    break;
  case GRID_FFT_LIB_FFTW:
    fft_fftw_1d_fw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                         grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
}

/*******************************************************************************
 * \brief Naive implementation of backwards FFT to transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_1d_bw_local(const int fft_size, const int number_of_ffts,
                     const bool transpose_rs, const bool transpose_gs,
                     double complex *grid_in, double complex *grid_out) {
  switch (grid_fft_lib_choice) {
  case GRID_FFT_LIB_REF:
    fft_ref_1d_bw_local(grid_in, grid_out, fft_size, number_of_ffts,
                        transpose_rs, transpose_gs);
    break;
  case GRID_FFT_LIB_FFTW:
    fft_fftw_1d_bw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                         grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
}

/*******************************************************************************
 * \brief Local transposition.
 * \author Frederick Stein
 ******************************************************************************/
void transpose_local(double complex *grid, double complex *grid_transposed,
                     const int number_of_columns_grid,
                     const int number_of_rows_grid) {
  switch (grid_fft_lib_choice) {
  case GRID_FFT_LIB_REF:
    fft_ref_transpose_local(grid, grid_transposed, number_of_columns_grid,
                            number_of_rows_grid);
    break;
  case GRID_FFT_LIB_FFTW:
    fft_fftw_transpose_local(grid, grid_transposed, number_of_columns_grid,
                             number_of_rows_grid);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
}

/*******************************************************************************
 * \brief Naive implementation of 2D FFT (transposed format, no normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_fw_local(const int fft_size[2], const int number_of_ffts,
                     const bool transpose_rs, const bool transpose_gs,
                     double complex *grid_in, double complex *grid_out) {
  switch (grid_fft_lib_choice) {
  case GRID_FFT_LIB_REF:
    fft_ref_2d_fw_local(grid_in, grid_out, fft_size, number_of_ffts,
                        transpose_rs, transpose_gs);
    break;
  case GRID_FFT_LIB_FFTW:
    fft_fftw_2d_fw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                         grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
}

/*******************************************************************************
 * \brief Performs local 2D FFT (reverse to fw routine, no normalization).
 * \note fft_2d_bw_local(grid_gs, grid_rs, n1, n2, m) is the reverse to
 * fft_2d_rw_local(grid_rs, grid_gs, n1, n2, m) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_bw_local(const int fft_size[2], const int number_of_ffts,
                     const bool transpose_rs, const bool transpose_gs,
                     double complex *grid_in, double complex *grid_out) {
  switch (grid_fft_lib_choice) {
  case GRID_FFT_LIB_REF:
    fft_ref_2d_bw_local(grid_in, grid_out, fft_size, number_of_ffts,
                        transpose_rs, transpose_gs);
    break;
  case GRID_FFT_LIB_FFTW:
    fft_fftw_2d_bw_local(fft_size, number_of_ffts, transpose_rs, transpose_gs,
                         grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
}

/*******************************************************************************
 * \brief Performs local 3D FFT (no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_local(const int fft_size[3], double complex *grid_in,
                     double complex *grid_out) {
  switch (grid_fft_lib_choice) {
  case GRID_FFT_LIB_REF:
    fft_ref_3d_fw_local(grid_in, grid_out, fft_size);
    break;
  case GRID_FFT_LIB_FFTW:
    fft_fftw_3d_fw_local(fft_size, grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
}

/*******************************************************************************
 * \brief Performs local 3D FFT (reverse to fw routine, no normalization).
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_local(const int fft_size[3], double complex *grid_in,
                     double complex *grid_out) {
  switch (grid_fft_lib_choice) {
  case GRID_FFT_LIB_REF:
    fft_ref_3d_bw_local(grid_in, grid_out, fft_size);
    break;
  case GRID_FFT_LIB_FFTW:
    fft_fftw_3d_bw_local(fft_size, grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
}

/*******************************************************************************
 * \brief Return buffer size and local sizes and start for distributed 2D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_2d_distributed_sizes(const int npts_global[2], const int number_of_ffts,
                             const grid_mpi_comm comm, int *local_n0,
                             int *local_n0_start) {
  if (grid_fft_lib_choice == GRID_FFT_LIB_REF) {
    assert(0 && "Distributed 2D FFT not available.");
    return -1;
  } else if (grid_fft_lib_choice == GRID_FFT_LIB_FFTW) {
    return fft_fftw_2d_distributed_sizes(npts_global, number_of_ffts, comm,
                                         local_n0, local_n0_start);
  } else {
    assert(0 && "Unknown FFT library.");
    return -1;
  }
}

/*******************************************************************************
 * \brief Return buffer size and local sizes and start for distributed 3D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_3d_distributed_sizes(const int npts_global[3], const grid_mpi_comm comm,
                             int *local_n2, int *local_n2_start, int *local_n1,
                             int *local_n1_start) {
  if (grid_fft_lib_choice == GRID_FFT_LIB_REF) {
    assert(0 && "Distributed 3D FFT not available.");
    return -1;
  } else if (grid_fft_lib_choice == GRID_FFT_LIB_FFTW) {
    return fft_fftw_3d_distributed_sizes(
        npts_global, comm, local_n2, local_n2_start, local_n1, local_n1_start);
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
                           const grid_mpi_comm comm, double complex *grid_in,
                           double complex *grid_out) {
  assert(fft_lib_use_mpi());
  switch (grid_fft_lib_choice) {
  case GRID_FFT_LIB_FFTW:
    fft_fftw_2d_fw_distributed(npts_global, number_of_ffts, comm, grid_in,
                               grid_out);
    break;
  default:
    assert(0 && "Distributed 2D FFT not available.");
  }
}

/*******************************************************************************
 * \brief Performs a distributed 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_fw_distributed(const int npts_global[3], const grid_mpi_comm comm,
                           double complex *grid_in, double complex *grid_out) {
  assert(fft_lib_use_mpi());
  switch (grid_fft_lib_choice) {
  case GRID_FFT_LIB_FFTW:
    fft_fftw_3d_fw_distributed(npts_global, comm, grid_in, grid_out);
    break;
  default:
    assert(0 && "Distributed 3D FFT not available.");
  }
}

/*******************************************************************************
 * \brief Performs a distributed 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_bw_distributed(const int npts_global[3], const grid_mpi_comm comm,
                           double complex *grid_in, double complex *grid_out) {
  assert(fft_lib_use_mpi());
  switch (grid_fft_lib_choice) {
  case GRID_FFT_LIB_FFTW:
    fft_fftw_3d_bw_distributed(npts_global, comm, grid_in, grid_out);
    break;
  default:
    assert(0 && "Distributed 3D FFT not available.");
  }
}

// EOF
