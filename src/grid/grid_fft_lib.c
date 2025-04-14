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
void fft_init_lib(const grid_fft_lib lib) {
  if (grid_fft_lib_initialized) {
    return;
  }
  grid_fft_lib_initialized = true;
  grid_fft_lib_choice = lib;
  fft_ref_init_lib();
  fft_fftw_init_lib();
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
 * \brief Create a plan for a 1D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_create_1d_plan(double complex *grid_rs, double complex *grid_gs,
                        const int fft_size, const int number_of_ffts,
                        grid_fft_plan *plan) {
  assert(plan != NULL);
  plan->fft_size[0] = fft_size;
  plan->number_of_ffts = number_of_ffts;
  if (grid_fft_lib_choice == GRID_FFT_LIB_FFTW) {
    fft_fftw_create_1d_plan(grid_rs, grid_gs, fft_size, number_of_ffts,
                            &plan->fftw_plan_fw, &plan->fftw_plan_bw);
  }
}

/*******************************************************************************
 * \brief Create a plan for a 1D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_create_2d_plan(double complex *grid_rs, double complex *grid_gs,
                        const int fft_size[2], const int number_of_ffts,
                        grid_fft_plan *plan) {
  assert(plan != NULL);
  plan->fft_size[0] = fft_size[0];
  plan->fft_size[1] = fft_size[1];
  plan->number_of_ffts = number_of_ffts;
  if (grid_fft_lib_choice == GRID_FFT_LIB_FFTW) {
    fft_fftw_create_2d_plan(grid_rs, grid_gs, fft_size, number_of_ffts,
                            &plan->fftw_plan_fw, &plan->fftw_plan_bw);
  }
}

/*******************************************************************************
 * \brief Create a plan for a 1D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_create_3d_plan(const int fft_size[3], grid_fft_plan *plan) {
  assert(plan != NULL);
  plan->fft_size[0] = fft_size[0];
  plan->fft_size[1] = fft_size[1];
  plan->fft_size[2] = fft_size[2];
  plan->fftw_plan_fw = NULL;
  plan->fftw_plan_bw = NULL;
}

/*******************************************************************************
 * \brief Frees FFT plans.
 * \author Frederick Stein
 ******************************************************************************/
void fft_free_plan(grid_fft_plan *plan) {
  if (grid_fft_lib_choice == GRID_FFT_LIB_FFTW) {
    fft_fftw_free_plan(&plan->fftw_plan_fw, &plan->fftw_plan_bw);
  }
}

/*******************************************************************************
 * \brief Naive implementation of FFT from transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_1d_fw_local(const grid_fft_plan *plan, double complex *grid_in,
                     double complex *grid_out) {
  switch (grid_fft_lib_choice) {
  case GRID_FFT_LIB_REF:
    fft_ref_1d_fw_local(grid_in, grid_out, plan->fft_size[0],
                        plan->number_of_ffts);
    break;
  case GRID_FFT_LIB_FFTW:
    fft_fftw_1d_fw_local(plan->fftw_plan_fw, grid_in, grid_out);
    break;
  default:
    assert(0 && "Unknown FFT library.");
  }
}

/*******************************************************************************
 * \brief Naive implementation of backwards FFT to transposed format (for easier
 *transposition). \author Frederick Stein
 ******************************************************************************/
void fft_1d_bw_local(const grid_fft_plan *plan, double complex *grid_in,
                     double complex *grid_out) {
  switch (grid_fft_lib_choice) {
  case GRID_FFT_LIB_REF:
    fft_ref_1d_bw_local(grid_in, grid_out, plan->fft_size[0],
                        plan->number_of_ffts);
    break;
  case GRID_FFT_LIB_FFTW:
    fft_fftw_1d_bw_local(plan->fftw_plan_bw, grid_in, grid_out);
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
void fft_2d_fw_local(const grid_fft_plan *plan, double complex *grid_in,
                     double complex *grid_out) {
  switch (grid_fft_lib_choice) {
  case GRID_FFT_LIB_REF:
    fft_ref_2d_fw_local(grid_in, grid_out, plan->fft_size[0], plan->fft_size[1],
                        plan->number_of_ffts);
    break;
  case GRID_FFT_LIB_FFTW:
    fft_fftw_2d_fw_local(plan->fftw_plan_fw, grid_in, grid_out);
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
void fft_2d_bw_local(const grid_fft_plan *plan, double complex *grid_in,
                     double complex *grid_out) {
  switch (grid_fft_lib_choice) {
  case GRID_FFT_LIB_REF:
    fft_ref_2d_bw_local(grid_in, grid_out, plan->fft_size[0], plan->fft_size[1],
                        plan->number_of_ffts);
    break;
  case GRID_FFT_LIB_FFTW:
    fft_fftw_2d_bw_local(plan->fftw_plan_bw, grid_in, grid_out);
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

// EOF
