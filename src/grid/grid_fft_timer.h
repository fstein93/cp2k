/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2025 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef GRID_FFT_TIMER_H
#define GRID_FFT_TIMER_H

#include "common/grid_mpi.h"

#define FFT_MAX_STRING_LENGTH 35

/*******************************************************************************
 * \brief Initializes the internal timer.
 * \author Frederick Stein
 ******************************************************************************/
void fft_init_timer();

/*******************************************************************************
 * \brief Prints the timing report.
 * \author Frederick Stein
 ******************************************************************************/
void fft_print_timing_report(const grid_mpi_comm comm);

/*******************************************************************************
 * \brief Finalizes the internal timer.
 * \author Frederick Stein
 ******************************************************************************/
void fft_finalize_timer();

/*******************************************************************************
 * \brief Start a timing section.
 * \author Frederick Stein
 ******************************************************************************/
int fft_start_timer(const char *routine_name);

/*******************************************************************************
 * \brief Stop a timing.
 * \author Frederick Stein
 ******************************************************************************/
void fft_stop_timer(const int handle);

#endif /* GRID_FFT_TIMER_H */

// EOF
