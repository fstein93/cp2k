/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef FFT_LIB_CUFFTMP_MPI_H
#define FFT_LIB_CUFFTMP_MPI_H

#include "../mpiwrap/cp_mpi.h"

#include <complex.h>
#include <stdbool.h>

/*******************************************************************************
 * \brief Initialize the cuFFTMp distributed FFT backend.
 * \author CP2K developers
 ******************************************************************************/
int fft_cufftmp_mpi_init_lib(void);

/*******************************************************************************
 * \brief Finalize the cuFFTMp distributed FFT backend and release all plans.
 * \author CP2K developers
 ******************************************************************************/
void fft_cufftmp_mpi_finalize_lib(void);

/*******************************************************************************
 * \brief Register (create) a distributed 2D C2C FFT.
 ******************************************************************************/
void fft_cufftmp_mpi_register_2d_distributed(const bool direction,
                                         const int npts_global[2],
                                         const int number_of_ffts,
                                         const cp_mpi_comm_t comm,
                                         double complex *restrict grid_in,
                                         double complex *restrict grid_out);

/*******************************************************************************
 * \brief Register (create) a distributed 2D R2C FFT.
 ******************************************************************************/
void fft_cufftmp_mpi_register_2d_r2c_distributed(const int npts_global[2],
                                             const int number_of_ffts,
                                             const cp_mpi_comm_t comm,
                                             double *restrict grid_in,
                                             double complex *restrict grid_out);

/*******************************************************************************
 * \brief Register (create) a distributed 2D C2R FFT.
 ******************************************************************************/
void fft_cufftmp_mpi_register_2d_c2r_distributed(const int npts_global[2],
                                             const int number_of_ffts,
                                             const cp_mpi_comm_t comm,
                                             double complex *restrict grid_in,
                                             double *restrict grid_out);

/*******************************************************************************
 * \brief Register (create) a distributed 3D C2C FFT.
 ******************************************************************************/
void fft_cufftmp_mpi_register_3d_distributed(const bool direction,
                                         const int npts_global[3],
                                         const cp_mpi_comm_t comm,
                                         double complex *restrict grid_in,
                                         double complex *restrict grid_out);

/*******************************************************************************
 * \brief Register (create) a distributed 3D R2C FFT.
 ******************************************************************************/
void fft_cufftmp_mpi_register_3d_r2c_distributed(const int npts_global[3],
                                             const cp_mpi_comm_t comm,
                                             double *restrict grid_in,
                                             double complex *restrict grid_out);

/*******************************************************************************
 * \brief Register (create) a distributed 3D C2R FFT.
 ******************************************************************************/
void fft_cufftmp_mpi_register_3d_c2r_distributed(const int npts_global[3],
                                             const cp_mpi_comm_t comm,
                                             double complex *restrict grid_in,
                                             double *restrict grid_out);

/*******************************************************************************
 * \brief Performs a distributed 2D C2C FFT.
 ******************************************************************************/
void fft_cufftmp_mpi_2d_distributed(const bool direction,
                                const int npts_global[2],
                                const int number_of_ffts,
                                const cp_mpi_comm_t comm,
                                double complex *restrict grid_in,
                                double complex *restrict grid_out);

/*******************************************************************************
 * \brief Performs a distributed 2D R2C FFT.
 ******************************************************************************/
void fft_cufftmp_mpi_2d_r2c_distributed(const int npts_global[2],
                                    const int number_of_ffts,
                                    const cp_mpi_comm_t comm,
                                    double *restrict grid_in,
                                    double complex *restrict grid_out);

/*******************************************************************************
 * \brief Performs a distributed 2D C2R FFT.
 ******************************************************************************/
void fft_cufftmp_mpi_2d_c2r_distributed(const int npts_global[2],
                                    const int number_of_ffts,
                                    const cp_mpi_comm_t comm,
                                    double complex *restrict grid_in,
                                    double *restrict grid_out);

/*******************************************************************************
 * \brief Performs a distributed 3D C2C FFT.
 ******************************************************************************/
void fft_cufftmp_mpi_3d_distributed(const bool direction,
                                const int npts_global[3],
                                const cp_mpi_comm_t comm,
                                double complex *restrict grid_in,
                                double complex *restrict grid_out);

/*******************************************************************************
 * \brief Performs a distributed 3D R2C FFT.
 ******************************************************************************/
void fft_cufftmp_mpi_3d_r2c_distributed(const int npts_global[3],
                                    const cp_mpi_comm_t comm,
                                    double *restrict grid_in,
                                    double complex *restrict grid_out);

/*******************************************************************************
 * \brief Performs a distributed 3D C2R FFT.
 ******************************************************************************/
void fft_cufftmp_mpi_3d_c2r_distributed(const int npts_global[3],
                                    const cp_mpi_comm_t comm,
                                    double complex *restrict grid_in,
                                    double *restrict grid_out);

/*******************************************************************************
 * \brief Return buffer size, local sizes and start for distributed 2D FFTs.
 ******************************************************************************/
int fft_cufftmp_mpi_2d_distributed_sizes(const int npts_global[2],
                                     const int number_of_ffts,
                                     const cp_mpi_comm_t comm, int *local_n0,
                                     int *local_n0_start, int *local_n1,
                                     int *local_n1_start);

/*******************************************************************************
 * \brief Return buffer size, local sizes and start for distributed 2D R2C FFTs.
 ******************************************************************************/
int fft_cufftmp_mpi_2d_distributed_sizes_r2c(const int npts_global[2],
                                         const int number_of_ffts,
                                         const cp_mpi_comm_t comm,
                                         int *local_n0, int *local_n0_start,
                                         int *local_n1, int *local_n1_start);

/*******************************************************************************
 * \brief Return buffer size, local sizes and start for distributed 3D FFTs.
 ******************************************************************************/
int fft_cufftmp_mpi_3d_distributed_sizes(const int npts_global[3],
                                     const cp_mpi_comm_t comm, int *local_n0,
                                     int *local_n0_start, int *local_n1,
                                     int *local_n1_start);

/*******************************************************************************
 * \brief Return buffer size, local sizes and start for distributed 3D R2C FFTs.
 ******************************************************************************/
int fft_cufftmp_mpi_3d_distributed_sizes_r2c(const int npts_global[3],
                                         const cp_mpi_comm_t comm,
                                         int *local_n0, int *local_n0_start,
                                         int *local_n1, int *local_n1_start);

#endif /* FFT_LIB_CUFFTMP_MPI_H */

// EOF
