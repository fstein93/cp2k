/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#ifndef FFT_LIB_H
#define FFT_LIB_H

#include "../mpiwrap/cp_mpi.h"
#include "fft_lib_fftw.h"

#include <complex.h>
#include <stdbool.h>

typedef enum { FFT_LIB_REF, FFT_LIB_FFTW, FFT_LIB_GPU } fft_lib;

#if defined(__OFFLOAD) && !defined(__NO_OFFLOAD_FFT)
static const fft_lib FFT_LIB_DEFAULT = FFT_LIB_GPU;
#elif defined(__FFTW3)
static const fft_lib FFT_LIB_DEFAULT = FFT_LIB_FFTW;
#else
static const fft_lib FFT_LIB_DEFAULT = FFT_LIB_REF;
#endif

/*******************************************************************************
 * \brief Initialize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_init_lib(const fft_lib lib, const int fftw_planning_flag,
                  const bool use_fft_mpi, const bool use_guru_interface,
                  const char *wisdom_file);

/*******************************************************************************
 * \brief Initialize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_init_acc_lib();

/*******************************************************************************
 * \brief Finalize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_finalize_lib(const char *wisdom_file);

/*******************************************************************************
 * \brief Finalize the FFT library (if not done externally).
 * \author Frederick Stein
 ******************************************************************************/
void fft_finalize_acc_lib();

/*******************************************************************************
 * \brief Get the default library (GPU if offloading was enabled, else FFTW3).
 * \author Frederick Stein
 ******************************************************************************/
int fft_lib_default_library();

/*******************************************************************************
 * \brief Inquire the library backend in use.
 * \author Frederick Stein
 ******************************************************************************/
int fft_lib_backend_in_use();

/*******************************************************************************
 * \brief Whether compound MPI implementations are available.
 * \author Frederick Stein
 ******************************************************************************/
bool fft_lib_use_mpi();

/*******************************************************************************
 * \brief Whether compound MPI implementations are available.
 * \author Frederick Stein
 ******************************************************************************/
bool fft_lib_has_guru_interface();

/*******************************************************************************
 * \brief Whether compound operations (FFT+copy) are available.
 * \author Frederick Stein
 ******************************************************************************/
bool fft_lib_has_compound_operations();

/*******************************************************************************
 * \brief Ensure that buffers have a required size (in units of complex numbers)
 * \author Frederick Stein
 ******************************************************************************/
void ensure_buffer_size(const int size);

/*******************************************************************************
 * \brief Get the first internal buffer
 * \author Frederick Stein
 ******************************************************************************/
double complex *get_buffer_1();

/*******************************************************************************
 * \brief Get the second internal buffer
 * \author Frederick Stein
 ******************************************************************************/
double complex *get_buffer_2();

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_allocate_double(const int length, double **buffer);

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_allocate_complex(const int length, double complex **buffer);

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_free_double(double *buffer);

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_free_complex(double complex *buffer);

/*******************************************************************************
 * \brief Register a local 1D C2C FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_register_1d_local(const bool dir,const int fft_size, const int number_of_ffts,
                     const bool transpose_rs, const bool transpose_gs,
                     const int leading_dimension_rs, const int leading_dimension_gs,
                     double complex *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief Register a local 1D R2C FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_register_1d_r2c_local(const int fft_size, const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs,
                     const int leading_dimension_rs, const int leading_dimension_gs,
                         double *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief Register a local 1D C2R FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_register_1d_c2r_local(const int fft_size, const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs,
                     const int leading_dimension_rs, const int leading_dimension_gs,
                         double complex *grid_in, double *grid_out);

/*******************************************************************************
 * \brief Register a local 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_register_2d_local(const bool dir, const int fft_size[2], const int number_of_ffts,
                     const bool transpose_rs, const bool transpose_gs,
                     double complex *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief Register a local 2D R2C FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_register_2d_r2c_local(const int fft_size[2], const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs,
                         double *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief Register a local 2D C2R FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_register_2d_c2r_local(const int fft_size[2], const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs,
                         double complex *grid_in, double *grid_out);

/*******************************************************************************
 * \brief Register a local 3D FFT.
 * \note fft_3d_bw_local(grid_gs, grid_rs, n) is the reverse to
 * fft_3d_rw_local(grid_rs, grid_gs, n) (ignoring normalization).
 * \author Frederick Stein
 ******************************************************************************/
void fft_register_3d_local(const bool dir, const int fft_size[3], double complex *grid_in,
                     double complex *grid_out);

/*******************************************************************************
 * \brief Register a local 3D R2C FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_register_3d_r2c_local(const int fft_size[3], double *grid_in,
                         double complex *grid_out);

/*******************************************************************************
 * \brief Register a local 3D C2R FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_register_3d_c2r_local(const int fft_size[3], double complex *grid_in,
                         double *grid_out);

/*******************************************************************************
 * \brief Register a local C2C FFT using the Guru interface.
 * \author Frederick Stein
 ******************************************************************************/
void fft_register_guru(const bool dir, const int rank, const fft_iodim *dims, int howmany_rank,
                 const fft_iodim *howmany_dims, const int number_of_threads,
                 double complex *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief Register a local forward R2C FFT using the Guru interface.
 * \author Frederick Stein
 ******************************************************************************/
void fft_register_guru_r2c(int rank, const fft_iodim *dims, int howmany_rank,
                     const fft_iodim *howmany_dims, const int number_of_threads,
                     double *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief Register a local backwards R2C FFT using the Guru interface.
 * \author Frederick Stein
 ******************************************************************************/
void fft_register_guru_c2r(int rank, const fft_iodim *dims, int howmany_rank,
                     const fft_iodim *howmany_dims, const int number_of_threads,
                     double complex *grid_in, double *grid_out);

/*******************************************************************************
 * \brief Register a distributed 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_register_2d_distributed(const bool dir, const int npts_global[2], const int number_of_ffts,
                           const cp_mpi_comm_t comm,
                           double complex *restrict grid_in,
                           double complex *restrict grid_out);

/*******************************************************************************
 * \brief Register a distributed 2D R2C FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_register_2d_r2c_distributed(const int npts_global[2],
                               const int number_of_ffts,
                               const cp_mpi_comm_t comm,
                               double *restrict grid_in,
                               double complex *restrict grid_out);

/*******************************************************************************
 * \brief Register a distributed 2D C2R FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_register_2d_c2r_distributed(const int npts_global[2],
                               const int number_of_ffts,
                               const cp_mpi_comm_t comm,
                               double complex *restrict grid_in,
                               double *restrict grid_out);

/*******************************************************************************
 * \brief Register a distributed 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_register_3d_distributed(const bool dir, const int npts_global[3], const cp_mpi_comm_t comm,
                           double complex *restrict grid_in,
                           double complex *restrict grid_out);

/*******************************************************************************
 * \brief Register a distributed 3D R2C FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_register_3d_r2c_distributed(const int npts_global[3],
                               const cp_mpi_comm_t comm,
                               double *restrict grid_in,
                               double complex *restrict grid_out);

/*******************************************************************************
 * \brief Register a distributed 3D C2R FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_register_3d_c2r_distributed(const int npts_global[3],
                               const cp_mpi_comm_t comm,
                               double complex *restrict grid_in,
                               double *restrict grid_out);

/*******************************************************************************
 * \brief Perform a local 1D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_1d_local(const bool dir, const int fft_size, const int number_of_ffts,
                     const bool transpose_rs, const bool transpose_gs,
                     const int leading_dimension_rs, const int leading_dimension_gs,
                     double complex *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief Perform a local 1D R2C FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_1d_r2c_local(const int fft_size, const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs,
                     const int leading_dimension_rs, const int leading_dimension_gs,
                         double *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief Perform a local 1D C2R FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_1d_c2r_local(const int fft_size, const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs,
                     const int leading_dimension_rs, const int leading_dimension_gs,
                         double complex *grid_in, double *grid_out);

/*******************************************************************************
 * \brief Perform a local 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_local(const bool dir, const int fft_size[2], const int number_of_ffts,
                     const bool transpose_rs, const bool transpose_gs,
                     double complex *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief Perform a local 2D R2C FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_r2c_local(const int fft_size[2], const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs,
                         double *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief Perform a local 2D C2R FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_c2r_local(const int fft_size[2], const int number_of_ffts,
                         const bool transpose_rs, const bool transpose_gs,
                         double complex *grid_in, double *grid_out);

/*******************************************************************************
 * \brief Perform a local 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_local(const bool dir, const int fft_size[3], double complex *grid_in,
                     double complex *grid_out);

/*******************************************************************************
 * \brief Perform a local 3D R2C FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_r2c_local(const int fft_size[3], double *grid_in,
                         double complex *grid_out);

/*******************************************************************************
 * \brief Perform a local 3D C2R FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_c2r_local(const int fft_size[3], double complex *grid_in,
                         double *grid_out);

/*******************************************************************************
 * \brief Perform a local C2C FFT using the Guru interface.
 * \author Frederick Stein
 ******************************************************************************/
void fft_guru(const bool dir, int rank, const fft_iodim *dims, int howmany_rank,
                 const fft_iodim *howmany_dims, const int number_of_threads,
                 double complex *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief Perform a local forward R2C FFT using the Guru interface.
 * \author Frederick Stein
 ******************************************************************************/
void fft_guru_r2c(int rank, const fft_iodim *dims, int howmany_rank,
                     const fft_iodim *howmany_dims, const int number_of_threads,
                     double *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief Perform a local backwards R2C FFT using the Guru interface.
 * \author Frederick Stein
 ******************************************************************************/
void fft_guru_c2r(int rank, const fft_iodim *dims, int howmany_rank,
                     const fft_iodim *howmany_dims, const int number_of_threads,
                     double complex *grid_in, double *grid_out);

/*******************************************************************************
 * \brief Return buffer size, local sizes and start for distributed 2D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_2d_distributed_sizes(const int npts_global[2], const int number_of_ffts,
                             const cp_mpi_comm_t comm, int *local_n0,
                             int *local_n0_start, int *local_n1,
                             int *local_n1_start);

/*******************************************************************************
 * \brief Return buffer size, local sizes and start for distributed 2D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_2d_distributed_sizes_r2c(const int npts_global[2],
                                 const int number_of_ffts,
                                 const cp_mpi_comm_t comm, int *local_n0,
                                 int *local_n0_start, int *local_n1,
                                 int *local_n1_start);

/*******************************************************************************
 * \brief Return buffer size, local sizes and start for distributed 3D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_3d_distributed_sizes(const int npts_global[3], const cp_mpi_comm_t comm,
                             int *local_n2, int *local_n2_start, int *local_n1,
                             int *local_n1_start);

/*******************************************************************************
 * \brief Return buffer size, local sizes and start for distributed 3D FFTs.
 * \author Frederick Stein
 ******************************************************************************/
int fft_3d_distributed_sizes_r2c(const int npts_global[3],
                                 const cp_mpi_comm_t comm, int *local_n0,
                                 int *local_n0_start, int *local_n1,
                                 int *local_n1_start);

/*******************************************************************************
 * \brief Perform a distributed 2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_distributed(const bool dir, const int npts_global[2], const int number_of_ffts,
                           const cp_mpi_comm_t comm,
                           double complex *restrict grid_in,
                           double complex *restrict grid_out);

/*******************************************************************************
 * \brief Perform a distributed R2C2D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_r2c_distributed(const int npts_global[2],
                               const int number_of_ffts,
                               const cp_mpi_comm_t comm,
                               double *restrict grid_in,
                               double complex *restrict grid_out);

/*******************************************************************************
 * \brief Perform a distributed 2D C2R FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_2d_c2r_distributed(const int npts_global[2],
                               const int number_of_ffts,
                               const cp_mpi_comm_t comm,
                               double complex *restrict grid_in,
                               double *restrict grid_out);

/*******************************************************************************
 * \brief Perform a distributed 3D FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_distributed(const bool dir, const int npts_global[3], const cp_mpi_comm_t comm,
                           double complex *restrict grid_in,
                           double complex *restrict grid_out);

/*******************************************************************************
 * \brief Perform a distributed 3D R2C FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_r2c_distributed(const int npts_global[3],
                            const cp_mpi_comm_t comm,
                            double *restrict grid_in,
                            double complex *restrict grid_out);

/*******************************************************************************
 * \brief Perform a distributed 3D C2R FFT.
 * \author Frederick Stein
 ******************************************************************************/
void fft_3d_c2r_distributed(const int npts_global[3],
                            const cp_mpi_comm_t comm,
                            double complex *restrict grid_in,
                            double *restrict grid_out);

#endif /* FFT_LIB_H */

// EOF
