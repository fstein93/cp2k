/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2026 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: BSD-3-Clause                                     */
/*----------------------------------------------------------------------------*/
#include <complex.h>
#include <stdbool.h>
#include <stddef.h>

/*******************************************************************************
 * \brief Initializes the fft_gpu library.
 * \author Ole Schuett
 ******************************************************************************/
void fft_gpu_init(void);

/*******************************************************************************
 * \brief Releases resources held by the fft_gpu library.
 * \author Ole Schuett
 ******************************************************************************/
void fft_gpu_finalize(void);

/*******************************************************************************
 * \brief Checks size of device buffers and re-allocates them if necessary.
 * \author Ole Schuett
 ******************************************************************************/
void ensure_memory_sizes(const size_t requested_buffer_size,
                         const size_t requested_map_size);

/*******************************************************************************
 * \brief Checks the size of the GPU-resident driver buffers and re-allocates
 *        them if necessary.
 * \author Frederick Stein
 ******************************************************************************/
void fft_gpu_ensure_buffer_size(const int size);

/*******************************************************************************
 * \brief Get the first GPU-resident driver buffer.
 * \author Frederick Stein
 ******************************************************************************/
double complex *fft_gpu_get_buffer_1(void);

/*******************************************************************************
 * \brief Get the second GPU-resident driver buffer.
 * \author Frederick Stein
 ******************************************************************************/
double complex *fft_gpu_get_buffer_2(void);

/*******************************************************************************
 * \brief Get the size of the GPU-resident driver buffers in elements.
 * \author Frederick Stein
 ******************************************************************************/
int fft_gpu_get_buffer_size(void);

/*******************************************************************************
 * \brief Ensure the central index-map buffer is large enough.
 * \author Frederick Stein
 ******************************************************************************/
void fft_gpu_ensure_map_size(const int size);

/*******************************************************************************
 * \brief Get the central index-map device buffer.
 * \author Frederick Stein
 ******************************************************************************/
int *fft_gpu_get_map_buffer(void);

/*******************************************************************************
 * \brief Get the GPU stream used by the FFT backend on the current thread.
 * \author Frederick Stein
 ******************************************************************************/
void *fft_gpu_get_stream(void);

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_gpu_allocate_double(const int length, double **buffer);

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_gpu_allocate_complex(const int length, double complex **buffer);

/*******************************************************************************
 * \brief Allocate buffer of type double.
 * \author Frederick Stein
 ******************************************************************************/
void fft_gpu_free_double(double *buffer);

/*******************************************************************************
 * \brief Allocate buffer of type double complex.
 * \author Frederick Stein
 ******************************************************************************/
void fft_gpu_free_complex(double complex *buffer);

/*******************************************************************************
 * \brief   Performs a (double precision complex) 1D-FFT on the GPU.
 * \author  Andreas Gloess, Ole Schuett
 ******************************************************************************/
void fft_gpu_register_1d_local(const bool dir, const int n, const int m,
                               const bool transpose_in,
                               const bool transpose_out,
                               const int leading_dimension_in,
                               const int leading_dimension_out,
                               double complex *grid_in,
                               double complex *grid_out);

/*******************************************************************************
 * \brief   Performs a (double precision complex) R2C 1D-FFT on the GPU.
 * \author  Andreas Gloess, Ole Schuett
 ******************************************************************************/
void fft_gpu_register_1d_r2c_local(const int n, const int m,
                                   const bool transpose_in,
                                   const bool transpose_out,
                                   const int leading_dimension_in,
                                   const int leading_dimension_out,
                                   double *grid_in, double complex *grid_out);

/*******************************************************************************
 * \brief   Performs a (double precision complex) C2R 1D-FFT on the GPU.
 * \author  Frederick Stein
 ******************************************************************************/
void fft_gpu_register_1d_c2r_local(const int n, const int m,
                                   const bool transpose_in,
                                   const bool transpose_out,
                                   const int leading_dimension_in,
                                   const int leading_dimension_out,
                                   double complex *grid_in, double *grid_out);

/*******************************************************************************
 * \brief   Performs a (double precision complex) 1D-FFT on the GPU.
 * \author  Frederick Stein
 ******************************************************************************/
void fft_gpu_register_2d_local(const bool dir, const int n[2], const int m,
                               const bool transpose_in,
                               const bool transpose_out,
                               double complex *grid_in,
                               double complex *grid_out);

/*******************************************************************************
 * \brief   Performs a (double precision complex) R2C 1D-FFT on the GPU.
 * \author  Frederick Stein
 ******************************************************************************/
void fft_gpu_register_2d_r2c_local(const bool dir, const int n[2], const int m,
                                   const bool transpose_in,
                                   const bool transpose_out,
                                   const bool inplace);

/*******************************************************************************
 * \brief   Performs a (double precision complex) 3D-FFT on the GPU.
 * \author  Andreas Gloess, Ole Schuett
 ******************************************************************************/
void fft_gpu_register_3d_local(const bool dir, const int *npts,
                               const int number_of_ffts,
                               const bool transpose_in,
                               const bool transpose_out,
                               double complex *grid_in,
                               double complex *grid_out);

/*******************************************************************************
 * \brief   Performs a 3D-R2C/C2R-FFT, on the GPU.
 * \author  Andreas Gloess, Ole Schuett
 ******************************************************************************/
void fft_gpu_register_3d_r2c_local(const bool dir, const int *npts,
                                   const bool inplace, const int number_of_ffts,
                                   const bool transpose_in,
                                   const bool transpose_out);

/*******************************************************************************
 * \brief   Performs a (double precision complex) 1D-FFT on the GPU.
 * \author  Andreas Gloess, Ole Schuett
 ******************************************************************************/
void fft_gpu_1d_local(const bool dir, const int n, const int m,
                      const bool transpose_in, const bool transpose_out,
                      const int leading_dimension_in,
                      const int leading_dimension_out,
                      const double complex *zin, double complex *zout);

/*******************************************************************************
 * \brief   Performs a (double precision complex) R2C 1D-FFT on the GPU.
 * \author  Andreas Gloess, Ole Schuett
 ******************************************************************************/
void fft_gpu_1d_r2c_local(const int n, const int m, const bool transpose_in,
                          const bool transpose_out,
                          const int leading_dimension_in,
                          const int leading_dimension_out, const double *zin,
                          double complex *zout);

/*******************************************************************************
 * \brief   Performs a (double precision complex) C2R 1D-FFT on the GPU.
 * \author  Frederick Stein
 ******************************************************************************/
void fft_gpu_1d_c2r_local(const int n, const int m, const bool transpose_in,
                          const bool transpose_out,
                          const int leading_dimension_in,
                          const int leading_dimension_out,
                          const double complex *zin, double *zout);

/*******************************************************************************
 * \brief   Performs a (double precision complex) 1D-FFT on the GPU.
 * \author  Frederick Stein
 ******************************************************************************/
void fft_gpu_2d_local(const bool dir, const int n[2], const int m,
                      const bool transpose_in, const bool transpose_out,
                      const double complex *zin, double complex *zout);

/*******************************************************************************
 * \brief   Performs a (double precision complex) R2C 1D-FFT on the GPU.
 * \author  Frederick Stein
 ******************************************************************************/
void fft_gpu_2d_r2c_local(const double *zin, double *zout, const bool dir,
                          const int n[2], const int m, const bool transpose_in,
                          const bool transpose_out);

/*******************************************************************************
 * \brief   Performs a (double precision complex) 3D-FFT on the GPU.
 * \author  Andreas Gloess, Ole Schuett
 ******************************************************************************/
void fft_gpu_3d_local(const bool dir, const int *npts,
                      const int number_of_ffts, const bool transpose_in,
                      const bool transpose_out, const double complex *zin,
                      double complex *zout);

/*******************************************************************************
 * \brief   Performs a 3D-R2C/C2R-FFT, on the GPU.
 * \author  Andreas Gloess, Ole Schuett
 ******************************************************************************/
void fft_gpu_3d_r2c_local(const double *zin, double *zout, const bool dir,
                          const int *npts, const int number_of_ffts,
                          const bool transpose_in, const bool transpose_out);

/*******************************************************************************
 * \brief   Performs a 1D-FFT on the GPU using GPU-resident buffers.
 * \author  Frederick Stein
 ******************************************************************************/
void fft_gpu_1d_local_device(const bool dir, const int n, const int m,
                             const bool transpose_in, const bool transpose_out,
                             const int leading_dimension_in,
                             const int leading_dimension_out,
                             const double complex *zin, double complex *zout);

/*******************************************************************************
 * \brief   Performs an R2C 1D-FFT on the GPU using GPU-resident buffers.
 * \author  Frederick Stein
 ******************************************************************************/
void fft_gpu_1d_r2c_local_device(const int n, const int m,
                                 const bool transpose_in,
                                 const bool transpose_out,
                                 const int leading_dimension_in,
                                 const int leading_dimension_out,
                                 const double *zin, double complex *zout);

/*******************************************************************************
 * \brief   Performs a C2R 1D-FFT on the GPU using GPU-resident buffers.
 * \author  Frederick Stein
 ******************************************************************************/
void fft_gpu_1d_c2r_local_device(const int n, const int m,
                                 const bool transpose_in,
                                 const bool transpose_out,
                                 const int leading_dimension_in,
                                 const int leading_dimension_out,
                                 const double complex *zin, double *zout);

/*******************************************************************************
 * \brief   Performs a 2D-FFT on the GPU using GPU-resident buffers.
 * \author  Frederick Stein
 ******************************************************************************/
void fft_gpu_2d_local_device(const bool dir, const int n[2], const int m,
                             const bool transpose_in, const bool transpose_out,
                             const double complex *zin, double complex *zout);

/*******************************************************************************
 * \brief   Performs a 3D-FFT on the GPU using GPU-resident buffers.
 * \author  Frederick Stein
 ******************************************************************************/
void fft_gpu_3d_local_device(const bool dir, const int *npts,
                             const int number_of_ffts, const bool transpose_in,
                             const bool transpose_out,
                             const double complex *zin, double complex *zout);

// EOF
