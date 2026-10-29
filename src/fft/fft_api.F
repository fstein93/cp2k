!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: BSD-3-Clause                                                          !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Fortran API for the FFT package, which is written in C.
!> \author Frederick Stein
! **************************************************************************************************
MODULE fft_api
   USE ISO_C_BINDING, ONLY: &
      C_ASSOCIATED, C_BOOL, C_CHAR, C_DOUBLE, C_DOUBLE_COMPLEX, C_F_POINTER, C_INT, C_LOC, C_NULL_CHAR, &
      C_NULL_PTR, C_PTR
   USE kinds, ONLY: dp
   USE message_passing, ONLY: mp_comm_type
#include "../base/base_uses.f90"

   IMPLICIT NONE

   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'fft_api'

   PUBLIC :: fft_library_init, fft_library_finalize
   PUBLIC :: fft_print_timing_report
   PUBLIC :: fft_backend_in_use, fft_has_guru_interface
   PUBLIC :: fft_allocate, fft_deallocate
   PUBLIC :: fft_register_1d_local, fft_register_2d_local, fft_register_3d_local
   PUBLIC :: fft_register_1d_local_inplace, fft_register_2d_local_inplace, fft_register_3d_local_inplace
   PUBLIC :: fft_1d_local, fft_2d_local, fft_3d_local
   PUBLIC :: fft_1d_local_inplace, fft_2d_local_inplace, fft_3d_local_inplace
   PUBLIC :: fft_grid_type, fft_create_grid, fft_create_grid_from_reference, fft_free_grid
   PUBLIC :: fft_grid_get_npts_global, fft_grid_get_comm
   PUBLIC :: fft_grid_get_proc2local_rs, fft_grid_get_proc2local_ms, fft_grid_get_proc2local_gs

   ! Keep it synchronized with fft_api_F.c
   INTEGER, PARAMETER, PUBLIC :: FFT_LIBRARY_BACKEND_DEFAULT = 1
   INTEGER, PARAMETER, PUBLIC :: FFT_LIBRARY_BACKEND_FFTW = 2
   INTEGER, PARAMETER, PUBLIC :: FFT_LIBRARY_BACKEND_GPU = 3
   INTEGER, PARAMETER, PUBLIC :: FFT_LIBRARY_FFTW_MODE_ESTIMATE = 11
   INTEGER, PARAMETER, PUBLIC :: FFT_LIBRARY_FFTW_MODE_MEASURE = 12
   INTEGER, PARAMETER, PUBLIC :: FFT_LIBRARY_FFTW_MODE_PATIENT = 13
   INTEGER, PARAMETER, PUBLIC :: FFT_LIBRARY_FFTW_MODE_EXHAUSTIVE = 14

   INTERFACE fft_allocate
      MODULE PROCEDURE :: fft_allocate_real_1
      MODULE PROCEDURE :: fft_allocate_real_2
      MODULE PROCEDURE :: fft_allocate_real_3
      MODULE PROCEDURE :: fft_allocate_complex_1
      MODULE PROCEDURE :: fft_allocate_complex_2
      MODULE PROCEDURE :: fft_allocate_complex_3
   END INTERFACE fft_allocate

   INTERFACE fft_deallocate
      MODULE PROCEDURE :: fft_free_real_1
      MODULE PROCEDURE :: fft_free_real_2
      MODULE PROCEDURE :: fft_free_real_3
      MODULE PROCEDURE :: fft_free_complex_1
      MODULE PROCEDURE :: fft_free_complex_2
      MODULE PROCEDURE :: fft_free_complex_3
   END INTERFACE fft_deallocate

   INTERFACE
      SUBROUTINE fft_allocate_double_c(length, buffer) BIND(C, name="fft_allocate_double_F")
         IMPORT :: C_PTR, C_INT
         INTEGER(C_INT), VALUE :: length
         TYPE(C_PTR) :: buffer
      END SUBROUTINE fft_allocate_double_c
   END INTERFACE

   INTERFACE
      SUBROUTINE fft_allocate_complex_c(length, buffer) BIND(C, name="fft_allocate_complex_F")
         IMPORT :: C_PTR, C_INT
         INTEGER(C_INT), VALUE :: length
         TYPE(C_PTR) :: buffer
      END SUBROUTINE fft_allocate_complex_c
   END INTERFACE

   INTERFACE
      SUBROUTINE fft_free_double_c(buffer) BIND(C, name="fft_free_double_F")
         IMPORT :: C_PTR
         TYPE(C_PTR), VALUE :: buffer
      END SUBROUTINE fft_free_double_c
   END INTERFACE

   INTERFACE
      SUBROUTINE fft_free_complex_c(buffer) BIND(C, name="fft_free_complex_F")
         IMPORT :: C_PTR
         TYPE(C_PTR), VALUE :: buffer
      END SUBROUTINE fft_free_complex_c
      END INTERFACE

   TYPE fft_grid_type
      PRIVATE
      TYPE(C_PTR) :: fft_grid = C_NULL_PTR
   END TYPE fft_grid_type

CONTAINS

! **************************************************************************************************
!> \brief Initialize FFT library
!> \param backend ...
!> \param planning_mode ...
!> \param use_guru_interface ...
!> \param wisdom_file ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_library_init(backend, planning_mode, use_fft_mpi, use_guru_interface, wisdom_file, status)
      INTEGER, INTENT(IN)                                :: backend, planning_mode
      LOGICAL, INTENT(IN)                                :: use_fft_mpi, use_guru_interface
      CHARACTER(LEN=*), INTENT(IN)                       :: wisdom_file
      INTEGER, INTENT(OUT)                               :: status

      CHARACTER(KIND=C_CHAR, LEN=1), ALLOCATABLE, &
         DIMENSION(:), TARGET                            :: wisdom_file_c
      INTEGER                                            :: i
      INTERFACE
         INTEGER FUNCTION fft_library_init_c(backend, planning_mode, use_fft_mpi, use_guru_interface, &
                                             wisdom_file) BIND(C, name="fft_library_init_F")
            IMPORT :: C_INT, C_BOOL, C_PTR
            INTEGER(C_INT), VALUE :: backend, planning_mode
            LOGICAL(C_BOOL), VALUE :: use_fft_mpi, use_guru_interface
            TYPE(C_PTR), VALUE :: wisdom_file

         END FUNCTION fft_library_init_c
      END INTERFACE

      ALLOCATE (wisdom_file_c(LEN_TRIM(wisdom_file) + 1))
      DO i = 1, LEN_TRIM(wisdom_file)
         wisdom_file_c(i) = wisdom_file(i:i)
      END DO
      wisdom_file_c(LEN_TRIM(wisdom_file) + 1) = C_NULL_CHAR

      status = INT(fft_library_init_c(backend, planning_mode, &
                                      LOGICAL(use_fft_mpi, C_BOOL), LOGICAL(use_guru_interface, C_BOOL), C_LOC(wisdom_file_c)))

      DEALLOCATE (wisdom_file_c)

   END SUBROUTINE fft_library_init

! **************************************************************************************************
!> \brief Finalize FFT library
!> \param wisdom_file ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_library_finalize(wisdom_file)
      CHARACTER(LEN=*), INTENT(IN)                       :: wisdom_file

      CHARACTER(KIND=C_CHAR, LEN=1), ALLOCATABLE, &
         DIMENSION(:), TARGET                            :: wisdom_file_c
      INTEGER                                            :: i
      INTERFACE
         SUBROUTINE fft_library_finalize_c(wisdom_file) BIND(C, name="fft_library_finalize_F")
            IMPORT :: C_PTR
            TYPE(C_PTR), VALUE :: wisdom_file
         END SUBROUTINE fft_library_finalize_c
      END INTERFACE

      ALLOCATE (wisdom_file_c(LEN_TRIM(wisdom_file) + 1))
      DO i = 1, LEN_TRIM(wisdom_file)
         wisdom_file_c(i) = wisdom_file(i:i)
      END DO
      wisdom_file_c(LEN_TRIM(wisdom_file) + 1) = C_NULL_CHAR

      CALL fft_library_finalize_c(C_LOC(wisdom_file_c))

      DEALLOCATE (wisdom_file_c)

   END SUBROUTINE fft_library_finalize

! **************************************************************************************************
!> \brief Print FFT library statistics
!> \param threshold ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_print_timing_report(threshold)
      REAL(KIND=dp), INTENT(IN)                          :: threshold

      INTERFACE
         SUBROUTINE fft_print_timing_report_c(threshold) &
            BIND(C, name="fft_print_timing_report_F")
            IMPORT :: C_DOUBLE
            REAL(C_DOUBLE), VALUE :: threshold
         END SUBROUTINE fft_print_timing_report_c
      END INTERFACE

      CALL fft_print_timing_report_c(threshold)

   END SUBROUTINE fft_print_timing_report

! **************************************************************************************************
!> \brief Inquire the library backend in use
!> \return ...
!> \author Frederick Stein
! **************************************************************************************************
   INTEGER FUNCTION fft_backend_in_use()

      INTERFACE
         INTEGER(C_INT) FUNCTION fft_backend_in_use_c() &
            BIND(C, name="fft_backend_in_use_F")
            IMPORT :: C_INT
         END FUNCTION fft_backend_in_use_c
      END INTERFACE

      fft_backend_in_use = fft_backend_in_use_c()

   END FUNCTION fft_backend_in_use

! **************************************************************************************************
!> \brief Inquire whether a guru interface is available
!> \return ...
!> \author Frederick Stein
! **************************************************************************************************
   LOGICAL FUNCTION fft_has_guru_interface()

      INTERFACE
         LOGICAL(C_BOOL) FUNCTION fft_has_guru_interface_c() &
            BIND(C, name="fft_has_guru_interface_F")
            IMPORT :: C_BOOL
         END FUNCTION fft_has_guru_interface_c
      END INTERFACE

      fft_has_guru_interface = fft_has_guru_interface_c()

   END FUNCTION fft_has_guru_interface

   #:set maxdim = 3
   #:for dim in range(1, maxdim+1)
      #:for type in ["COMPLEX", "REAL"]
! **************************************************************************************************
!> \brief Allocate an array for use with the FFT library
!> \param buffer ...
!> \param length ...
!> \author Frederick Stein
! **************************************************************************************************
         SUBROUTINE fft_allocate_${type}$_${dim}$ (buffer, length)
            ${type}$ (KIND=dp), CONTIGUOUS, DIMENSION(:${", :"*(dim-1)}$), &
               POINTER                                         :: buffer
            INTEGER, DIMENSION(${dim}$), INTENT(IN)                                :: length

            TYPE(C_PTR)                                        :: buffer_c

            buffer_c = C_NULL_PTR

            CALL fft_allocate_complex_c(INT(PRODUCT(length), C_INT), buffer_c)

            CALL C_F_POINTER(buffer_c, buffer, length)

         END SUBROUTINE fft_allocate_${type}$_${dim}$
      #:endfor
   #:endfor

! **************************************************************************************************
!> \brief Free a complex array allocated with fft_allocate_complex
!> \param buffer ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_free_complex_1(buffer)
      COMPLEX(KIND=dp), CONTIGUOUS, DIMENSION(:), &
         POINTER                                         :: buffer

      CALL fft_free_complex_c(C_LOC(buffer))

      NULLIFY (buffer)

   END SUBROUTINE fft_free_complex_1

! **************************************************************************************************
!> \brief Free a complex array allocated with fft_allocate_complex
!> \param buffer ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_free_complex_2(buffer)
      COMPLEX(KIND=dp), CONTIGUOUS, DIMENSION(:, :), &
         POINTER                                         :: buffer

      CALL fft_free_complex_c(C_LOC(buffer))

      NULLIFY (buffer)

   END SUBROUTINE fft_free_complex_2

! **************************************************************************************************
!> \brief Free a complex array allocated with fft_allocate_complex
!> \param buffer ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_free_complex_3(buffer)
      COMPLEX(KIND=dp), CONTIGUOUS, DIMENSION(:, :, :), &
         POINTER                                         :: buffer

      CALL fft_free_complex_c(C_LOC(buffer))

      NULLIFY (buffer)

   END SUBROUTINE fft_free_complex_3

! **************************************************************************************************
!> \brief Free a double array allocated with fft_allocate_double
!> \param buffer ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_free_real_1(buffer)
      REAL(KIND=dp), CONTIGUOUS, DIMENSION(:), POINTER   :: buffer

      CALL fft_free_double_c(C_LOC(buffer))

      NULLIFY (buffer)

   END SUBROUTINE fft_free_real_1

! **************************************************************************************************
!> \brief Free a double array allocated with fft_allocate_double
!> \param buffer ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_free_real_2(buffer)
      REAL(KIND=dp), CONTIGUOUS, DIMENSION(:, :), &
         POINTER                                         :: buffer

      CALL fft_free_double_c(C_LOC(buffer))

      NULLIFY (buffer)

   END SUBROUTINE fft_free_real_2

! **************************************************************************************************
!> \brief Free a double array allocated with fft_allocate_double
!> \param buffer ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_free_real_3(buffer)
      REAL(KIND=dp), CONTIGUOUS, DIMENSION(:, :, :), &
         POINTER                                         :: buffer

      CALL fft_free_double_c(C_LOC(buffer))

      NULLIFY (buffer)

   END SUBROUTINE fft_free_real_3

! **************************************************************************************************
!> \brief Registers a complex-to-complex 1D FFT
!> \param direction ...
!> \param fft_size ...
!> \param number_of_ffts ...
!> \param transpose_in ...
!> \param transpose_out ...
!> \param leading_dimension_in ...
!> \param leading_dimension_out ...
!> \param grid_in ...
!> \param grid_out ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_register_1d_local(direction, fft_size, number_of_ffts, transpose_in, transpose_out, &
                                    leading_dimension_in, leading_dimension_out, grid_in, grid_out)
      LOGICAL, INTENT(IN)                                :: direction
      INTEGER, INTENT(IN)                                :: fft_size, number_of_ffts
      LOGICAL, INTENT(IN)                                :: transpose_in, transpose_out
      INTEGER, INTENT(IN)                                :: leading_dimension_in, leading_dimension_out
      COMPLEX(KIND=dp), DIMENSION(leading_dimension_in*MERGE(fft_size, number_of_ffts, transpose_in)), &
         INTENT(INOUT), TARGET      :: grid_in
      COMPLEX(KIND=dp), DIMENSION(leading_dimension_out*MERGE(fft_size, number_of_ffts, transpose_out)), &
         INTENT(OUT), TARGET        :: grid_out

      INTERFACE
         SUBROUTINE fft_register_1d_local_c(direction, fft_size, number_of_ffts, transpose_in, transpose_out, &
                                            leading_dimension_in, leading_dimension_out, grid_in, grid_out) &
            BIND(C, name="fft_register_1d_local_F")
            IMPORT :: C_INT, C_BOOL, C_DOUBLE_COMPLEX, C_PTR
            LOGICAL(C_BOOL), VALUE :: direction
            INTEGER(C_INT), VALUE :: fft_size, number_of_ffts
            LOGICAL(C_BOOL), VALUE :: transpose_in, transpose_out
            INTEGER(C_INT), VALUE :: leading_dimension_in, leading_dimension_out
            TYPE(C_PTR), VALUE :: grid_in, grid_out
         END SUBROUTINE fft_register_1d_local_c
      END INTERFACE

      CALL fft_register_1d_local_c(LOGICAL(direction, C_BOOL), INT(fft_size, C_INT), &
                                   INT(number_of_ffts, C_INT), LOGICAL(transpose_in, C_BOOL), &
                                   LOGICAL(transpose_out, C_BOOL), leading_dimension_in, leading_dimension_out, &
                                   C_LOC(grid_in), C_LOC(grid_out))

   END SUBROUTINE fft_register_1d_local

! **************************************************************************************************
!> \brief Registers a complex-to-complex 2D FFT
!> \param direction ...
!> \param fft_size ...
!> \param number_of_ffts ...
!> \param transpose_in ...
!> \param transpose_out ...
!> \param grid_in ...
!> \param grid_out ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_register_2d_local(direction, fft_size, number_of_ffts, &
                                    transpose_in, transpose_out, grid_in, grid_out)
      LOGICAL, INTENT(IN)                                :: direction
      INTEGER, DIMENSION(2), INTENT(IN)                  :: fft_size
      INTEGER, INTENT(IN)                                :: number_of_ffts
      LOGICAL, INTENT(IN)                                :: transpose_in, transpose_out
      COMPLEX(KIND=dp), DIMENSION(PRODUCT(fft_size)*number_of_ffts), INTENT(INOUT), TARGET      :: grid_in
      COMPLEX(KIND=dp), DIMENSION(PRODUCT(fft_size)*number_of_ffts), INTENT(OUT), TARGET        :: grid_out

      INTERFACE
         SUBROUTINE fft_register_2d_local_c(direction, fft_size, number_of_ffts, &
                                            transpose_in, transpose_out, grid_in, grid_out) &
            BIND(C, name="fft_register_2d_local_F")
            IMPORT :: C_INT, C_BOOL, C_DOUBLE_COMPLEX, C_PTR
            LOGICAL(C_BOOL), VALUE :: direction
            INTEGER(C_INT), DIMENSION(2) :: fft_size
            INTEGER(C_INT), VALUE :: number_of_ffts
            LOGICAL(C_BOOL), VALUE :: transpose_in, transpose_out
            TYPE(C_PTR), VALUE :: grid_in, grid_out
         END SUBROUTINE fft_register_2d_local_c
      END INTERFACE

      CALL fft_register_2d_local_c(LOGICAL(direction, C_BOOL), INT(fft_size, C_INT), &
                                   INT(number_of_ffts, C_INT), LOGICAL(transpose_in, C_BOOL), &
                                   LOGICAL(transpose_out, C_BOOL), C_LOC(grid_in), C_LOC(grid_out))

   END SUBROUTINE fft_register_2d_local

! **************************************************************************************************
!> \brief Registers a complex-to-complex 3D FFT
!> \param direction ...
!> \param fft_size ...
!> \param grid_in ...
!> \param grid_out ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_register_3d_local(direction, fft_size, grid_in, grid_out)
      LOGICAL, INTENT(IN)                                :: direction
      INTEGER, DIMENSION(3), INTENT(IN)                  :: fft_size
      COMPLEX(KIND=dp), DIMENSION(*), INTENT(INOUT), TARGET      :: grid_in
      COMPLEX(KIND=dp), DIMENSION(*), INTENT(OUT), TARGET        :: grid_out

      INTERFACE
         SUBROUTINE fft_register_3d_local_c(direction, fft_size, grid_in, grid_out) &
            BIND(C, name="fft_register_3d_local_F")
            IMPORT :: C_INT, C_DOUBLE_COMPLEX, C_BOOL, C_PTR
            LOGICAL(C_BOOL), VALUE :: direction
            INTEGER(C_INT), DIMENSION(3), INTENT(IN) :: fft_size
            TYPE(C_PTR), VALUE :: grid_in, grid_out
         END SUBROUTINE fft_register_3d_local_c
      END INTERFACE

      CALL fft_register_3d_local_c(LOGICAL(direction, C_BOOL), INT(fft_size, C_INT), &
                                   C_LOC(grid_in), C_LOC(grid_out))

   END SUBROUTINE fft_register_3d_local

! **************************************************************************************************
!> \brief Registers a complex-to-complex 1D FFT
!> \param direction ...
!> \param fft_size ...
!> \param number_of_ffts ...
!> \param transpose ...
!> \param leading_dimension ...
!> \param grid ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_register_1d_local_inplace(direction, fft_size, number_of_ffts, &
                                            transpose, leading_dimension, grid)
      LOGICAL, INTENT(IN)                                :: direction
      INTEGER, INTENT(IN)                                :: fft_size, number_of_ffts
      LOGICAL, INTENT(IN)                                :: transpose
      INTEGER, INTENT(IN)                                :: leading_dimension
      COMPLEX(KIND=dp), DIMENSION(leading_dimension*MERGE(fft_size, number_of_ffts, transpose)), INTENT(INOUT), TARGET      :: grid

      INTERFACE
         SUBROUTINE fft_register_1d_local_inplace_c(direction, fft_size, number_of_ffts, &
                                                    transpose, leading_dimension, grid) &
            BIND(C, name="fft_register_1d_local_inplace_F")
            IMPORT :: C_INT, C_BOOL, C_DOUBLE_COMPLEX, C_PTR
            LOGICAL(C_BOOL), VALUE :: direction
            INTEGER(C_INT), VALUE :: fft_size, number_of_ffts
            LOGICAL(C_BOOL), VALUE :: transpose
            INTEGER(C_INT), VALUE :: leading_dimension
            TYPE(C_PTR), VALUE :: grid
         END SUBROUTINE fft_register_1d_local_inplace_c
      END INTERFACE

      CALL fft_register_1d_local_inplace_c(LOGICAL(direction, C_BOOL), INT(fft_size, C_INT), &
                                           INT(number_of_ffts, C_INT), &
                                           LOGICAL(transpose, C_BOOL), leading_dimension, C_LOC(grid))

   END SUBROUTINE fft_register_1d_local_inplace

! **************************************************************************************************
!> \brief Registers a complex-to-complex 2D FFT
!> \param direction ...
!> \param fft_size ...
!> \param number_of_ffts ...
!> \param transpose ...
!> \param grid ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_register_2d_local_inplace(direction, fft_size, number_of_ffts, transpose, grid)
      LOGICAL, INTENT(IN)                                :: direction
      INTEGER, DIMENSION(2), INTENT(IN)                  :: fft_size
      INTEGER, INTENT(IN)                                :: number_of_ffts
      LOGICAL, INTENT(IN)                                :: transpose
      COMPLEX(KIND=dp), DIMENSION(PRODUCT(fft_size)*number_of_ffts), INTENT(INOUT), TARGET      :: grid

      INTERFACE
         SUBROUTINE fft_register_2d_local_inplace_c(direction, fft_size, number_of_ffts, transpose, grid) &
            BIND(C, name="fft_register_2d_local_inplace_F")
            IMPORT :: C_INT, C_BOOL, C_DOUBLE_COMPLEX, C_PTR
            LOGICAL(C_BOOL), VALUE :: direction
            INTEGER(C_INT), DIMENSION(2) :: fft_size
            INTEGER(C_INT), VALUE :: number_of_ffts
            LOGICAL(C_BOOL), VALUE :: transpose
            TYPE(C_PTR), VALUE :: grid
         END SUBROUTINE fft_register_2d_local_inplace_c
      END INTERFACE

      CALL fft_register_2d_local_inplace_c(LOGICAL(direction, C_BOOL), INT(fft_size, C_INT), &
                                           INT(number_of_ffts, C_INT), &
                                           LOGICAL(transpose, C_BOOL), C_LOC(grid))

   END SUBROUTINE fft_register_2d_local_inplace

! **************************************************************************************************
!> \brief Registers a complex-to-complex 3D FFT
!> \param direction ...
!> \param fft_size ...
!> \param grid ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_register_3d_local_inplace(direction, fft_size, grid)
      LOGICAL, INTENT(IN)                                :: direction
      INTEGER, DIMENSION(3), INTENT(IN)                  :: fft_size
      COMPLEX(KIND=dp), DIMENSION(*), INTENT(INOUT), TARGET      :: grid

      INTERFACE
         SUBROUTINE fft_register_3d_local_inplace_c(direction, fft_size, grid) &
            BIND(C, name="fft_register_3d_local_inplace_F")
            IMPORT :: C_INT, C_DOUBLE_COMPLEX, C_BOOL, C_PTR
            LOGICAL(C_BOOL), VALUE :: direction
            INTEGER(C_INT), DIMENSION(3), INTENT(IN) :: fft_size
            TYPE(C_PTR), VALUE :: grid
         END SUBROUTINE fft_register_3d_local_inplace_c
      END INTERFACE

      CALL fft_register_3d_local_inplace_c(LOGICAL(direction, C_BOOL), INT(fft_size, C_INT), C_LOC(grid))

   END SUBROUTINE fft_register_3d_local_inplace

! **************************************************************************************************
!> \brief Performs a complex-to-complex 1D FFT
!> \param direction ...
!> \param fft_size ...
!> \param number_of_ffts ...
!> \param transpose_in ...
!> \param transpose_out ...
!> \param leading_dimension_in ...
!> \param leading_dimension_out ...
!> \param grid_in ...
!> \param grid_out ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_1d_local(direction, fft_size, number_of_ffts, transpose_in, transpose_out, &
                           leading_dimension_in, leading_dimension_out, grid_in, grid_out)
      LOGICAL, INTENT(IN)                                :: direction
      INTEGER, INTENT(IN)                                :: fft_size, number_of_ffts
      LOGICAL, INTENT(IN)                                :: transpose_in, transpose_out
      INTEGER, INTENT(IN)                                :: leading_dimension_in, leading_dimension_out
      COMPLEX(KIND=dp), DIMENSION(leading_dimension_in*MERGE(fft_size, number_of_ffts, transpose_in)), &
         INTENT(INOUT), TARGET      :: grid_in
      COMPLEX(KIND=dp), DIMENSION(leading_dimension_out*MERGE(fft_size, number_of_ffts, transpose_out)), &
         INTENT(OUT), TARGET        :: grid_out

      INTERFACE
         SUBROUTINE fft_1d_local_c(direction, fft_size, number_of_ffts, transpose_in, transpose_out, &
                                   leading_dimension_in, leading_dimension_out, grid_in, grid_out) &
            BIND(C, name="fft_1d_local_F")
            IMPORT :: C_INT, C_BOOL, C_DOUBLE_COMPLEX, C_PTR
            LOGICAL(C_BOOL), VALUE :: direction
            INTEGER(C_INT), VALUE :: fft_size, number_of_ffts
            LOGICAL(C_BOOL), VALUE :: transpose_in, transpose_out
            INTEGER(C_INT), VALUE :: leading_dimension_in, leading_dimension_out
            TYPE(C_PTR), VALUE :: grid_in, grid_out
         END SUBROUTINE fft_1d_local_c
      END INTERFACE

      CALL fft_1d_local_c(LOGICAL(direction, C_BOOL), INT(fft_size, C_INT), &
                          INT(number_of_ffts, C_INT), LOGICAL(transpose_in, C_BOOL), &
                          LOGICAL(transpose_out, C_BOOL), leading_dimension_in, leading_dimension_out, &
                          C_LOC(grid_in), C_LOC(grid_out))

   END SUBROUTINE fft_1d_local

! **************************************************************************************************
!> \brief Performs a complex-to-complex 2D FFT
!> \param direction ...
!> \param fft_size ...
!> \param number_of_ffts ...
!> \param transpose_in ...
!> \param transpose_out ...
!> \param grid_in ...
!> \param grid_out ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_2d_local(direction, fft_size, number_of_ffts, transpose_in, transpose_out, grid_in, grid_out)
      LOGICAL, INTENT(IN)                                :: direction
      INTEGER, DIMENSION(2), INTENT(IN)                  :: fft_size
      INTEGER, INTENT(IN)                                :: number_of_ffts
      LOGICAL, INTENT(IN)                                :: transpose_in, transpose_out
      COMPLEX(KIND=dp), DIMENSION(PRODUCT(fft_size)*number_of_ffts), INTENT(INOUT), TARGET      :: grid_in
      COMPLEX(KIND=dp), DIMENSION(PRODUCT(fft_size)*number_of_ffts), INTENT(OUT), TARGET        :: grid_out

      INTERFACE
         SUBROUTINE fft_2d_local_c(direction, fft_size, number_of_ffts, transpose_in, transpose_out, grid_in, grid_out) &
            BIND(C, name="fft_2d_local_F")
            IMPORT :: C_INT, C_BOOL, C_DOUBLE_COMPLEX, C_PTR
            LOGICAL(C_BOOL), VALUE :: direction
            INTEGER(C_INT), DIMENSION(2) :: fft_size
            INTEGER(C_INT), VALUE :: number_of_ffts
            LOGICAL(C_BOOL), VALUE :: transpose_in, transpose_out
            TYPE(C_PTR), VALUE :: grid_in, grid_out
         END SUBROUTINE fft_2d_local_c
      END INTERFACE

      CALL fft_2d_local_c(LOGICAL(direction, C_BOOL), INT(fft_size, C_INT), &
                          INT(number_of_ffts, C_INT), LOGICAL(transpose_in, C_BOOL), &
                          LOGICAL(transpose_out, C_BOOL), C_LOC(grid_in), C_LOC(grid_out))

   END SUBROUTINE fft_2d_local

! **************************************************************************************************
!> \brief Performs a complex-to-complex 3D FFT
!> \param direction ...
!> \param fft_size ...
!> \param grid_in ...
!> \param grid_out ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_3d_local(direction, fft_size, grid_in, grid_out)
      LOGICAL, INTENT(IN)                                :: direction
      INTEGER, DIMENSION(3), INTENT(IN)                  :: fft_size
      COMPLEX(KIND=dp), DIMENSION(*), INTENT(INOUT), TARGET      :: grid_in
      COMPLEX(KIND=dp), DIMENSION(*), INTENT(OUT), TARGET        :: grid_out

      INTERFACE
         SUBROUTINE fft_3d_local_c(direction, fft_size, grid_in, grid_out) &
            BIND(C, name="fft_3d_local_F")
            IMPORT :: C_INT, C_DOUBLE_COMPLEX, C_BOOL, C_PTR
            LOGICAL(C_BOOL), VALUE :: direction
            INTEGER(C_INT), DIMENSION(3), INTENT(IN) :: fft_size
            TYPE(C_PTR), VALUE :: grid_in, grid_out
         END SUBROUTINE fft_3d_local_c
      END INTERFACE

      CALL fft_3d_local_c(LOGICAL(direction, C_BOOL), INT(fft_size, C_INT), C_LOC(grid_in), C_LOC(grid_out))

   END SUBROUTINE fft_3d_local

! **************************************************************************************************
!> \brief Performs a complex-to-complex 1D FFT
!> \param direction ...
!> \param fft_size ...
!> \param number_of_ffts ...
!> \param transpose ...
!> \param leading_dimension ...
!> \param grid ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_1d_local_inplace(direction, fft_size, number_of_ffts, transpose, leading_dimension, grid)
      LOGICAL, INTENT(IN)                                :: direction
      INTEGER, INTENT(IN)                                :: fft_size, number_of_ffts
      LOGICAL, INTENT(IN)                                :: transpose
      INTEGER, INTENT(IN)                                :: leading_dimension
      COMPLEX(KIND=dp), DIMENSION(leading_dimension*MERGE(fft_size, number_of_ffts, transpose)), &
         INTENT(INOUT), TARGET      :: grid

      INTERFACE
         SUBROUTINE fft_1d_local_inplace_c(direction, fft_size, number_of_ffts, transpose, leading_dimension, grid) &
            BIND(C, name="fft_1d_local_inplace_F")
            IMPORT :: C_INT, C_BOOL, C_DOUBLE_COMPLEX, C_PTR
            LOGICAL(C_BOOL), VALUE :: direction
            INTEGER(C_INT), VALUE :: fft_size, number_of_ffts
            LOGICAL(C_BOOL), VALUE :: transpose
            INTEGER(C_INT), VALUE :: leading_dimension
            TYPE(C_PTR), VALUE :: grid
         END SUBROUTINE fft_1d_local_inplace_c
      END INTERFACE

      CALL fft_1d_local_inplace_c(LOGICAL(direction, C_BOOL), INT(fft_size, C_INT), &
                                  INT(number_of_ffts, C_INT), &
                                  LOGICAL(transpose, C_BOOL), leading_dimension, C_LOC(grid))

   END SUBROUTINE fft_1d_local_inplace

! **************************************************************************************************
!> \brief Performs a complex-to-complex 2D FFT
!> \param direction ...
!> \param fft_size ...
!> \param number_of_ffts ...
!> \param transpose ...
!> \param grid ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_2d_local_inplace(direction, fft_size, number_of_ffts, transpose, grid)
      LOGICAL, INTENT(IN)                                :: direction
      INTEGER, DIMENSION(2), INTENT(IN)                  :: fft_size
      INTEGER, INTENT(IN)                                :: number_of_ffts
      LOGICAL, INTENT(IN)                                :: transpose
      COMPLEX(KIND=dp), DIMENSION(PRODUCT(fft_size)*number_of_ffts), INTENT(INOUT), TARGET      :: grid

      INTERFACE
         SUBROUTINE fft_2d_local_inplace_c(direction, fft_size, number_of_ffts, transpose, grid) &
            BIND(C, name="fft_2d_local_inplace_F")
            IMPORT :: C_INT, C_BOOL, C_DOUBLE_COMPLEX, C_PTR
            LOGICAL(C_BOOL), VALUE :: direction
            INTEGER(C_INT), DIMENSION(2) :: fft_size
            INTEGER(C_INT), VALUE :: number_of_ffts
            LOGICAL(C_BOOL), VALUE :: transpose
            TYPE(C_PTR), VALUE :: grid
         END SUBROUTINE fft_2d_local_inplace_c
      END INTERFACE

      CALL fft_2d_local_inplace_c(LOGICAL(direction, C_BOOL), INT(fft_size, C_INT), &
                                  INT(number_of_ffts, C_INT), LOGICAL(transpose, C_BOOL), C_LOC(grid))

   END SUBROUTINE fft_2d_local_inplace

! **************************************************************************************************
!> \brief Performs a complex-to-complex 3D FFT
!> \param direction ...
!> \param fft_size ...
!> \param grid ...
!> \author Frederick Stein
! **************************************************************************************************
   SUBROUTINE fft_3d_local_inplace(direction, fft_size, grid)
      LOGICAL, INTENT(IN)                                :: direction
      INTEGER, DIMENSION(3), INTENT(IN)                  :: fft_size
      COMPLEX(KIND=dp), DIMENSION(*), INTENT(INOUT), TARGET      :: grid

      INTERFACE
         SUBROUTINE fft_3d_local_inplace_c(direction, fft_size, grid) &
            BIND(C, name="fft_3d_local_inplace_F")
            IMPORT :: C_INT, C_DOUBLE_COMPLEX, C_BOOL, C_PTR
            LOGICAL(C_BOOL), VALUE :: direction
            INTEGER(C_INT), DIMENSION(3), INTENT(IN) :: fft_size
            TYPE(C_PTR), VALUE :: grid
         END SUBROUTINE fft_3d_local_inplace_c
      END INTERFACE

      CALL fft_3d_local_inplace_c(LOGICAL(direction, C_BOOL), INT(fft_size, C_INT), C_LOC(grid))

   END SUBROUTINE fft_3d_local_inplace

! **************************************************************************************************
!> \brief Create a FFT grid layout anew
!> \param fft_grid ...
!> \param fft_size ...
!> \param comm ...
!> \param dh_inv ...
!> \param use_halfspace ...
!> \param cutoff ...
!> \param pgrid_guess ...
!> \param external_local_bounds ...
! **************************************************************************************************
   SUBROUTINE fft_create_grid(fft_grid, fft_size, comm, dh_inv, use_halfspace, cutoff, pgrid_guess, external_local_bounds)
      TYPE(fft_grid_type), INTENT(INOUT)                 :: fft_grid
      INTEGER, DIMENSION(3), INTENT(IN)                  :: fft_size
      CLASS(mp_comm_type), INTENT(IN)                    :: comm
      REAL(KIND=dp), DIMENSION(3, 3), INTENT(IN)         :: dh_inv
      LOGICAL, INTENT(IN)                                :: use_halfspace
      REAL(KIND=dp), INTENT(IN)                          :: cutoff
      INTEGER, DIMENSION(2), INTENT(IN), OPTIONAL        :: pgrid_guess
      INTEGER, DIMENSION(2, 3), INTENT(IN), OPTIONAL     :: external_local_bounds

      INTEGER(C_INT), DIMENSION(2), TARGET               :: pgrid_guess_c
      INTEGER(C_INT), DIMENSION(2, 3), TARGET            :: external_local_bounds_c
      TYPE(C_PTR)                                        :: pgrid_guess_ptr, external_local_bounds_ptr

      INTERFACE
         SUBROUTINE fft_create_grid_c(layout, comm, fft_size, dh_inv, use_halfspace, cutoff, &
                                      pgrid_guess, external_local_bounds) BIND(C, name="fft_create_grid_F")
            IMPORT :: C_INT, C_DOUBLE, C_PTR, C_BOOL
            TYPE(C_PTR), INTENT(INOUT) :: layout
            INTEGER(C_INT), VALUE :: comm
            INTEGER(C_INT), DIMENSION(3), INTENT(IN) :: fft_size
            REAL(C_DOUBLE), DIMENSION(3, 3), INTENT(IN) :: dh_inv
            LOGICAL(C_BOOL), VALUE :: use_halfspace
            REAL(C_DOUBLE), VALUE :: cutoff
            TYPE(C_PTR), VALUE :: pgrid_guess
            TYPE(C_PTR), VALUE :: external_local_bounds
         END SUBROUTINE fft_create_grid_c
      END INTERFACE

      IF (C_ASSOCIATED(fft_grid%fft_grid)) &
         CPABORT("The grid layout must be unassociated before creation")

      pgrid_guess_ptr = C_NULL_PTR
      IF (PRESENT(pgrid_guess)) THEN
         pgrid_guess_c = INT(pgrid_guess, C_INT)
         pgrid_guess_ptr = C_LOC(pgrid_guess_c)
      END IF

      external_local_bounds_ptr = C_NULL_PTR
      IF (PRESENT(external_local_bounds)) THEN
         external_local_bounds_c = INT(external_local_bounds, C_INT)
         external_local_bounds_ptr = C_LOC(external_local_bounds_c)
      END IF

      CALL fft_create_grid_c(fft_grid%fft_grid, INT(comm%get_handle(), C_INT), INT(fft_size, C_INT), &
                             REAL(dh_inv, C_DOUBLE), LOGICAL(use_halfspace, C_BOOL), REAL(cutoff, C_DOUBLE), &
                             pgrid_guess_ptr, external_local_bounds_ptr)

      IF (.NOT. C_ASSOCIATED(fft_grid%fft_grid)) &
         CPABORT("Grid creation failed")

   END SUBROUTINE fft_create_grid

! **************************************************************************************************
!> \brief Create a FFT grid layout from a reference grid layout
!> \param fft_grid ...
!> \param fft_size ...
!> \param cutoff ...
!> \param external_local_bounds ...
!> \param fft_grid_ref ...
! **************************************************************************************************
   SUBROUTINE fft_create_grid_from_reference(fft_grid, fft_size, cutoff, external_local_bounds, fft_grid_ref)
      TYPE(fft_grid_type), INTENT(INOUT)                 :: fft_grid
      INTEGER, DIMENSION(3), INTENT(IN)                  :: fft_size
      REAL(KIND=dp), INTENT(IN)                          :: cutoff
      INTEGER, DIMENSION(2, 3), INTENT(IN), OPTIONAL     :: external_local_bounds
      TYPE(fft_grid_type), INTENT(IN)                    :: fft_grid_ref

      INTEGER(C_INT), DIMENSION(2, 3), TARGET            :: external_local_bounds_c
      TYPE(C_PTR)                                        :: external_local_bounds_ptr

      INTERFACE
         SUBROUTINE fft_create_grid_from_reference_c(layout, fft_size, cutoff, external_local_bounds, layout_ref) &
            BIND(C, name="fft_create_grid_from_reference_F")
            IMPORT :: C_INT, C_PTR, C_DOUBLE
            TYPE(C_PTR), INTENT(INOUT) :: layout
            INTEGER(C_INT), DIMENSION(3), INTENT(IN) :: fft_size
            REAL(C_DOUBLE), VALUE :: cutoff
            TYPE(C_PTR), VALUE :: external_local_bounds
            TYPE(C_PTR), VALUE :: layout_ref
         END SUBROUTINE fft_create_grid_from_reference_c
      END INTERFACE

      IF (C_ASSOCIATED(fft_grid%fft_grid)) &
         CPABORT("The grid layout must be unassociated before creation")
      IF (.NOT. C_ASSOCIATED(fft_grid_ref%fft_grid)) &
         CPABORT("The reference grid layout must be associated")

      external_local_bounds_ptr = C_NULL_PTR
      IF (PRESENT(external_local_bounds)) THEN
         external_local_bounds_c = INT(external_local_bounds, C_INT)
         external_local_bounds_ptr = C_LOC(external_local_bounds_c)
      END IF

      CALL fft_create_grid_from_reference_c(fft_grid%fft_grid, INT(fft_size, C_INT), REAL(cutoff, C_DOUBLE), &
                                            external_local_bounds_ptr, fft_grid_ref%fft_grid)

      IF (.NOT. C_ASSOCIATED(fft_grid%fft_grid)) &
         CPABORT("Grid creation failed")

   END SUBROUTINE fft_create_grid_from_reference

! **************************************************************************************************
!> \brief Get the number of global points of a grid layout
!> \param fft_grid ...
!> \return ...
! **************************************************************************************************
   FUNCTION fft_grid_get_npts_global(fft_grid) RESULT(npts_global)
      TYPE(fft_grid_type), INTENT(IN)                    :: fft_grid
      INTEGER, DIMENSION(3)                              :: npts_global

      INTEGER(C_INT), DIMENSION(3)                       :: npts_global_c

      INTERFACE
         SUBROUTINE fft_grid_get_npts_global_c(layout, npts_global) &
            BIND(C, name="fft_grid_get_npts_global_F")
            IMPORT :: C_PTR, C_INT
            TYPE(C_PTR), VALUE :: layout
            INTEGER(C_INT), DIMENSION(*) :: npts_global
         END SUBROUTINE fft_grid_get_npts_global_c
      END INTERFACE

      CPASSERT(C_ASSOCIATED(fft_grid%fft_grid))

      CALL fft_grid_get_npts_global_c(fft_grid%fft_grid, npts_global_c)

      npts_global = INT(npts_global_c)

   END FUNCTION fft_grid_get_npts_global

! **************************************************************************************************
!> \brief Get the cartesian communicator of a grid layout
!> \param fft_grid ...
!> \param comm ...
! **************************************************************************************************
   SUBROUTINE fft_grid_get_comm(fft_grid, comm)
      TYPE(fft_grid_type), INTENT(IN)                    :: fft_grid
      CLASS(mp_comm_type)                                :: comm

      INTEGER(C_INT)                                     :: comm_handle
      TYPE(mp_comm_type)                                 :: my_comm

      INTERFACE
         SUBROUTINE fft_grid_get_comm_c(layout, comm) &
            BIND(C, name="fft_grid_get_comm_F")
            IMPORT :: C_PTR, C_INT
            TYPE(C_PTR), VALUE :: layout
            INTEGER(C_INT), INTENT(OUT) :: comm
         END SUBROUTINE fft_grid_get_comm_c
      END INTERFACE

      CPASSERT(C_ASSOCIATED(fft_grid%fft_grid))

      CALL fft_grid_get_comm_c(fft_grid%fft_grid, comm_handle)

      CALL my_comm%set_handle(INT(comm_handle), 2)
      CALL comm%from_dup(my_comm)

   END SUBROUTINE fft_grid_get_comm

! **************************************************************************************************
!> \brief Get the process-to-local-bounds mapping in real space
!> \param fft_grid ...
!> \param proc2local_rs ...
! **************************************************************************************************
   SUBROUTINE fft_grid_get_proc2local_rs(fft_grid, proc2local_rs)
      TYPE(fft_grid_type), INTENT(IN)                    :: fft_grid
      INTEGER, DIMENSION(:, :, 0:), INTENT(OUT)          :: proc2local_rs

      INTEGER(C_INT), DIMENSION(:), TARGET, ALLOCATABLE  :: proc2local_rs_c

      INTERFACE
         SUBROUTINE fft_grid_get_proc2local_rs_c(layout, proc2local_rs) &
            BIND(C, name="fft_grid_get_proc2local_rs_F")
            IMPORT :: C_PTR
            TYPE(C_PTR), VALUE :: layout
            TYPE(C_PTR), VALUE :: proc2local_rs
         END SUBROUTINE fft_grid_get_proc2local_rs_c
      END INTERFACE

      CPASSERT(C_ASSOCIATED(fft_grid%fft_grid))

      ALLOCATE (proc2local_rs_c(0:SIZE(proc2local_rs) - 1))
      CALL fft_grid_get_proc2local_rs_c(fft_grid%fft_grid, C_LOC(proc2local_rs_c))
      CALL fft_grid_copy_proc2local(proc2local_rs, proc2local_rs_c)
      DEALLOCATE (proc2local_rs_c)

   END SUBROUTINE fft_grid_get_proc2local_rs

! **************************************************************************************************
!> \brief Get the process-to-local-bounds mapping in mixed space
!> \param fft_grid ...
!> \param proc2local_ms ...
! **************************************************************************************************
   SUBROUTINE fft_grid_get_proc2local_ms(fft_grid, proc2local_ms)
      TYPE(fft_grid_type), INTENT(IN)                    :: fft_grid
      INTEGER, DIMENSION(:, :, 0:), INTENT(OUT)          :: proc2local_ms

      INTEGER(C_INT), DIMENSION(:), TARGET, ALLOCATABLE  :: proc2local_ms_c

      INTERFACE
         SUBROUTINE fft_grid_get_proc2local_ms_c(layout, proc2local_ms) &
            BIND(C, name="fft_grid_get_proc2local_ms_F")
            IMPORT :: C_PTR
            TYPE(C_PTR), VALUE :: layout
            TYPE(C_PTR), VALUE :: proc2local_ms
         END SUBROUTINE fft_grid_get_proc2local_ms_c
      END INTERFACE

      CPASSERT(C_ASSOCIATED(fft_grid%fft_grid))

      ALLOCATE (proc2local_ms_c(0:SIZE(proc2local_ms) - 1))
      CALL fft_grid_get_proc2local_ms_c(fft_grid%fft_grid, C_LOC(proc2local_ms_c))
      CALL fft_grid_copy_proc2local(proc2local_ms, proc2local_ms_c)
      DEALLOCATE (proc2local_ms_c)

   END SUBROUTINE fft_grid_get_proc2local_ms

! **************************************************************************************************
!> \brief Get the process-to-local-bounds mapping in reciprocal space
!> \param fft_grid ...
!> \param proc2local_gs ...
! **************************************************************************************************
   SUBROUTINE fft_grid_get_proc2local_gs(fft_grid, proc2local_gs)
      TYPE(fft_grid_type), INTENT(IN)                    :: fft_grid
      INTEGER, DIMENSION(:, :, 0:), INTENT(OUT)          :: proc2local_gs

      INTEGER(C_INT), DIMENSION(:), TARGET, ALLOCATABLE  :: proc2local_gs_c

      INTERFACE
         SUBROUTINE fft_grid_get_proc2local_gs_c(layout, proc2local_gs) &
            BIND(C, name="fft_grid_get_proc2local_gs_F")
            IMPORT :: C_PTR
            TYPE(C_PTR), VALUE :: layout
            TYPE(C_PTR), VALUE :: proc2local_gs
         END SUBROUTINE fft_grid_get_proc2local_gs_c
      END INTERFACE

      CPASSERT(C_ASSOCIATED(fft_grid%fft_grid))

      ALLOCATE (proc2local_gs_c(0:SIZE(proc2local_gs) - 1))
      CALL fft_grid_get_proc2local_gs_c(fft_grid%fft_grid, C_LOC(proc2local_gs_c))
      CALL fft_grid_copy_proc2local(proc2local_gs, proc2local_gs_c)
      DEALLOCATE (proc2local_gs_c)

   END SUBROUTINE fft_grid_get_proc2local_gs

! **************************************************************************************************
!> \brief Convert the C-side process distribution to one-based local bounds
!> \param proc2local ...
!> \param proc2local_c ...
! **************************************************************************************************
   SUBROUTINE fft_grid_copy_proc2local(proc2local, proc2local_c)
      INTEGER, DIMENSION(:, :, 0:), INTENT(OUT)          :: proc2local
      INTEGER(C_INT), DIMENSION(0:), INTENT(IN)          :: proc2local_c

      INTEGER                                            :: c_dir, dir, lb_dir, ndir, proc, proc_start, &
                                                            proc_length, num_proc

      CPASSERT(SIZE(proc2local, 1) == 2)
      CPASSERT(SIZE(proc2local_c) == 6*SIZE(proc2local, 3))
      ndir = SIZE(proc2local, 2)
      lb_dir = LBOUND(proc2local, 2)
      num_proc = SIZE(proc2local_c)/6

      DO proc = 0, num_proc - 1
         DO dir = 1, ndir
            ! The C layout stores directions as (z, y, x), so revert the order.
            c_dir = ndir - dir
            proc_start = INT(proc2local_c(6*proc + 2*c_dir))
            proc_length = INT(proc2local_c(6*proc + 2*c_dir + 1))
            proc2local(1, lb_dir + dir - 1, proc) = proc_start + 1
            proc2local(2, lb_dir + dir - 1, proc) = proc_start + proc_length
         END DO
      END DO

   END SUBROUTINE fft_grid_copy_proc2local

! **************************************************************************************************
!> \brief Free a FFT grid layout
!> \param fft_grid ...
! **************************************************************************************************
   SUBROUTINE fft_free_grid(fft_grid)
      TYPE(fft_grid_type), INTENT(INOUT)                 :: fft_grid

      INTERFACE
         SUBROUTINE fft_free_grid_c(layout) &
            BIND(C, name="fft_free_grid_F")
            IMPORT :: C_PTR
            TYPE(C_PTR), VALUE :: layout
         END SUBROUTINE fft_free_grid_c
      END INTERFACE

      IF (C_ASSOCIATED(fft_grid%fft_grid)) THEN
         CALL fft_free_grid_c(fft_grid%fft_grid)
         fft_grid%fft_grid = C_NULL_PTR
      END IF

   END SUBROUTINE fft_free_grid

END MODULE fft_api
