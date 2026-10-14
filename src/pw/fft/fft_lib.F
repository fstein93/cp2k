!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: GPL-2.0-or-later                                                      !
!--------------------------------------------------------------------------------------------------!
MODULE fft_lib

   USE fft_kinds,                       ONLY: dp
   USE fft_plan,                        ONLY: fft_plan_type
   USE fftsg_lib,                       ONLY: fftsg1dm,&
                                              fftsg3d,&
                                              fftsg_do_cleanup,&
                                              fftsg_do_init
   USE fftw3_lib,                       ONLY: &
        fftw31dm, fftw33d, fftw3_create_plan_1d, fftw3_create_plan_3d, fftw3_destroy_plan
#include "../../base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'fft_lib'

   INTEGER, SAVE :: fft_type = 0

   PUBLIC :: fft_create_plan_3d
   PUBLIC :: fft_create_plan_1d, fft_1d, fft_3d, fft_destroy_plan

CONTAINS

! **************************************************************************************************
!> \brief ...
!> \param plan ...
!> \param fft_in_place ...
!> \param fsign ...
!> \param n ...
!> \param zin ...
!> \param zout ...
! **************************************************************************************************
   SUBROUTINE fft_create_plan_3d(plan, fft_in_place, fsign, n, zin, zout)

      TYPE(fft_plan_type), INTENT(INOUT)                 :: plan
      LOGICAL, INTENT(IN)                                :: fft_in_place
      INTEGER, INTENT(IN)                                :: fsign
      INTEGER, DIMENSION(3), INTENT(IN)                  :: n
      COMPLEX(KIND=dp), DIMENSION(*), INTENT(INOUT)      :: zin, zout

      plan%fsign = fsign
      plan%fft_in_place = fft_in_place
      plan%n_3d = n
!$    plan%need_alt_plan = .FALSE.

      ! Planning only needed for FFTW3
      IF (fft_type == 3) THEN
         CALL fftw3_create_plan_3d(plan, zin, zout)
         plan%valid = .TRUE.
      END IF

   END SUBROUTINE fft_create_plan_3d

!
! really ugly, plan is intent out, because plan%fsign is also a status flag
! if something goes wrong, plan%fsign is set to zero, and the plan becomes invalid
!
! **************************************************************************************************
!> \brief ...
!> \param plan ...
!> \param scale ...
!> \param zin ...
!> \param zout ...
!> \param stat ...
! **************************************************************************************************
   SUBROUTINE fft_3d(plan, scale, zin, zout, stat)
      TYPE(fft_plan_type), INTENT(IN)                    :: plan
      REAL(KIND=dp), INTENT(IN)                          :: scale
      COMPLEX(KIND=dp), DIMENSION(*), INTENT(INOUT)      :: zin, zout
      INTEGER, INTENT(OUT)                               :: stat

      stat = plan%fsign
      IF (plan%n_3d(1)*plan%n_3d(2)*plan%n_3d(3) > 0) THEN
         SELECT CASE (fft_type)
         CASE DEFAULT
            CPABORT("fft_3d")
         CASE (1)
            CALL fftsg3d(plan%fft_in_place, stat, scale, plan%n_3d, zin, zout)
         CASE (3)
            CALL fftw33d(plan, scale, zin, zout, stat)
         END SELECT
      END IF
      ! stat is set to zero on error, -1,+1 are OK
      IF (stat == 0) THEN
         stat = 1
      ELSE
         stat = 0
      END IF

   END SUBROUTINE fft_3d

! **************************************************************************************************

! **************************************************************************************************
!> \brief ...
!> \param plan ...
!> \param fsign ...
!> \param trans_in ...
!> \param trans_out ...
!> \param ldx_in ...
!> \param ldx_out ...
!> \param n ...
!> \param m ...
!> \param zin ...
!> \param zout ...
! **************************************************************************************************
   SUBROUTINE fft_create_plan_1d(plan, fsign, trans_in, trans_out, ldx_in, ldx_out, n, m, zin, zout)
      TYPE(fft_plan_type), INTENT(INOUT)                 :: plan
      INTEGER, INTENT(IN)                                :: fsign
      LOGICAL, INTENT(IN)                                :: trans_in, trans_out
      INTEGER, INTENT(IN)                                :: ldx_in, ldx_out, n, m
      COMPLEX(KIND=dp), DIMENSION(*), INTENT(IN)         :: zin, zout

      plan%fsign = fsign
      plan%trans_in = trans_in
      plan%trans_out = trans_out
      IF (plan%trans_in) THEN
         plan%ldx_in = ldx_in
         plan%ldy_in = n
      ELSE
         plan%ldx_in = ldx_in
         plan%ldy_in = m
      END IF
      IF (plan%trans_out) THEN
         plan%ldx_out = ldx_out
         plan%ldy_out = n
      ELSE
         plan%ldx_out = ldx_out
         plan%ldy_out = m
      END IF

      plan%n = n
      plan%m = m
!$    plan%need_alt_plan = .FALSE.

      ! Planning only needed for FFTW3
      IF ((fft_type == 3) .AND. (n*m /= 0)) THEN
         CALL fftw3_create_plan_1d(plan, zin, zout)
         plan%valid = .TRUE.
      ELSE
         plan%valid = .FALSE.
      END IF

   END SUBROUTINE fft_create_plan_1d

! **************************************************************************************************
!> \brief ...
!> \param plan ...
! **************************************************************************************************
   SUBROUTINE fft_destroy_plan(plan)
      TYPE(fft_plan_type), INTENT(INOUT)                 :: plan

! Planning only needed for FFTW3

      IF (plan%valid .AND. fft_type == 3) THEN
         CALL fftw3_destroy_plan(plan)
         plan%valid = .FALSE.
      END IF

   END SUBROUTINE fft_destroy_plan

! **************************************************************************************************
!> \brief ...
!> \param plan ...
!> \param zin ...
!> \param zout ...
!> \param scale ...
!> \param stat ...
! **************************************************************************************************
   SUBROUTINE fft_1d(plan, zin, zout, scale, stat)
      TYPE(fft_plan_type), INTENT(IN)                    :: plan
      COMPLEX(KIND=dp), DIMENSION(*), INTENT(INOUT)      :: zin, zout
      REAL(KIND=dp), INTENT(IN)                          :: scale
      INTEGER, INTENT(OUT)                               :: stat

      stat = plan%fsign
      IF (plan%n*plan%m > 0) THEN
         SELECT CASE (fft_type)
         CASE DEFAULT
            CPABORT("fft_1d")
         CASE (1)
            CALL fftsg1dm(stat, plan%trans_in, plan%trans_out, plan%n, plan%m, &
                          plan%ldx_in, plan%ldy_in, plan%ldx_out, plan%ldy_out, zin, zout, scale)
         CASE (3)
            CALL fftw31dm(plan, zin, zout, scale, stat)
         END SELECT
      END IF
      ! stat is set to zero on error, -1,+1 are OK
      IF (stat == 0) THEN
         stat = 1
      ELSE
         stat = 0
      END IF

   END SUBROUTINE fft_1d

END MODULE fft_lib
