!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright 2000-2026 CP2K developers group <https://cp2k.org>                                   !
!                                                                                                  !
!   SPDX-License-Identifier: BSD-3-Clause                                                          !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief FFT grid-size utility routines (radix selection and FFT direction constants).
! **************************************************************************************************
MODULE fft_tools
#include "../base/base_uses.f90"
   IMPLICIT NONE

   PRIVATE

   PUBLIC :: fft_radix_operations
   PUBLIC :: FWFFT, BWFFT
   PUBLIC :: FFT_RADIX_CLOSEST, FFT_RADIX_NEXT, FFT_RADIX_NEXT_ODD

   LOGICAL, PARAMETER :: FWFFT = .TRUE., BWFFT = .FALSE.
   INTEGER, PARAMETER :: FFT_RADIX_CLOSEST = 493, FFT_RADIX_NEXT = 494
   INTEGER, PARAMETER :: FFT_RADIX_ALLOWED = 495, FFT_RADIX_DISALLOWED = 496
   INTEGER, PARAMETER :: FFT_RADIX_NEXT_ODD = 497

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'fft_tools'

CONTAINS

! **************************************************************************************************
!> \brief ...
!> \param DATA ...
!> \param max_length ...
! **************************************************************************************************
   SUBROUTINE fft_get_lengths(DATA, max_length)

      INTEGER, DIMENSION(:), INTENT(INOUT)               :: DATA
      INTEGER, INTENT(INOUT)                             :: max_length

      INTEGER                                            :: h, i, j, k, m, maxn, maxn_elevens, &
                                                            maxn_fives, maxn_sevens, &
                                                            maxn_thirteens, maxn_threes, &
                                                            maxn_twos, ndata, nmax, number
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: dlocal, idx

!------------------------------------------------------------------------------
! compute ndata
!! FFTW can do arbitrary(?) lengths, maybe you want to limit them to some
!!    powers of small prime numbers though...

      maxn_twos = 15
      maxn_threes = 3
      maxn_fives = 2
      maxn_sevens = 1
      maxn_elevens = 1
      maxn_thirteens = 0
      maxn = 37748736

      ndata = 0
      DO h = 0, maxn_twos
         nmax = HUGE(0)/2**h
         DO i = 0, maxn_threes
            DO j = 0, maxn_fives
               DO k = 0, maxn_sevens
                  DO m = 0, maxn_elevens
                     number = (3**i)*(5**j)*(7**k)*(11**m)

                     IF (number > nmax) CYCLE

                     number = number*2**h
                     IF (number >= maxn) CYCLE

                     ndata = ndata + 1
                  END DO
               END DO
            END DO
         END DO
      END DO

      ALLOCATE (dlocal(ndata), idx(ndata))

      ndata = 0
      dlocal(:) = 0
      DO h = 0, maxn_twos
         nmax = HUGE(0)/2**h
         DO i = 0, maxn_threes
            DO j = 0, maxn_fives
               DO k = 0, maxn_sevens
                  DO m = 0, maxn_elevens
                     number = (3**i)*(5**j)*(7**k)*(11**m)

                     IF (number > nmax) CYCLE

                     number = number*2**h
                     IF (number >= maxn) CYCLE

                     ndata = ndata + 1
                     dlocal(ndata) = number
                  END DO
               END DO
            END DO
         END DO
      END DO

      CALL sortint(dlocal, ndata, idx)
      ndata = MIN(ndata, max_length)
      DATA(1:ndata) = dlocal(1:ndata)
      max_length = ndata

      DEALLOCATE (dlocal, idx)

   END SUBROUTINE fft_get_lengths

! **************************************************************************************************
!> \brief ...
!> \param iarr ...
!> \param n ...
!> \param index ...
! **************************************************************************************************
   SUBROUTINE sortint(iarr, n, index)

      INTEGER, INTENT(IN)                                :: n
      INTEGER, INTENT(INOUT)                             :: iarr(1:n)
      INTEGER, INTENT(OUT)                               :: INDEX(1:n)

      INTEGER, PARAMETER                                 :: m = 7, nstack = 50

      INTEGER                                            :: a, i, ib, ir, istack(1:nstack), itemp, &
                                                            j, jstack, k, l, temp

!------------------------------------------------------------------------------

      DO i = 1, n
         INDEX(i) = i
      END DO
      jstack = 0
      l = 1
      ir = n
      DO WHILE (.TRUE.)
      IF (ir - l < m) THEN
         DO j = l + 1, ir
            a = iarr(j)
            ib = INDEX(j)
            DO i = j - 1, 0, -1
               IF (i == 0) EXIT
               IF (iarr(i) <= a) EXIT
               iarr(i + 1) = iarr(i)
               INDEX(i + 1) = INDEX(i)
            END DO
            iarr(i + 1) = a
            INDEX(i + 1) = ib
         END DO
         IF (jstack == 0) RETURN
         ir = istack(jstack)
         l = istack(jstack - 1)
         jstack = jstack - 2
      ELSE
         k = (l + ir)/2
         temp = iarr(k)
         iarr(k) = iarr(l + 1)
         iarr(l + 1) = temp
         itemp = INDEX(k)
         INDEX(k) = INDEX(l + 1)
         INDEX(l + 1) = itemp
         IF (iarr(l + 1) > iarr(ir)) THEN
            temp = iarr(l + 1)
            iarr(l + 1) = iarr(ir)
            iarr(ir) = temp
            itemp = INDEX(l + 1)
            INDEX(l + 1) = INDEX(ir)
            INDEX(ir) = itemp
         END IF
         IF (iarr(l) > iarr(ir)) THEN
            temp = iarr(l)
            iarr(l) = iarr(ir)
            iarr(ir) = temp
            itemp = INDEX(l)
            INDEX(l) = INDEX(ir)
            INDEX(ir) = itemp
         END IF
         IF (iarr(l + 1) > iarr(l)) THEN
            temp = iarr(l + 1)
            iarr(l + 1) = iarr(l)
            iarr(l) = temp
            itemp = INDEX(l + 1)
            INDEX(l + 1) = INDEX(l)
            INDEX(l) = itemp
         END IF
         i = l + 1
         j = ir
         a = iarr(l)
         ib = INDEX(l)
         DO WHILE (.TRUE.)
            i = i + 1
            DO WHILE (iarr(i) < a)
               i = i + 1
            END DO
            j = j - 1
            DO WHILE (iarr(j) > a)
               j = j - 1
            END DO
            IF (j < i) EXIT
            temp = iarr(i)
            iarr(i) = iarr(j)
            iarr(j) = temp
            itemp = INDEX(i)
            INDEX(i) = INDEX(j)
            INDEX(j) = itemp
         END DO
         iarr(l) = iarr(j)
         iarr(j) = a
         INDEX(l) = INDEX(j)
         INDEX(j) = ib
         jstack = jstack + 2
         IF (jstack > nstack) CPABORT("Nstack too small in sortint")
         IF (ir - i + 1 >= j - l) THEN
            istack(jstack) = ir
            istack(jstack - 1) = i
            ir = j - 1
         ELSE
            istack(jstack) = j - 1
            istack(jstack - 1) = l
            l = i
         END IF
      END IF

      END DO

   END SUBROUTINE sortint

! **************************************************************************************************
!> \brief Determine the allowed lengths of FFT's   '''
!> \param radix_in ...
!> \param radix_out ...
!> \param operation ...
!> \par History
!>      new library structure (JGH)
!> \author Ari Seitsonen
! **************************************************************************************************
   SUBROUTINE fft_radix_operations(radix_in, radix_out, operation)

      INTEGER, INTENT(IN)                                :: radix_in
      INTEGER, INTENT(OUT)                               :: radix_out
      INTEGER, INTENT(IN)                                :: operation

      INTEGER, PARAMETER                                 :: number_of_small_radices = 81
      INTEGER, DIMENSION(number_of_small_radices), PARAMETER :: small_radices = [2, 4, 6, 8, 9, 12,&
         15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 54, 60, 64, 72, 75, 80, 81, 90, 96, &
         100, 108, 120, 125, 128, 135, 144, 150, 160, 162, 180, 192, 200, 216, 225, 240, 243, 256, &
         270, 288, 300, 320, 324, 360, 375, 384, 400, 405, 432, 450, 480, 486, 500, 512, 540, 576, &
         600, 625, 640, 648, 675, 720, 729, 750, 768, 800, 810, 864, 900, 960, 972, 1000, 1024]

      INTEGER                                            :: i, iloc, ldata, trial_number
      INTEGER, ALLOCATABLE, DIMENSION(:)                 :: DATA

      ldata = 1024
      ALLOCATE (DATA(ldata))
      DATA = -1

      DATA(1:number_of_small_radices) = small_radices(1:number_of_small_radices)

      ! First run: try smaller radices
      ! Second run: Fetch extended list of radices and try these
      DO trial_number = 1, 2
         IF (trial_number == 2) CALL fft_get_lengths(DATA, ldata)
         iloc = 0
         DO i = 1, ldata
            IF (DATA(i) == radix_in) THEN
               iloc = i
               EXIT
            ELSE
               IF (OPERATION == FFT_RADIX_ALLOWED) THEN
                  CYCLE
               ELSE IF (DATA(i) > radix_in) THEN
                  iloc = i
                  EXIT
               END IF
            END IF
         END DO

         IF (iloc == 0) THEN
            IF (trial_number == 1) THEN
               ! We haven't found any suitable size in the list of smaller radices
               ! because the requested radix is too large (large cell or large cutoff)
               ! So, try again with the larger radices
               CYCLE
            ELSE
               IF (OPERATION == FFT_RADIX_CLOSEST) THEN
                  radix_out = MAXVAL(DATA)
               ELSE
                  CALL cp_abort(__LOCATION__, &
                                "No suitable FFT lengths found! Either enable FFTW or reduce the grid size!")
               END IF
               RETURN
            END IF

         END IF

         IF (OPERATION == FFT_RADIX_ALLOWED) THEN
            IF (DATA(iloc) == radix_in) THEN
               radix_out = FFT_RADIX_ALLOWED
            ELSE
               radix_out = FFT_RADIX_DISALLOWED
            END IF

         ELSE IF (OPERATION == FFT_RADIX_CLOSEST) THEN
            IF (DATA(iloc) == radix_in) THEN
               radix_out = DATA(iloc)
            ELSE
               IF (ABS(DATA(iloc - 1) - radix_in) <= &
                   ABS(DATA(iloc) - radix_in)) THEN
                  radix_out = DATA(iloc - 1)
               ELSE
                  radix_out = DATA(iloc)
               END IF
            END IF

         ELSE IF (OPERATION == FFT_RADIX_NEXT) THEN
            radix_out = DATA(iloc)

         ELSE IF (OPERATION == FFT_RADIX_NEXT_ODD) THEN
            DO i = iloc, ldata
               IF (MOD(DATA(i), 2) == 1) THEN
                  radix_out = DATA(i)
                  EXIT
               END IF
            END DO
            IF (MOD(radix_out, 2) == 0) THEN
               ! No larger odd radix found
               IF (trial_number == 1) THEN
                  CYCLE
               ELSE
                  radix_out = radix_in + 1 - MOD(radix_in, 2)
               END IF
            END IF

         END IF
         EXIT
      END DO

      DEALLOCATE (DATA)

   END SUBROUTINE fft_radix_operations
END MODULE fft_tools
