!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright (C) 2000 - 2020  CP2K developers group                                               !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Calculates the short range correlation LDA energy (Improved version of Paola Gori-Giorgi's Code
!> \par History
!>      18-MAR-2002, TCH, working version
!>      fawzi (04.2004)  : adapted to the new xc interface
!> \see functionals_utilities
! **************************************************************************************************
MODULE xc_sr_lda
#:include "xc_sr_lda.fypp"

   USE bibliography,                    ONLY: Paziani2006,&
                                              Toulouse2004,&
                                              cite_reference
   USE input_section_types,             ONLY: section_vals_type,&
                                              section_vals_val_get
   USE kinds,                           ONLY: dp
   USE mathconstants,                   ONLY: pi
   USE xc_input_constants,              ONLY: pw_dmc,&
                                              pw_orig,&
                                              pw_vmc
   USE xc_derivative_set_types,         ONLY: xc_derivative_set_type,&
                                              xc_dset_get_derivative
   USE xc_derivative_types,             ONLY: xc_derivative_get,&
                                              xc_derivative_type
   USE xc_rho_cflags_types,             ONLY: xc_rho_cflags_type
   USE xc_rho_set_types,                ONLY: xc_rho_set_get,&
                                              xc_rho_set_type
   USE xc_functionals_utilities,        ONLY: set_util
#include "../base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'xc_sr_lda'

@:global_var_srlda()

   PUBLIC :: sr_lda_info, sr_lda_eval, sr_lsd_eval

CONTAINS

! **************************************************************************************************
!> \brief Return some info on the functionals.
!> \param reference full reference
!> \param shortform short reference
!> \param lsd ...
!> \param needs ...
!> \param max_deriv ...
! **************************************************************************************************
   SUBROUTINE sr_lda_info(reference, shortform, lsd, needs, max_deriv)
      CHARACTER(LEN=*), INTENT(OUT), OPTIONAL            :: reference, shortform
      LOGICAL, INTENT(IN), OPTIONAL                      :: lsd
      TYPE(xc_rho_cflags_type), INTENT(INOUT), OPTIONAL  :: needs
      INTEGER, INTENT(OUT), OPTIONAL                     :: max_deriv

      CHARACTER(len=*), PARAMETER :: routineN = 'sr_lda_info', routineP = moduleN//':'//routineN

      CALL cite_reference(Toulouse2004)
      CALL cite_reference(Paziani2006)
      IF (PRESENT(reference)) THEN
         reference = "J. Toulouse, A. Savin, and H.-J. Flad," &
                     //" Int. J. Quantum Chem. 100, 1074-1056 (2004)"
      END IF
      IF (PRESENT(shortform)) THEN
         shortform = "J. Toulouse et al., IJQC 100, 1074-1056 (2004)"
      END IF
      IF (PRESENT(needs)) THEN
         IF (lsd) THEN
            needs%rho_spin = .TRUE.
         ELSE
            needs%rho = .TRUE.
         END IF
      END IF
      IF (PRESENT(max_deriv)) max_deriv = 2

   END SUBROUTINE sr_lda_info

! **************************************************************************************************
!> \brief Calculate the correlation energy and its derivatives
!>      wrt to rho (the electron density) up to 3rd order. This
!>      is the short-range LDA version of the Perdew-Wang correlation energy
!>      If no order argument is given, then the routine calculates
!>      just the energy.
!> \param rho_set ...
!> \param deriv_set ...
!> \param order order of derivatives to calculate
!>      order must lie between -2 and 2. If it is negative then only
!>      that order will be calculated, otherwise all derivatives up to
!>      that order will be calculated.
!> \param sr_section ...
! **************************************************************************************************
   SUBROUTINE sr_lda_eval(rho_set, deriv_set, order, sr_section)

      TYPE(xc_rho_set_type), POINTER                     :: rho_set
      TYPE(xc_derivative_set_type), POINTER              :: deriv_set
      INTEGER, INTENT(in)                                :: order
      TYPE(section_vals_type), POINTER                   :: sr_section

      CHARACTER(len=*), PARAMETER :: routineN = 'sr_lda_eval', routineP = moduleN//':'//routineN

      INTEGER                                            :: npoints, handle, method
      INTEGER, DIMENSION(:, :), POINTER                  :: bo
      REAL(KIND=dp)                                      :: omega, rho_cutoff, sc, sx
      REAL(KIND=dp), DIMENSION(:, :, :), POINTER         :: dummy, e_0, e_rho, rho, e_rho_rho
      TYPE(xc_derivative_type), POINTER                  :: deriv

      CALL timeset(routineN, handle)

      CALL section_vals_val_get(sr_section, 'SCALE_X', r_val=sx)
      CALL section_vals_val_get(sr_section, 'SCALE_C', r_val=sc)
      CALL section_vals_val_get(sr_section, 'OMEGA', r_val=omega)
      CALL section_vals_val_get(sr_section, 'PARAMETRIZATION', i_val=method)

      NULLIFY (bo, rho, e_0, e_rho, e_rho_rho, dummy)
      CPASSERT(ASSOCIATED(rho_set))
      CPASSERT(rho_set%ref_count > 0)
      CPASSERT(ASSOCIATED(deriv_set))
      CPASSERT(deriv_set%ref_count > 0)
      CALL xc_rho_set_get(rho_set, rho=rho, &
                          local_bounds=bo, rho_cutoff=rho_cutoff)

      CALL perdew_wang_init(method, rho_cutoff)

      npoints = (bo(2, 1) - bo(1, 1) + 1)*(bo(2, 2) - bo(1, 2) + 1)*(bo(2, 3) - bo(1, 3) + 1)

      dummy => rho

      e_0 => dummy
      e_rho => dummy
      e_rho_rho => dummy

      IF (order >= 0) THEN
         deriv => xc_dset_get_derivative(deriv_set, "", &
                                         allocate_deriv=.TRUE.)
         CALL xc_derivative_get(deriv, deriv_data=e_0)
      END IF
      IF (order >= 1 .OR. order == -1) THEN
         deriv => xc_dset_get_derivative(deriv_set, "(rho)", &
                                         allocate_deriv=.TRUE.)
         CALL xc_derivative_get(deriv, deriv_data=e_rho)
      END IF
      IF (order >= 2 .OR. order == -2) THEN
         deriv => xc_dset_get_derivative(deriv_set, "(rho)(rho)", &
                                         allocate_deriv=.TRUE.)
         CALL xc_derivative_get(deriv, deriv_data=e_rho_rho)
      END IF
      IF (order > 2 .OR. order < -2) THEN
         CPABORT("derivatives bigger than 2 not implemented")
      END IF

      CALL sr_lda_calc(rho, omega, rho_cutoff, e_0, e_rho, e_rho_rho, &
                       npoints, order, sx, sc)

      CALL timestop(handle)

   END SUBROUTINE sr_lda_eval

! **************************************************************************************************
!> \brief ...
!> \param rho ...
!> \param omega ...
!> \param rho_cutoff ...
!> \param e_0 ...
!> \param e_rho ...
!> \param npoints ...
!> \param order ...
!> \param sx ...
!> \param sc ...
! **************************************************************************************************
   SUBROUTINE sr_lda_calc(rho, omega, rho_cutoff, e_0, e_rho, e_rho_rho, npoints, order, sx, sc)
      !FM low level calc routine
      REAL(KIND=dp), DIMENSION(*), INTENT(IN)            :: rho
      REAL(KIND=dp), INTENT(IN)                          :: omega, rho_cutoff
      REAL(KIND=dp), DIMENSION(*), INTENT(INOUT)         :: e_0, e_rho, e_rho_rho
      INTEGER, INTENT(IN)                                :: npoints, order
      REAL(KIND=dp), INTENT(IN)                          :: sx, sc

      CHARACTER(len=*), PARAMETER :: routineN = 'sr_lda_calc', routineP = moduleN//':'//routineN

      INTEGER                                            :: handle, k, order_
      REAL(KIND=dp)                                      :: my_rho, rs, e, e_r, e_r_r

      CALL timeset(routineN, handle)

      order_ = ABS(order)

      IF (sc /= 0.0_dp .OR. sx /= 0.0_dp) THEN
!$OMP PARALLEL DO PRIVATE (k, e, e_r, e_r_r, my_rho, rs) DEFAULT(NONE)&
!$OMP SHARED(npoints,rho,rho_cutoff,omega,e_0,e_rho,order,sc,sx)
         DO k = 1, npoints

            my_rho = rho(k)
            IF (rho(k) > rho_cutoff) THEN
               rs = (3.0_dp/4.0_dp/pi/my_rho)**(1.0_dp/3.0_dp)

               IF (sc /= 0.0_dp) THEN
                  CALL ldasr(rs, omega, e, e_r)

                  IF (order >= 0) THEN
                     e_0(k) = e_0(k) + sc*e*my_rho
                  END IF
                  IF (order >= 1 .OR. order == -1) THEN
                     e_rho(k) = e_rho(k) + sc*e_r
                  END IF
               END IF

               IF (sx /= 0.0_dp) THEN
                  CALL calc_sr_lda_ex(rs, omega, order_, e, e_r, e_r_r)

                  IF (order >= 0) THEN
                     e_0(k) = e_0(k) + sx*e*my_rho
                  END IF
                  IF (order >= 1 .OR. order == -1) THEN
                     e_rho(k) = e_rho(k) + sx*(e_r*my_rho+e)
                  END IF
                  IF (order >= 2 .OR. order == -2) THEN
                     e_rho_rho(k) = e_rho_rho(k) + sx*(e_r_r*my_rho+e_r)
                  END IF
               END IF
            END IF

         END DO
!$OMP END PARALLEL DO
      END IF

      CALL timestop(handle)

   END SUBROUTINE sr_lda_calc

! **************************************************************************************************
!> \brief Calculate the correlation energy and its derivatives
!>      wrt to rho (the electron density) up to 3rd order. This
!>      is the short-range LSD version of the Perdew-Wang correlation energy
!>      If no order argument is given, then the routine calculates
!>      just the energy.
!> \param rho_set ...
!> \param deriv_set ...
!> \param order order of derivatives to calculate
!>      order must lie between -3 and 3. If it is negative then only
!>      that order will be calculated, otherwise all derivatives up to
!>      that order will be calculated.
!> \param sr_section ...
! **************************************************************************************************
   SUBROUTINE sr_lsd_eval(rho_set, deriv_set, order, sr_section)

      TYPE(xc_rho_set_type), POINTER                     :: rho_set
      TYPE(xc_derivative_set_type), POINTER              :: deriv_set
      INTEGER, INTENT(IN), OPTIONAL                      :: order
      TYPE(section_vals_type), POINTER                   :: sr_section

      CHARACTER(len=*), PARAMETER :: routineN = 'sr_lsd_eval', routineP = moduleN//':'//routineN

      INTEGER                                            :: npoints, handle, method
      INTEGER, DIMENSION(:, :), POINTER                  :: bo
      REAL(KIND=dp)                                      :: omega, rho_cutoff, sc, sx
      REAL(KIND=dp), DIMENSION(:, :, :), POINTER         :: a, b, dummy, e_0, ea, eb, eaa, eab, ebb
      TYPE(xc_derivative_type), POINTER                  :: deriv

      CALL timeset(routineN, handle)

      CALL section_vals_val_get(sr_section, 'SCALE_X', r_val=sx)
      CALL section_vals_val_get(sr_section, 'SCALE_C', r_val=sc)
      CALL section_vals_val_get(sr_section, 'OMEGA', r_val=omega)
      CALL section_vals_val_get(sr_section, 'PARAMETRIZATION', i_val=method)

      NULLIFY (bo, a, b, e_0, ea, eb, eaa, eab, ebb)
      CPASSERT(ASSOCIATED(rho_set))
      CPASSERT(rho_set%ref_count > 0)
      CPASSERT(ASSOCIATED(deriv_set))
      CPASSERT(deriv_set%ref_count > 0)
      CALL xc_rho_set_get(rho_set, rhoa=a, rhob=b, &
                          local_bounds=bo, rho_cutoff=rho_cutoff)

      CALL perdew_wang_init(method, rho_cutoff)

      npoints = (bo(2, 1) - bo(1, 1) + 1)*(bo(2, 2) - bo(1, 2) + 1)*(bo(2, 3) - bo(1, 3) + 1)

      ! meaningful default for the arrays we don't need: let us make compiler
      ! and debugger happy...
      dummy => a

      e_0 => dummy
      ea => dummy
      eb => dummy
      eaa => dummy
      eab => dummy
      ebb => dummy

      IF (order >= 0) THEN
         deriv => xc_dset_get_derivative(deriv_set, "", &
                                         allocate_deriv=.TRUE.)
         CALL xc_derivative_get(deriv, deriv_data=e_0)
      END IF
      IF (order >= 1 .OR. order == -1) THEN
         deriv => xc_dset_get_derivative(deriv_set, "(rhoa)", &
                                         allocate_deriv=.TRUE.)
         CALL xc_derivative_get(deriv, deriv_data=ea)
         deriv => xc_dset_get_derivative(deriv_set, "(rhob)", &
                                         allocate_deriv=.TRUE.)
         CALL xc_derivative_get(deriv, deriv_data=eb)
      END IF
      IF (order >= 2 .OR. order == -2) THEN
         deriv => xc_dset_get_derivative(deriv_set, "(rhoa)(rhoa)", &
                                         allocate_deriv=.TRUE.)
         CALL xc_derivative_get(deriv, deriv_data=eaa)
         deriv => xc_dset_get_derivative(deriv_set, "(rhoa)(rhob)", &
                                         allocate_deriv=.TRUE.)
         CALL xc_derivative_get(deriv, deriv_data=eab)
         deriv => xc_dset_get_derivative(deriv_set, "(rhob)(rhob)", &
                                         allocate_deriv=.TRUE.)
         CALL xc_derivative_get(deriv, deriv_data=ebb)
      END IF
      IF (order > 2 .OR. order < -2) THEN
         CPABORT("derivatives bigger than 2 not implemented")
      END IF

      CALL sr_lsd_calc(a, b, omega, rho_cutoff, e_0, ea, eb, eaa, eab, ebb, npoints, order, sx, sc)

      CALL timestop(handle)

   END SUBROUTINE sr_lsd_eval

! **************************************************************************************************
!> \brief ...
!> \param rhoa ...
!> \param rhob ...
!> \param omega ...
!> \param rho_cutoff ...
!> \param e_0 ...
!> \param ea ...
!> \param eb ...
!> \param npoints ...
!> \param order ...
!> \param sx ...
!> \param sc ...
! **************************************************************************************************
   SUBROUTINE sr_lsd_calc(rhoa, rhob, omega, rho_cutoff, e_0, ea, eb, eaa, eab, ebb, npoints, order, sx, sc)
      !FM low-level computation routine
      REAL(KIND=dp), DIMENSION(*), INTENT(IN)            :: rhoa, rhob
      REAL(KIND=dp), INTENT(IN)                          :: omega, rho_cutoff
      REAL(KIND=dp), DIMENSION(*), INTENT(INOUT)         :: e_0, ea, eb, eaa, eab, ebb
      INTEGER, INTENT(IN)                                :: npoints, order
      REAL(KIND=dp), INTENT(IN)                          :: sx, sc

      CHARACTER(len=*), PARAMETER :: routineN = 'sr_lsd_calc', routineP = moduleN//':'//routineN

      INTEGER                                            :: handle, k, order_
      REAL(KIND=dp)                                      :: my_rhoa, my_rhob, rho, rs, zeta, e, e_ra, e_rb, &
         e_ra_ra, e_ra_rb, e_rb_rb

      MARK_USED(e_ra_rb)

      CALL timeset(routineN, handle)

      order_ = ABS(order)

      IF (sc /= 0.0_dp .OR. sx /= 0.0_dp) THEN
!$OMP PARALLEL DO PRIVATE (k, rho, e, e_ra, e_rb, e_ra_ra, e_ra_rb, e_rb_rb, my_rhoa, my_rhob, rs, zeta) DEFAULT(NONE)&
!$OMP SHARED(npoints,rhoa,rhob,rho_cutoff,omega,order,e_0,ea,eb,sx,sc, order_)
         DO k = 1, npoints
            eab(k) = eab(k) + 0.0_dp

            my_rhoa = rhoa(k)
            my_rhob = rhob(k)
            rho = my_rhoa + my_rhob
            IF (rho > rho_cutoff .AND. sc /= 0.0_dp) THEN
               rs = (3.0_dp/4.0_dp/pi/rho)**(1.0_dp/3.0_dp)
               zeta = (my_rhoa - my_rhob)/rho

               CALL lsdsr(rs, zeta, omega, e, e_ra, e_rb)
               IF (order >= 0) THEN
                  e_0(k) = e_0(k) + sc*e*rho
               END IF
               IF (order >= 1 .OR. order == -1) THEN
                  ea(k) = ea(k) + sc*e_ra
                  eb(k) = eb(k) + sc*e_rb
               END IF
            END IF
            IF (2.0_dp*my_rhoa > rho_cutoff .AND. sx /= 0.0_dp) THEN
               rs = (3.0_dp/8.0_dp/pi/my_rhoa)**(1.0_dp/3.0_dp)

               CALL calc_sr_lda_ex(rs, omega, order_, e, e_ra, e_ra_ra)
               IF (order >= 0) THEN
                  e_0(k) = e_0(k) + sx*e*my_rhoa
               END IF
               IF (order >= 1 .OR. order == -1) THEN
                  ea(k) = ea(k) + sx*(2.0_dp*e_ra*my_rhoa+e)
               END IF
               IF (order >= 2 .OR. order == -2) THEN
                  eaa(k) = eaa(k) + sx*(4.0_dp*e_ra_ra*my_rhoa+2.0_dp*e_ra)
               END IF
            END IF
            IF (2.0_dp*my_rhob > rho_cutoff .AND. sx /= 0.0_dp) THEN
               rs = (3.0_dp/8.0_dp/pi/my_rhob)**(1.0_dp/3.0_dp)

               CALL calc_sr_lda_ex(rs, omega, order_, e, e_rb, e_rb_rb)
               IF (order >= 0) THEN
                  e_0(k) = e_0(k) + sx*e*my_rhob
               END IF
               IF (order >= 1 .OR. order == -1) THEN
                  eb(k) = eb(k) + sx*(2.0_dp*e_rb*my_rhob+e)
               END IF
               IF (order >= 2 .OR. order == -2) THEN
                  ebb(k) = ebb(k) + sx*(4.0_dp*e_rb_rb*my_rhob+2.0_dp*e_rb)
               END IF
            END IF
         END DO
!$OMP END PARALLEL DO

         CALL timestop(handle)

      END IF

   END SUBROUTINE sr_lsd_calc

@:ldasr()

END MODULE xc_sr_lda
