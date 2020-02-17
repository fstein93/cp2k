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
#:include "xc_perdew_wang.fypp"

   USE bibliography, ONLY: Paziani2006, &
                           Toulouse2004, &
                           cite_reference
   USE input_section_types, ONLY: section_vals_type, &
                                  section_vals_val_get
   USE kinds, ONLY: dp
   USE mathconstants, ONLY: pi
   USE xc_input_constants, ONLY: pw_dmc, &
                                 pw_orig, &
                                 pw_vmc
   USE xc_derivative_set_types, ONLY: xc_derivative_set_type, &
                                      xc_dset_get_derivative
   USE xc_derivative_types, ONLY: xc_derivative_get, &
                                  xc_derivative_type
   USE xc_rho_cflags_types, ONLY: xc_rho_cflags_type
   USE xc_rho_set_types, ONLY: xc_rho_set_get, &
                               xc_rho_set_type
   USE xc_functionals_utilities, ONLY: set_util, calc_fx, calc_z, calc_rs
#include "../base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'xc_sr_lda'

@:global_var_pw92()
   REAL(KIND=dp), PARAMETER, PRIVATE :: &
      Acoul = 2._dp*(LOG(2._dp) - 1._dp)/pi**2, &
      aQ2 = 5.84605_dp, &
      cQ2 = 3.91744_dp, &
      dQ2 = 3.44851_dp, &
      bQ2 = dQ2 - 3._dp/(2._dp*pi*Acoul)*(4._dp/(9._dp*pi))**(1._dp/3._dp), &
      f02 = 4._dp/(9._dp*(2._dp**(1._dp/3._dp) - 1._dp)), &
      cf = (9._dp*pi/4._dp)**(1._dp/3._dp), &
      p2p = 0.04_dp, &
      p3p = 0.4319_dp, &
      Cg0 = 0.0819306_dp, &
      Fg0 = 0.752411_dp, &
      Dg0 = -0.0127713_dp, &
      Eg0 = 0.00185898_dp, &
      Bg0 = 0.7317_dp - Fg0, &
      adib = 0.784949_dp, &
      q1a = -0.388_dp, &
      q2a = 0.676_dp, &
      q3a = 0.547_dp, &
      t1a = -4.95_dp, &
      t2a = 1._dp, &
      t3a = 0.31_dp, &
      d3_1 = 0.31_dp, &
      d3_2 = 4.95_dp, &
      g2_1 = 0.02267_dp, &
      g2_2 = 0.4319_dp, &
      g2_3 = 0.04_dp, &
      g0_1 = 0.0207_dp, &
      g0_2 = 0.08193_dp, &
      g0_3 = 0.01277_dp, &
      g0_4 = 0.001859_dp, &
      g0_5 = 0.7524_dp, &
      alpha = (4.0_dp/(9.0_dp*pi))**(1.0_dp/3.0_dp), &
      aa = 5.84605_dp, &
      cc = 3.91744_dp, &
      dd = 3.44851_dp, &
      qq = (2.0_dp*LOG(2.0_dp) - 2.0_dp)/pi**2, &
      bb = dd - 3.0_dp*pi*alpha/(4.0_dp*LOG(2.0_dp) - 4.0_dp), &
      beta = 0.784949_dp, &
      c3_1 = 1.0_dp/SQRT(2.0_dp*pi), &
      d2_1 = 0.547_dp, &
      d2_2 = 0.388_dp, &
      d2_3 = 0.676_dp, &
      facx_A = (18.0_dp*pi)**(-1.0_dp/3.0_dp), &
      facx = (18.0_dp/pi**2)**(1.0_dp/3.0_dp)

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
      REAL(KIND=dp), DIMENSION(:, :, :), POINTER         :: dummy, e_0, e_rho, e_rho_rho, rho
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
!$OMP PARALLEL DO PRIVATE (k, my_rho, rs, e, e_r, e_r_r) DEFAULT(NONE)&
!$OMP SHARED(npoints,rho,rho_cutoff,omega,e_0,e_rho,order,sc,sx,order_)
         DO k = 1, npoints

            my_rho = rho(k)
            IF (my_rho > rho_cutoff) THEN
               rs = (3.0_dp/4.0_dp/pi/my_rho)**(1.0_dp/3.0_dp)

               IF (sc /= 0.0_dp) THEN
                  CALL calc_sr_lda_ec(rs, omega, order_, e, e_r, e_r_r)

                  IF (order >= 0) THEN
                     e_0(k) = e_0(k) + sc*e*my_rho
                  END IF
                  IF (order >= 1 .OR. order == -1) THEN
                     e_rho(k) = e_rho(k) + sc*(e_r*my_rho + e)
                  END IF
                  IF (order >= 2 .OR. order == -2) THEN
                     e_rho_rho(k) = e_rho_rho(k) + sc*(e_r_r*my_rho + 2.0_dp*e_r)
                  END IF
               END IF

               IF (sx /= 0.0_dp) THEN
                  CALL calc_sr_lda_ex(rs, omega, order_, e, e_r, e_r_r)

                  IF (order >= 0) THEN
                     e_0(k) = e_0(k) + sx*e*my_rho
                  END IF
                  IF (order >= 1 .OR. order == -1) THEN
                     e_rho(k) = e_rho(k) + sx*(e_r*my_rho + e)
                  END IF
                  IF (order >= 2 .OR. order == -2) THEN
                     e_rho_rho(k) = e_rho_rho(k) + sx*(e_r_r*my_rho + 2.0_dp*e_r)
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

      NULLIFY (bo, a, b, e_0, ea, eb)
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
!> \param eaa ...
!> \param eab ...
!> \param ebb ...
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

      CALL timeset(routineN, handle)

      order_ = ABS(order)

      IF (sc /= 0.0_dp .OR. sx /= 0.0_dp) THEN
!$OMP PARALLEL DO PRIVATE (k, rho, my_rhoa, my_rhob, e, e_ra, e_rb, e_ra_ra, e_ra_rb, e_rb_rb, rs, zeta) DEFAULT(NONE)&
!$OMP SHARED(npoints,rhoa,rhob,rho_cutoff,omega,order,e_0,ea,eb,sx,sc,order_)
         DO k = 1, npoints

            my_rhoa = rhoa(k)
            my_rhob = rhob(k)
            rho = my_rhoa + my_rhob

            IF (rho > rho_cutoff .AND. sc /= 0.0_dp) THEN
               rs = (3.0_dp/4.0_dp/pi/rho)**(1.0_dp/3.0_dp)
               zeta = (my_rhoa - my_rhob)/rho
               CALL calc_sr_lsd_ec(rs, zeta, omega, order_, e, e_ra, e_rb, e_ra_ra, e_ra_rb, e_rb_rb)
               IF (order >= 0) THEN
                  e_0(k) = e_0(k) + sc*e*rho
               END IF
               IF (order >= 1 .OR. order == -1) THEN
                  ea(k) = ea(k) + sc*(e_ra*rho + e)
                  eb(k) = eb(k) + sc*(e_rb*rho + e)
               END IF
               IF (order >= 2 .OR. order == -2) THEN
                  eaa(k) = eaa(k) + sc*(e_ra_ra*rho + 2.0_dp*e_ra)
                  eab(k) = eab(k) + sc*(e_ra_rb*rho + e_ra + e_rb)
                  ebb(k) = ebb(k) + sc*(e_rb_rb*rho + 2.0_dp*e_rb)
               END IF
            END IF
            IF (2.0_dp*my_rhoa > rho_cutoff .AND. sx /= 0.0_dp) THEN
               rs = (3.0_dp/8.0_dp/pi/my_rhoa)**(1.0_dp/3.0_dp)
               CALL calc_sr_lda_ex(rs, omega, order_, e, e_ra, e_ra_ra)
               IF (order >= 0) THEN
                  e_0(k) = e_0(k) + sx*e*my_rhoa
               END IF
               IF (order >= 1 .OR. order == -1) THEN
                  ea(k) = ea(k) + sx*(2.0_dp*e_ra*my_rhoa + e)
               END IF
               IF (order >= 2 .OR. order == -2) THEN
                  eaa(k) = eaa(k) + sx*(4.0_dp*e_ra_ra*my_rhoa + 2.0_dp*e_ra)
               END IF
            END IF
            IF (2.0_dp*my_rhob > rho_cutoff .AND. sx /= 0.0_dp) THEN
               rs = (3.0_dp/8.0_dp/pi/my_rhob)**(1.0_dp/3.0_dp)
               CALL calc_sr_lda_ex(rs, omega, order_, e, e_rb, e_rb_rb)
               IF (order >= 0) THEN
                  e_0(k) = e_0(k) + sx*e*my_rhob
               END IF
               IF (order >= 1 .OR. order == -1) THEN
                  eb(k) = eb(k) + sx*(2.0_dp*e_rb*my_rhob + e)
               END IF
               IF (order >= 2 .OR. order == -2) THEN
                  ebb(k) = ebb(k) + sx*4.0_dp*(e_rb_rb*my_rhob + e_rb)
               END IF
            END IF
         END DO
!$OMP END PARALLEL DO

         CALL timestop(handle)

      END IF

   END SUBROUTINE sr_lsd_calc

   ELEMENTAL SUBROUTINE calc_D3(rs, max_deriv, D3, D3_rs, D3_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: D3, D3_rs, D3_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_D3", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7

      x0 = -d3_2 + rs
      x1 = rs**(-2)
      x2 = exp(-d3_1*rs)
      x3 = x1*x2
      x4 = x0*x3
      D3 = x4
      IF (max_deriv >= 1) THEN
         x5 = x2/rs**3
         x6 = d3_1*x1*x2
         D3_rs = -x0*x6 + x3 + 2.0_dp*x5*(d3_2 - rs)
      END IF
      IF (max_deriv >= 2) THEN
         x7 = 4.0_dp*x5
         D3_rs_rs = d3_1**2*x4 + (d3_1*x0 - 1.0_dp)*x7 - 2.0_dp*x6 + 6.0_dp*x0*x2/rs**4
      END IF

   END SUBROUTINE calc_D3

   ELEMENTAL SUBROUTINE calc_D2(rs, max_deriv, D2, D2_rs, D2_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: D2, D2_rs, D2_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_D2", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6

      x0 = 1.0_dp/rs
      x1 = exp(-d2_1*rs)
      x2 = x1*(-d2_2 + d2_3*rs)
      x3 = x0*x2
      D2 = x3
      IF (max_deriv >= 1) THEN
         x4 = d2_3*x1
         x5 = x0*x4
         x6 = rs**(-2)
         D2_rs = -d2_1*x3 - (x2*x6 - x5)
      END IF
      IF (max_deriv >= 2) THEN
         D2_rs_rs = d2_1**2*x3 + 2.0_dp*(d2_1*(x2*x6 - x5) - x4*x6 + x2/rs**3)
      END IF

   END SUBROUTINE calc_D2

   ELEMENTAL SUBROUTINE calc_g2(rs, max_deriv, g2, g2_rs, g2_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: g2, g2_rs, g2_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_g2", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18

      x0 = rs**2
      x1 = 1.0_dp/x0
      x2 = (1.0_dp/5.0_dp)*x1
      x3 = alpha**(-2)
      x4 = 2.0_dp**(1.0_dp/3.0_dp)
      x5 = -g2_1*rs*x4 + 1.0_dp
      x6 = g2_2*x4
      x7 = 2.0_dp**(2.0_dp/3.0_dp)
      x8 = g2_3*x0*x7 + rs*x6 + 1.0_dp
      x9 = 1.0_dp/x8
      x10 = x3*x5*x9
      g2 = x10*x2
      IF (max_deriv >= 1) THEN
         x11 = g2_1*x3*x4*x9
         x12 = rs**(-3)
         x13 = g2_3*rs*x7
         x14 = -2.0_dp*x13 - x6
         x15 = x8**(-2)
         x16 = x14*x15*x3*x5
         g2_rs = -2.0_dp/5.0_dp*x10*x12 - x11*x2 + x16*x2
      END IF
      IF (max_deriv >= 2) THEN
         x17 = (4.0_dp/5.0_dp)*x12
         x18 = (2.0_dp/5.0_dp)*x1*x15*x3
         g2_rs_rs = -g2_1*x14*x18*x4 - g2_3*x18*x5*x7 + x11*x17 + x14*x2*x3*x5*( &
                    -4.0_dp*x13 - 2.0_dp*x6)/x8**3 - x16*x17 + (6.0_dp/5.0_dp)*x10/rs**4
      END IF

   END SUBROUTINE calc_g2

   ELEMENTAL SUBROUTINE calc_g2z_ab(rs, zeta, max_deriv, g2z_ab, g2z_ab_rs, g2z_ab_zeta, &
                                    g2z_ab_rs_rs, g2z_ab_rs_zeta, g2z_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: g2z_ab, g2z_ab_rs, g2z_ab_zeta, g2z_ab_rs_rs, g2z_ab_rs_zeta, g2z_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_g2z_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, &
                       x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, &
                       x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, &
                       x56, x57, x58, x59, x60, x61, x62, x63, x64, x65, x66, x67, x68, x69, x70, x71, x72, x73, x74, &
                       x75, x76, x77, x78, x79, x80, x81, x82, x83, x84, x85

      IF (ABS(zeta) >= 1.0_dp) THEN
         x0 = 2.0_dp**(2.0_dp/3.0_dp)
         x1 = alpha**(-2)
         x2 = rs**2
         x3 = 1.0_dp/x2
         x4 = -g2_1*rs + 1.0_dp
         x5 = g2_2*rs
         x6 = g2_3*x2
         x7 = x5 + x6 + 1.0_dp
         x8 = 1.0_dp/x7
         x9 = x0*x1*x3*x4*x8
         g2z_ab = (2.0_dp/5.0_dp)*x9
         IF (max_deriv >= 1) THEN
            x10 = g2_1*x0*x1*x3*x8
            x11 = rs**(-3)
            x12 = x0*x1*x11*x4*x8
            x13 = g2_3*rs
            x14 = -g2_2 - 2.0_dp*x13
            x15 = x7**(-2)
            x16 = (2.0_dp/5.0_dp)*x0*x1*x15*x3
            x17 = 1.0_dp/rs
            x18 = g2_1*x0*x1*x17*x8
            x19 = (1.0_dp/6.0_dp)*g2_2
            x20 = rs*x19 + (1.0_dp/3.0_dp)*x6
            x21 = (2.0_dp/5.0_dp)*x0*x1*x15*x3*x4
            g2z_ab_rs = -2.0_dp/5.0_dp*x10 - 4.0_dp/5.0_dp*x12 + x14*x16*x4
            g2z_ab_zeta = (1.0_dp/15.0_dp)*x18 + x20*x21 + (8.0_dp/15.0_dp)*x9
            IF (zeta < 0.0_dp) g2z_ab_zeta = -g2z_ab_zeta
         END IF
         IF (max_deriv >= 2) THEN
            x22 = (8.0_dp/5.0_dp)*x11
            x23 = (4.0_dp/5.0_dp)*x0*x1*x15*x3
            x24 = x0*x1*x14*x15*x4
            x25 = x7**(-3)
            x26 = (2.0_dp/5.0_dp)*x0*x1*x14*x25*x3*x4
            x27 = g2_1*x20
            x28 = x0*x1*x15*x20*x4
            x29 = (1.0_dp/3.0_dp)*x5 + (2.0_dp/3.0_dp)*x6
            g2z_ab_rs_rs = g2_1*x0*x1*x22*x8 - g2_1*x14*x23 - g2_3*x23*x4 - x22* &
                           x24 + x26*(-2.0_dp*g2_2 - 4.0_dp*x13) + (12.0_dp/5.0_dp)*x0*x1*x4*x8/rs**4
            g2z_ab_rs_zeta = (1.0_dp/15.0_dp)*g2_1*x0*x1*x14*x15*x17 - 3.0_dp/5.0_dp* &
                             x10 - 4.0_dp/5.0_dp*x11*x28 - 16.0_dp/15.0_dp*x12 - x16*x27 + x21*(( &
                                                                       2.0_dp/3.0_dp)*x13 + x19) + (8.0_dp/15.0_dp)*x24*x3 + x26*x29
            IF (zeta < 0.0_dp) g2z_ab_rs_zeta = -g2z_ab_rs_zeta
            g2z_ab_zeta_zeta = (2.0_dp/15.0_dp)*x0*x1*x15*x17*x27 + (2.0_dp/5.0_dp)* &
                               x0*x1*x20*x25*x29*x3*x4 + (2.0_dp/15.0_dp)*x18 + x21*(-1.0_dp/9.0_dp* &
                                                             x5 - 5.0_dp/18.0_dp*x6) + (16.0_dp/15.0_dp)*x28*x3 + (4.0_dp/9.0_dp)*x9
         END IF

      ELSE
         x0 = (1.0_dp/2.0_dp)*zeta
         x1 = x0 + 1.0_dp/2.0_dp
         x2 = x1**2
         x3 = alpha**(-2)
         x4 = rs**2
         x5 = 1.0_dp/x4
         x6 = zeta + 1.0_dp
         x7 = 1.0_dp/x6
         x8 = x7**(2.0_dp/3.0_dp)
         x9 = 1.0_dp/x8
         x10 = 2.0_dp**(1.0_dp/3.0_dp)
         x11 = g2_1*rs*x10
         x12 = x7**(1.0_dp/3.0_dp)
         x13 = -x11*x12 + 1.0_dp
         x14 = g2_2*x10*x12
         x15 = rs*x14
         x16 = 2.0_dp**(2.0_dp/3.0_dp)
         x17 = g2_3*x16*x4
         x18 = x17*x8
         x19 = x15 + x18 + 1.0_dp
         x20 = 1.0_dp/x19
         x21 = (2.0_dp/5.0_dp)*x13*x20*x3*x5*x9
         x22 = (-x0 + 1.0_dp/2.0_dp)**2
         x23 = -zeta + 1.0_dp
         x24 = 1.0_dp/x23
         x25 = x24**(2.0_dp/3.0_dp)
         x26 = 1.0_dp/x25
         x27 = x24**(1.0_dp/3.0_dp)
         x28 = -x11*x27 + 1.0_dp
         x29 = g2_2*x10*x27
         x30 = rs*x29
         x31 = x17*x25
         x32 = x30 + x31 + 1.0_dp
         x33 = 1.0_dp/x32
         x34 = (2.0_dp/5.0_dp)*x26*x28*x3*x33*x5
         g2z_ab = x2*x21 + x22*x34
         IF (max_deriv >= 1) THEN
            x35 = 1.0_dp/x12
            x36 = (2.0_dp/5.0_dp)*g2_1*x10*x20*x3*x35*x5
            x37 = x2*x36
            x38 = 1.0_dp/x27
            x39 = (2.0_dp/5.0_dp)*g2_1*x10*x3*x33*x38*x5
            x40 = x22*x39
            x41 = rs**(-3)
            x42 = x13*x20*x3*x41*x9
            x43 = x26*x28*x3*x33*x41
            x44 = g2_3*rs*x16*x8
            x45 = -x14 - 2.0_dp*x44
            x46 = x19**(-2)
            x47 = (2.0_dp/5.0_dp)*x13*x3*x45*x46*x5*x9
            x48 = g2_3*rs*x16*x25
            x49 = -x29 - 2.0_dp*x48
            x50 = x32**(-2)
            x51 = (2.0_dp/5.0_dp)*x26*x28*x3*x49*x5*x50
            x52 = 1.0_dp/rs
            x53 = g2_1*x10*x2*x20*x3*x35
            x54 = g2_1*x10*x22*x3*x33*x38
            x55 = x13*x2*x20*x3*x5*x9
            x56 = x0 - 1.0_dp/2.0_dp
            x57 = x22*x26*x28*x3*x33*x5
            x58 = (1.0_dp/3.0_dp)*x7
            x59 = (2.0_dp/3.0_dp)*x7
            x60 = x15*x58 + x18*x59
            x61 = (2.0_dp/5.0_dp)*x13*x2*x3*x46*x5*x9
            x62 = (1.0_dp/3.0_dp)*x24
            x63 = (2.0_dp/3.0_dp)*x24
            x64 = -x30*x62 - x31*x63
            x65 = (2.0_dp/5.0_dp)*x22*x26*x28*x3*x5*x50
            g2z_ab_rs = -4.0_dp/5.0_dp*x2*x42 + x2*x47 - 4.0_dp/5.0_dp*x22*x43 + x22*x51 &
                        - x37 - x40
            g2z_ab_zeta = x1*x21 - 2.0_dp/15.0_dp*x24*x52*x54 - 4.0_dp/15.0_dp*x24*x57 + &
                          x34*x56 + (2.0_dp/15.0_dp)*x52*x53*x7 + (4.0_dp/15.0_dp)*x55*x7 + x60 &
                          *x61 + x64*x65
         END IF
         IF (max_deriv >= 2) THEN
            x66 = (8.0_dp/5.0_dp)*x41
            x67 = rs**(-4)
            x68 = x19**(-3)
            x69 = (2.0_dp/5.0_dp)*x13*x2*x3*x45*x5*x68*x9
            x70 = x32**(-3)
            x71 = (2.0_dp/5.0_dp)*x22*x26*x28*x3*x49*x5*x70
            x72 = (2.0_dp/5.0_dp)*zeta
            x73 = x72 + 2.0_dp/5.0_dp
            x74 = (8.0_dp/15.0_dp)*x2*x7
            x75 = x72 - 2.0_dp/5.0_dp
            x76 = (8.0_dp/15.0_dp)*x22*x24
            x77 = (4.0_dp/3.0_dp)*x7
            x78 = x3*x41
            x79 = (4.0_dp/3.0_dp)*x24
            x80 = x15*x59 + x18*x77
            x81 = -x30*x63 - x31*x79
            x82 = x6**(-2)
            x83 = x23**(-2)
            x84 = x13*x3*x46*x5*x60*x9
            x85 = x26*x28*x3*x5*x50*x64
            g2z_ab_rs_rs = -4.0_dp/5.0_dp*g2_1*x10*x2*x3*x35*x45*x46*x5 - 4.0_dp/5.0_dp* &
                           g2_1*x10*x22*x3*x38*x49*x5*x50 - 4.0_dp/5.0_dp*g2_3*x13*x16*x2*x3* &
                           x46*x5 - 4.0_dp/5.0_dp*g2_3*x16*x22*x28*x3*x5*x50 + (12.0_dp/5.0_dp)* &
                           x13*x2*x20*x3*x67*x9 - 8.0_dp/5.0_dp*x13*x2*x3*x41*x45*x46*x9 + ( &
                           12.0_dp/5.0_dp)*x22*x26*x28*x3*x33*x67 - 8.0_dp/5.0_dp*x22*x26*x28*x3 &
                           *x41*x49*x50 + x53*x66 + x54*x66 + x69*(-2*x14 - 4*x44) + x71*(-2.0_dp*x29 - 4.0_dp*x48)
            g2z_ab_rs_zeta = (2.0_dp/15.0_dp)*g2_1*x10*x2*x3*x35*x45*x46*x52*x7 - &
                             2.0_dp/5.0_dp*g2_1*x10*x2*x3*x35*x46*x5*x60 - 2.0_dp/15.0_dp*g2_1*x10 &
                             *x22*x24*x3*x38*x49*x50*x52 - 2.0_dp/5.0_dp*g2_1*x10*x22*x3*x38*x5* &
                             x50*x64 - x1*x36 + x1*x47 + (4.0_dp/15.0_dp)*x13*x2*x3*x45*x46*x5* &
                             x7*x9 - 4.0_dp/5.0_dp*x13*x2*x46*x60*x78*x9 - 4.0_dp/15.0_dp*x22*x24* &
                             x26*x28*x3*x49*x5*x50 - 4.0_dp/5.0_dp*x22*x26*x28*x50*x64*x78 + x24 &
                             *x40 - x37*x7 - x39*x56 - x42*x73 - x42*x74 - x43*x75 + x43*x76 + &
                             x51*x56 + x61*(x14*x58 + x44*x77) + x65*(-x29*x62 - x48*x79) + &
                             x69*x80 + x71*x81
            g2z_ab_zeta_zeta = (4.0_dp/15.0_dp)*g2_1*x1*x10*x20*x3*x35*x52*x7 + (4.0_dp &
                                                                /15.0_dp)*g2_1*x10*x2*x3*x35*x46*x52*x60*x7 - 4.0_dp/15.0_dp*g2_1* &
                               x10*x22*x24*x3*x38*x50*x52*x64 - 4.0_dp/15.0_dp*g2_1*x10*x24*x3*x33 &
                               *x38*x52*x56 + (8.0_dp/15.0_dp)*x1*x13*x20*x3*x5*x7*x9 + (2.0_dp/ &
                                                              5.0_dp)*x13*x2*x3*x5*x60*x68*x80*x9 + (1.0_dp/5.0_dp)*x13*x20*x3*x5* &
                               x9 + (2.0_dp/5.0_dp)*x22*x26*x28*x3*x5*x64*x70*x81 - 8.0_dp/15.0_dp* &
                               x24*x26*x28*x3*x33*x5*x56 + (1.0_dp/5.0_dp)*x26*x28*x3*x33*x5 - &
                               4.0_dp/45.0_dp*x55*x82 - 4.0_dp/45.0_dp*x57*x83 + x61*(-4.0_dp/9.0_dp* &
                                                                 x15*x82 - 10.0_dp/9.0_dp*x18*x82) + x65*(-4.0_dp/9.0_dp*x30*x83 - &
                                                                     10.0_dp/9.0_dp*x31*x83) + x73*x84 + x74*x84 + x75*x85 - x76*x85
         END IF
      END IF

   END SUBROUTINE calc_g2z_ab

   ELEMENTAL SUBROUTINE calc_g0(rs, max_deriv, g0, g0_rs, g0_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: g0, g0_rs, g0_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_g0", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5

      x0 = exp(-g0_5*rs)
      x1 = (1.0_dp/2.0_dp)*g0_1
      x2 = rs**2
      x3 = rs**3
      x4 = x0*((1.0_dp/2.0_dp)*g0_2*x2 - 1.0_dp/2.0_dp*g0_3*x3 + (1.0_dp/2.0_dp)* &
               g0_4*rs**4 + rs*x1 + 1.0_dp/2.0_dp)
      g0 = x4
      IF (max_deriv >= 1) THEN
         x5 = x0*(g0_2*rs - 3.0_dp/2.0_dp*g0_3*x2 + 2.0_dp*g0_4*x3 + x1)
         g0_rs = -g0_5*x4 + x5
      END IF
      IF (max_deriv >= 2) THEN
         g0_rs_rs = g0_5**2*x4 - 2.0_dp*g0_5*x5 + x0*(g0_2 - 3.0_dp*g0_3*rs + 6.0_dp*g0_4*x2)
      END IF

   END SUBROUTINE calc_g0

   ELEMENTAL SUBROUTINE calc_C2(rs, max_deriv, C2, C2_rs, C2_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: C2, C2_rs, C2_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C2", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, g0, g0_rs, g0_rs_rs

      CALL calc_g0(rs, max_deriv, g0, g0_rs, g0_rs_rs)

      x0 = rs**(-3)
      C2 = x0*(-3.0_dp/8.0_dp*g0 + 3.0_dp/16.0_dp)
      IF (max_deriv >= 1) THEN
         x1 = (3.0_dp/8.0_dp)*x0
         x2 = rs**(-4)
         C2_rs = -g0_rs*x1 + x2*((9.0_dp/8.0_dp)*g0 - 9.0_dp/16.0_dp)
      END IF
      IF (max_deriv >= 2) THEN
         C2_rs_rs = (9.0_dp/4.0_dp)*g0_rs*x2 - g0_rs_rs*x1 + (-9.0_dp/2.0_dp*g0 + &
                                                              9.0_dp/4.0_dp)/rs**5
      END IF

   END SUBROUTINE calc_C2

   ELEMENTAL SUBROUTINE calc_C3(rs, max_deriv, C3, C3_rs, C3_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: C3, C3_rs, C3_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C3", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, g0, g0_rs, g0_rs_rs

      CALL calc_g0(rs, max_deriv, g0, g0_rs, g0_rs_rs)

      x0 = c3_1/rs**3
      C3 = -g0*x0
      IF (max_deriv >= 1) THEN
         x1 = c3_1*g0
         x2 = rs**(-4)
         C3_rs = -g0_rs*x0 + 3.0_dp*x1*x2
      END IF
      IF (max_deriv >= 2) THEN
         C3_rs_rs = 6.0_dp*c3_1*g0_rs*x2 - g0_rs_rs*x0 - 12.0_dp*x1/rs**5
      END IF

   END SUBROUTINE calc_C3

   ELEMENTAL SUBROUTINE calc_C4(rs, max_deriv, C4, C4_rs, C4_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: C4, C4_rs, C4_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C4", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, g2, g2_rs, g2_rs_rs, D2, D2_rs, D2_rs_rs

      CALL calc_g2(rs, max_deriv, g2, g2_rs, g2_rs_rs)
      CALL calc_D2(rs, max_deriv, D2, D2_rs, D2_rs_rs)

      x0 = rs**(-3)
      x1 = alpha**(-2)
      x2 = x1/rs**2
      x3 = rs**(-4)
      x4 = x0*x1
      C4 = x0*(-9.0_dp/64.0_dp*D2 - 9.0_dp/64.0_dp*g2 + (9.0_dp/320.0_dp)*x2)
      IF (max_deriv >= 1) THEN
         C4_rs = x0*(-9.0_dp/64.0_dp*D2_rs - 9.0_dp/64.0_dp*g2_rs - 9.0_dp/160.0_dp*x4) &
                 + x3*((27.0_dp/64.0_dp)*D2 + (27.0_dp/64.0_dp)*g2 - 27.0_dp/320.0_dp*x2)
      END IF
      IF (max_deriv >= 2) THEN
         C4_rs_rs = x0*(-9.0_dp/64.0_dp*D2_rs_rs - 9.0_dp/64.0_dp*g2_rs_rs + (27.0_dp/160.0_dp)*x1*x3) &
                    + x3*((27.0_dp/32.0_dp)*D2_rs + (27.0_dp/32.0_dp)*g2_rs + (27.0_dp/80.0_dp)*x4) &
                    + (-27.0_dp/16.0_dp*D2 - 27.0_dp/16.0_dp*g2 + (27.0_dp/80.0_dp)*x2)/rs**5
      END IF

   END SUBROUTINE calc_C4

   ELEMENTAL SUBROUTINE calc_C5(rs, max_deriv, C5, C5_rs, C5_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: C5, C5_rs, C5_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C5", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, g2, g2_rs, g2_rs_rs, D3, D3_rs, D3_rs_rs

      CALL calc_g2(rs, max_deriv, g2, g2_rs, g2_rs_rs)
      CALL calc_D3(rs, max_deriv, D3, D3_rs, D3_rs_rs)

      x0 = sqrt(2.0_dp)
      x1 = 1.0_dp/sqrt(pi)
      x2 = x0*x1/rs**3
      C5 = -9.0_dp/80.0_dp*x2*(D3 + g2)
      IF (max_deriv >= 1) THEN
         x3 = x0*x1/rs**4
         C5_rs = -90.0_dp/80.0_dp*x2*(D3_rs + g2_rs) + (27.0_dp/80.0_dp)*x3*(D3 + g2)
      END IF
      IF (max_deriv >= 2) THEN
         C5_rs_rs = -9.0_dp/80.0_dp*x2*(D3_rs_rs + g2_rs_rs) + (27.0_dp/40.0_dp)*x3*( &
                    D3_rs + g2_rs) - 27.0_dp/20.0_dp*x0*x1*(D3 + g2)/rs**5
      END IF

   END SUBROUTINE calc_C5

   ELEMENTAL SUBROUTINE calc_a1(rs, max_deriv, a1, a1_rs, a1_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a1, a1_rs, a1_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a1", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, C3, C3_rs, C3_rs_rs, C5, C5_rs, C5_rs_rs, b0

      CALL calc_C3(rs, max_deriv, C3, C3_rs, C3_rs_rs)
      CALL calc_C5(rs, max_deriv, C5, C5_rs, C5_rs_rs)

      b0 = beta*rs

      x0 = b0**8
      x1 = C5*x0
      x2 = b0**6
      x3 = 4.0_dp*x2
      a1 = C3*x3 + x1
      IF (max_deriv >= 1) THEN
         x4 = C5_rs*x0
         x5 = 1.0_dp/rs
         x6 = C3*x2
         a1_rs = C3_rs*x3 + 8.0_dp*x1*x5 + x4 + 24.0_dp*x5*x6
      END IF
      IF (max_deriv >= 2) THEN
         x7 = rs**(-2)
         a1_rs_rs = 48.0_dp*C3_rs*x2*x5 + C3_rs_rs*x3 + C5_rs_rs*x0 + 56.0_dp*x1*x7 + 16.0_dp*x4 &
                    *x5 + 120.0_dp*x6*x7
      END IF

   END SUBROUTINE calc_a1

   ELEMENTAL SUBROUTINE calc_a2(rs, max_deriv, a2, a2_rs, a2_rs_rs, Ec, Ec_rs, Ec_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs, Ec, Ec_rs, Ec_rs_rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a2, a2_rs, a2_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a2", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, C2, C2_rs, &
                       C2_rs_rs, C4, C4_rs, C4_rs_rs, b0

      CALL calc_C2(rs, max_deriv, C2, C2_rs, C2_rs_rs)
      CALL calc_C4(rs, max_deriv, C4, C4_rs, C4_rs_rs)

      b0 = beta*rs

      x0 = b0**8
      x1 = C4*x0
      x2 = b0**6
      x3 = 4.0_dp*x2
      x4 = b0**4
      x5 = 6.0_dp*x4
      a2 = C2*x3 + Ec*x5 + x1
      IF (max_deriv >= 1) THEN
         x6 = C4_rs*x0
         x7 = 1.0_dp/rs
         x8 = 24.0_dp*x7
         x9 = C2*x2
         x10 = Ec*x4
         a2_rs = C2_rs*x3 + Ec_rs*x5 + 8.0_dp*x1*x7 + x10*x8 + x6 + x8*x9
      END IF
      IF (max_deriv >= 2) THEN
         x11 = rs**(-2)
         x12 = 48.0_dp*x7
         a2_rs_rs = C2_rs*x12*x2 + C2_rs_rs*x3 + C4_rs_rs*x0 + Ec_rs*x12*x4 + &
                    Ec_rs_rs*x5 + 56.0_dp*x1*x11 + 72.0_dp*x10*x11 + 120.0_dp*x11*x9 + 16.0_dp*x6*x7
      END IF

   END SUBROUTINE calc_a2

   ELEMENTAL SUBROUTINE calc_a3(rs, max_deriv, a3, a3_rs, a3_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a3, a3_rs, a3_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a3", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, C3, C3_rs, C3_rs_rs, b0

      CALL calc_C3(rs, max_deriv, C3, C3_rs, C3_rs_rs)

      b0 = beta*rs

      x0 = b0**8
      x1 = C3*x0
      a3 = x1
      IF (max_deriv >= 1) THEN
         x2 = C3_rs*x0
         x3 = 1.0_dp/rs
         a3_rs = x2 + 8.0_dp*x1*x3
      END IF
      IF (max_deriv >= 2) THEN
         a3_rs_rs = 16.0_dp*x2*x3 + C3_rs_rs*x0 + 56.0_dp*x1/rs**2
      END IF

   END SUBROUTINE calc_a3

   ELEMENTAL SUBROUTINE calc_a4(rs, max_deriv, a4, a4_rs, a4_rs_rs, Ec, Ec_rs, Ec_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs, Ec, Ec_rs, Ec_rs_rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a4, a4_rs, a4_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a4", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, C2, C2_rs, C2_rs_rs, b0

      CALL calc_C2(rs, max_deriv, C2, C2_rs, C2_rs_rs)

      b0 = beta*rs

      x0 = b0**8
      x1 = C2*x0
      x2 = b0**6
      x3 = 4.0_dp*x2
      a4 = Ec*x3 + x1
      IF (max_deriv >= 1) THEN
         x4 = C2_rs*x0
         x5 = 1.0_dp/rs
         x6 = Ec*x2
         a4_rs = Ec_rs*x3 + 8.0_dp*x1*x5 + x4 + 24.0_dp*x5*x6
      END IF
      IF (max_deriv >= 2) THEN
         x7 = rs**(-2)
         a4_rs_rs = C2_rs_rs*x0 + 48.0_dp*Ec_rs*x2*x5 + Ec_rs_rs*x3 + 56.0_dp*x1*x7 + 16.0_dp*x4 &
                    *x5 + 120.0_dp*x6*x7
      END IF

   END SUBROUTINE calc_a4

   ELEMENTAL SUBROUTINE calc_a5(rs, max_deriv, a5, a5_rs, a5_rs_rs, Ec, Ec_rs, Ec_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs, Ec, Ec_rs, Ec_rs_rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a5, a5_rs, a5_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a5", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, b0

      b0 = beta*rs

      x0 = b0**8
      x1 = Ec*x0
      a5 = x1
      IF (max_deriv >= 1) THEN
         x2 = Ec_rs*x0
         x3 = 1.0_dp/rs
         a5_rs = 8.0_dp*x1*x3 + x2
      END IF
      IF (max_deriv >= 2) THEN
         a5_rs_rs = Ec_rs_rs*x0 + 16.0_dp*x2*x3 + 56.0_dp*x1/rs**2
      END IF

   END SUBROUTINE calc_a5

   ELEMENTAL SUBROUTINE calc_Q(rs, mu, max_deriv, Q, Q_rs, Q_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: Q, Q_rs, Q_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_Q", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, xx

      xx = mu*sqrt(rs)

      x0 = xx**2
      x1 = aa*xx + 1.0_dp
      x2 = dd*x0 + x1
      x3 = 1.0_dp/x2
      x4 = bb*x0 + cc*xx**3 + x1
      Q = qq*log(x3*x4)
      IF (max_deriv >= 1) THEN
         x5 = 2.0_dp*xx
         x6 = aa + dd*x5
         x7 = x4*x6
         x8 = 3.0_dp*cc*x0
         x9 = aa + bb*x5 + x8
         x10 = x2*x9
         x11 = xx*(-x10 + x7)
         Q_rs = -1.0_dp/2.0_dp*qq*x11*x3/(rs*x4)
      END IF
      IF (max_deriv >= 2) THEN
         x12 = x2**2
         Q_rs_rs = (1.0_dp/4.0_dp)*qq*xx*(x10*x11 - x11*x7 + x4*(x12*(-aa + x8) + &
                                                                 x2*(aa*x4 - x5*x6*x9) + x4*x5*x6**2))/(rs**2*x12*x4**2)
      END IF

   END SUBROUTINE calc_Q

   ELEMENTAL SUBROUTINE calc_sr_lda_ec(rs, mu, max_deriv, ec_mu, ec_mu_rho, ec_mu_rho_rho)
      REAL(KIND=dp), INTENT(IN) :: rs, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: ec_mu, ec_mu_rho, ec_mu_rho_rho

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_ec_mu", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, &
                       x18, Ec, Ec_rs, Ec_rs_rs, Q, Q_rs, Q_rs_rs, a1, a1_rs, a1_rs_rs, a2, a2_rs, a2_rs_rs, a3, a3_rs, &
                       a3_rs_rs, a4, a4_rs, a4_rs_rs, a5, a5_rs, a5_rs_rs, ed(0:2), ec_mu_rs, ec_mu_rs_rs, xx

      CALL calc_g(rs, 0, ed, max_deriv)
      Ec = ed(0)
      Ec_rs = ed(1)
      Ec_rs_rs = ed(2)

      xx = mu*sqrt(rs)

      CALL calc_Q(rs, mu, max_deriv, Q, Q_rs, Q_rs_rs)
      CALL calc_a1(rs, max_deriv, a1, a1_rs, a1_rs_rs)
      CALL calc_a2(rs, max_deriv, a2, a2_rs, a2_rs_rs, Ec, Ec_rs, Ec_rs_rs)
      CALL calc_a3(rs, max_deriv, a3, a3_rs, a3_rs_rs)
      CALL calc_a4(rs, max_deriv, a4, a4_rs, a4_rs_rs, Ec, Ec_rs, Ec_rs_rs)
      CALL calc_a5(rs, max_deriv, a5, a5_rs, a5_rs_rs, Ec, Ec_rs, Ec_rs_rs)

      x0 = rs**2
      x1 = beta**2
      x2 = mu**2
      x3 = x1*x2
      x4 = x0*x3 + 1.0_dp
      x5 = x4**(-4)
      x6 = mu**3
      x7 = a1*x6
      x8 = mu**4
      x9 = a2*x8
      x10 = mu**5
      x11 = a3*x10
      x12 = mu**6
      x13 = a4*x12
      x14 = mu**8
      x15 = a5*x14
      ec_mu = Ec - x5*(Q + mu**3*(a1 + mu*(a2 + mu*(a3 + mu*(a4 + a5*mu**2)))))
      IF (max_deriv >= 1) THEN
         x16 = x4**(-5)
         x17 = rs*x1*x16*x2
         x18 = 8.0_dp*(Q + x11 + x13 + x15 + x7 + x9)
         ec_mu_rs = Ec_rs + x17*x18 - x5*(Q_rs + mu**3*(a1_rs + mu*(a2_rs + mu*(a3_rs + mu*(a4_rs + a5_rs*mu**2)))))
      END IF
      IF (max_deriv >= 2) THEN
         ec_mu_rs_rs = Ec_rs_rs - 80.0_dp*beta**4*x0*x8*(Q + mu**3*(a1 + mu*(a2 + mu*(a3 + mu*(a4 + a5*mu**2)))))/x4**6 &
                      + x16*x18*x3 + 16.0_dp*x17*(Q_rs + mu**3*(a1_rs + mu*(a2_rs + mu*(a3_rs + mu*(a4_rs + a5_rs*mu**2))))) - x5* &
                       (Q_rs_rs + mu**3*(a1_rs_rs + mu*(a2_rs_rs + mu*(a3_rs_rs + mu*(a4_rs_rs + a5_rs_rs*mu**2)))))
      END IF

      CALL calc_rs2rho(rs, max_deriv, ec_mu_rs, ec_mu_rs_rs, ec_mu_rho, ec_mu_rho_rho)

   END SUBROUTINE calc_sr_lda_ec

   ELEMENTAL SUBROUTINE calc_phin(zeta, n, max_deriv, phin, phin_zeta, phin_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: zeta
      INTEGER, INTENT(IN) :: n, max_deriv
      REAL(KIND=dp), INTENT(OUT) :: phin, phin_zeta, phin_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_phin", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, n_

      n_ = REAL(n, KIND=dp)

      IF (ABS(zeta) >= 1.0_dp) THEN
         phin = 1.0_dp
         phin_zeta = 0.0_dp
         phin_zeta_zeta = 0.0_dp
      ELSE
         x0 = zeta + 1.0_dp
         x1 = (1.0_dp/3.0_dp)*n_
         x2 = x0**x1
         x3 = -zeta + 1.0_dp
         x4 = x3**x1
         phin = (1.0_dp/2.0_dp)*x2 + (1.0_dp/2.0_dp)*x4
         IF (max_deriv >= 1) THEN
            x5 = (1.0_dp/6.0_dp)*n_*x2
            x6 = (1.0_dp/6.0_dp)*n_*x4
            phin_zeta = -x6/x3 + x5/x0
         END IF
         IF (max_deriv >= 2) THEN
            x7 = x0**(-2)
            x8 = (1.0_dp/18.0_dp)*n_**2
            x9 = x3**(-2)
            phin_zeta_zeta = x2*x7*x8 + x4*x8*x9 - x5*x7 - x6*x9
         END IF
      END IF

   END SUBROUTINE calc_phin

   ELEMENTAL SUBROUTINE calc_C2_ab(rs, zeta, max_deriv, C2_ab, C2_ab_rs, C2_ab_zeta, C2_ab_rs_rs, C2_ab_rs_zeta, C2_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: C2_ab, C2_ab_rs, C2_ab_zeta, C2_ab_rs_rs, C2_ab_rs_zeta, C2_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C2_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, g0, g0_rs, g0_rs_rs

      CALL calc_g0(rs, max_deriv, g0, g0_rs, g0_rs_rs)

      x0 = g0 - 1.0_dp/2.0_dp
      x1 = rs**(-3)
      x2 = zeta**2
      x3 = (3.0_dp/8.0_dp)*x2 - 3.0_dp/8.0_dp
      x4 = x1*x3
      C2_ab = x0*x4
      IF (max_deriv >= 1) THEN
         x5 = rs**(-4)
         x6 = x0*x5
         x7 = (3.0_dp/4.0_dp)*x1*zeta
         C2_ab_rs = g0_rs*x4 - x6*((9.0_dp/8.0_dp)*x2 - 9.0_dp/8.0_dp)
         C2_ab_zeta = x0*x7
      END IF
      IF (max_deriv >= 2) THEN
         C2_ab_rs_rs = -6.0_dp*g0_rs*x3*x5 + g0_rs_rs*x4 + 12.0_dp*x0*x3/rs**5
         C2_ab_rs_zeta = g0_rs*x7 - 9.0_dp/4.0_dp*x6*zeta
         C2_ab_zeta_zeta = x1*((3.0_dp/4.0_dp)*g0 - 3.0_dp/8.0_dp)
      END IF

   END SUBROUTINE calc_C2_ab

   ELEMENTAL SUBROUTINE calc_C3_ab(rs, zeta, max_deriv, C3_ab, C3_ab_rs, C3_ab_zeta, C3_ab_rs_rs, C3_ab_rs_zeta, C3_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: C3_ab, C3_ab_rs, C3_ab_zeta, C3_ab_rs_rs, C3_ab_rs_zeta, C3_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C3_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, g0, g0_rs, g0_rs_rs

      CALL calc_g0(rs, max_deriv, g0, g0_rs, g0_rs_rs)

      x0 = c3_1*g0
      x1 = rs**(-3)
      x2 = zeta**2
      x3 = x2 - 1.0_dp
      x4 = x1*x3
      C3_ab = x0*x4
      IF (max_deriv >= 1) THEN
         x5 = c3_1*g0_rs*x1
         x6 = rs**(-4)
         x7 = c3_1*g0*x6
         x8 = 2.0_dp*x0*x1
         C3_ab_rs = x3*x5 - x7*(3.0_dp*x2 - 3.0_dp)
         C3_ab_zeta = x8*zeta
      END IF
      IF (max_deriv >= 2) THEN
         C3_ab_rs_rs = -6.0_dp*c3_1*g0_rs*x3*x6 + c3_1*g0_rs_rs*x4 + 12.0_dp*x0*x3/rs**5
         C3_ab_rs_zeta = 2.0_dp*x5*zeta - 6.0_dp*x7*zeta
         C3_ab_zeta_zeta = x8
      END IF

   END SUBROUTINE calc_C3_ab

   ELEMENTAL SUBROUTINE calc_C4_ab(rs, zeta, max_deriv, C4_ab, C4_ab_rs, C4_ab_zeta, C4_ab_rs_rs, C4_ab_rs_zeta, C4_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: C4_ab, C4_ab_rs, C4_ab_zeta, C4_ab_rs_rs, C4_ab_rs_zeta, C4_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C4_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, g2z_ab, g2z_ab_rs, g2z_ab_zeta, g2z_ab_rs_rs, &
                       g2z_ab_rs_zeta, g2z_ab_zeta_zeta, D2, D2_rs, D2_rs_rs, phi8, phi8_zeta, phi8_zeta_zeta

      CALL calc_g2z_ab(rs, zeta, max_deriv, g2z_ab, g2z_ab_rs, g2z_ab_zeta, g2z_ab_rs_rs, g2z_ab_rs_zeta, g2z_ab_zeta_zeta)
      CALL calc_D2(rs, max_deriv, D2, D2_rs, D2_rs_rs)
      CALL calc_phin(zeta, 8, max_deriv, phi8, phi8_zeta, phi8_zeta_zeta)

      x0 = rs**(-3)
      x1 = alpha**(-2)
      x2 = rs**(-2)
      x3 = phi8*x1*x2
      x4 = -zeta**2 + 1.0_dp
      x5 = D2*x4
      C4_ab = x0*(-9.0_dp/64.0_dp*g2z_ab + (9.0_dp/320.0_dp)*x3 - 9.0_dp/64.0_dp*x5)
      IF (max_deriv >= 1) THEN
         x6 = rs**(-4)
         x7 = phi8*x0*x1
         x8 = D2_rs*x4
         x9 = (9.0_dp/32.0_dp)*D2
         x10 = phi8_zeta*x1*x2
         C4_ab_rs = x0*(-9.0_dp/64.0_dp*g2z_ab_rs - 9.0_dp/160.0_dp*x7 - 9.0_dp/64.0_dp &
                        *x8) + x6*((27.0_dp/64.0_dp)*g2z_ab - 27.0_dp/320.0_dp*x3 + (27.0_dp/ &
                                                                                     64.0_dp)*x5)
         C4_ab_zeta = x0*(-9.0_dp/64.0_dp*g2z_ab_zeta + (9.0_dp/320.0_dp)*x10 + x9* &
                          zeta)
      END IF
      IF (max_deriv >= 2) THEN
         C4_ab_rs_rs = x0*(-9.0_dp/64.0_dp*D2_rs_rs*x4 - 9.0_dp/64.0_dp*g2z_ab_rs_rs &
                           + (27.0_dp/160.0_dp)*phi8*x1*x6) + x6*((27.0_dp/32.0_dp)*g2z_ab_rs + &
                                                           (27.0_dp/80.0_dp)*x7 + (27.0_dp/32.0_dp)*x8) + (-27.0_dp/16.0_dp*g2z_ab &
                                                                                  + (27.0_dp/80.0_dp)*x3 - 27.0_dp/16.0_dp*x5)/rs**5
         C4_ab_rs_zeta = x0*((9.0_dp/32.0_dp)*D2_rs*zeta - 9.0_dp/64.0_dp* &
                             g2z_ab_rs_zeta - 9.0_dp/160.0_dp*phi8_zeta*x0*x1) + x6*(-27.0_dp/ &
                                                             32.0_dp*D2*zeta + (27.0_dp/64.0_dp)*g2z_ab_zeta - 27.0_dp/320.0_dp*x10)
         C4_ab_zeta_zeta = x0*(-9.0_dp/64.0_dp*g2z_ab_zeta_zeta + (9.0_dp/320.0_dp)* &
                               phi8_zeta_zeta*x1*x2 + x9)
      END IF

   END SUBROUTINE calc_C4_ab

   ELEMENTAL SUBROUTINE calc_C5_ab(rs, zeta, max_deriv, C5_ab, C5_ab_rs, C5_ab_zeta, C5_ab_rs_rs, C5_ab_rs_zeta, C5_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: C5_ab, C5_ab_rs, C5_ab_zeta, C5_ab_rs_rs, C5_ab_rs_zeta, C5_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C5_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, g2z_ab, g2z_ab_rs, g2z_ab_zeta, &
                       g2z_ab_rs_rs, g2z_ab_rs_zeta, g2z_ab_zeta_zeta, D3, D3_rs, D3_rs_rs

      CALL calc_g2z_ab(rs, zeta, max_deriv, g2z_ab, g2z_ab_rs, g2z_ab_zeta, g2z_ab_rs_rs, g2z_ab_rs_zeta, g2z_ab_zeta_zeta)
      CALL calc_D3(rs, max_deriv, D3, D3_rs, D3_rs_rs)

      x0 = sqrt(2.0_dp)
      x1 = 1.0_dp/sqrt(pi)
      x2 = x0*x1/rs**3
      x3 = -zeta**2 + 1.0_dp
      x4 = D3*x3
      C5_ab = x2*(-9.0_dp/80.0_dp*g2z_ab - 9.0_dp/80.0_dp*x4)
      IF (max_deriv >= 1) THEN
         x5 = x0*x1/rs**4
         x6 = D3_rs*x3
         x7 = (9.0_dp/40.0_dp)*D3
         C5_ab_rs = x2*(-9.0_dp/80.0_dp*g2z_ab_rs - 9.0_dp/80.0_dp*x6) + x5*((27.0_dp/ &
                                                                              80.0_dp)*g2z_ab + (27.0_dp/80.0_dp)*x4)
         C5_ab_zeta = x2*(-9.0_dp/80.0_dp*g2z_ab_zeta + x7*zeta)
      END IF
      IF (max_deriv >= 2) THEN
         C5_ab_rs_rs = x2*(-9.0_dp/80.0_dp*D3_rs_rs*x3 - 9.0_dp/80.0_dp*g2z_ab_rs_rs &
                           ) + x5*((27.0_dp/40.0_dp)*g2z_ab_rs + (27.0_dp/40.0_dp)*x6) + x0*x1*( &
                       -27.0_dp/20.0_dp*g2z_ab - 27.0_dp/20.0_dp*x4)/rs**5
         C5_ab_rs_zeta = x2*((9.0_dp/40.0_dp)*D3_rs*zeta - 9.0_dp/80.0_dp* &
                             g2z_ab_rs_zeta) + x5*(-27.0_dp/40.0_dp*D3*zeta + (27.0_dp/80.0_dp)* &
                                                   g2z_ab_zeta)
         C5_ab_zeta_zeta = x2*(-9.0_dp/80.0_dp*g2z_ab_zeta_zeta + x7)
      END IF

   END SUBROUTINE calc_C5_ab

   ELEMENTAL SUBROUTINE calc_a1_ab(rs, zeta, max_deriv, a1_ab, a1_ab_rs, a1_ab_zeta, a1_ab_rs_rs, a1_ab_rs_zeta, a1_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a1_ab, a1_ab_rs, a1_ab_zeta, a1_ab_rs_rs, a1_ab_rs_zeta, a1_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a1_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, C3_ab, C3_ab_rs, C3_ab_zeta, &
           C3_ab_rs_rs, C3_ab_rs_zeta, C3_ab_zeta_zeta, C5_ab, C5_ab_rs, C5_ab_zeta, C5_ab_rs_rs, C5_ab_rs_zeta, C5_ab_zeta_zeta, b0

      CALL calc_C3_ab(rs, zeta, max_deriv, C3_ab, C3_ab_rs, C3_ab_zeta, C3_ab_rs_rs, C3_ab_rs_zeta, C3_ab_zeta_zeta)
      CALL calc_C5_ab(rs, zeta, max_deriv, C5_ab, C5_ab_rs, C5_ab_zeta, C5_ab_rs_rs, C5_ab_rs_zeta, C5_ab_zeta_zeta)

      b0 = beta*rs

      x0 = b0**8
      x1 = C5_ab*x0
      x2 = b0**6
      x3 = 4.0_dp*x2
      a1_ab = C3_ab*x3 + x1
      IF (max_deriv >= 1) THEN
         x4 = C5_ab_rs*x0
         x5 = 1.0_dp/rs
         x6 = 24.0_dp*x5
         x7 = C3_ab*x2
         x8 = 8.0_dp*x5
         x9 = C5_ab_zeta*x0
         a1_ab_rs = C3_ab_rs*x3 + x1*x8 + x4 + x6*x7
         a1_ab_zeta = C3_ab_zeta*x3 + x9
      END IF
      IF (max_deriv >= 2) THEN
         x10 = rs**(-2)
         a1_ab_rs_rs = 48.0_dp*C3_ab_rs*x2*x5 + C3_ab_rs_rs*x3 + C5_ab_rs_rs*x0 + 56.0_dp* &
                       x1*x10 + 120.0_dp*x10*x7 + 16.0_dp*x4*x5
         a1_ab_rs_zeta = C3_ab_rs_zeta*x3 + C3_ab_zeta*x2*x6 + C5_ab_rs_zeta*x0 + x8*x9
         a1_ab_zeta_zeta = C3_ab_zeta_zeta*x3 + C5_ab_zeta_zeta*x0
      END IF

   END SUBROUTINE calc_a1_ab

   ELEMENTAL SUBROUTINE calc_a2_ab(rs, zeta, max_deriv, a2_ab, a2_ab_rs, a2_ab_zeta, a2_ab_rs_rs, a2_ab_rs_zeta, a2_ab_zeta_zeta, &
                                   Ec_ab, Ec_ab_rs, Ec_ab_zeta, Ec_ab_rs_rs, Ec_ab_rs_zeta, Ec_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta, Ec_ab, Ec_ab_rs, Ec_ab_zeta, Ec_ab_rs_rs, Ec_ab_rs_zeta, Ec_ab_zeta_zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a2_ab, a2_ab_rs, a2_ab_zeta, a2_ab_rs_rs, a2_ab_rs_zeta, a2_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a2_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, &
                       C2_ab, C2_ab_rs, C2_ab_zeta, C2_ab_rs_rs, C2_ab_rs_zeta, C2_ab_zeta_zeta, C4_ab, C4_ab_rs, &
                       C4_ab_zeta, C4_ab_rs_rs, C4_ab_rs_zeta, C4_ab_zeta_zeta, b0

      CALL calc_C2_ab(rs, zeta, max_deriv, C2_ab, C2_ab_rs, C2_ab_zeta, C2_ab_rs_rs, C2_ab_rs_zeta, C2_ab_zeta_zeta)
      CALL calc_C4_ab(rs, zeta, max_deriv, C4_ab, C4_ab_rs, C4_ab_zeta, C4_ab_rs_rs, C4_ab_rs_zeta, C4_ab_zeta_zeta)

      b0 = beta*rs

      x0 = b0**8
      x1 = C4_ab*x0
      x2 = b0**6
      x3 = 4.0_dp*x2
      x4 = b0**4
      x5 = 6.0_dp*x4
      a2_ab = C2_ab*x3 + Ec_ab*x5 + x1
      IF (max_deriv >= 1) THEN
         x6 = C4_ab_rs*x0
         x7 = 1.0_dp/rs
         x8 = 24.0_dp*x7
         x9 = C2_ab*x2
         x10 = 8.0_dp*x7
         x11 = Ec_ab*x4
         x12 = C4_ab_zeta*x0
         a2_ab_rs = C2_ab_rs*x3 + Ec_ab_rs*x5 + x1*x10 + x11*x8 + x6 + x8*x9
         a2_ab_zeta = C2_ab_zeta*x3 + Ec_ab_zeta*x5 + x12
      END IF
      IF (max_deriv >= 2) THEN
         x13 = rs**(-2)
         x14 = 48.0_dp*x7
         a2_ab_rs_rs = C2_ab_rs*x14*x2 + C2_ab_rs_rs*x3 + C4_ab_rs_rs*x0 + &
                       Ec_ab_rs*x14*x4 + Ec_ab_rs_rs*x5 + 56.0_dp*x1*x13 + 72.0_dp*x11*x13 + 120.0_dp* &
                       x13*x9 + 16.0_dp*x6*x7
         a2_ab_rs_zeta = C2_ab_rs_zeta*x3 + C2_ab_zeta*x2*x8 + C4_ab_rs_zeta*x0 + &
                         Ec_ab_rs_zeta*x5 + Ec_ab_zeta*x4*x8 + x10*x12
         a2_ab_zeta_zeta = C2_ab_zeta_zeta*x3 + C4_ab_zeta_zeta*x0 + Ec_ab_zeta_zeta*x5
      END IF

   END SUBROUTINE calc_a2_ab

   ELEMENTAL SUBROUTINE calc_a3_ab(rs, zeta, max_deriv, a3_ab, a3_ab_rs, a3_ab_zeta, a3_ab_rs_rs, a3_ab_rs_zeta, a3_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a3_ab, a3_ab_rs, a3_ab_zeta, a3_ab_rs_rs, a3_ab_rs_zeta, a3_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a3_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, C3_ab, C3_ab_rs, C3_ab_zeta, C3_ab_rs_rs, C3_ab_rs_zeta, C3_ab_zeta_zeta, b0

      CALL calc_C3_ab(rs, zeta, max_deriv, C3_ab, C3_ab_rs, C3_ab_zeta, C3_ab_rs_rs, C3_ab_rs_zeta, C3_ab_zeta_zeta)

      b0 = beta*rs

      x0 = b0**8
      x1 = C3_ab*x0
      a3_ab = x1
      IF (max_deriv >= 1) THEN
         x2 = C3_ab_rs*x0
         x3 = 1.0_dp/rs
         x4 = 8.0_dp*x3
         x5 = C3_ab_zeta*x0
         a3_ab_rs = x1*x4 + x2
         a3_ab_zeta = x5
      END IF
      IF (max_deriv >= 2) THEN
         a3_ab_rs_rs = C3_ab_rs_rs*x0 + 16.0_dp*x2*x3 + 56.0_dp*x1/rs**2
         a3_ab_rs_zeta = C3_ab_rs_zeta*x0 + x4*x5
         a3_ab_zeta_zeta = C3_ab_zeta_zeta*x0
      END IF

   END SUBROUTINE calc_a3_ab

   ELEMENTAL SUBROUTINE calc_a4_ab(rs, zeta, max_deriv, a4_ab, a4_ab_rs, a4_ab_zeta, a4_ab_rs_rs, a4_ab_rs_zeta, a4_ab_zeta_zeta, &
                                   Ec_ab, Ec_ab_rs, Ec_ab_zeta, Ec_ab_rs_rs, Ec_ab_rs_zeta, Ec_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta, Ec_ab, Ec_ab_rs, Ec_ab_zeta, Ec_ab_rs_rs, Ec_ab_rs_zeta, Ec_ab_zeta_zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a4_ab, a4_ab_rs, a4_ab_zeta, a4_ab_rs_rs, a4_ab_rs_zeta, a4_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a4_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, C2_ab, C2_ab_rs, C2_ab_zeta, C2_ab_rs_rs, &
                       C2_ab_rs_zeta, C2_ab_zeta_zeta, b0

      CALL calc_C2_ab(rs, zeta, max_deriv, C2_ab, C2_ab_rs, C2_ab_zeta, C2_ab_rs_rs, C2_ab_rs_zeta, C2_ab_zeta_zeta)

      b0 = beta*rs

      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4.0_dp*x2
      x4 = beta**6
      x5 = Ec_ab*x4
      a4_ab = C2_ab*x1 + x3*x5
      IF (max_deriv >= 1) THEN
         x6 = rs**7
         x7 = 8.0_dp*x6
         x8 = C2_ab*x0
         x9 = rs**5
         x10 = 24.0_dp*x9
         x11 = Ec_ab_rs*x4
         x12 = 4.0_dp*x2*x4
         a4_ab_rs = C2_ab_rs*x1 + x10*x5 + x11*x3 + x7*x8
         a4_ab_zeta = C2_ab_zeta*x1 + Ec_ab_zeta*x12
      END IF
      IF (max_deriv >= 2) THEN
         a4_ab_rs_rs = 16.0_dp*C2_ab_rs*x0*x6 + C2_ab_rs_rs*x1 + Ec_ab_rs_rs*x12 + 120.0_dp &
                       *rs**4*x5 + 48.0_dp*x11*x9 + 56.0_dp*x2*x8
         a4_ab_rs_zeta = C2_ab_rs_zeta*x1 + C2_ab_zeta*x0*x7 + Ec_ab_rs_zeta*x12 &
                         + Ec_ab_zeta*x10*x4
         a4_ab_zeta_zeta = C2_ab_zeta_zeta*x1 + Ec_ab_zeta_zeta*x12
      END IF

   END SUBROUTINE calc_a4_ab

   ELEMENTAL SUBROUTINE calc_a5_ab(rs, max_deriv, a5_ab, a5_ab_rs, a5_ab_zeta, a5_ab_rs_rs, a5_ab_rs_zeta, a5_ab_zeta_zeta, &
                                   Ec_ab, Ec_ab_rs, Ec_ab_zeta, Ec_ab_rs_rs, Ec_ab_rs_zeta, Ec_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, Ec_ab, Ec_ab_rs, Ec_ab_zeta, Ec_ab_rs_rs, Ec_ab_rs_zeta, Ec_ab_zeta_zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a5_ab, a5_ab_rs, a5_ab_zeta, a5_ab_rs_rs, a5_ab_rs_zeta, a5_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a5_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, b0

      b0 = beta*rs

      x0 = b0**8
      x1 = Ec_ab*x0
      a5_ab = x1
      IF (max_deriv >= 1) THEN
         x2 = Ec_ab_rs*x0
         x3 = 1.0_dp/rs
         x4 = 8.0_dp*x3
         x5 = Ec_ab_zeta*x0
         a5_ab_rs = x1*x4 + x2
         a5_ab_zeta = x5
      END IF
      IF (max_deriv >= 2) THEN
         a5_ab_rs_rs = Ec_ab_rs_rs*x0 + 16.0_dp*x2*x3 + 56.0_dp*x1/rs**2
         a5_ab_rs_zeta = Ec_ab_rs_zeta*x0 + x4*x5
         a5_ab_zeta_zeta = Ec_ab_zeta_zeta*x0
      END IF

   END SUBROUTINE calc_a5_ab

   ELEMENTAL SUBROUTINE calc_Q_ab(rs, zeta, mu, max_deriv, Q_ab, Q_ab_rs, Q_ab_zeta, Q_ab_rs_rs, Q_ab_rs_zeta, Q_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: Q_ab, Q_ab_rs, Q_ab_zeta, Q_ab_rs_rs, Q_ab_rs_zeta, Q_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_Q_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, &
                       x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, &
                       x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, &
                       phi2, phi2_zeta, phi2_zeta_zeta, xx

      CALL calc_phin(zeta, 2, max_deriv, phi2, phi2_zeta, phi2_zeta_zeta)

      xx = mu*sqrt(rs)/phi2

      x0 = phi2**(-4)
      x1 = xx**2
      x2 = x0*x1
      x3 = dd*x2
      x4 = aa*xx
      x5 = x4/phi2**2
      x6 = x5 + 1.0_dp
      x7 = x3 + x6
      x8 = 1.0_dp/x7
      x9 = bb*x2
      x10 = phi2**(-6)
      x11 = xx**3
      x12 = cc*x11
      x13 = x10*x12
      x14 = x13 + x6 + x9
      Q_ab = qq*log(x14*x8)
      IF (max_deriv >= 1) THEN
         x15 = 1.0_dp/x14
         x16 = 1.0_dp/rs
         x17 = x16*x5
         x18 = (1.0_dp/2.0_dp)*x17
         x19 = x16*x9
         x20 = (3.0_dp/2.0_dp)*x13*x16
         x21 = x18 + x19 + x20
         x22 = -x18
         x23 = x16*x3
         x24 = x22 - x23
         x25 = x7**(-2)
         x26 = x14*x25
         x27 = x21*x8 + x24*x26
         x28 = qq*x15*x27
         x29 = phi2**(-3)
         x30 = aa*x29*xx
         x31 = phi2_zeta*x30
         x32 = -x31
         x33 = phi2**(-5)
         x34 = 2.0_dp*bb*x1*x33
         x35 = phi2_zeta*x34
         x36 = phi2**(-7)
         x37 = 3.0_dp*cc*x11*x36
         x38 = phi2_zeta*x37
         x39 = x32 - x35 - x38
         x40 = dd*phi2_zeta*x1*x33
         x41 = 2.0_dp*x40
         x42 = x31 + x41
         x43 = x26*x42 + x39*x8
         x44 = qq*x15*x43
         Q_ab_rs = x28*x7
         Q_ab_zeta = x44*x7
      END IF
      IF (max_deriv >= 2) THEN
         x46 = qq*x15*x7
         x47 = rs**(-2)
         x48 = (1.0_dp/4.0_dp)*x47*x5
         x49 = x24*x25
         x50 = x7**(-3)
         x51 = x32 - x41
         x52 = qq*x45*x7*(x31 + x35 + x38)
         x53 = (1.0_dp/2.0_dp)*x16*x31
         x54 = x25*x42
         x55 = x14*x50*(2.0_dp*x31 + 4.0_dp*x40)
         x56 = phi2_zeta_zeta*x30
         x57 = phi2_zeta**2
         x58 = 2.0_dp*x0*x4*x57
         x59 = 6.0_dp*x10*x57
         x60 = dd*x1
         x56 = phi2_zeta_zeta*x27
         x57 = 2.0_dp*phi2_zeta_zeta*rs*x0*x12
         x58 = phi2_zeta**2
         x59 = 2.0_dp*x12*x4*x52*x58
         x60 = 6.0_dp*rs*x0*x33*x58
         x61 = cc*x11
         Q_ab_rs_rs = qq*x27*x45*x7*(-x19 - x20 + x22) + x28*(x18 + x23) + x46*( &
                      x14*x24*x50*(-x17 - 2.0_dp*x23) + x26*x48 + x49*(2.0_dp*bb*x0*x1*x16 + 3.0_dp*cc &
                                                                       *x10*x11*x16 + x17) + x8*((3.0_dp/4.0_dp)*x13*x47 - x48))
         Q_ab_rs_zeta = x27*x52 + x28*x51 + x46*(x21*x54 + x24*x55 + x26*(x16*x41 &
                                                                + x53) + x39*x49 + x8*(-9.0_dp/2.0_dp*phi2_zeta*x12*x16*x36 - x16* &
                                                                                                 x35 - x53))
         Q_ab_zeta_zeta = x43*x52 + x44*x51 + x46*(x26*(2*phi2_zeta_zeta*x33*x60 &
                                                        + x56 - x58 - x59*x60) + x42*x55 + x54*(-2.0_dp*aa*phi2_zeta*x29*xx - &
                                                       4.0_dp*bb*phi2_zeta*x1*x33 - 6.0_dp*cc*phi2_zeta*x11*x36) + x8*(bb*x1*x59 - &
                                                            phi2_zeta_zeta*x34 - phi2_zeta_zeta*x37 - x56 + x58 + 12.0_dp*x12*x57/ &
                                                                                                                           phi2**8))
      END IF

   END SUBROUTINE calc_Q_ab

   ELEMENTAL IMPURE SUBROUTINE calc_sr_lsd_ec(rs, zeta, mu, max_deriv, ec_mu_ab, ec_mu_ab_rs, ec_mu_ab_zeta, &
                                              ec_mu_ab_rs_rs, ec_mu_ab_rs_zeta, ec_mu_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: ec_mu_ab, ec_mu_ab_rs, ec_mu_ab_zeta, ec_mu_ab_rs_rs, ec_mu_ab_rs_zeta, ec_mu_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_ec_mu_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, &
                       x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, Q_ab, Q_ab_rs, Q_ab_zeta, Q_ab_rs_rs, &
                       Q_ab_rs_zeta, Q_ab_zeta_zeta, phi2, phi2_zeta, phi2_zeta_zeta, a1_ab, a1_ab_rs, a1_ab_zeta, &
                       a1_ab_rs_rs, a1_ab_rs_zeta, a1_ab_zeta_zeta, a2_ab, a2_ab_rs, a2_ab_zeta, a2_ab_rs_rs, a2_ab_rs_zeta, &
                       a2_ab_zeta_zeta, a3_ab, a3_ab_rs, a3_ab_zeta, a3_ab_rs_rs, a3_ab_rs_zeta, a3_ab_zeta_zeta, a4_ab, &
                       a4_ab_rs, a4_ab_zeta, a4_ab_rs_rs, a4_ab_rs_zeta, a4_ab_zeta_zeta, a5_ab, a5_ab_rs, a5_ab_zeta, &
                       a5_ab_rs_rs, a5_ab_rs_zeta, a5_ab_zeta_zeta, Ec_ab, Ec_ab_rs, Ec_ab_zeta, Ec_ab_rs_rs, Ec_ab_rs_zeta, &
                       Ec_ab_zeta_zeta, ed(0:5), fx(0:3)

      CALL calc_Q_ab(rs, zeta, mu, max_deriv, Q_ab, Q_ab_rs, Q_ab_zeta, Q_ab_rs_rs, Q_ab_rs_zeta, Q_ab_zeta_zeta)
      CALL calc_phin(zeta, 2, max_deriv, phi2, phi2_zeta, phi2_zeta_zeta)
      CALL calc_fx(zeta, fx, max_deriv)
      CALL pw_lsd_ed_loc_rz(rs, zeta, fx, ed, max_deriv)

      Ec_ab = ed(0)
      IF (max_deriv >= 1) THEN
         Ec_ab_rs = ed(1)!Ec_ab_rho*(-9.0_dp/(4.0_dp*pi))/rs**4
         Ec_ab_zeta = ed(2)!3.0_dp/8.0_dp/pi/rs**3*(ed(1)-ed(2))
      END IF
      IF (max_deriv >= 2) THEN
         Ec_ab_rs_rs = ed(3)!*(81.0_dp/(16.0_dp*pi**2))/rs**8+ed(1)*(9.0_dp/pi)/rs**5
         Ec_ab_rs_zeta = ed(4)!*(-9.0_dp/(4.0_dp*pi))/rs**4
         Ec_ab_zeta_zeta = ed(5)
      END IF

      CALL calc_a1_ab(rs, zeta, max_deriv, a1_ab, a1_ab_rs, a1_ab_zeta, a1_ab_rs_rs, a1_ab_rs_zeta, a1_ab_zeta_zeta)
      CALL calc_a2_ab(rs, zeta, max_deriv, a2_ab, a2_ab_rs, a2_ab_zeta, a2_ab_rs_rs, a2_ab_rs_zeta, a2_ab_zeta_zeta, &
                      Ec_ab, Ec_ab_rs, Ec_ab_zeta, Ec_ab_rs_rs, Ec_ab_rs_zeta, Ec_ab_zeta_zeta)
      CALL calc_a3_ab(rs, zeta, max_deriv, a3_ab, a3_ab_rs, a3_ab_zeta, a3_ab_rs_rs, a3_ab_rs_zeta, a3_ab_zeta_zeta)
      CALL calc_a4_ab(rs, zeta, max_deriv, a4_ab, a4_ab_rs, a4_ab_zeta, a4_ab_rs_rs, a4_ab_rs_zeta, a4_ab_zeta_zeta, &
                      Ec_ab, Ec_ab_rs, Ec_ab_zeta, Ec_ab_rs_rs, Ec_ab_rs_zeta, Ec_ab_zeta_zeta)
      CALL calc_a5_ab(rs, max_deriv, a5_ab, a5_ab_rs, a5_ab_zeta, a5_ab_rs_rs, a5_ab_rs_zeta, a5_ab_zeta_zeta, &
                      Ec_ab, Ec_ab_rs, Ec_ab_zeta, Ec_ab_rs_rs, Ec_ab_rs_zeta, Ec_ab_zeta_zeta)

      x0 = rs**2
      x1 = beta**2
      x2 = mu**2
      x3 = x1*x2
      x4 = x0*x3 + 1.0_dp
      x5 = x4**(-4)
      x6 = phi2**3
      x7 = Q_ab*x6
      x8 = mu**3
      x9 = a1_ab*x8
      x10 = mu**4
      x11 = a2_ab*x10
      x12 = mu**5
      x13 = a3_ab*x12
      x14 = mu**6
      x15 = a4_ab*x14
      x16 = mu**8
      x17 = a5_ab*x16
      ec_mu_ab = Ec_ab - x5*(x11 + x13 + x15 + x17 + x7 + x9)
      IF (max_deriv >= 1) THEN
         x18 = x4**(-5)
         x19 = 8*rs*x1*x18*x2
         x20 = Q_ab_zeta*x6
         x21 = a1_ab_zeta*x8
         x22 = a2_ab_zeta*x10
         x23 = a3_ab_zeta*x12
         x24 = a4_ab_zeta*x14
         x25 = a5_ab_zeta*x16
         x26 = phi2**2
         x27 = 3.0_dp*Q_ab*x26
         x28 = phi2_zeta*x27
         ec_mu_ab_rs = Ec_ab_rs + x19*(x11 + x13 + x15 + x17 + x7 + x9) - x5*( &
                       Q_ab_rs*x6 + a1_ab_rs*x8 + a2_ab_rs*x10 + a3_ab_rs*x12 + &
                       a4_ab_rs*x14 + a5_ab_rs*x16)
         ec_mu_ab_zeta = Ec_ab_zeta - x5*(x20 + x21 + x22 + x23 + x24 + x25 + x28)
      END IF
      IF (max_deriv >= 2) THEN
         x29 = phi2_zeta*x26
         ec_mu_ab_rs_rs = Ec_ab_rs_rs - 80.0_dp*beta**4*x0*x10*(x11 + x13 + x15 &
                                                                + x17 + x7 + x9)/x4**6 + 16.0_dp*rs*x1*x18*x2*(Q_ab_rs*x6 + &
                                                                             a1_ab_rs*x8 + a2_ab_rs*x10 + a3_ab_rs*x12 + a4_ab_rs* &
                                                                      x14 + a5_ab_rs*x16) + 8.0_dp*x18*x3*(x11 + x13 + x15 + x17 + &
                                                                      x7 + x9) - x5*(Q_ab_rs_rs*x6 + a1_ab_rs_rs*x8 + a2_ab_rs_rs* &
                                                                          x10 + a3_ab_rs_rs*x12 + a4_ab_rs_rs*x14 + a5_ab_rs_rs*x16)
         ec_mu_ab_rs_zeta = Ec_ab_rs_zeta + x19*(x20 + x21 + x22 + x23 + x24 + &
                                                 x25 + x28) - x5*(3.0_dp*Q_ab_rs*x29 + Q_ab_rs_zeta*x6 + a1_ab_rs_zeta &
                                                                 *x8 + a2_ab_rs_zeta*x10 + a3_ab_rs_zeta*x12 + a4_ab_rs_zeta*x14 + &
                                                                  a5_ab_rs_zeta*x16)
         ec_mu_ab_zeta_zeta = Ec_ab_zeta_zeta - x5*(6*Q_ab*phi2*phi2_zeta**2 + 6 &
                                                    *Q_ab_zeta*x29 + Q_ab_zeta_zeta*x6 + a1_ab_zeta_zeta*x8 + &
                                                    a2_ab_zeta_zeta*x10 + a3_ab_zeta_zeta*x12 + a4_ab_zeta_zeta*x14 + &
                                                    a5_ab_zeta_zeta*x16 + phi2_zeta_zeta*x27)
      END IF

   END SUBROUTINE calc_sr_lsd_ec

! **************************************************************************************************
!> \brief ...
!> \param rs ...
!> \param max_deriv ...
!> \param ex_mu ...
!> \param ex_mu_rho ...
!> \param ex_mu_rho_rho ...
! **************************************************************************************************
   ELEMENTAL SUBROUTINE calc_sr_lda_ex(rs, mu, max_deriv, ex_mu, ex_mu_rho, ex_mu_rho_rho)
      REAL(KIND=dp), INTENT(IN) :: rs, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: ex_mu, ex_mu_rho, ex_mu_rho_rho

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_sr_lda_ex", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, ex_mu_rs, ex_mu_rs_rs, Ax, exp_Ax

      Ax = facx_A*rs*mu

      x0 = Ax**2
      exp_Ax = exp(-0.25_dp/x0)
      ex_mu = (1.0_dp/8.0_dp)*facx*(8.0_dp*Ax*(4.0_dp*Ax**3 - 2.0_dp*Ax*exp_Ax*(2.0_dp*x0 - 1) - 3.0_dp* &
                                               Ax + sqrt(pi)*erf(0.5_dp/Ax)) - 3.0_dp)/rs
      IF (max_deriv >= 1) THEN
         x1 = 8.0_dp*x0
         x2 = 32.0_dp*Ax**4
         x3 = exp_Ax*x2 - x2
         ex_mu_rs = -3.0_dp/8.0_dp*facx*(x1 + x3 - 1.0_dp)/rs**2
      END IF
      IF (max_deriv >= 2) THEN
         ex_mu_rs_rs = -3.0_dp/4.0_dp*facx*(exp_Ax*x1 + x3 + 1.0_dp)/rs**3
      END IF

      CALL calc_rs2rho(rs, max_deriv, ex_mu_rs, ex_mu_rs_rs, ex_mu_rho, ex_mu_rho_rho)

   END SUBROUTINE calc_sr_lda_ex

! **************************************************************************************************
!> \brief Transform derivatives wrt rs to derivatives wrt rho
!> \param rs ...
!> \param max_deriv ...
!> \param e_rs ...
!> \param e_rs_rs ...
!> \param e_rho ...
!> \param e_rho_rho ...
! **************************************************************************************************
   ELEMENTAL SUBROUTINE calc_rs2rho(rs, max_deriv, e_rs, e_rs_rs, e_rho, e_rho_rho)
      REAL(KIND=dp), INTENT(IN) :: rs, e_rs, e_rs_rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: e_rho, e_rho_rho

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_rs2rho", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: rs_rho, rs_rho_rho

      IF (max_deriv >= 1) THEN
         rs_rho = -4.0_dp/9.0_dp*pi*rs**4
         e_rho = e_rs*rs_rho
      END IF
      IF (max_deriv >= 2) THEN
         rs_rho_rho = 4.0_dp*rs_rho**2/rs
         e_rho_rho = e_rs_rs*rs_rho**2 + e_rs*rs_rho_rho
      END IF

   END SUBROUTINE calc_rs2rho

@:calc_ec_pw92()

END MODULE xc_sr_lda
