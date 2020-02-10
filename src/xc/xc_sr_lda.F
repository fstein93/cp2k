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
   USE xc_functionals_utilities, ONLY: set_util
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
      d2_3 = 0.676_dp

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
      IF (PRESENT(max_deriv)) max_deriv = 1

   END SUBROUTINE sr_lda_info

@:init_pw92()

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
      REAL(KIND=dp), DIMENSION(:, :, :), POINTER         :: dummy, e_0, e_rho, rho
      TYPE(xc_derivative_type), POINTER                  :: deriv

      CALL timeset(routineN, handle)

      CALL section_vals_val_get(sr_section, 'SCALE_X', r_val=sx)
      CALL section_vals_val_get(sr_section, 'SCALE_C', r_val=sc)
      CALL section_vals_val_get(sr_section, 'OMEGA', r_val=omega)
      CALL section_vals_val_get(sr_section, 'PARAMETRIZATION', i_val=method)

      NULLIFY (bo, rho, e_0, e_rho, dummy)
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
      IF (order > 1 .OR. order < -1) THEN
         CPABORT("derivatives bigger than 1 not implemented")
      END IF

      CALL sr_lda_calc(rho, omega, rho_cutoff, e_0, e_rho, &
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
   SUBROUTINE sr_lda_calc(rho, omega, rho_cutoff, e_0, e_rho, npoints, order, sx, sc)
      !FM low level calc routine
      REAL(KIND=dp), DIMENSION(*), INTENT(IN)            :: rho
      REAL(KIND=dp), INTENT(IN)                          :: omega, rho_cutoff
      REAL(KIND=dp), DIMENSION(*), INTENT(INOUT)         :: e_0, e_rho
      INTEGER, INTENT(IN)                                :: npoints, order
      REAL(KIND=dp), INTENT(IN)                          :: sx, sc

      CHARACTER(len=*), PARAMETER :: routineN = 'sr_lda_calc', routineP = moduleN//':'//routineN

      INTEGER                                            :: handle, k
      REAL(KIND=dp)                                      :: my_rho, rs
      REAL(KIND=dp), DIMENSION(0:1)                      :: ed

      CALL timeset(routineN, handle)

      IF (sc /= 0.0_dp .OR. sx /= 0.0_dp) THEN
!$OMP PARALLEL DO PRIVATE (k, ed, my_rho, rs) DEFAULT(NONE)&
!$OMP SHARED(npoints,rho,rho_cutoff,omega,e_0,e_rho,order,sc,sx)
         DO k = 1, npoints

            my_rho = rho(k)
            IF (rho(k) > rho_cutoff) THEN
               rs = (3.0_dp/4.0_dp/pi/my_rho)**(1.0_dp/3.0_dp)

               CALL ldasr(rs, omega, ed(0), ed(1), sx, sc)

               IF (order >= 0) THEN
                  e_0(k) = e_0(k) + ed(0)*my_rho
               END IF
               IF (order >= 1 .OR. order == -1) THEN
                  e_rho(k) = e_rho(k) + ed(1)
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
      REAL(KIND=dp), DIMENSION(:, :, :), POINTER         :: a, b, dummy, e_0, ea, eb
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
      ea => dummy; eb => dummy

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
      IF (order > 1 .OR. order < -1) THEN
         CPABORT("derivatives bigger than 1 not implemented")
      END IF

      CALL sr_lsd_calc(a, b, omega, rho_cutoff, e_0, ea, eb, npoints, order, sx, sc)

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
   SUBROUTINE sr_lsd_calc(rhoa, rhob, omega, rho_cutoff, e_0, ea, eb, npoints, order, sx, sc)
      !FM low-level computation routine
      REAL(KIND=dp), DIMENSION(*), INTENT(IN)            :: rhoa, rhob
      REAL(KIND=dp), INTENT(IN)                          :: omega, rho_cutoff
      REAL(KIND=dp), DIMENSION(*), INTENT(INOUT)         :: e_0, ea, eb
      INTEGER, INTENT(IN)                                :: npoints, order
      REAL(KIND=dp), INTENT(IN)                          :: sx, sc

      CHARACTER(len=*), PARAMETER :: routineN = 'sr_lsd_calc', routineP = moduleN//':'//routineN

      INTEGER                                            :: handle, k
      REAL(KIND=dp)                                      :: my_rhoa, my_rhob, rho, rs, zeta
      REAL(KIND=dp), DIMENSION(0:5)                      :: ed

      CALL timeset(routineN, handle)

      IF (sc /= 0.0_dp .OR. sx /= 0.0_dp) THEN
!$OMP PARALLEL DO PRIVATE (k, rho, ed, my_rhoa, my_rhob, rs, zeta) DEFAULT(NONE)&
!$OMP SHARED(npoints,rhoa,rhob,rho_cutoff,omega,order,e_0,ea,eb,sx,sc)
         DO k = 1, npoints

            my_rhoa = rhoa(k)
            my_rhob = rhob(k)
            rho = my_rhoa + my_rhob
            IF (rho > rho_cutoff) THEN
               rs = (3.0_dp/4.0_dp/pi/rho)**(1.0_dp/3.0_dp)
               zeta = (my_rhoa - my_rhob)/rho

               CALL lsdsr(rs, zeta, omega, ed(0), ed(1), ed(2), sx, sc)
               IF (order >= 0) THEN
                  e_0(k) = e_0(k) + ed(0)*rho
               END IF
               IF (order >= 1 .OR. order == -1) THEN
                  ea(k) = ea(k) + ed(1)
                  eb(k) = eb(k) + ed(2)
               END IF
            END IF

         END DO
!$OMP END PARALLEL DO

         CALL timestop(handle)

      END IF

   END SUBROUTINE sr_lsd_calc

! **************************************************************************************************
!> \brief ...
!> \param rs ...
!> \param mu ...
!> \param excsr ...
!> \param vxcsr ...
!> \param sx ...
!> \param sc ...
! **************************************************************************************************
   ELEMENTAL SUBROUTINE ldasr(rs, mu, excsr, vxcsr, sx, sc)
      REAL(KIND=dp), INTENT(IN)                          :: rs, mu
      REAL(KIND=dp), INTENT(OUT)                         :: excsr, vxcsr
      REAL(KIND=dp), INTENT(IN)                          :: sx, sc

      CHARACTER(len=*), PARAMETER :: routineN = 'ldasr', routineP = moduleN//':'//routineN

      REAL(KIND=dp)                                      :: ec, ecd, eclr, ex, exlr, vc, vclr, vx, &
                                                            vxlr

      IF (sx /= 0.0_dp) THEN
         ex = -3._dp*cf/rs/4._dp/pi
         vx = -(3._dp/2._dp/pi)**(2._dp/3._dp)/rs

         CALL exchangelr_lda(rs, mu, exlr, vxlr)
      ELSE
         ex = 0.0_dp
         vx = 0.0_dp

         exlr = 0.0_dp
         vxlr = 0.0_dp
      END IF

      IF (sc /= 0.0_dp) THEN
         CALL ecPW_lda(rs, ec, ecd)
         vc = ec - rs/3._dp*ecd

         CALL ecorrlr_lda(rs, mu, eclr, vclr, ec, ecd)
      ELSE
         ec = 0.0_dp
         vc = 0.0_dp

         eclr = 0.0_dp
         vclr = 0.0_dp
      END IF

      excsr = sx*ex + sc*ec - (sx*exlr + sc*eclr)
      vxcsr = sx*vx + sc*vc - (sx*vxlr + sc*vclr)

   END SUBROUTINE

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
                        *x41*x49*x50 + x53*x66 + x54*x66 + x69*(-2*x14 - 4*x44) + x71*(-2.0_dp &
                                                                                       *x29 - 4.0_dp*x48)
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

! **************************************************************************************************
!> \brief ...
!> \param rs ...
!> \param mu ...
!> \param eclr ...
!> \param vclr ...
!> \param ec ...
!> \param ecd ...
! **************************************************************************************************
   ELEMENTAL SUBROUTINE ecorrlr_lda(rs, mu, eclr, vclr, ec, ecd)
      REAL(KIND=dp), INTENT(IN)                          :: rs, mu
      REAL(KIND=dp), INTENT(OUT)                         :: eclr, vclr
      REAL(KIND=dp), INTENT(IN)                          :: ec, ecd

      CHARACTER(len=*), PARAMETER :: routineN = 'ecorrlr_lda', routineP = moduleN//':'//routineN

      REAL(KIND=dp) :: a1, a1rs, a2, a2rs, a3, a3rs, a4, a4rs, a5, a5rs, b0, coe2, coe2rs, coe3, &
                       coe3rs, coe4, coe4rs, coe5, coe5rs, d2anti, d2antid, d3anti, d3antid, eclrrs, &
                       x, z, dummy, g0, g0d, dpol, dpold

      b0 = adib*rs
      z = 0._dp

      CALL calc_D2(rs, 1, d2anti, d2antid, dummy)
      CALL calc_D3(rs, 1, d3anti, d3antid, dummy)

      CALL calc_g0(rs, 1, g0, g0d, dummy)
      CALL calc_g2(rs, 1, dpol, dpold, dummy)

      coe2 = -3._dp/8._dp/rs**3*(g0 - 0.5_dp)
      coe2rs = -3._dp/8._dp/rs**3*g0d+9._dp/8._dp/rs**4*(g0 - 0.5_dp)

      coe3 = -g0/SQRT(2._dp*pi)/rs**3
      coe3rs = -g0d/SQRT(2._dp*pi)/rs**3 + 3._dp*g0/SQRT(2._dp*pi)/rs**4

      coe4 = -9._dp/64._dp/rs**3*(dpol + d2anti - cf**2/5._dp/rs**2)
      coe4rs = -3._dp/rs*coe4 - 9._dp/64._dp/rs**3*(dpold + d2antid &
                                                    + cf**2/5._dp*2._dp/rs**3)

      coe5 = -9._dp/40._dp/SQRT(2._dp*pi)/rs**3*(dpol + d3anti)
      coe5rs = -3._dp/rs*coe5 - 9._dp/(40._dp*SQRT(2._dp*pi)*rs**3)*(dpold + d3antid)

      a1 = 4._dp*b0**6*coe3 + b0**8*coe5
      a1rs = 24._dp*adib*b0**5*coe3 + 4._dp*b0**6*coe3rs + 8._dp*adib*b0**7*coe5 + b0**8*coe5rs

      a2 = 4._dp*b0**6*coe2 + b0**8*coe4 + 6._dp*b0**4*ec
      a2rs = 24._dp*adib*b0**5*coe2 + 4._dp*b0**6*coe2rs + 8._dp*adib*b0**7*coe4 + b0**8*coe4rs &
             + 24._dp*adib*b0**3*ec + 6._dp*b0**4*ecd

      a3 = b0**8*coe3
      a3rs = 8._dp*adib*b0**7*coe3 + b0**8*coe3rs

      a4 = b0**6*(b0**2*coe2 + 4._dp*ec)
      a4rs = 8._dp*adib*b0**7*coe2 + b0**8*coe2rs + 24._dp*adib*b0**5*ec + 4._dp*b0**6*ecd

      a5 = b0**8*ec
      a5rs = 8._dp*adib*b0**7*ec + b0**8*ecd

      x = mu*SQRT(rs)

      eclr = (Qrpa(x) + mu**3*(a1 + mu*(a2 + mu*(a3 + mu*(a4 + a5*mu**2)))))/((1._dp + b0**2*mu**2)**4)

      eclrrs = -8._dp*adib/(1._dp + b0**2*mu**2)*b0*mu**2*eclr + &
               1._dp/((1._dp + b0**2*mu**2)**4)*(mu/(2._dp*SQRT(rs))*Qrpad(x) + &
                                                 mu**3*(a1rs + mu*(a2rs + mu*(a3rs + mu*(a4rs + a5rs*mu**2)))))

      vclr = eclr - rs/3._dp*eclrrs

   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param rs ...
!> \param z ...
!> \param mu ...
!> \param vxlrup ...
!> \param vxlrdown ...
! **************************************************************************************************
   ELEMENTAL SUBROUTINE exchangelr_lda(rs, mu, exlr, vxlr)
      REAL(KIND=dp), INTENT(IN)                          :: rs, mu
      REAL(KIND=dp), INTENT(OUT)                         :: exlr, vxlr

      CHARACTER(len=*), PARAMETER :: routineN = 'exchangelr_lda', &
                                     routineP = moduleN//':'//routineN

      REAL(KIND=dp)                                      :: derrs, fx, fx1, y

      y = alpha/2._dp*mu*rs
      fx = -((y*(-3._dp + 4._dp*y**2 + (2._dp - 4._dp*y**2)*EXP(-.25_dp/y**2)) + SQRT(pi)*ERF(.5_dp/y))/pi)
      exlr = mu*fx
      fx1 = (3._dp*(1._dp + (-4._dp + 4._dp*EXP(-.25_dp/y**2))*y**2))/pi
      derrs = alpha/4._dp*mu**2*fx1
      vxlr = 2._dp/3._dp*rs*derrs

      vxlr = exlr - vxlr

   END SUBROUTINE

! **************************************************************************************************
!> \brief PW92 energy functional
!> \param rs ...
!> \param ec ...
!> \param ecd ...
! **************************************************************************************************
   ELEMENTAL SUBROUTINE ecPW_lda(rs, ec, ecd)
      REAL(KIND=dp), INTENT(IN)                          :: rs
      REAL(KIND=dp), INTENT(OUT)                         :: ec, ecd

      REAL(KIND=dp) :: G(0:1)

      CALL calc_g(rs, 0, G, 1)
      ec = G(0)
      ecd = G(1)

   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param rs ...
!> \param z ...
!> \param mu ...
!> \param excsr ...
!> \param vxcsrup ...
!> \param vxcsrdown ...
!> \param sx ...
!> \param sc ...
! **************************************************************************************************
   ELEMENTAL SUBROUTINE lsdsr(rs, z, mu, excsr, vxcsrup, vxcsrdown, sx, sc)
      REAL(KIND=dp), INTENT(IN)                          :: rs, z, mu
      REAL(KIND=dp), INTENT(OUT)                         :: excsr, vxcsrup, vxcsrdown
      REAL(KIND=dp), INTENT(IN)                          :: sx, sc

      CHARACTER(len=*), PARAMETER :: routineN = 'lsdsr', routineP = moduleN//':'//routineN

      REAL(KIND=dp)                                      :: ec, ecd, eclr, ecz, ex, exlr, vcdown, &
                                                            vclrdown, vclrup, vcup, vxdown, &
                                                            vxlrdown, vxlrup, vxup

      IF (sx /= 0.0_dp) THEN
         ex = -3._dp*cf/rs/8._dp/pi*(MIN(1._dp + z, 0.0_dp)**(4._dp/3._dp) + &
                                     MIN(1._dp - z, 0.0_dp)**(4._dp/3._dp))

         vxup = -MIN(1._dp + z, 0.0_dp)**(1._dp/3._dp)*(3._dp/2._dp/pi)**(2._dp/3._dp)/rs
         vxdown = -MIN(1._dp - z, 0.0_dp)**(1._dp/3._dp)*(3._dp/2._dp/pi)**(2._dp/3._dp)/rs

         CALL exchangelr_lsd(rs, z, mu, exlr, vxlrup, vxlrdown)
      ELSE
         ex = 0.0_dp
         vxup = 0.0_dp
         vxdown = 0.0_dp

         exlr = 0.0_dp
         vxlrup = 0.0_dp
         vxlrdown = 0.0_dp
      END IF

      IF (sc /= 0.0_dp) THEN
         CALL ecPW_lsd(rs, z, ec, ecd, ecz)
         vcup = ec - rs/3._dp*ecd - (z - 1._dp)*ecz
         vcdown = ec - rs/3._dp*ecd - (z + 1._dp)*ecz

         CALL ecorrlr_lsd(rs, z, mu, eclr, vclrup, vclrdown, ec, ecd, ecz)
      ELSE
         ec = 0.0_dp
         vcup = 0.0_dp
         vcdown = 0.0_dp

         eclr = 0.0_dp
         vclrup = 0.0_dp
         vclrdown = 0.0_dp
      END IF

      excsr = sx*ex + sc*ec - (sx*exlr + sc*eclr)
      vxcsrup = sx*vxup + sc*vcup - (sx*vxlrup + sc*vclrup)
      vxcsrdown = sx*vxdown + sc*vcdown - (sx*vxlrdown + sc*vclrdown)

   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param rs ...
!> \param z ...
!> \param mu ...
!> \param eclr ...
!> \param ec ...
! **************************************************************************************************
   ELEMENTAL SUBROUTINE ecorrlr_lsd(rs, z, mu, eclr, vclrup, vclrdown, ec, ecd, ecz)
      REAL(KIND=dp), INTENT(IN)                          :: rs, z, mu
      REAL(KIND=dp), INTENT(OUT)                         :: eclr, vclrup, vclrdown
      REAL(KIND=dp), INTENT(IN)                          :: ec, ecd, ecz

      CHARACTER(len=*), PARAMETER :: routineN = 'ecorrlr_lsd', routineP = moduleN//':'//routineN

      REAL(KIND=dp) :: a1, a1rs, a1z, a2, a2rs, a2z, a3, a3rs, a3z, a4, a4rs, a4z, a5, a5rs, a5z, &
                       b0, coe2, coe2rs, coe2z, coe3, coe3rs, coe3z, coe4, coe4rs, coe4z, coe5, coe5rs, coe5z, &
                       d2anti, d2antid, d3anti, d3antid, eclrrs, eclrz, phi, x, dummy

      phi = ((1._dp + z)**(2._dp/3._dp) + (1._dp - z)**(2._dp/3._dp))/2._dp

      b0 = adib*rs

      CALL calc_D2(rs, 1, d2anti, d2antid, dummy)
      CALL calc_D3(rs, 1, d3anti, d3antid, dummy)

      coe2 = -3._dp/8._dp/rs**3*(1._dp - z**2)*(g0(rs) - 0.5_dp)
      coe2rs = -3._dp/8._dp/rs**3*(1._dp - z**2)*g0d(rs) + 9._dp/8._dp/rs**4*(1._dp - z**2)*(g0(rs) - 0.5_dp)
      coe2z = -3._dp/8._dp/rs**3*(-2._dp*z)*(g0(rs) - 0.5_dp)

      coe3 = -(1._dp - z**2)*g0(rs)/SQRT(2._dp*pi)/rs**3
      coe3rs = -(1._dp - z**2)*g0d(rs)/SQRT(2._dp*pi)/rs**3 + 3._dp*(1._dp - z**2)*g0(rs)/SQRT(2._dp*pi)/rs**4
      coe3z = 2._dp*z*g0(rs)/(SQRT(2._dp*pi)*rs**3)

      IF (ABS(z) >= 1._dp) THEN

         coe4 = -9._dp/64._dp/rs**3*(dpol(rs) - cf**2*2**(5._dp/3._dp)/5._dp/rs**2)
         coe4rs = -3._dp/rs*coe4 - 9._dp/64._dp/rs**3*(dpold(rs) + 2._dp*cf**2*2**(5._dp/3._dp)/5._dp/rs**3)
         coe4z = -9._dp/64._dp/rs**3*(dpol(rs) - rs/6._dp*dpold(rs) - 2._dp*d2anti &
                                      - 4._dp/15._dp*cf**2*2._dp**(5._dp/3._dp)/rs**2)*z

         coe5 = -9._dp/40._dp/SQRT(2._dp*pi)/rs**3*dpol(rs)
         coe5rs = -3._dp/rs*coe5 - 9._dp/40._dp/SQRT(2._dp*pi)/rs**3*dpold(rs)
         coe5z = -9._dp/40._dp/SQRT(2._dp*pi)/rs**3*(dpol(rs) - rs/6._dp*dpold(rs) - 2._dp*d3anti)*z

      ELSE

         coe4 = -9._dp/64._dp/rs**3*(((1._dp + z)/2._dp)**2* &
                                     dpol(rs*(2._dp/(1._dp + z))**(1._dp/3._dp)) + ((1._dp - z)/2._dp)**2 &
                                     *dpol(rs*(2._dp/(1._dp - z))**(1._dp/3._dp)) + &
                                     (1._dp - z**2)*d2anti - cf**2/10._dp*((1._dp + z)**(8._dp/3._dp) &
                                                                           + (1._dp - z)**(8._dp/3._dp))/rs**2)
         coe4rs = -3._dp/rs*coe4 - 9._dp/64._dp/rs**3*(((1._dp + z)/2._dp)**(5._dp/3._dp)*dpold(rs*(2._dp/(1._dp + z))** &
                                                                              (1._dp/3._dp)) + ((1._dp - z)/2._dp)**(5._dp/3._dp)* &
                                                       dpold(rs*(2._dp/(1._dp - z))**(1._dp/3._dp)) + (1._dp - z**2)*d2antid &
                                                      + cf**2/5._dp*((1._dp + z)**(8._dp/3._dp) + (1._dp - z)**(8._dp/3._dp))/rs**3)
         coe4z = -9._dp/64._dp/rs**3*(1._dp/2._dp*(1._dp + z)*dpol(rs*(2._dp/(1._dp + z))**(1._dp/3._dp)) &
                                      - 1._dp/2._dp*(1._dp - z)*dpol(rs*(2._dp/(1._dp - z))**(1._dp/3._dp)) &
                                      - rs/6._dp*((1._dp + z)/2._dp)**(2._dp/3._dp)*dpold(rs*(2/(1._dp + z))**(1._dp/3._dp)) &
                                      + rs/6._dp*((1._dp - z)/2._dp)**(2._dp/3._dp)*dpold(rs*(2._dp/(1._dp - z))**(1._dp/3._dp)) &
                                      - 2._dp*z*d2anti - 4._dp/15._dp*cf**2/rs**2*((1._dp + z)**(5._dp/3._dp) &
                                                                                   - (1._dp - z)**(5._dp/3._dp)))

         coe5 = -9._dp/40._dp/SQRT(2._dp*pi)/rs**3*(((1._dp + z)/2._dp)**2*dpol(rs*(2._dp/(1._dp + z))**(1._dp/3._dp)) &
                                                    + ((1._dp - z)/2._dp)**2*dpol(rs*(2._dp/(1._dp - z))**(1._dp/3._dp)) &
                                                    + (1._dp - z**2)*d3anti)
         coe5rs = -3._dp/rs*coe5 - 9._dp/(40._dp*SQRT(2._dp*pi)*rs**3)*( &
                  ((1._dp + z)/2._dp)**(5._dp/3._dp)*dpold(rs*(2._dp/(1._dp + z))**(1._dp/3._dp)) &
                  + ((1._dp - z)/2._dp)**(5._dp/3._dp)*dpold(rs*(2._dp/(1._dp - z))**(1._dp/3._dp)) + (1._dp - z**2)*d3antid)
         coe5z = -9._dp/40._dp/SQRT(2._dp*pi)/rs**3*(1._dp/2._dp*(1._dp + z)*dpol(rs*(2/(1._dp + z))**(1._dp/3._dp)) &
                                                     - 1._dp/2._dp*(1._dp - z)*dpol(rs*(2/(1._dp - z))**(1._dp/3._dp)) &
                                                     - rs/6._dp*((1._dp + z)/2._dp)**(2._dp/3._dp)*dpold(rs*(2/(1._dp + z)) &
                                                                    **(1._dp/3._dp)) + rs/6._dp*((1._dp - z)/2._dp)**(2._dp/3._dp) &
                                                     *dpold(rs*(2/(1._dp - z))**(1._dp/3._dp)) - 2._dp*z*d3anti)

      END IF

      a1 = 4._dp*b0**6*coe3 + b0**8*coe5
      a1rs = 24._dp*adib*b0**5*coe3 + 4._dp*b0**6*coe3rs + 8._dp*adib*b0**7*coe5 + b0**8*coe5rs
      a1z = 4._dp*b0**6*coe3z + b0**8*coe5z

      a2 = 4._dp*b0**6*coe2 + b0**8*coe4 + 6._dp*b0**4*ec
      a2rs = 24._dp*adib*b0**5*coe2 + 4._dp*b0**6*coe2rs + 8._dp*adib*b0**7*coe4 + b0**8*coe4rs &
             + 24._dp*adib*b0**3*ec + 6._dp*b0**4*ecd
      a2z = 4._dp*b0**6*coe2z + b0**8*coe4z + 6._dp*b0**4*ecz

      a3 = b0**8*coe3
      a3rs = 8._dp*adib*b0**7*coe3 + b0**8*coe3rs
      a3z = b0**8*coe3z

      a4 = b0**6*(b0**2*coe2 + 4._dp*ec)
      a4rs = 8._dp*adib*b0**7*coe2 + b0**8*coe2rs + 24._dp*adib*b0**5*ec + 4._dp*b0**6*ecd
      a4z = b0**6*(b0**2*coe2z + 4._dp*ecz)

      a5 = b0**8*ec
      a5rs = 8._dp*adib*b0**7*ec + b0**8*ecd
      a5z = b0**8*ecz

      x = mu*SQRT(rs)/phi

      eclr = (phi**3*Qrpa(x) + mu**3*(a1 + mu*(a2 + mu*(a3 + mu*(a4 + a5*mu**2)))))/((1._dp + b0**2*mu**2)**4)

      eclrrs = -8._dp*adib/(1._dp + b0**2*mu**2)*b0*mu**2*eclr + &
               1._dp/((1._dp + b0**2*mu**2)**4)*(phi**2*mu/(2._dp*SQRT(rs))*Qrpad(x) + &
                                                 mu**3*(a1rs + mu*(a2rs + mu*(a3rs + mu*(a4rs + a5rs*mu**2)))))

      IF (z >= 1._dp) THEN
         vclrup = eclr - rs/3._dp*eclrrs
         vclrdown = 0._dp
      ELSE IF (z <= -1._dp) THEN
         vclrup = 0._dp
         vclrdown = eclr - rs/3._dp*eclrrs
      ELSE

         eclrz = (phi**2*((1._dp + z)**(-1._dp/3._dp) - (1._dp - z)**(-1._dp/3._dp)) &
                  *Qrpa(x) - phi*Qrpad(x)*mu*SQRT(rs)*((1._dp + z)**(-1._dp/3._dp) &
                                                       - (1._dp - z)**(-1._dp/3._dp))/3._dp + &
                  mu**3*(a1z + mu*(a2z + mu*(a3z + mu*(a4z + a5z*mu**2)))))/((1._dp + b0**2*mu**2)**4)

         vclrup = eclr - rs/3._dp*eclrrs - (z - 1._dp)*eclrz
         vclrdown = eclr - rs/3._dp*eclrrs - (z + 1._dp)*eclrz
      END IF

   END SUBROUTINE

! **************************************************************************************************
!> \brief ...
!> \param rs ...
!> \return ...
! **************************************************************************************************
   ELEMENTAL FUNCTION g0(rs)
      REAL(KIND=dp), INTENT(IN)                          :: rs
      REAL(KIND=dp)                                      :: g0

      g0 = (1._dp - (0.7317_dp - Fg0)*rs + Cg0*rs**2 + Dg0*rs**3 + Eg0*rs**4)*EXP(-ABS(Fg0)*rs)/2._dp

   END FUNCTION

! **************************************************************************************************
!> \brief ...
!> \param rs ...
!> \return ...
! **************************************************************************************************
   ELEMENTAL FUNCTION g0d(rs)
      REAL(KIND=dp), INTENT(IN)                          :: rs
      REAL(KIND=dp)                                      :: g0d

      g0d = (-Bg0 + 2.0_dp*Cg0*rs + 3.0_dp*Dg0*rs**2 + 4.0_dp*Eg0*rs**3)/2._dp*EXP(-Fg0*rs) &
            - (Fg0*(1.0_dp - Bg0*rs + Cg0*rs**2 + Dg0*rs**3 + Eg0*rs**4))/ &
            2._dp*EXP(-Fg0*rs)

   END FUNCTION

! **************************************************************************************************
!> \brief ...
!> \param rs ...
!> \return ...
! **************************************************************************************************
   ELEMENTAL FUNCTION dpol(rs)
      REAL(KIND=dp), INTENT(IN)                          :: rs
      REAL(KIND=dp)                                      :: dpol

      dpol = 2._dp**(5._dp/3._dp)/5._dp*cf**2/rs**2*(1._dp + (p3p - 0.454555_dp)*rs) &
             /(1._dp + p3p*rs + p2p*rs**2)

   END FUNCTION

! **************************************************************************************************
!> \brief ...
!> \param rs ...
!> \return ...
! **************************************************************************************************
   ELEMENTAL FUNCTION dpold(rs)
      REAL(KIND=dp), INTENT(IN)                          :: rs
      REAL(KIND=dp)                                      :: dpold

      dpold = 2._dp**(5._dp/3._dp)/5._dp*cf**2* &
              (-2._dp + (0.454555 - 4._dp*p3p)*rs + &
               (-4._dp*p2p + (0.90911 - 2.*p3p)*p3p)*rs**2 &
               + p2p*(1.363665 - 3._dp*p3p)*rs**3)/ &
              (rs**3*(1._dp + p3p*rs + p2p*rs**2)**2)

   END FUNCTION

! **************************************************************************************************
!> \brief ...
!> \param x ...
!> \return ...
! **************************************************************************************************
   ELEMENTAL FUNCTION Qrpa(x)
      REAL(KIND=dp), INTENT(IN)                          :: x
      REAL(KIND=dp)                                      :: Qrpa

      Qrpa = Acoul*LOG((1._dp + aQ2*x + bQ2*x**2 + cQ2*x**3)/(1._dp + aQ2*x + dQ2*x**2))

   END FUNCTION

! **************************************************************************************************
!> \brief ...
!> \param x ...
!> \return ...
! **************************************************************************************************
   ELEMENTAL FUNCTION Qrpad(x)
      REAL(KIND=dp), INTENT(IN)                          :: x
      REAL(KIND=dp)                                      :: Qrpad

      Qrpad = Acoul*((x*(bQ2*(2._dp + aQ2*x) + cQ2*x*(3._dp + 2._dp*aQ2*x) + dQ2*(-2._dp - aQ2*x + cQ2*x**3)))/ &
                     ((1._dp + aQ2*x + dQ2*x**2)*(1._dp + aQ2*x + bQ2*x**2 + cQ2*x**3)))

   END FUNCTION

! **************************************************************************************************
!> \brief ...
!> \param rs ...
!> \param z ...
!> \param mu ...
!> \param vxlrup ...
!> \param vxlrdown ...
! **************************************************************************************************
   ELEMENTAL SUBROUTINE exchangelr_lsd(rs, z, mu, exlr, vxlrup, vxlrdown)
      REAL(KIND=dp), INTENT(IN)                          :: rs, z, mu
      REAL(KIND=dp), INTENT(OUT)                         :: exlr, vxlrup, vxlrdown

      CHARACTER(len=*), PARAMETER :: routineN = 'exchangelr_lsd', &
                                     routineP = moduleN//':'//routineN

      REAL(KIND=dp)                                      :: derrs, derz, fx, fx1, x, y

      IF (z >= 1._dp) THEN
         x = rs*alpha*mu
         y = .5_dp**(4._dp/3._dp)*x
         fx = -((y*(-3._dp + 4._dp*y**2 + (2._dp - 4._dp*y**2)*EXP(-.25/y**2)) + SQRT(pi)*ERF(.5_dp/y))/pi)
         exlr = mu*fx
         vxlrup = mu*(x/(2._dp**(1._dp/3._dp)*pi) - x/(2._dp**(1._dp/3._dp)*pi)* &
                      EXP(-2._dp**(2._dp/3._dp)/x**2) - &
                      ERF(2._dp**(1._dp/3._dp)/x)/SQRT(pi))
         vxlrdown = 0._dp
      ELSE IF (z <= -1._dp) THEN
         x = rs*alpha*mu
         y = .5_dp**(4._dp/3._dp)*x
         fx = -((y*(-3._dp + 4._dp*y**2 + (2._dp - 4._dp*y**2)*EXP(-.25/y**2)) + SQRT(pi)*ERF(.5_dp/y))/pi)
         exlr = mu*fx
         vxlrdown = mu*(x/(2._dp**(1._dp/3._dp)*pi) - x/(2._dp**(1._dp/3._dp)*pi)* &
                        EXP(-2._dp**(2._dp/3._dp)/x**2) - &
                        ERF(2._dp**(1._dp/3._dp)/x)/SQRT(pi))
         vxlrup = 0._dp
      ELSE
         y = alpha/2._dp/(1.+z)**(1._dp/3._dp)*mu*rs
         fx = -((y*(-3._dp + 4._dp*y**2 + (2._dp - 4._dp*y**2)*EXP(-.25_dp/y**2)) + &
                 SQRT(pi)*ERF(.5_dp/y))/pi)
         exlr = (1._dp + z)*mu*fx/2._dp
         fx1 = (3._dp*(1._dp + (-4._dp + 4._dp*EXP(-.25_dp/y**2))*y**2))/pi
         derrs = alpha/4._dp*(1._dp + z)**(2._dp/3._dp)*mu**2*fx1
         derz = 1._dp/2._dp*mu*fx - 1._dp/6._dp*fx1*mu*y
         vxlrup = rs/3._dp*derrs + (z - 1._dp)*derz
         vxlrdown = rs/3._dp*derrs + (z + 1._dp)*derz

         y = alpha/2._dp/(1.-z)**(1._dp/3._dp)*mu*rs
         fx = -((y*(-3._dp + 4._dp*y**2 + (2._dp - 4._dp*y**2)*EXP(-.25_dp/y**2)) + &
                 SQRT(pi)*ERF(.5_dp/y))/pi)
         exlr = exlr + (1._dp - z)*mu*fx/2._dp
         fx1 = (3._dp*(1._dp + (-4._dp + 4._dp*EXP(-.25_dp/y**2))*y**2))/pi
         derrs = alpha/4._dp*(1._dp - z)**(2._dp/3._dp)*mu**2*fx1
         derz = -1._dp/2._dp*mu*fx + 1._dp/6._dp*fx1*mu*y
         vxlrup = vxlrup + rs/3._dp*derrs + (z - 1._dp)*derz
         vxlrdown = vxlrdown + rs/3._dp*derrs + (z + 1._dp)*derz

         vxlrup = exlr - vxlrup
         vxlrdown = exlr - vxlrdown
      ENDIF

   END SUBROUTINE

! **************************************************************************************************
!> \brief PW92 energy functional
!> \param rs ...
!> \param z ...
!> \param ec ...
!> \param ecd ...
!> \param ecz ...
! **************************************************************************************************
   ELEMENTAL SUBROUTINE ecPW_lsd(rs, z, ec, ecd, ecz)
      REAL(KIND=dp), INTENT(IN)                          :: rs, z
      REAL(KIND=dp), INTENT(OUT)                         :: ec, ecd, ecz

      CHARACTER(len=*), PARAMETER :: routineN = 'ecPW_lsd', routineP = moduleN//':'//routineN

      REAL(KIND=dp)                                      :: alfac(0:1), ec0(0:1), ec1(0:1), ff

      IF (ABS(z) >= 1._dp) THEN
         CALL calc_g(rs, 0, ec0, 0)
         CALL calc_g(rs, 1, ec1, 1)
         CALL calc_g(rs, -1, alfac, 0)
         alfac = -alfac

         ec = ec1(0)
         ecd = ec1(1)
         ecz = SIGN(-4._dp/f02*alfac(0) + (ec1(0) - ec0(0))*(4._dp + 2._dp**(4._dp/3._dp)/3._dp/ &
                                                             (2._dp**(1._dp/3._dp) - 1._dp)), z)
      ELSE
         ff = ((1._dp + z)**(4._dp/3._dp) + (1._dp - z)**(4._dp/3._dp) - &
               2._dp)/(2._dp**(4._dp/3._dp) - 2._dp)

         CALL calc_g(rs, 0, ec0, 1)
         CALL calc_g(rs, 1, ec1, 1)
         CALL calc_g(rs, -1, alfac, 1)
         alfac = -alfac

         ec = ec0(0) + alfac(0)*ff/f02*(1._dp - z**4) + (ec1(0) - ec0(0))*ff*z**4
         ecd = ec0(1) + alfac(1)*ff/f02*(1._dp - z**4) + (ec1(1) - ec0(1))*ff*z**4
         ecz = alfac(0)*(-4._dp*z**3)*ff/f02 + alfac(0)*(1._dp - z**4)/f02* &
               4._dp/3._dp*((1._dp + z)**(1._dp/3._dp) - (1._dp - z)**(1._dp/3._dp))/ &
               (2._dp**(4._dp/3._dp) - 2._dp) + (ec1(0) - ec0(0))*(4._dp*z**3*ff + &
                                                            4._dp/3._dp*((1._dp + z)**(1._dp/3._dp) - (1._dp - z)**(1._dp/3._dp))/ &
                                                                   (2._dp**(4._dp/3._dp) - 2._dp)*z**4)
      END IF

   END SUBROUTINE

@:calc_g()

END MODULE xc_sr_lda
