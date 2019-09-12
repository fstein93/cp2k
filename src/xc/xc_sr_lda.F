!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright (C) 2000 - 2019  CP2K developers group                                               !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief Calculates the short range correlation LDA energy
!> \par History
!>      18-MAR-2002, TCH, working version
!>      fawzi (04.2004)  : adapted to the new xc interface
!> \see functionals_utilities
! **************************************************************************************************
MODULE xc_sr_lda
   USE bibliography,                    ONLY: Paziani2006,&
                                              Toulouse2004,&
                                              cite_reference
   USE input_section_types,             ONLY: section_vals_type,&
                                              section_vals_val_get
   USE kinds,                           ONLY: dp
   USE mathconstants,                   ONLY: oorootpi,&
                                              rootpi,&
                                              pi,&
                                              sqrt2
   USE xc_derivative_set_types,         ONLY: xc_derivative_set_type,&
                                              xc_dset_get_derivative
   USE xc_derivative_types,             ONLY: xc_derivative_get,&
                                              xc_derivative_type
   USE xc_functionals_utilities,        ONLY: calc_fx,&
                                              calc_rs,&
                                              calc_z
   USE xc_rho_cflags_types,             ONLY: xc_rho_cflags_type
   USE xc_rho_set_types,                ONLY: xc_rho_set_get,&
                                              xc_rho_set_type
   USE xc_sr_gori_giorgi,               ONLY: lsdsr
#include "../base/base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'xc_sr_lda'

   PUBLIC :: sr_lda_info, sr_lda_eval, sr_lsd_eval

   REAL(KIND=dp), PRIVATE, PARAMETER :: alpha2 = (4.0_dp/9.0_dp/pi)**(2.0_dp/3.0_dp), &
                                        aa = 5.84605_dp, &
                                        cc = 3.91744_dp, &
                                        dd = 3.44851_dp, &
                                        qq = (-2.0_dp+2.0_dp*0.693147180559945_dp)/pi**2, &
                                        bb = 3.44851_dp-17.53815_dp*pi/(-4.0_dp+4.0_dp*0.693147180559945_dp), &
                                        g2_1 = 0.02267_dp, &
                                        g2_2 = 0.4319_dp, &
                                        g2_3 = 0.04_dp, &
                                        g0_1 = 0.0207_dp, &
                                        g0_2 = 0.08193_dp, &
                                        g0_3 = 0.01277_dp, &
                                        g0_4 = 0.001859_dp, &
                                        g0_5 = 0.7524_dp, &
                                        beta = 0.784949_dp, &
                                        d3_1 = 0.31_dp, &
                                        d3_2 = 4.95_dp, &
                                        d2_1 = 0.547_dp, &
                                        d2_2 = 0.388_dp, &
                                        d2_3 = 0.676_dp

   REAL(KIND=dp), DIMENSION(-1:1) :: A, a1, b1, b2, b3, b4
   REAL(KIND=dp), DIMENSION(-1:1) :: c0, c1, c2, c3
   REAL(KIND=dp), DIMENSION(-1:1) :: d0, d1
   REAL(KIND=dp), PARAMETER :: &
      fpp = 0.584822362263464620726223866376013788782_dp ! d^2f(0)/dz^2

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

      CHARACTER(len=3)                                   :: p_string

      CALL cite_reference(Toulouse2004)
      CALL cite_reference(Paziani2006)
      IF (PRESENT(reference)) THEN
         reference = "J. Toulouse, A. Savin, and H.-J. Flad," &
                     //" Int. J. Quantum Chem. 100, 1074-1056 (2004)" &
                     //"["//TRIM(p_string)//"]"
      END IF
      IF (PRESENT(shortform)) THEN
         shortform = "J. Toulouse et al., IJQC 100, 1074-1056 (2004)" &
                     //"["//TRIM(p_string)//"]"
      END IF
      IF (PRESENT(needs)) THEN
         IF (lsd) THEN
            needs%rho_spin = .TRUE.
         ELSE
            needs%rho = .TRUE.
         END IF
      END IF
      IF (PRESENT(max_deriv)) max_deriv = 2

      CALL test_sr_lda()

   END SUBROUTINE sr_lda_info

   SUBROUTINE test_sr_lda()

      REAL(KIND=dp) :: rhoa(67), rhob(67), mu(17), rs, zeta, rho, ed(0:5), e_lda, my_mu, my_rhoa, my_rhob, &
         e_lsd, e_gg, va_gg, va_lsd, v_lda, vb_gg, vb_lsd, err_e_lsd, err_va_lsd, err_vb_lsd, err_e_lda, err_v_lda, &
         err_rel_e_lsd, err_rel_va_lsd, err_rel_vb_lsd, err_rel_e_lda, err_rel_v_lda, sx, sc
      INTEGER :: i, j, k

      mu = (/0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0/)

      rhoa = 0.0_dp
      rhob = 0.0_dp
      DO i = 1, 67
         rhoa(i) = 2.0_dp**(i-33)
      END DO
      rhob = rhoa

      CALL perdew_wang_init()

      sx = 0.0_dp
      sc = 1.0_dp
      
      DO i = 1, 17
         my_mu = mu(i)

         err_e_lsd = 0.0_dp
         err_va_lsd = 0.0_dp
         err_vb_lsd = 0.0_dp

         err_e_lda = 0.0_dp
         err_v_lda = 0.0_dp

         err_rel_e_lsd = 0.0_dp
         err_rel_va_lsd = 0.0_dp
         err_rel_vb_lsd = 0.0_dp

         err_rel_e_lda = 0.0_dp
         err_rel_v_lda = 0.0_dp

         DO j = 1, 67
            my_rhoa = rhoa(j)
            DO k = j, 67
               my_rhob = rhob(k)
               rho = my_rhoa+my_rhob
               rs = (3.0_dp/4.0_dp/pi/rho)**(1.0_dp/3.0_dp)
               zeta = (my_rhoa-my_rhob)/rho

               CALL lsdsr(rs, zeta, my_mu, e_gg, va_gg, vb_gg, sx, sc)
               va_gg = va_gg
               vb_gg = vb_gg
               e_gg = e_gg

               ! Test LSD
               CALL ec_lsd_sr(my_rhoa, my_rhob, my_mu, ed, 1)
               e_lsd = sc*ed(0)
               va_lsd = sc*(ed(0)+rho*ed(1))
               vb_lsd = sc*(ed(0)+rho*ed(2))

               CALL ex_lda_sr(2.0_dp*my_rhoa, my_mu, ed(0:2), 1)
               e_lsd = e_lsd+ed(0)*sx*0.5_dp
               va_lsd = va_lsd+ed(1)*sx

               CALL ex_lda_sr(2.0_dp*my_rhob, my_mu, ed(0:2), 1)
               e_lsd = e_lsd+ed(0)*sx*0.5_dp
               vb_lsd = vb_lsd+ed(1)*sx
WRITE(*,*) rho, rs, zeta, e_gg, va_gg, vb_gg, e_lsd, va_lsd, vb_lsd

               err_e_lsd = err_e_lsd+(e_gg-e_lsd)**2
               err_va_lsd = err_va_lsd+(va_gg-va_lsd)**2
               err_vb_lsd = err_vb_lsd+(vb_gg-vb_lsd)**2

               IF (ABS(e_gg) > 0.0_dp) err_rel_e_lsd = err_rel_e_lsd+(1.0_dp-e_lsd/e_gg)**2
               IF (ABS(va_gg) > 0.0_dp) err_rel_va_lsd = err_rel_va_lsd+(1.0_dp-va_lsd/va_gg)**2
               IF (ABS(vb_gg) > 0.0_dp) err_rel_vb_lsd = err_rel_vb_lsd+(1.0_dp-vb_lsd/vb_gg)**2

               IF (j == k) THEN
                  CALL ec_lda_sr(rho, my_mu, ed, 1)
                  e_lda = sc*ed(0)
                  v_lda = sc*(rho*ed(1)+ed(0))

                  CALL ex_lda_sr(rho, my_mu, ed(0:2), 1)
                  e_lda = e_lda+ed(0)*sx
                  v_lda = v_lda+ed(1)*sx
WRITE(*,*) rho, e_gg, va_gg, vb_gg, e_lda, v_lda

                  err_e_lda = err_e_lda+(e_gg-e_lda)**2
                  err_v_lda = err_v_lda+(va_gg-v_lda)**2

                  IF (ABS(e_gg) > 0.0_dp) err_rel_e_lda = err_rel_e_lda+(1.0_dp-e_lda/e_gg)**2
                  IF (ABS(va_gg) > 0.0_dp) err_rel_v_lda = err_rel_v_lda+(1.0_dp-v_lda/va_gg)**2

               END IF
            END DO
         END DO

         err_e_lsd = SQRT(err_e_lsd/2278.0_dp)
         err_va_lsd = SQRT(err_va_lsd/2278.0_dp)
         err_vb_lsd = SQRT(err_vb_lsd/2278.0_dp)

         err_e_lda = SQRT(err_e_lda/67.0_dp)
         err_v_lda = SQRT(err_v_lda/67.0_dp)

         err_rel_e_lsd = SQRT(err_rel_e_lsd/2278.0_dp)
         err_rel_va_lsd = SQRT(err_rel_va_lsd/2278.0_dp)
         err_rel_vb_lsd = SQRT(err_rel_vb_lsd/2278.0_dp)

         err_rel_e_lda = SQRT(err_rel_e_lda/67.0_dp)
         err_rel_v_lda = SQRT(err_rel_v_lda/67.0_dp)

         WRITE(*,*)
         WRITE(*,*) "Mu no.", i, "value: ", my_mu
         WRITE(*,*) "err_e_lsd", err_e_lsd
         WRITE(*,*) "err_va_lsd", err_va_lsd
         WRITE(*,*) "err_vb_lsd", err_vb_lsd
         WRITE(*,*) "err_e_lda", err_e_lda
         WRITE(*,*) "err_v_lda", err_v_lda
         WRITE(*,*) "err_rel_e_lsd", err_rel_e_lsd
         WRITE(*,*) "err_rel_va_lsd", err_rel_va_lsd
         WRITE(*,*) "err_rel_vb_lsd", err_rel_vb_lsd
         WRITE(*,*) "err_rel_e_lda", err_rel_e_lda
         WRITE(*,*) "err_rel_v_lda", err_rel_v_lda
      END DO

   END SUBROUTINE test_sr_lda

! **************************************************************************************************
!> \brief Initializes the functionals
! **************************************************************************************************
   SUBROUTINE perdew_wang_init()

      CHARACTER(len=*), PARAMETER :: routineN = 'perdew_wang_init', &
         routineP = moduleN//':'//routineN

      INTEGER                                            :: k

      ! values for -ac are the same for all methods
      A(-1) = 0.016887_dp
      a1(-1) = 0.11125_dp
      b1(-1) = 10.357_dp
      b2(-1) = 3.6231_dp
      b3(-1) = 0.88026_dp
      b4(-1) = 0.49671_dp

      A(0) = 0.031091_dp; A(1) = 0.015545_dp
      a1(0) = 0.21370_dp; a1(1) = 0.20548_dp
      b1(0) = 7.5957_dp; b1(1) = 14.1189_dp
      b2(0) = 3.5876_dp; b2(1) = 6.1977_dp
      b3(0) = 1.6382_dp; b3(1) = 3.3662_dp
      b4(0) = 0.49294_dp; b4(1) = 0.62517_dp

      DO k = -1, 1, 1
         c0(k) = A(k)
         c1(k) = -2.0_dp*c0(k)*LOG(2.0_dp*A(k)*b1(k))
         c2(k) = A(k)*a1(k)
         c3(k) = -2.0_dp*A(k)*(a1(k)*LOG(2.0_dp*A(k)*b1(k)) &
                               -(b2(k)/b1(k))**2+(b3(k)/b1(k)))
         d0(k) = a1(k)/b4(k)
         d1(k) = a1(k)*b3(k)/(b4(k)**2)
      END DO

   END SUBROUTINE perdew_wang_init

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

      INTEGER                                            :: npoints, timer_handle
      INTEGER, DIMENSION(:, :), POINTER                  :: bo
      REAL(KIND=dp)                                      :: omega, rho_cutoff, sc, sx
      REAL(KIND=dp), DIMENSION(:, :, :), POINTER         :: dummy, e_0, e_rho, e_rho_rho, rho
      TYPE(xc_derivative_type), POINTER                  :: deriv

      CALL timeset(routineN, timer_handle)

      CALL section_vals_val_get(sr_section, 'SCALE_X', r_val=sx)
      CALL section_vals_val_get(sr_section, 'SCALE_C', r_val=sc)
      CALL section_vals_val_get(sr_section, 'OMEGA', r_val=omega)

      CALL perdew_wang_init()

      NULLIFY (bo, rho, e_0, e_rho, e_rho_rho, dummy)
      CPASSERT(ASSOCIATED(rho_set))
      CPASSERT(rho_set%ref_count > 0)
      CPASSERT(ASSOCIATED(deriv_set))
      CPASSERT(deriv_set%ref_count > 0)
      CALL xc_rho_set_get(rho_set, rho=rho, &
                          local_bounds=bo)
      npoints = (bo(2, 1)-bo(1, 1)+1)*(bo(2, 2)-bo(1, 2)+1)*(bo(2, 3)-bo(1, 3)+1)

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

      CALL timestop(timer_handle)

   END SUBROUTINE sr_lda_eval

! **************************************************************************************************
!> \brief ...
!> \param rho ...
!> \param omega ...
!> \param rho_cutoff ...
!> \param e_0 ...
!> \param e_rho ...
!> \param e_rho_rho ...
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

      INTEGER                                            :: abs_order, k
      REAL(KIND=dp)                                      :: my_rho
      REAL(KIND=dp), DIMENSION(0:2)                      :: ed

      abs_order = ABS(order)

!$OMP PARALLEL DO PRIVATE (k, ed, my_rho) DEFAULT(NONE)&
!$OMP SHARED(npoints,rho,rho_cutoff,abs_order,omega,e_0,e_rho,e_rho_rho,order,sc,sx)
      DO k = 1, npoints

         my_rho = rho(k)
         IF (rho(k) > rho_cutoff) THEN
!! order_ is positive as it must be in this case:
!! ec(:,2) needs ed(:,1) for example
            IF (sc /= 0.0_dp) THEN
               CALL ec_lda_sr(my_rho, omega, ed, abs_order)
               ed = sc*ed

               IF (order >= 0) THEN
                  e_0(k) = e_0(k)+ed(0)
               END IF
               IF (order >= 1 .OR. order == -1) THEN
                  e_rho(k) = e_rho(k)+ed(0)+my_rho*ed(1)
               END IF
               IF (order >= 2 .OR. order == -2) THEN
                  e_rho_rho(k) = e_rho_rho(k)+2.0_dp*ed(1)+my_rho*ed(2)
               END IF
            END IF

            IF (sx /= 0.0_dp) THEN
               CALL ex_lda_sr(my_rho, omega, ed, abs_order)
               ed = sx*ed

               IF (order >= 0) THEN
                  e_0(k) = e_0(k)+ed(0)
               END IF
               IF (order >= 1 .OR. order == -1) THEN
                  e_rho(k) = e_rho(k)+ed(1)
               END IF
               IF (order >= 2 .OR. order == -2) THEN
                  e_rho_rho(k) = e_rho_rho(k)+ed(2)
               END IF

            END IF

         END IF

      END DO
!$OMP END PARALLEL DO

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

      INTEGER                                            :: npoints, timer_handle
      INTEGER, DIMENSION(:, :), POINTER                  :: bo
      REAL(KIND=dp)                                      :: omega, rho_cutoff, sc, sx
      REAL(KIND=dp), DIMENSION(:, :, :), POINTER         :: a, b, dummy, e_0, ea, eaa, eab, eb, ebb
      TYPE(xc_derivative_type), POINTER                  :: deriv

      CALL timeset(routineN, timer_handle)

      CALL section_vals_val_get(sr_section, 'SCALE_X', r_val=sx)
      CALL section_vals_val_get(sr_section, 'SCALE_C', r_val=sc)
      CALL section_vals_val_get(sr_section, 'OMEGA', r_val=omega)

      CALL perdew_wang_init()

      NULLIFY (bo, a, b, e_0, ea, eb, eaa, eab, ebb)
      CPASSERT(ASSOCIATED(rho_set))
      CPASSERT(rho_set%ref_count > 0)
      CPASSERT(ASSOCIATED(deriv_set))
      CPASSERT(deriv_set%ref_count > 0)
      CALL xc_rho_set_get(rho_set, rhoa=a, rhob=b, &
                          local_bounds=bo, rho_cutoff=rho_cutoff)
      npoints = (bo(2, 1)-bo(1, 1)+1)*(bo(2, 2)-bo(1, 2)+1)*(bo(2, 3)-bo(1, 3)+1)

      ! meaningful default for the arrays we don't need: let us make compiler
      ! and debugger happy...
      dummy => a

      e_0 => dummy
      ea => dummy; eb => dummy
      eaa => dummy; eab => dummy; ebb => dummy

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

      CALL timestop(timer_handle)

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

      INTEGER                                            :: abs_order, k
      REAL(KIND=dp)                                      :: my_rhoa, my_rhob, rho
      REAL(KIND=dp), DIMENSION(0:5)                      :: ed

      abs_order = ABS(order)

!$OMP PARALLEL DO PRIVATE (k, rho, ed, my_rhoa, my_rhob) DEFAULT(NONE)&
!$OMP SHARED(npoints,rhoa,rhob,rho_cutoff,abs_order,omega,order,e_0,ea,eb,eaa,eab,ebb,sx,sc)
      DO k = 1, npoints

         my_rhoa = rhoa(k)
         my_rhob = rhob(k)
         rho = my_rhoa+my_rhob
         IF (rho > rho_cutoff) THEN

            IF (sc /= 0.0_dp) THEN
               CALL ec_lsd_sr(my_rhoa, my_rhob, omega, ed, abs_order)
               ed = ed*sc
               IF (order >= 0) THEN
                  e_0(k) = e_0(k)+ed(0)
               END IF
               IF (order >= 1 .OR. order == -1) THEN
                  ea(k) = ea(k)+ed(1)
                  eb(k) = eb(k)+ed(2)
               END IF
               IF (order >= 2 .OR. order == -2) THEN
                  eaa(k) = eaa(k)+ed(3)
                  eab(k) = eab(k)+ed(4)
                  ebb(k) = ebb(k)+ed(5)
               END IF
            END IF
            IF (sx /= 0.0_dp) THEN
               IF (2.0_dp*my_rhoa > rho_cutoff) THEN
                  CALL ex_lda_sr(2.0_dp*my_rhoa, omega, ed(0:2), abs_order)
                  ed(0:2) = ed(0:2)*sx
                  IF (order >= 0) THEN
                     e_0(k) = e_0(k)+ed(0)*0.5_dp
                  END IF
                  IF (order >= 1 .OR. order == -1) THEN
                     ea(k) = ea(k)+ed(1)
                  END IF
                  IF (order >= 2 .OR. order == -2) THEN
                     eaa(k) = eaa(k)+ed(2)
                  END IF
               END IF
               IF (2.0_dp*my_rhob > rho_cutoff) THEN
                  CALL ex_lda_sr(2.0_dp*my_rhob, omega, ed(0:2), abs_order)
                  ed(0:2) = ed(0:2)*sx
                  IF (order >= 0) THEN
                     e_0(k) = e_0(k)+ed(0)*0.5_dp
                  END IF
                  IF (order >= 1 .OR. order == -1) THEN
                     eb(k) = eb(k)+ed(1)
                  END IF
                  IF (order >= 2 .OR. order == -2) THEN
                     ebb(k) = ebb(k)+ed(2)
                  END IF
               END IF
            END IF

         END IF

      END DO

   END SUBROUTINE sr_lsd_calc

! **************************************************************************************************
!> \brief ...
!> \param r ...
!> \param z ...
!> \param g ...
!> \param order ...
! **************************************************************************************************
   SUBROUTINE calc_g(r, z, g, order)

!   Calculates g and its derivatives wrt r up to 3rd order, where:
!
!   g = .... for r < 1
!   g = .... for r > 100 and everywhere else
!   g = 2A(1+a1*r)ln(1+1/(2A(b1*r^1/2 + b2*r + b3*r^(3/2) + b4*r^2))).

      REAL(KIND=dp), INTENT(IN)                          :: r
      INTEGER, INTENT(IN)                                :: z
      REAL(KIND=dp), DIMENSION(0:), INTENT(OUT)          :: g
      INTEGER, INTENT(IN)                                :: order

      REAL(KIND=dp)                                      :: a1_, A_, b1_, b2_, b3_, b4_, rr, rsr, &
                                                            sr, t11, t12, t15, t16, t20, t3, t40

      A_ = A(z); a1_ = a1(z)
      b1_ = b1(z); b2_ = b2(z); b3_ = b3(z); b4_ = b4(z)

      sr = SQRT(r)
      rsr = r*sr
      rr = r*r

      IF (r < 1.0_dp) THEN

         ! order 0 must always be calculated
         g(0) = c0(z)*LOG(r)-c1(z)+c2(z)*r*LOG(r)-c3(z)*r
         IF (order >= 1) g(1) = c0(z)/r+c2(z)*LOG(r)+c2(z)-c3(z)
         IF (order >= 2) g(2) = -c0(z)/rr+c2(z)/r
         IF (order >= 3) g(3) = 2.0_dp*c0(z)/(rr*r)-c2(z)/rr

      ELSE IF (r <= 100.0_dp) THEN

         t3 = 1.0_dp+a1_*r
         t11 = b1_*sr+b2_*r+b3_*rsr+b4_*rr
         t12 = t11**2
         t15 = 1.0_dp+0.5_dp/A_/t11
         t16 = LOG(t15)
         t20 = 0.5_dp*b1_/sr+b2_+1.5_dp*b3_*sr+2.0_dp*b4_*r

         ! order 0 must always be calculated
         g(0) = -2.0_dp*A_*t3*t16

         IF (order >= 1) THEN

            g(1) = -2.0_dp*A_*a1_*t16+t3*t20/(t12*t15)

         END IF

         IF (order >= 2) THEN

            t40 = -0.25_dp*b1_/rsr+0.75_dp*b3_/sr+2.0_dp*b4_

            g(2) = 2.0_dp*a1_*t20/(t12*t15) &
                   -2.0_dp*(t20**2)*t3/(t12*t11*t15) &
                   +t3*t40/(t12*t15) &
                   +0.5_dp*t3*(t20**2)/(A_*(t12**2)*(t15**2))

         END IF

      ELSE

         ! order 0 must always be calculated
         g(0) = -d0(z)/r+d1(z)/rsr
         IF (order >= 1) g(1) = d0(z)/rr-1.5_dp*d1(z)/(rsr*r)
         IF (order >= 2) g(2) = -2.0_dp*d0(z)/(rr*r)+3.75_dp*d1(z)/(rsr*rr)

      END IF

   END SUBROUTINE calc_g

! **************************************************************************************************
!> \brief ...
!> \param rho ...
!> \param ed ...
!> \param order ...
! **************************************************************************************************
   SUBROUTINE pw_lda_ed_loc(rho, ed, order)

      REAL(KIND=dp), INTENT(IN)                          :: rho
      REAL(KIND=dp), DIMENSION(0:2), INTENT(OUT)         :: ed
      INTEGER, INTENT(IN)                                :: order

      INTEGER                                            :: m, order_
      LOGICAL, DIMENSION(0:2)                            :: calc
      REAL(KIND=dp), DIMENSION(0:2)                      :: e0, r

      ed = 0
      calc = .FALSE.

      IF (order >= 0) THEN
         calc(0:order) = .TRUE.
      ELSE
         order_ = -1*order_
         calc(order) = .TRUE.
      END IF

      CALL calc_rs(rho, r(0))
      CALL calc_g(r(0), 0, e0, order_)

      IF (order >= 1) r(1) = (-1.0_dp/3.0_dp)*r(0)/rho
      IF (order >= 2) r(2) = (-4.0_dp/3.0_dp)*r(1)/rho

      m = 0
      IF (calc(0)) THEN
         ed(m) = e0(0)
         m = m+1
      END IF
      IF (calc(1)) THEN
         ed(m) = e0(1)*r(1)
         m = m+1
      END IF
      IF (calc(2)) THEN
         ed(m) = e0(2)*r(1)**2+e0(1)*r(2)
         m = m+1
      END IF

   END SUBROUTINE pw_lda_ed_loc

! **************************************************************************************************
!> \brief ...
!> \param a ...
!> \param b ...
!> \param ed ...
!> \param order ...
! **************************************************************************************************
   SUBROUTINE pw_lsd_ed_loc(a, b, ed, order)

      REAL(KIND=dp), INTENT(IN)                          :: a, b
      REAL(KIND=dp), DIMENSION(0:2), INTENT(OUT)         :: ed
      INTEGER, INTENT(IN)                                :: order

      INTEGER                                            :: m, order_
      LOGICAL, DIMENSION(0:2)                            :: calc
      REAL(KIND=dp)                                      :: rho, tr, trr, trz, tz, tzz
      REAL(KIND=dp), DIMENSION(0:2)                      :: ac, e0, e1, f, r
      REAL(KIND=dp), DIMENSION(0:2, 0:2)                 :: z

      order_ = order
      calc = .FALSE.

      IF (order_ > 0) THEN
         calc(0:order_) = .TRUE.
      ELSE
         order_ = -1*order_
         calc(order_) = .TRUE.
      END IF

      rho = a+b

      CALL calc_fx(a, b, f(0:order_), order_)
      CALL calc_rs(rho, r(0))
      CALL calc_g(r(0), -1, ac(0:order_), order_)
      CALL calc_g(r(0), 0, e0(0:order_), order_)
      CALL calc_g(r(0), 1, e1(0:order_), order_)
      CALL calc_z(a, b, z(0:order_, 0:order_), order_)

!! calculate first partial derivatives
      IF (order_ >= 1) THEN
         r(1) = (-1.0_dp/3.0_dp)*r(0)/rho
         tr = e0(1) &
              +fpp*ac(1)*f(0) &
              -fpp*ac(1)*f(0)*z(0, 0)**4 &
              +(e1(1)-e0(1))*f(0)*z(0, 0)**4
         tz = fpp*ac(0)*f(1) &
              -fpp*ac(0)*f(1)*z(0, 0)**4 &
              -fpp*ac(0)*f(0)*4.0_dp*z(0, 0)**3 &
              +(e1(0)-e0(0))*f(1)*z(0, 0)**4 &
              +(e1(0)-e0(0))*f(0)*4.0_dp*z(0, 0)**3
      END IF

!! calculate second partial derivatives
      IF (order_ >= 2) THEN
         r(2) = (-4.0_dp/3.0_dp)*r(1)/rho
         trr = e0(2) &
               +fpp*ac(2)*f(0) &
               -fpp*ac(2)*f(0)*z(0, 0)**4 &
               +(e1(2)-e0(2))*f(0)*z(0, 0)**4
         trz = fpp*ac(1)*f(1) &
               -fpp*ac(1)*f(1)*z(0, 0)**4 &
               -fpp*ac(1)*f(0)*4.0_dp*z(0, 0)**3 &
               +(e1(1)-e0(1))*f(1)*z(0, 0)**4 &
               +(e1(1)-e0(1))*f(0)*4.0_dp*z(0, 0)**3
         tzz = fpp*ac(0)*f(2) &
               -fpp*ac(0)*f(2)*z(0, 0)**4 &
               -fpp*ac(0)*f(1)*8.0_dp*z(0, 0)**3 &
               -fpp*ac(0)*f(0)*12.0_dp*z(0, 0)**2 &
               +(e1(0)-e0(0))*f(2)*z(0, 0)**4 &
               +(e1(0)-e0(0))*f(1)*8.0_dp*z(0, 0)**3 &
               +(e1(0)-e0(0))*f(0)*12.0_dp*z(0, 0)**2
      END IF

      m = 0
      IF (calc(0)) THEN
         ed(m) = e0(0) &
                 +fpp*ac(0)*f(0)*(1.0_dp-z(0, 0)**4) &
                 +(e1(0)-e0(0))*f(0)*z(0, 0)**4
         m = m+1
      END IF
      IF (calc(1)) THEN
         ed(m) = tr*r(1)+tz*z(1, 0)
         ed(m+1) = tr*r(1)+tz*z(0, 1)
         m = m+2
      END IF
      IF (calc(2)) THEN
         ed(m) = trr*r(1)**2+2.0_dp*trz*r(1)*z(1, 0) &
                 +tr*r(2)+tzz*z(1, 0)**2+tz*z(2, 0)
         ed(m+1) = trr*r(1)**2+trz*r(1)*(z(0, 1)+z(1, 0)) &
                   +tr*r(2)+tzz*z(1, 0)*z(0, 1)+tz*z(1, 1)
         ed(m+2) = trr*r(1)**2+2.0_dp*trz*r(1)*z(0, 1) &
                   +tr*r(2)+tzz*z(0, 1)**2+tz*z(0, 2)
         m = m+3
      END IF

   END SUBROUTINE pw_lsd_ed_loc

   PURE SUBROUTINE calc_D3(rs, D3, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs
      REAL(KIND=dp), INTENT(OUT) :: D3(0:)
      INTEGER, INTENT(IN) :: max_deriv

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_D3", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7

      D3 = 0.0_dp

      x0 = -d3_2 + rs
      x1 = rs**(-2)
      x2 = exp(-d3_1*rs)
      x3 = x1*x2
      x4 = x0*x3
      
      D3(0) = x4

      IF (max_deriv >= 1) THEN
         x5 = x2/rs**3
         x6 = d3_1*x1*x2

         D3(1) = -x0*x6 + x3 + 2.0_dp*x5*(d3_2 - rs)
      END IF

      IF (max_deriv >= 2) THEN
         x7 = 4.0_dp*x5

         D3(2) = d3_1**2*x4 + d3_1*x0*x7 - 2.0_dp*x6 - x7 + 6.0_dp*x0*x2/rs**4
      END IF

   END SUBROUTINE calc_D3

   PURE SUBROUTINE calc_D2(rs, D2, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs
      REAL(KIND=dp), INTENT(OUT) :: D2(0:)
      INTEGER, INTENT(IN) :: max_deriv

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_D2", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6

      D2 = 0.0_dp

      x0 = 1.0_dp/rs
      x1 = exp(-d2_1*rs)
      x2 = x1*(-d2_2 + d2_3*rs)
      x3 = x0*x2
      
      D2(0) = x3

      IF (max_deriv >= 1) THEN
         x4 = d2_3*x1
         x5 = x0*x4
         x6 = rs**(-2)

         D2(1) = -d2_1*x3 - x2*x6 + x5
      END IF

      IF (max_deriv >= 2) THEN 
         D2(2) = d2_1**2*x3 + 2.0_dp*(d2_1*x2*x6 - d2_1*x5 - x4*x6 + x2/rs**3)
      END IF

   END SUBROUTINE calc_D2

   PURE SUBROUTINE calc_g2z_ab(rs, zeta, g2z, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      REAL(KIND=dp), INTENT(OUT) :: g2z(0:)
      INTEGER, INTENT(IN) :: max_deriv

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_g2z_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46

      g2z = 0.0_dp
      
      IF (zeta > -1.0_dp) THEN

      x0 = (1.0_dp/2.0_dp)*zeta + 1.0_dp/2.0_dp
      x1 = x0**2
      x2 = alpha**(-2)
      x3 = rs**2
      x4 = 1.0_dp/x3
      x5 = zeta + 1.0_dp
      x6 = 1.0_dp/x5
      x7 = x6**(2.0_dp/3.0_dp)
      x8 = 1.0_dp/x7
      x9 = 2.0_dp**(1.0_dp/3.0_dp)
      x10 = x6**(1.0_dp/3.0_dp)
      x11 = -g2_1*rs*x10*x9 + 1.0_dp
      x12 = g2_2*x10*x9
      x13 = rs*x12
      x14 = 2.0_dp**(2.0_dp/3.0_dp)
      x15 = g2_3*x14*x3*x7
      x16 = x13 + x15 + 1.0_dp
      x17 = 1.0_dp/x16
      x18 = (2.0_dp/5.0_dp)*x11*x17*x2*x4*x8
      
      g2z(0) = x1*x18

      IF (max_deriv >= 1) THEN
         x19 = 1.0_dp/x10
         x20 = (2.0_dp/5.0_dp)*g2_1*x17*x19*x2*x4*x9
         x21 = x1*x20
         x22 = rs**(-3)
         x23 = x11*x17*x2*x22*x8
         x24 = x16**(-2)
         x25 = x1*x24
         x26 = g2_3*rs*x14*x7
         x27 = -x12 - 2.0_dp*x26
         x28 = (2.0_dp/5.0_dp)*x11*x2*x27*x4*x8
         x29 = 1.0_dp/rs
         x30 = g2_1*x1*x17*x19*x2*x9
         x31 = x1*x11*x17*x2*x4*x8
         x32 = (1.0_dp/3.0_dp)*x6
         x33 = (2.0_dp/3.0_dp)*x6
         x34 = x13*x32 + x15*x33
         x35 = (2.0_dp/5.0_dp)*x1*x11*x2*x24*x4*x8

         g2z(1) = -4.0_dp/5.0_dp*x1*x23 - x21 + x25*x28
         g2z(2) = x0*x18 + (2.0_dp/15.0_dp)*x29*x30*x6 + (4.0_dp/15.0_dp)*x31*x6 + &
                  x34*x35
      END IF

      IF (max_deriv >= 2) THEN
         x36 = x16**(-3)
         x37 = (2.0_dp/5.0_dp)*x1*x11*x2*x27*x36*x4*x8
         x38 = (2.0_dp/5.0_dp)*zeta + 2.0_dp/5.0_dp
         x39 = (8.0_dp/15.0_dp)*x1*x6
         x40 = g2_1*x1*x19*x2*x24*x34*x9
         x41 = (4.0_dp/3.0_dp)*x6
         x42 = x34*x8
         x43 = x13*x33 + x15*x41
         x44 = (4.0_dp/15.0_dp)*x29*x6
         x45 = x5**(-2)
         x46 = x11*x2*x24*x34*x4*x8

         g2z(3) = -4.0_dp/5.0_dp*g2_1*x1*x19*x2*x24*x27*x4*x9 - 4.0_dp/5.0_dp* &
                  g2_3*x1*x11*x14*x2*x24*x4 - 8.0_dp/5.0_dp*x1*x11*x2*x22*x24*x27*x8 &
                  + (8.0_dp/5.0_dp)*x22*x30 + x37*(-2.0_dp*x12 - 4.0_dp*x26) + (12.0_dp/5.0_dp)* &
                  x1*x11*x17*x2*x8/rs**4
         g2z(4) = (2.0_dp/15.0_dp)*g2_1*x19*x2*x25*x27*x29*x6*x9 - x0*x20 + &
                  x0*x24*x28 - 4.0_dp/5.0_dp*x1*x11*x2*x22*x24*x42 + (4.0_dp/15.0_dp)* &
                  x1*x11*x2*x24*x27*x4*x6*x8 - x21*x6 - x23*x38 - x23*x39 + x35*( &
                  x12*x32 + x26*x41) + x37*x43 - 2.0_dp/5.0_dp*x4*x40
         g2z(5) = g2_1*x0*x17*x19*x2*x44*x9 + (8.0_dp/15.0_dp)*x0*x11*x17* &
                  x2*x4*x6*x8 + (2.0_dp/5.0_dp)*x1*x11*x2*x36*x4*x42*x43 + (1.0_dp/ &
                  5.0_dp)*x11*x17*x2*x4*x8 - 4.0_dp/45.0_dp*x31*x45 + x35*(-4.0_dp/ &
         9.0_dp*x13*x45 - 10.0_dp/9.0_dp*x15*x45) + x38*x46 + x39*x46 + x40*x44
      END IF
      
      END IF

   END SUBROUTINE calc_g2z_ab

   PURE SUBROUTINE calc_g0(rs, g0, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs
      REAL(KIND=dp), INTENT(OUT) :: g0(0:)
      INTEGER, INTENT(IN) :: max_deriv

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_g0", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5
      
      g0 = 0.0_dp

      x0 = exp(-g0_5*rs)
      x1 = (1.0_dp/2.0_dp)*g0_1
      x2 = rs**2
      x3 = rs**3
      x4 = 0.5_dp*x0*(g0_2*x2 - g0_3*x3 + g0_4*rs**4 + 2.0_dp*rs*x1 + 1.0_dp)
      x5 = x0*(g0_2*rs - 3.0_dp/2.0_dp*g0_3*x2 + 2.0_dp*g0_4*x3 + x1)
      
      g0(0) = x4

      IF (max_deriv >= 1) THEN
         x5 = x0*(g0_2*rs - 3.0_dp/2.0_dp*g0_3*x2 + 2.0_dp*g0_4*x3 + x1)

         g0(1) = -g0_5*x4 + x5
      END IF

      IF (max_deriv >= 2) THEN
         g0(2) = g0_5**2*x4 - 2.0_dp*g0_5*x5 + x0*(g0_2 - 3.0_dp*g0_3*rs + 6.0_dp*g0_4*x2)
      END IF

   END SUBROUTINE calc_g0

   PURE SUBROUTINE calc_phin(zeta, n, phin, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: zeta
      INTEGER, INTENT(IN) :: n
      REAL(KIND=dp), INTENT(OUT) :: phin(0:)
      INTEGER, INTENT(IN) :: max_deriv

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_phin", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9

      phin = 0.0_dp

      IF (ABS(zeta)>=1.0_dp) THEN
         ! If the density is fully spin-polarized, we have to use different formulae for numerical reasons
         If (n>-1) phin(0) = 2.0_dp**(REAL(n-3, KIND=dp)/3.0_dp)
         IF (n>2 .AND. max_deriv >=1) phin(1) = SIGN(zeta)*REAL(n, KIND=dp)/3.0_dp*2.0_dp**(REAL(n-6, KIND=dp)/3.0_dp)
         IF (n>5 .AND. max_deriv >=2) phin(2) = REAL(n*(n-3), KIND=dp)/9.0_dp*2.0_dp**(REAL(n-9, KIND=dp)/3.0_dp)
      ELSE

      x0 = zeta + 1.0_dp
      x1 = (1.0_dp/3.0_dp)*REAL(n, KIND=dp)
      x2 = x0**x1
      x3 = -zeta + 1.0_dp
      x4 = x3**x1
      
      phin(0) = (1.0_dp/2.0_dp)*(x2 + x4)

      IF (max_deriv >= 1) THEN
         x5 = (1.0_dp/6.0_dp)*n*x2
         x6 = (1.0_dp/6.0_dp)*n*x4

         phin(1) = -x6/x3 + x5/x0
      END IF

      IF (max_deriv >= 2) THEN
         x7 = x0**(-2)
         x8 = (1.0_dp/18.0_dp)*n**2
         x9 = x3**(-2)

         phin(2) = x2*x7*x8 + x4*x8*x9 - x5*x7 - x6*x9
      END IF
      END IF

   END SUBROUTINE calc_phin

   PURE SUBROUTINE calc_C2_ab(rs, zeta, C2, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      REAL(KIND=dp), INTENT(OUT) :: C2(0:)
      INTEGER, INTENT(IN) :: max_deriv

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C2_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, g0(0:2)

      CALL calc_g0(rs, zeta, g0, max_deriv)

      C2 = 0.0_dp

      x0 = g0(0) - 1.0_dp/2.0_dp
      x1 = rs**(-3)
      x2 = zeta**2
      x3 = (3.0_dp/8.0_dp)*x2 - 3.0_dp/8.0_dp
      x4 = x1*x3
      x5 = rs**(-4)
      x6 = x0*x5
      x7 = (3.0_dp/4.0_dp)*x1*zeta
      
      C2(0) = x0*x4

      IF (max_deriv >= 1) THEN
         x5 = rs**(-4)
         x6 = x0*x5
         x7 = (3.0_dp/4.0_dp)*x1*zeta

         C2(1) = g0(1)*x4 - (9.0_dp/8.0_dp)*x6*(x2 - 1.0_dp)
         C2(2) = x0*x7
      END IF

      IF (max_deriv >= 2) THEN
         C2(3) = -6.0_dp*g0(1)*x3*x5 + g0(2)*x4 + 12.0_dp*x0*x3/rs**5
         C2(4) = g0(1)*x7 - 9.0_dp/4.0_dp*x6*zeta
         C2(5) = x1*((3.0_dp/4.0_dp)*g0(0) - 3.0_dp/8.0_dp)
      END IF

   END SUBROUTINE calc_C2_ab

   PURE SUBROUTINE calc_C3_ab(rs, zeta, C3, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      REAL(KIND=dp), INTENT(OUT) :: C3(0:)
      INTEGER, INTENT(IN) :: max_deriv

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C3", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x3, x4, x5, x6, x7, x8, x9, g0(0:2)

      CALL calc_g0(rs, zeta, g0, max_deriv)

      C3 = 0.0_dp

      x0 = zeta**2
      x3 = rs**(-3)
      x4 = oorootpi*root2/g0(0)*x3
      
      C3(0) = x4*((1.0_dp/2.0_dp)*x0 - 1.0_dp/2.0_dp)

      IF (max_deriv >= 1) THEN
      x5 = rs**(-4)
      x6 = x0 - 1.0_dp
      x7 = oorootpi/root2*x6
      x8 = g0(1)*x3
      x9 = (1.0_dp/2.0_dp)*oorootpi*root2*x6

      C3(1) = -3.0_dp/2.0_dp*g0(0)*x5*x7 + x8*x9
      C3(2) = x4*zeta
      END IF

      IF (max_deriv >= 2) THEN
      C3(3) = 6.0_dp*g0(0)*x7/rs**5 - 3.0_dp*g0(1)*x5*x7 + g0(2)*x3*x9
      C3(4) = -3.0_dp*oorootpi*root2*g0(0)*x5*zeta + oorootpi*root2*x8*zeta
      C3(5) = x4
      END IF

   END SUBROUTINE calc_C3_ab

   PURE SUBROUTINE calc_C4_ab(rs, zeta, C4, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      REAL(KIND=dp), INTENT(OUT) :: C4(0:)
      INTEGER, INTENT(IN) :: max_deriv

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C4_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, phi8(0:2), g2zp(0:2), g2zm(0:2), D2(0:2)
      
      CALL calc_phin(zeta, phi8, 8, max_deriv)
      CALL calc_g2z(rs, zeta, g2zp, max_deriv)
      CALL calc_g2z(rs, -zeta, g2zm, max_deriv)
      CALL calc_D2(rs, D2, max_deriv)

      C4 = 0.0_dp

      x0 = rs**(-3)
      x1 = alpha**(-2)
      x2 = rs**(-2)
      x3 = phi8(0)*x1*x2
      x4 = -zeta**2 + 1.0_dp
      x5 = D2(0)*x4
      
      C4(0) = x0*(-9.0_dp/64.0_dp*g2zm(0) - 9.0_dp/64.0_dp*g2zp(0) + (9.0_dp/320.0_dp)*x3 - &
      9.0_dp/64.0_dp*x5)
      
      IF (max_deriv >= 1) THEN
      x6 = rs**(-4)
      x7 = phi8(0)*x0*x1
      x8 = D2(1)*x4
      x9 = (9.0_dp/32.0_dp)*D2
      x10 = phi8(1)*x1*x2
      C4(1) = x0*(-9.0_dp/64.0_dp*g2zm(1) - 9.0_dp/64.0_dp*g2zp(1) - 9.0_dp/ &
      160.0_dp*x7 - 9.0_dp/64.0_dp*x8) + x6*((27.0_dp/64.0_dp)*g2zm(0) + ( &
      27.0_dp/64.0_dp)*g2zp(0) - 27.0_dp/320.0_dp*x3 + (27.0_dp/64.0_dp)*x5)
      C4(2) = x0*(-9.0_dp/64.0_dp*g2zm(2) - 9.0_dp/64.0_dp*g2zp(2) + (9.0_dp &
      /320.0_dp)*x10 + x9*zeta)
      END IF
      
      IF (max_deriv >= 2) THEN
      C4(3) = x0*(-9.0_dp/64.0_dp*D2(3)*x4 - 9.0_dp/64.0_dp*g2zm(3) - &
      9.0_dp/64.0_dp*g2zp(3) + (27.0_dp/160.0_dp)*phi8(0)*x1*x6) + x6*(( &
      27.0_dp/32.0_dp)*g2zm(1) + (27.0_dp/32.0_dp)*g2zp(1) + (27.0_dp/80.0_dp &
      )*x7 + (27.0_dp/32.0_dp)*x8) + (-27.0_dp/16.0_dp*g2zm(0) - 27.0_dp/16.0_dp &
      *g2zp(0) + (27.0_dp/80.0_dp)*x3 - 27.0_dp/16.0_dp*x5)/rs**5
      C4(4) = x0*((9.0_dp/32.0_dp)*D2(1)*zeta - 9.0_dp/64.0_dp*g2zm(4) &
      - 9.0_dp/64.0_dp*g2zp(4) - 9.0_dp/160.0_dp*phi8(1)*x0*x1) + x6 &
      *(-27.0_dp/32.0_dp*D2(0)*zeta + (27.0_dp/64.0_dp)*g2zm(2) + (27.0_dp/ &
      64.0_dp)*g2zp(2) - 27.0_dp/320.0_dp*x10)
      C4(5) = x0*(-9.0_dp/64.0_dp*g2zm(5) - 9.0_dp/64.0_dp* &
      g2zp(5) + (9.0_dp/320.0_dp)*phi8(2)*x1*x2 + x9)
      END IF

   END SUBROUTINE calc_C4_ab

   PURE SUBROUTINE calc_C5_ab(rs, zeta, C5, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      REAL(KIND=dp), INTENT(INOUT) :: C5(0:)
      INTEGER, INTENT(IN) :: max_deriv

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C5_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x2, x3, x4, x5, x6, x7, g2zp(0:2), g2zm(0:2), D3(0:2)
      
      CALL calc_g2z(rs, zeta, g2zp, max_deriv)
      CALL calc_g2z(rs, -zeta, g2zm, max_deriv)
      CALL calc_D3(rs, D3, max_deriv)
      
      C5 = 0.0_dp

      x2 = oorootpi*x0/rs**3
      x3 = -zeta**2 + 1.0_dp
      x4 = D3(0)*x3
      
      C5(0) = -9.0_dp/80.0_dp*x2*(g2zm(0) + g2zp(0) + x4)
      
      IF (max_deriv >= 1) THEN
      x5 = oorootpi*root2/rs**4
      x6 = D3(1)*x3
      x7 = (9.0_dp/40.0_dp)*D3(0)
      
      C5(1) = -9.0_dp/80.0_dp*x2*(g2zm(1) + g2zp(1) + x6)
       + (27.0_dp/80.0_dp)*x5*(g2zm(0) + g2zp(0) + x4)
      C5(2) = x2*(-9.0_dp/80.0_dp*g2zm(1) - 9.0_dp/80.0_dp*g2zp(1) + x7*zeta)
      END IF
      
      IF (max_deriv >= 2) THEN
      C5(3) = x2*(-9.0_dp/80.0_dp*D3(2)*x3 - 9.0_dp/80.0_dp*g2zm(3) - &
      9.0_dp/80.0_dp*g2zp(3)) + (27.0_dp/40.0_dp)*x5*(g2zm(1) + g2zp(1) + x6) &
      - 27.0_dp/20.0_dp*oorootpi*root2*(g2zm(0) + g2zp(0) + x4)/rs**5
      C5(4) = x2*((9.0_dp/40.0_dp)*D3(1)*zeta - 9.0_dp/80.0_dp*g2zm(4) &
      - 9.0_dp/80.0_dp*g2zp(4)) + x5*(-27.0_dp/40.0_dp*D3(0)*zeta + ( &
      27.0_dp/80.0_dp)*g2zm(2) + (27.0_dp/80.0_dp)*g2zp(2))
      C5(5) = x2*(-9.0_dp/80.0_dp*g2zm(5) - 9.0_dp/80.0_dp*g2zp(5) + x7)
      END IF

   END SUBROUTINE calc_C5_ab

   PURE SUBROUTINE calc_a1_ab(rs, zeta, a1, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a1(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a1_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, C3(0:5), C5(0:5)
      
      a1 = 0.0_dp

      CALL calc_C3_ab(rs, zeta, C3, max_deriv)
      CALL calc_C5_ab(rs, zeta, C5, max_deriv)

      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4.0_dp*x2
      x4 = beta**6
      x5 = C3(0)*x4
      
      a1(0) = C5(0)*x1 + x3*x5
      
      IF (max_deriv >=1) THEN
      x6 = rs**5
      x7 = 24_dp*x6
      x8 = C3(1)*x4
      x9 = rs**7
      x10 = 8_dp*x9
      x11 = C5(0)*x0
      x12 = C3(2)*x4
      
      a1(1) = C5(1)*x1 + x10*x11 + x3*x8 + x5*x7
      a1(2) = C5(2)*x1 + x12*x3
      END IF
      
      IF (max_deriv >= 2) THEN
      x13 = 4_dp*x2*x4
      
      a1(3) = C3(3)*x13 + 16_dp*C5(1)*x0*x9 + C5(3)*x1 + 120_dp*rs**4*x5 + 56_dp*x11* &
      x2 + 48_dp*x6*x8
      a1(4) = C3(4)*x13 + C5(2)*x0*x10 + C5(4)*x1 + x12*x7
      a1(5) = C3(5)*x13 + C5(5)*x1
      END IF

   END SUBROUTINE calc_a1_ab

   PURE SUBROUTINE calc_a2_ab(rs, zeta, a2, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a2(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a2_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, Ec(0:5), C2(0:5), C4(0:5)
      
      a2 = 0.0_dp

      CALL pw_lsd_ed_loc(rs, zeta, Ec, max_deriv)
      CALL calc_C2_ab(rs, zeta, C2, max_deriv)
      CALL calc_C4_ab(rs, zeta, C4, max_deriv)

      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4.0_dp*x2
      x4 = beta**6
      x5 = C2(0)*x4
      x6 = rs**4
      x7 = 6.0_dp*x6
      x8 = beta**4
      x9 = Ec(0)*x8
      
      a2(0) = C4(0)*x1 + x3*x5 + x7*x9
      
      IF (max_deriv >= 1) THEN
      x10 = rs**5
      x11 = 24.0_dp*x10
      x12 = C2(1)*x4
      x13 = rs**7
      x14 = 8.0_dp*x13
      x15 = C4(0)*x0
      x16 = rs**3
      x17 = 24.0_dp*x16
      x18 = Ec(1)*x8
      x19 = C2(2)*x4
      x20 = Ec(2)*x8
      
      a2(1) = C4(1)*x1 + x11*x5 + x12*x3 + x14*x15 + x17*x9 + x18*x7
      a2(2) = C4(2)*x1 + x19*x3 + x20*x7
      END IF
      
      IF (max_deriv >= 2) THEN
      x21 = 4.0_dp*x2*x4
      x22 = 6.0_dp*x6*x8
      
      a2(3) = C2(3)*x21 + 16.0_dp*C4(1)*x0*x13 + C4(3)*x1 + Ec(3)*x22 + 72.0_dp*rs**2* &
      x9 + 48.0_dp*x10*x12 + 56.0_dp*x15*x2 + 48.0_dp*x16*x18 + 120.0_dp*x5*x6
      a2(4) = C2(4)*x21 + C4(2)*x0*x14 + C4(4)*x1 + Ec(4)*x22 + x11*x19 + x17 &
      *x20
      a2(5) = C2(5)*x21 + C4(5)*x1 + Ec(5)*x22
      END IF

   END SUBROUTINE calc_a2_ab

   PURE SUBROUTINE calc_a3_ab(rs, zeta, a3, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a3(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a3_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, C3(0:5)
      
      a3 = 0.0_dp

      CALL calc_C3_ab(rs, zeta, C3, max_deriv)
      
      x0 = beta**8
      x1 = rs**8*x0
      
      a3(0) = C3(0)*x1
      
      IF (max_deriv >= 1) THEN
      x2 = rs**7
      x3 = 8.0_dp*x2
      x4 = C3(0)*x0
      
      a3(1) = C3(1)*x1 + x3*x4
      a3(2) = C3(2)*x1
      END IF
      
      IF (max_deriv >= 2) THEN
      a3(3) = 16.0_dp*C3(1)*x0*x2 + C3(3)*x1 + 56.0_dp*rs**6*x4
      a3(4) = C3(2)*x0*x3 + C3(4)*x1
      a3(5) = C3(5)*x1
      END IF

   END SUBROUTINE calc_a3_ab

   PURE SUBROUTINE calc_a4_ab(rs, zeta, a4, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a4(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a4_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, C2(0:5), Ec(0:5)
      
      a4 = 0.0_dp

      CALL calc_C2_ab(rs, zeta, C2, max_deriv)
      CALL pw_lsd_ed_loc(rs, zeta, Ec, max_deriv)
      
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4.0_dp*x2
      x4 = beta**6
      x5 = Ec(0)*x4
      
      a4(0) = C2(0)*x1 + x3*x5
      
      IF (max_deriv >= 1) THEN
      x6 = rs**7
      x7 = 8.0_dp*x6
      x8 = C2(0)*x0
      x9 = rs**5
      x10 = 24.0_dp*x9
      x11 = Ec(1)*x4
      x12 = Ec(2)*x4
      
      a4(1) = C2(1)*x1 + x10*x5 + x11*x3 + x7*x8
      a4(2) = C2(2)*x1 + x12*x3
      END IF
      
      IF (max_deriv >= 2) THEN
      x13 = 4.0_dp*x2*x4
      
      a4(3) = 16*C2(1)*x0*x6 + C2(3)*x1 + Ec(3)*x13 + 120.0_dp*rs**4*x5 + 48.0_dp*x11* &
      x9 + 56.0_dp*x2*x8
      a4(4) = C2(2)*x0*x7 + C2(4)*x1 + Ec(4)*x13 + x10*x12
      a4(5) = C2(5)*x1 + Ec(5)*x13
      END IF

   END SUBROUTINE calc_a4_ab

   PURE SUBROUTINE calc_a5_ab(rs, zeta, a5, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a5(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a5_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, Ec(0:5)
      
      a5 = 0.0_dp

      CALL pw_lsd_ed_loc(rs, zeta, Ec, max_deriv)
      
      x0 = beta**8
      x1 = rs**8*x0
      
      a5(0) = Ec(0)*x1
      
      IF (max_deriv >= 1) THEN
      x2 = rs**7
      x3 = 8.0_dp*x2
      x4 = Ec(0)*x0
      
      a5(1) = Ec(1)*x1 + x3*x4
      a5(2) = Ec(2)*x1
      END IF
      
      IF (max_deriv >= 2) THEN
      a5(3) = 16.0_dp*Ec(1)*x0*x2 + Ec(3)*x1 + 56.0_dp*rs**6*x4
      a5(4) = Ec(2)*x0*x3 + Ec(4)*x1
      a5(5) = Ec(5)*x1
      END IF

   END SUBROUTINE calc_a5_ab

   PURE SUBROUTINE calc_Q_ab(rs, zeta, mu, Q, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: Q(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_Q_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64, x65, x66, x67, phi2(0:2)
      
      Q = 0.0_dp

      CALL calc_phin(zeta, phi2, 2, max_deriv)
      
      x0 = phi2(0)**3
      x1 = mu**2
      x2 = phi2(0)**2
      x3 = 1.0_dp/x2
      x4 = x1*x3
      x5 = dd*x4
      x6 = sqrt(rs)
      x7 = aa*mu/phi2(0)
      x8 = x6*x7 + 1.0_dp
      x9 = rs*x5 + x8
      x10 = 1.0_dp/x9
      x11 = bb*x4
      x12 = rs**(3.0_dp/2.0_dp)
      x13 = mu**3
      x14 = 1.0_dp/x0
      x15 = cc*x13*x14
      x16 = rs*x11 + x12*x15 + x8
      x17 = log(x10*x16)
      x18 = qq*x17
      
      Q(0) = x0*x18
      
      IF (max_deriv >= 1) THEN
      x19 = 1.0_dp/x16
      x20 = 1.0_dp/x6
      x21 = x20*x7
      x22 = (1.0_dp/2.0_dp)*x21
      x23 = (3.0_dp/2.0_dp)*x15*x6
      x24 = x11 + x22 + x23
      x25 = -x22
      x26 = x25 - x5
      x27 = x9**(-2)
      x28 = x16*x27
      x29 = x10*x24 + x26*x28
      x30 = qq*x0*x19*x29
      x31 = 3.0_dp*qq*x17*x2
      x32 = aa*mu*x3*x6
      x33 = phi2(1)*x32
      x34 = -x33
      x35 = 2.0_dp*phi2(1)*x1*x14
      x36 = bb*x35
      x37 = rs*x36
      x38 = phi2(0)**(-4)
      x39 = cc*phi2(1)*x13*x38
      x40 = 3.0_dp*x12*x39
      x41 = x34 - x37 - x40
      x42 = d.0_dp*x35
      x43 = rs*x42
      x44 = x33 + x43
      x45 = x10*x41 + x28*x44
      
      Q(1) = x30*x9
      Q(2) = phi2(1)*x31 + qq*x0*x19*x45*x9
      END IF
      
      IF (max_deriv >= 2) THEN
      x46 = x16**(-2)
      x47 = qq*x0*x29*x46*x9
      x48 = qq*x0*x19*x9
      x49 = (1.0_dp/4.0_dp)*x7/x12
      x50 = -x21 - 2.0_dp*x5
      x51 = x24*x27
      x52 = x9**(-3)
      x53 = x16*x26*x52
      x54 = phi2(1)*qq*x19*x2*x9
      x55 = x34 - x43
      x56 = x33 + x37 + x40
      x57 = aa*mu
      x58 = (1.0_dp/2.0_dp)*phi2(1)*x20*x3*x57
      x59 = x27*x41
      x60 = 4.0_dp*d*phi2(1)*rs*x1*x14 + 2*x33
      x61 = phi2(1)**2
      x62 = qq*x0*x45
      x63 = phi2(2)*x32
      x64 = 2.0_dp*phi2(2)*rs*x1*x14
      x65 = 2.0_dp*x14*x57*x6*x61
      x66 = 6.0_dp*rs*x1*x38*x61
      x67 = cc*x13
      
      Q(3) = x30*(x22 + x5) + x47*(-x11 - x23 + x25) + x48*(x10*((3.0_dp/4.0_dp &
      )*x15*x20 - x49) + x28*x49 + x50*x51 + x50*x53)
      Q(4) = x30*x55 + x47*x56 + x48*(x10*(-x36 - 9.0_dp/2.0_dp*x39*x6 - x58) + &
      x26*x59 + x28*(x42 + x58) + x44*x51 + x53*x60) + x54*(3*x10*x24 + &
      3*x16*x26*x27)
      Q(5) = 6.0_dp*phi2(0)*x18*x61 + phi2(2)*x31 + x19*x55*x62 + x46*x56*x62*x9 + &
      x48*(x10*(bb*(-x64 + x66) - 3.0_dp*phi2(2)*x12*x38*x67 - x63 + x65 + 12.0_dp &
      *x12*x61*x67/phi2(0)**5) + x16*x44*x52*x60 + x28*(dd*x64 - d*x66 + &
      x63 - x65) + x59*x60) + 6.0_dp*x54*(x10*x41 + x16*x27*x44)
      END IF

   END SUBROUTINE calc_Q_ab

   PURE SUBROUTINE ec_mu_ab(rs, zeta, mu, ec_mu, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: ec_mu(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "ec_mu_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, Ec(0:5), Q(0:5), a1(0:5), a2(0:5), a3(0:5), a4(0:5), a5(0:5)
      
      ec_mu = 0.0_dp

      CALL pw_lsd_ed_loc(rs, zeta, Ec, max_deriv)
      CALL calc_Q_ab(rs, zeta, Q, max_deriv)
      CALL calc_a1_ab(rs, zeta, a1, max_deriv)
      CALL calc_a2_ab(rs, zeta, a2, max_deriv)
      CALL calc_a3_ab(rs, zeta, a3, max_deriv)
      CALL calc_a4_ab(rs, zeta, a4, max_deriv)
      CALL calc_a5_ab(rs, zeta, a5, max_deriv)

      x0 = rs**2
      x1 = beta**2
      x2 = mu**2
      x3 = x1*x2
      x4 = x0*x3 + 1.0_dp
      x5 = x4**(-4)
      x6 = mu**3
      x7 = a1(0)*x6
      x8 = mu**4
      x9 = a2(0)*x8
      x10 = mu**5
      x11 = a3(0)*x10
      x12 = mu**6
      x13 = a4(0)*x12
      x14 = mu**8
      x15 = a5(0)*x14
      
      ec_mu(0) = Ec(0) - x5*(Q(0) + x11 + x13 + x15 + x7 + x9)
      
      IF (max_deriv >= 1) THEN
      x16 = x4**(-5)
      x17 = 8.0_dp*rs*x1*x16*x2
      x18 = a1(2)*x6
      x19 = a2(2)*x8
      x20 = a3(2)*x10
      x21 = a4(2)*x12
      x22 = a5(2)*x14
      
      ec_mu(1) = Ec(1) + x17*(Q(0) + x11 + x13 + x15 + x7 + x9) - x5*(Q(1) + &
      a1(1)*x6 + a2(1)*x8 + a3(1)*x10 + a4(1)*x12 + a5(1)*x14)
      ec_mu(2) = Ec(2) - x5*(Q(2) + x18 + x19 + x20 + x21 + x22)
      END IF
      
      IF (max_deriv >= 2) THEN
      ec_mu(3) = Ec(3) - 80.0_dp*beta**4*x0*x8*(Q(0) + x11 + x13 + x15 + &
      x7 + x9)/x4**6 + 16.0_dp*rs*x1*x16*x2*(Q(1) + a1(1)*x6 +  &
      a2(1)*x8 + a3(1)*x10 + a4(1)*x12 + a5(1)*x14) + 8.0_dp*x16*x3*( &
      Q(0) + x11 + x13 + x15 + x7 + x9) - x5*(Q(3) + a1(3) &
      *x6 + a2(3)*x8 + a3(3)*x10 + a4(3)*x12 + a5(3)*x14)
      ec_mu(4) = Ec(4) + x17*(Q(2) + x18 + x19 + x20 + x21 + x22) - x5*(Q(4) &
      + a1(4)*x6 + a2(4)*x8 + a3(4)*x10 + a4(4)*x12 + a5(4)*x14)
      ec_mu(5) = Ec(5) - x5*(Q(5) + a1(5)*x6 + a2(5)*x8 + a3(5)*x10 + a4(5)* &
      x12 + a5(5)*x14)
      END IF

   END SUBROUTINE ec_mu_ab

   PURE SUBROUTINE calc_g2z(rs, g2z, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: g2z(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_g2z", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18
      
      g2z = 0.0_dp

      x0 = rs**2
      x1 = 1.0_dp/x0
      x2 = (1.0_dp/10.0_dp)*x1
      x3 = alpha**(-2)
      x4 = 2.0_dp**(1.0_dp/3.0_dp)
      x5 = -g2_1*rs*x4 + 1.0_dp
      x6 = g2_2*x4
      x7 = 2.0_dp**(2.0_dp/3.0_dp)
      x8 = g2_3*x0*x7 + rs*x6 + 1.0_dp
      x9 = 1.0_dp/x8
      x10 = x3*x5*x9
      
      g2z(0) = x10*x2
      
      IF (max_deriv >= 1) THEN
      x11 = g2_1*x3*x4*x9
      x12 = rs**(-3)
      x13 = g2_3*rs*x7
      x14 = -2.0_dp*x13 - x6
      x15 = x8**(-2)
      x16 = x14*x15*x3*x5
      
      g2z(1) = -1.0_dp/5.0_dp*x10*x12 - x11*x2 + x16*x2
      END IF
      
      IF (max_deriv >= 2) THEN
      x17 = (2.0_dp/5.0_dp)*x12
      x18 = (1.0_dp/5.0_dp)*x1*x15*x3
      
      g2z(2) = -g2_1*x14*x18*x4 - g2_3*x18*x5*x7 + x11*x17 - x14*x2*x3*x5*(4.0_dp &
      *x13 + 2.0_dp*x6)/x8**3 - x16*x17 + (3.0_dp/5.0_dp)*x10/rs**4
      END IF

   END SUBROUTINE calc_g2z

   PURE SUBROUTINE calc_C2(rs, C2, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: C2(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C2", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, g0(0:2)
      
      C2 = 0.0_dp

      CALL calc_g0(rs, g0, max_deriv)

      x0 = rs**(-3)
      
      C2(0) = x0*(-3.0_dp/8.0_dp*g0(0) + 3.0_dp/16.0_dp)
      
      IF (max_deriv >= 1) THEN
      x1 = (3.0_dp/8.0_dp)*x0
      x2 = rs**(-4)
      
      C2(1) = -g0(1)*x1 + x2*((9.0_dp/8.0_dp)*g0(0) - 9.0_dp/16.0_dp)
      END IF
      
      IF (max_deriv >= 2) THEN
      C2(2) = (9.0_dp/4.0_dp)*g0(1)*x2 - g0(2)*x1 + (-9.0_dp/2.0_dp*g0(0) + 9.0_dp &
      /4.0_dp)/rs**5
      END IF

   END SUBROUTINE calc_C2

   PURE SUBROUTINE calc_C3(rs, C3, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: C3(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C3", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x3, x4, x5, g0(0:2)
      
      C3 = 0.0_dp

      CALL calc_g0(rs, g0, max_deriv)

      x0 = (1.0_dp/2.0_dp)/rs**3
      x3 = oorootpi*g0(0)*x1
      x4 = rs**(-4)
      x5 = oorootpi*root2*g0(1)
      C3(0) = -x0*x3
      
      IF (max_deriv >= 1) THEN
      C3(1) = -x0*x5 + (3.0_dp/2.0_dp)*x3*x4
      END IF
      
      IF (max_deriv >= 2) THEN
      C3(2) = -oorootpi*root2*g0(2)*x0 + 3.0_dp*x4*x5 - 6.0_dp*x3/rs**5
      END IF

   END SUBROUTINE calc_C3

   PURE SUBROUTINE calc_C4(rs, C4, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: C4(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C4", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, g2z(0:2), D2(0:2)
      
      C4 = 0.0_dp

      CALL calc_d2z(rs, g2z, max_deriv)
      CALL calc_D2(rs, D2, max_deriv)

      x0 = rs**(-3)
      C4(0) = x0*(-9.0_dp/64.0_dp*D2(0) - 9.0_dp/32.0_dp*g2z(0) + (9.0_dp/320.0_dp) &
      *x2)
      
      IF (max_deriv >= 1) THEN
      x1 = alpha**(-2)
      x2 = x1/rs**2
      x3 = rs**(-4)
      
      C4(1) = -x0*(9.0_dp/64.0_dp*D2(1) + 9.0_dp/32.0_dp*g2z(1) + 9.0_dp/160.0_dp* &
      x4) + x3*((27.0_dp/64.0_dp)*D2(0) + (27.0_dp/32.0_dp)*g2z(0) - 27.0_dp &
      /320.0_dp*x2)
      END IF
      
      IF (max_deriv >= 2) THEN
      x4 = x0*x1
      
      C4(2) = x0*(-9.0_dp/64.0_dp*D2(2) - 9.0_dp/32.0_dp*g2z(2) + (27.0_dp/160.0_dp &
      )*x1*x3) + x3*((27.0_dp/32.0_dp)*D2(1) + (27.0_dp/16.0_dp)*g2z(1) + ( &
      27.0_dp/80.0_dp)*x4) + (-27.0_dp/16.0_dp*D2(0) - 27.0_dp/8.0_dp*g2z(0) &
      + (27.0_dp/80.0_dp)*x2)/rs**5
      END IF

   END SUBROUTINE calc_C4

   PURE SUBROUTINE calc_C5(rs, C5, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: C5(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C5", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x2, x3, g2z(0:2), D3(0:2)
      
      C5 = 0.0_dp

      CALL calc_d2z(rs, g2z, max_deriv)
      CALL calc_D3(rs, D3, max_deriv)

      x2 = oorootpi*root2/rs**3
      C5(0) = -x2*(9.0_dp/80.0_dp*D3(0) + 9.0_dp/40.0_dp*g2z(0))
      
      IF (max_deriv >= 1) THEN
      x3 = x2/rs
      
      C5(1) = -x2*(9.0_dp/80.0_dp*D3(1) + 9.0_dp/40.0_dp*g2z(1)) + x3*((27.0_dp/ &
      80.0_dp)*D3(0) + (27.0_dp/40.0_dp)*g2z(0))
      END IF
      
      IF (max_deriv >=2) THEN
      C5(2) = x2*(-9.0_dp/80.0_dp*D3(2) - 9.0_dp/40.0_dp*g2z(2)) + x3*((27.0_dp/ &
      40.0_dp)*D3(1) + (27.0_dp/20.0_dp)*g2z(1)) + oorootpi*root2*(-27.0_dp/20.0_dp* &
      D3(0) - 27.0_dp/10.0_dp*g2z(0))/rs**5
      END IF

   END SUBROUTINE calc_C5

   PURE SUBROUTINE calc_a1(rs, a1, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a1(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a1", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, C3(0:2), C5(0:2)
      
      a1 = 0.0_dp

      CALL calc_C3(rs, C3, max_deriv)
      CALL calc_C5(rs, C5, max_deriv)

      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4.0_dp*x2
      x4 = beta**6
      x5 = C3(0)*x4
      a1(0) = C5(0)*x1 + x3*x5
      
      IF (max_deriv >= 1) THEN
      x6 = rs**5
      x7 = C3(1)*x4
      x8 = rs**7
      x9 = C5(0)*x0
      
      a1(1) = C5(1)*x1 + x3*x7 + 24.0_dp*x5*x6 + 8.0_dp*x8*x9
      END IF
      
      IF (max_deriv >= 2) THEN
      a1(2) = C3(2)*x3*x4 + 16.0_dp*C5(1)*x0*x8 + C5(2)*x1 + 120.0_dp*rs**4*x5 + 56.0_dp*x2* &
      x9 + 48.0_dp*x6*x7
      END IF

   END SUBROUTINE calc_a1

   PURE SUBROUTINE calc_a2(rs, a2, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a2(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a2", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, Ec(0:2), C2(0:2), C4(0:2)
      
      a2 = 0.0_dp

      CALL pw_lda_ed_loc(rs, Ec, max_deriv)
      CALL calc_C2(rs, C2, max_deriv)
      CALL calc_C4(rs, C4, max_deriv)

      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4.0_dp*x2
      x4 = beta**6
      x5 = C2(0)*x4
      x6 = rs**4
      x7 = 6.0_dp*x6
      x8 = beta**4
      x9 = Ec(0)*x8
      a2(0) = C4(0)*x1 + x3*x5 + x7*x9
      
      IF (max_deriv >= 1) THEN
      x10 = rs**5
      x11 = C2(1)*x4
      x12 = rs**7
      x13 = C4(0)*x0
      x14 = rs**3
      x15 = Ec(1)*x8
      
      a2(1) = C4(1)*x1 + 24.0_dp*x10*x5 + x11*x3 + 8.0_dp*x12*x13 + 24.0_dp*x14*x9 + x15*x7
      END IF
      
      IF (max_deriv >= 2) THEN
      a2(2) = C2(2)*x3*x4 + 16.0_dp*C4(1)*x0*x12 + C4(2)*x1 + Ec(2)*x7*x8 + 72.0_dp*rs &
      **2*x9 + 48.0_dp*x10*x11 + 56.0_dp*x13*x2 + 48.0_dp*x14*x15 + 120.0_dp*x5*x6
      END IF

   END SUBROUTINE calc_a2

   PURE SUBROUTINE calc_a3(rs, a3, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a3(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a3", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, C3(0:2)
      
      a3 = 0.0_dp

      CALL calc_C3(rs, C3, max_deriv)

      x0 = beta**8
      x1 = rs**8*x0
      a3(0) = C3(0)*x1
      
      IF (max_deriv >= 1) THEN
      x2 = rs**7
      x3 = C3(0)*x0
      
      a3(1) = C3(1)*x1 + 8.0_dp*x2*x3
      END IF
      
      IF (max_deriv >= 2) THEN
      a3(2) = 16.0_dp*C3(1)*x0*x2 + C3(2)*x1 + 56.0_dp*rs**6*x3
      END IF

   END SUBROUTINE calc_a3

   PURE SUBROUTINE calc_a4(rs, a4, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a4(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a4", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, C2(0:2), Ec(0:2)

      a4 = 0.0_dp

      CALL calc_C2(rs, C2, max_deriv)
      CALL pw_lda_ed_loc(rs, Ec, max_deriv)

      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4.0_dp*x2
      x4 = beta**6
      x5 = Ec(0)*x4
      a4(0) = C2(0)*x1 + x3*x5
      
      IF (max_deriv >= 1) THEN
      x6 = rs**7
      x7 = C2(0)*x0
      x8 = rs**5
      x9 = Ec(1)*x4
      
      a4(1) = C2(1)*x1 + x3*x9 + 24.0_dp*x5*x8 + 8.0_dp*x6*x7
      END IF
      
      IF (max_deriv >= 2) THEN
      a4(2) = 16*C2(1)*x0*x6 + C2(2)*x1 + Ec(2)*x3*x4 + 120.0_dp*rs**4*x5 + 56.0_dp*x2* &
      x7 + 48.0_dp*x8*x9
      END IF

   END SUBROUTINE calc_a4

   PURE SUBROUTINE calc_a5(rs, a5, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a5(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a5", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, Ec(0:2)
      
      a5 = 0.0_dp

      CALL pw_lda_ed_loc(rs, Ec, max_deriv)

      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**7
      x3 = Ec(0)*x0
      a5(0) = Ec(0)*x1
      
      IF (max_deriv >= 1) THEN
      a5(1) = Ec(1)*x1 + 8.0_dp*x2*x3
      END IF
      
      IF (max_deriv >= 2) THEN
      a5(2) = 16.0_dp*Ec(1)*x0*x2 + Ec(2)*x1 + 56.0_dp*rs**6*x3
      END IF

   END SUBROUTINE calc_a5

   PURE SUBROUTINE calc_Q(rs, mu, Q, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: Q(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_Q", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24
      
      Q = 0.0_dp

      x0 = mu**2
      x1 = dd*x0
      x2 = sqrt(rs)
      x3 = aa*mu
      x4 = x2*x3 + 1.0_dp
      x5 = rs*x1 + x4
      x6 = 1.0_dp/x5
      x7 = bb*x0
      x8 = rs**(3.0_dp/2.0_dp)
      x9 = cc*mu**3
      x10 = rs*x7 + x4 + x8*x9
      Q(0) = qq*log(x10*x6)
      
      IF (max_deriv >= 1) THEN
      x11 = 1.0_dp/x10
      x12 = 1.0_dp/x2
      x13 = x12*x3
      x14 = (1.0_dp/2.0_dp)*x13
      x15 = (3.0_dp/2.0_dp)*x2*x9
      x16 = x14 + x15 + x7
      x17 = -x14
      x18 = -x1 + x17
      x19 = x5**(-2)
      x20 = x10*x19
      x21 = x16*x6 + x18*x20
      x22 = qq*x11*x21
      
      Q(1) = x22*x5
      END IF
      
      IF (max_deriv >= 2) THEN
      x23 = qq*x5
      x24 = (1.0_dp/4.0_dp)*x3/x8
      
      Q(2) = x11*x23*(-x10*x18*(2.0_dp*x1 + x13)/x5**3 + 2.0_dp*x16*x18*x19 + x20*x24 + &
      x6*((3.0_dp/4.0_dp)*x12*x9 - x24)) + x22*(x1 + x14) + x21*x23*(-x15 &
      + x17 - x7)/x10**2
      END IF

   END SUBROUTINE calc_Q

   PURE SUBROUTINE ec_mu(rs, mu, ec_mu, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: ec_mu(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "ec_mu", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, Ec(0:2), Q(0:2), a1(0:2), a2(0:2), a3(0:2), a4(0:2), a5(0:2)
      
      ec_mu = 0.0_dp

      CALL pw_lda_ed_loc(rs, Ec, max_deriv)
      CALL calc_Q(rs, Q, max_deriv)
      CALL calc_a1(rs, a1, max_deriv)
      CALL calc_a2(rs, a2, max_deriv)
      CALL calc_a3(rs, a3, max_deriv)
      CALL calc_a4(rs, a4, max_deriv)
      CALL calc_a5(rs, a5, max_deriv)

      x0 = rs**2
      x1 = beta**2
      x2 = mu**2
      x3 = x1*x2
      x4 = x0*x3 + 1.0_dp
      x5 = x4**(-4)
      x6 = mu**3
      x7 = a1(0)*x6
      x8 = mu**4
      x9 = a2(0)*x8
      x10 = mu**5
      x11 = a3(0)*x10
      x12 = mu**6
      x13 = a4(0)*x12
      x14 = mu**8
      x15 = a5(0)*x14
      
      ec_mu(0) = Ec(0) - x5*(Q(0) + x11 + x13 + x15 + x7 + x9)
      
      IF (max_deriv >= 1) THEN
      x16 = x4**(-5)
      x17 = rs*x1*x16*x2
      x18 = 8.0_dp*(Q(0) + x11 + x13 + x15 + x7 + x9)
      
      ec_mu(1) = Ec(1) + x17*x18 + x5*(-Q(1) - a1(1)*x6 - a2(1)*x8 - a3(1)*x10 &
      - a4(1)*x12 - a5(1)*x14)
      END IF
      
      IF (max_deriv >= 2) THEN
      ec_mu(2) = Ec(2) - 80.0_dp*beta**4*x0*x8*(Q(0) + x11 + x13 + x15 + &
      x7 + x9)/x4**6 + x16*x18*x3 + 16.0_dp*x17*(Q(1) + a1(1)*x6 + &
      a2(1)*x8 + a3(1)*x10 + a4(1)*x12 + a5(1)*x14) - x5*( &
      Q(2) + a1(2)*x6 + a2(2)*x8 + a3(2)*x10 + a4(2)*x12 + a5(2)*x14)
      END IF

   END SUBROUTINE ec_mu
   
   PURE SUBROUTINE ex_mu(rs, mu, ex_mu, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: ex_mu(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "ex_mu", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31


      x0 = 1.0_dp/rs
      x1 = 18**(1.0_dp/3.0_dp)
      x2 = pi**(2.0_dp/3.0_dp)
      x3 = 1.0_dp/x2
      x4 = 12**(1.0_dp/3.0_dp)
      x5 = pi**(1.0_dp/3.0_dp)
      x6 = 1.0_dp/x5
      x7 = mu*x4*x6
      x8 = 1.0_dp/pi
      x9 = mu**3
      x10 = rs**3
      x11 = 1.0_dp/mu
      x12 = rs**2
      x13 = 1.0_dp/x12
      x14 = mu**(-2)
      x15 = exp(-0.75_dp*x13*x14*x2*x4)
      x16 = (1.0_dp/3.0_dp)*x7
      x17 = x8*x9
      x18 = rs*x16 - 2.0_dp/9.0_dp*x10*x17
      x19 = x7*(-1.0_dp/12.0_dp*mu*rs*x4*x6 + (1.0_dp/27.0_dp)*x10*x8*x9 + (1.0_dp/ &
      6.0_dp)*x15*x18 + (1.0_dp/6.0_dp)*rootpi*erf(0.5_dp*x0*x1*x11*x5))
      x20 = x1*x3*(-rs*x19 + 3.0_dp/8.0_dp)
      x21 = (1.0_dp/6.0_dp)*mu*rs*x4*x6
      x22 = (2.0_dp/3.0_dp)*x12*x17
      x23 = x1*x11*x15*x5
      x24 = x16 - x22
      x25 = 1.0_dp/x10
      x26 = x14*x2*x25*x4
      x27 = -x13*x23 + (3.0_dp/2.0_dp)*x15*x18*x26 + x15*x24 + x22 - 1.0_dp/2.0_dp*x7
      x28 = x1*x3*(-x19 - x21*x27)
      x29 = 2.0_dp*x25
      x30 = (4.0_dp/3.0_dp)*rs*x17
      x31 = (9.0_dp/2.0_dp)*x15*x18
      ex_mu(0) = -x0*x20
      ex_mu(1) = -x0*x28 + x13*x20
      ex_mu(2) = -x0*x1*x3*(-x16*x27 - x21*(3*x15*x24*x26 - x15*x30 + x23*x29 &
      + x30 - x14*x2*x31*x4/rs**4 - 9.0_dp*pi*x15/(rs**5*x9) + pi**(4.0_dp/ &
      3.0_dp)*x1*x31/(mu**4*rs**6))) + 2.0_dp*x13*x28 - x20*x29

   END SUBROUTINE ex_mu




END MODULE xc_sr_lda
