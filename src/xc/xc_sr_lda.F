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

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, d3_1, d3_2

      D3 = 0.0_dp

      d3_1 = 0.31_dp
      d3_2 = 4.95_dp

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

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, d2_1, d2_2, d2_3

      d2_1 = 0.547_dp
      d2_2 = 0.388_dp
      d2_3 = 0.676_dp

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

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_g2z", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, g2_1, g2_2, g2_3

      g2_1 = 0.02267_dp
      g2_2 = 0.4319_dp
      g2_3 = 0.04_dp

      g2z = 0.0_dp

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

   END SUBROUTINE calc_g2z_ab

   PURE SUBROUTINE calc_g0(rs, g0, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs
      REAL(KIND=dp), INTENT(OUT) :: g0(0:)
      INTEGER, INTENT(IN) :: max_deriv

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_g0", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, g0_1, g0_2, g0_3, g0_4, g0_5

      g0_1 = 0.0207_dp
      g0_2 = 0.08193_dp
      g0_3 = 0.01277_dp
      g0_4 = 0.001859_dp
      g0_5 = 0.7524_dp

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
         phin(0) = 2.0_dp**(REAL(n-3, KIND=dp)/3.0_dp)
         IF (n>2 .AND. max_deriv >=1) phin(1) = SIGN(zeta)*REAL(n, KIND=dp)/3.0_dp*2.0_dp**(REAL(n-6, KIND=dp)/3.0_dp)
         IF (n>5 .AND. max_deriv >=2) phin(2) = REAL(n*(n-3), KIND=dp)/9.0_dp*2.0_dp**(REAL(n-9, KIND=dp)/3.0_dp)
      ELSE

      x0 = zeta + 1.0_dp
      x1 = (1.0_dp/3.0_dp)*n
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

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C2", routineP = moduleN//":"//routineN

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

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, g0(0:2)

      CALL calc_g0(rs, zeta, g0, max_deriv)

      C3 = 0.0_dp

      x0 = zeta**2
      x1 = sqrt(2.0_dp)
      x2 = 1.0_dp/sqrt(pi)
      x3 = rs**(-3)
      x4 = g0(0)*x1*x2*x3
      C3(0) = x4*((1.0_dp/2.0_dp)*x0 - 1.0_dp/2.0_dp)

      IF (max_deriv >= 1) THEN
      x5 = rs**(-4)
      x6 = x0 - 1.0_dp
      x7 = x1*x2*x6
      x8 = g0(1)*x3
      x9 = (1.0_dp/2.0_dp)*x1*x2*x6

      C3(1) = -3.0_dp/2.0_dp*g0(0)*x5*x7 + x8*x9
      C3(2) = x4*zeta
      END IF

      IF (max_deriv >= 3) THEN
      C3(3) = 6.0_dp*g0(0)*x7/rs**5 - 3.0_dp*g0(1)*x5*x7 + g0(2)*x3*x9
      C3(4) = -3.0_dp*g0(0)*x1*x2*x5*zeta + x1*x2*x8*zeta
      C3(5) = x4
      END IF

   END SUBROUTINE calc_C3

   SUBROUTINE calc_C4(rs, zeta, C4, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      REAL(KIND=dp), INTENT(OUT) :: C4(0:)
      INTEGER, INTENT(IN) :: max_deriv

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C4", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10

      C4 = 0.0_dp

      x0 = rs**(-3)
      x1 = alpha**(-2)
      x2 = rs**(-2)
      x3 = phi8*x1*x2
      x4 = -zeta**2 + 1
      x5 = D2*x4
      x6 = rs**(-4)
      x7 = phi8*x0*x1
      x8 = D2_rs*x4
      x9 = (9.0_dp/32.0_dp)*D2
      x10 = phi8_zeta*x1*x2
      C4 = x0*(-9.0_dp/64.0_dp*g2zm - 9.0_dp/64.0_dp*g2zp + (9.0_dp/320.0_dp)*x3 - &
      9.0_dp/64.0_dp*x5)
      C4_rs = x0*(-9.0_dp/64.0_dp*g2zm_rs - 9.0_dp/64.0_dp*g2zp_rs - 9.0_dp/ &
      160.0_dp*x7 - 9.0_dp/64.0_dp*x8) + x6*((27.0_dp/64.0_dp)*g2zm + ( &
      27.0_dp/64.0_dp)*g2zp - 27.0_dp/320.0_dp*x3 + (27.0_dp/64.0_dp)*x5)
      C4_zeta = x0*(-9.0_dp/64.0_dp*g2zm_zeta - 9.0_dp/64.0_dp*g2zp_zeta + (9.0_dp &
      /320.0_dp)*x10 + x9*zeta)
      C4_rs_rs = x0*(-9.0_dp/64.0_dp*D2_rs_rs*x4 - 9.0_dp/64.0_dp*g2zm_rs_rs - &
      9.0_dp/64.0_dp*g2zp_rs_rs + (27.0_dp/160.0_dp)*phi8*x1*x6) + x6*(( &
      27.0_dp/32.0_dp)*g2zm_rs + (27.0_dp/32.0_dp)*g2zp_rs + (27.0_dp/80.0_dp &
      )*x7 + (27.0_dp/32.0_dp)*x8) + (-27.0_dp/16.0_dp*g2zm - 27.0_dp/16.0_dp &
      *g2zp + (27.0_dp/80.0_dp)*x3 - 27.0_dp/16.0_dp*x5)/rs**5
      C4_rs_zeta = x0*((9.0_dp/32.0_dp)*D2_rs*zeta - 9.0_dp/64.0_dp*g2zm_rs_zeta &
      - 9.0_dp/64.0_dp*g2zp_rs_zeta - 9.0_dp/160.0_dp*phi8_zeta*x0*x1) + x6 &
      *(-27.0_dp/32.0_dp*D2*zeta + (27.0_dp/64.0_dp)*g2zm_zeta + (27.0_dp/ &
      64.0_dp)*g2zp_zeta - 27.0_dp/320.0_dp*x10)
      C4_zeta_zeta = x0*(-9.0_dp/64.0_dp*g2zm_zeta_zeta - 9.0_dp/64.0_dp* &
      g2zp_zeta_zeta + (9.0_dp/320.0_dp)*phi8_zeta_zeta*x1*x2 + x9)

   END SUBROUTINE calc_C4

   SUBROUTINE calc_C5(rs, zeta, C5, C5_rs, C5_zeta, C5_rs_rs, C5_rs_zeta, C5_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      REAL(KIND=dp), INTENT(INOUT) :: C5, C5_rs, C5_zeta, C5_rs_rs, C5_rs_zeta, C5_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C5", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7

      x0 = sqrt(2.0_dp)
      x1 =1.0_dp/sqrt(pi)
      x2 = x0*x1/rs**3
      x3 = -zeta**2 + 1
      x4 = D3*x3
      x5 = x0*x1/rs**4
      x6 = D3_rs*x3
      x7 = (9.0_dp/40.0_dp)*D3
      C5 = x2*(-9.0_dp/80.0_dp*g2zm - 9.0_dp/80.0_dp*g2zp - 9.0_dp/80.0_dp*x4)
      C5_rs = x2*(-9.0_dp/80.0_dp*g2zm_rs - 9.0_dp/80.0_dp*g2zp_rs - 9.0_dp/80.0_dp &
      *x6) + x5*((27.0_dp/80.0_dp)*g2zm + (27.0_dp/80.0_dp)*g2zp + (27.0_dp/ &
      80.0_dp)*x4)
      C5_zeta = x2*(-9.0_dp/80.0_dp*g2zm_zeta - 9.0_dp/80.0_dp*g2zp_zeta + x7* &
      zeta)
      C5_rs_rs = x2*(-9.0_dp/80.0_dp*D3_rs_rs*x3 - 9.0_dp/80.0_dp*g2zm_rs_rs - &
      9.0_dp/80.0_dp*g2zp_rs_rs) + x5*((27.0_dp/40.0_dp)*g2zm_rs + (27.0_dp/ &
      40.0_dp)*g2zp_rs + (27.0_dp/40.0_dp)*x6) + x0*x1*(-27.0_dp/20.0_dp* &
      g2zm - 27.0_dp/20.0_dp*g2zp - 27.0_dp/20.0_dp*x4)/rs**5
      C5_rs_zeta = x2*((9.0_dp/40.0_dp)*D3_rs*zeta - 9.0_dp/80.0_dp*g2zm_rs_zeta &
      - 9.0_dp/80.0_dp*g2zp_rs_zeta) + x5*(-27.0_dp/40.0_dp*D3*zeta + ( &
      27.0_dp/80.0_dp)*g2zm_zeta + (27.0_dp/80.0_dp)*g2zp_zeta)
      C5_zeta_zeta = x2*(-9.0_dp/80.0_dp*g2zm_zeta_zeta - 9.0_dp/80.0_dp* &
      g2zp_zeta_zeta + x7)

   END SUBROUTINE calc_C5

   SUBROUTINE calc_a1(rs, zeta, a1, a1_rs, a1_zeta, a1_rs_rs, a1_rs_zeta, a1_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      REAL(KIND=dp), INTENT(INOUT) :: a1, a1_rs, a1_zeta, a1_rs_rs, a1_rs_zeta, a1_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a1", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, beta

      beta = 0.784949_dp
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = C3*x4
      x6 = rs**5
      x7 = 24*x6
      x8 = C3_rs*x4
      x9 = rs**7
      x10 = 8*x9
      x11 = C5*x0
      x12 = 4*x2*x4
      a1 = C5*x1 + x3*x5
      a1_rs = C5_rs*x1 + x10*x11 + x3*x8 + x5*x7
      a1_zeta = C3_zeta*x12 + C5_zeta*x1
      a1_rs_rs = C3_rs_rs*x12 + 16*C5_rs*x0*x9 + C5_rs_rs*x1 + 120*rs**4*x5 + &
      56*x11*x2 + 48*x6*x8
      a1_rs_zeta = C3_rs_zeta*x12 + C3_zeta*x4*x7 + C5_rs_zeta*x1 + C5_zeta* &
      x0*x10
      a1_zeta_zeta = C3_zeta_zeta*x12 + C5_zeta_zeta*x1

   END SUBROUTINE calc_a1

   SUBROUTINE calc_a2(rs, zeta, a2, a2_rs, a2_zeta, a2_rs_rs, a2_rs_zeta, a2_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      REAL(KIND=dp), INTENT(INOUT) :: a2, a2_rs, a2_zeta, a2_rs_rs, a2_rs_zeta, a2_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a2", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, beta, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, a1_, a1__rs, a1__zeta, a1__rs_rs, a1__rs_zeta, a1__zeta_zeta

      beta = 0.784949_dp
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = C3*x4
      x6 = rs**5
      x7 = 24*x6
      x8 = C3_rs*x4
      x9 = rs**7
      x10 = 8*x9
      x11 = C5*x0
      x12 = 4*x2*x4
      a1 = C5*x1 + x3*x5
      a1_rs = C5_rs*x1 + x10*x11 + x3*x8 + x5*x7
      a1_zeta = C3_zeta*x12 + C5_zeta*x1
      a1_rs_rs = C3_rs_rs*x12 + 16*C5_rs*x0*x9 + C5_rs_rs*x1 + 120*rs**4*x5 + &
      56*x11*x2 + 48*x6*x8
      a1_rs_zeta = C3_rs_zeta*x12 + C3_zeta*x4*x7 + C5_rs_zeta*x1 + C5_zeta* &
      x0*x10
      a1_zeta_zeta = C3_zeta_zeta*x12 + C5_zeta_zeta*x1
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = C2*x4
      x6 = rs**4
      x7 = 6*x6
      x8 = beta**4
      x9 = Ec*x8
      x10 = rs**5
      x11 = 24*x10
      x12 = C2_rs*x4
      x13 = rs**7
      x14 = 8*x13
      x15 = C4*x0
      x16 = rs**3
      x17 = 24*x16
      x18 = Ec_rs*x8
      x19 = 4*x2*x4
      x20 = 6*x6*x8
      a2 = C4*x1 + x3*x5 + x7*x9
      a2_rs = C4_rs*x1 + x11*x5 + x12*x3 + x14*x15 + x17*x9 + x18*x7
      a2_zeta = C2_zeta*x19 + C4_zeta*x1 + Ec_zeta*x20
      a2_rs_rs = C2_rs_rs*x19 + 16*C4_rs*x0*x13 + C4_rs_rs*x1 + Ec_rs_rs*x20 &
      + 72*rs**2*x9 + 48*x10*x12 + 56*x15*x2 + 48*x16*x18 + 120*x5*x6
      a2_rs_zeta = C2_rs_zeta*x19 + C2_zeta*x11*x4 + C4_rs_zeta*x1 + C4_zeta* &
      x0*x14 + Ec_rs_zeta*x20 + Ec_zeta*x17*x8
      a2_zeta_zeta = C2_zeta_zeta*x19 + C4_zeta_zeta*x1 + Ec_zeta_zeta*x20

   END SUBROUTINE calc_a2

   SUBROUTINE calc_a3(rs, zeta, a3, a3_rs, a3_zeta, a3_rs_rs, a3_rs_zeta, a3_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      REAL(KIND=dp), INTENT(INOUT) :: a3, a3_rs, a3_zeta, a3_rs_rs, a3_rs_zeta, a3_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a3", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, beta, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, a1_, a1__rs, a1__zeta, a1__rs_rs, a1__rs_zeta, a1__zeta_zeta, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, a2_, a2__rs, a2__zeta, a2__rs_rs, a2__rs_zeta, a2__zeta_zeta

      beta = 0.784949_dp
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = C3*x4
      x6 = rs**5
      x7 = 24*x6
      x8 = C3_rs*x4
      x9 = rs**7
      x10 = 8*x9
      x11 = C5*x0
      x12 = 4*x2*x4
      a1 = C5*x1 + x3*x5
      a1_rs = C5_rs*x1 + x10*x11 + x3*x8 + x5*x7
      a1_zeta = C3_zeta*x12 + C5_zeta*x1
      a1_rs_rs = C3_rs_rs*x12 + 16*C5_rs*x0*x9 + C5_rs_rs*x1 + 120*rs**4*x5 + &
      56*x11*x2 + 48*x6*x8
      a1_rs_zeta = C3_rs_zeta*x12 + C3_zeta*x4*x7 + C5_rs_zeta*x1 + C5_zeta* &
      x0*x10
      a1_zeta_zeta = C3_zeta_zeta*x12 + C5_zeta_zeta*x1
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = C2*x4
      x6 = rs**4
      x7 = 6*x6
      x8 = beta**4
      x9 = Ec*x8
      x10 = rs**5
      x11 = 24*x10
      x12 = C2_rs*x4
      x13 = rs**7
      x14 = 8*x13
      x15 = C4*x0
      x16 = rs**3
      x17 = 24*x16
      x18 = Ec_rs*x8
      x19 = 4*x2*x4
      x20 = 6*x6*x8
      a2 = C4*x1 + x3*x5 + x7*x9
      a2_rs = C4_rs*x1 + x11*x5 + x12*x3 + x14*x15 + x17*x9 + x18*x7
      a2_zeta = C2_zeta*x19 + C4_zeta*x1 + Ec_zeta*x20
      a2_rs_rs = C2_rs_rs*x19 + 16*C4_rs*x0*x13 + C4_rs_rs*x1 + Ec_rs_rs*x20 &
      + 72*rs**2*x9 + 48*x10*x12 + 56*x15*x2 + 48*x16*x18 + 120*x5*x6
      a2_rs_zeta = C2_rs_zeta*x19 + C2_zeta*x11*x4 + C4_rs_zeta*x1 + C4_zeta* &
      x0*x14 + Ec_rs_zeta*x20 + Ec_zeta*x17*x8
      a2_zeta_zeta = C2_zeta_zeta*x19 + C4_zeta_zeta*x1 + Ec_zeta_zeta*x20
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**7
      x3 = 8*x2
      x4 = C3*x0
      a3 = C3*x1
      a3_rs = C3_rs*x1 + x3*x4
      a3_zeta = C3_zeta*x1
      a3_rs_rs = 16*C3_rs*x0*x2 + C3_rs_rs*x1 + 56*rs**6*x4
      a3_rs_zeta = C3_rs_zeta*x1 + C3_zeta*x0*x3
      a3_zeta_zeta = C3_zeta_zeta*x1

   END SUBROUTINE calc_a3

   SUBROUTINE calc_a4(rs, zeta, a4, a4_rs, a4_zeta, a4_rs_rs, a4_rs_zeta, a4_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      REAL(KIND=dp), INTENT(INOUT) :: a4, a4_rs, a4_zeta, a4_rs_rs, a4_rs_zeta, a4_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a4", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, beta, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, a1_, a1__rs, a1__zeta, a1__rs_rs, a1__rs_zeta, a1__zeta_zeta, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, a2_, a2__rs, a2__zeta, a2__rs_rs, a2__rs_zeta, a2__zeta_zeta, x0, x1, x2, x3, x4, a3_, a3__rs, a3__zeta, a3__rs_rs, a3__rs_zeta, a3__zeta_zeta

      beta = 0.784949_dp
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = C3*x4
      x6 = rs**5
      x7 = 24*x6
      x8 = C3_rs*x4
      x9 = rs**7
      x10 = 8*x9
      x11 = C5*x0
      x12 = 4*x2*x4
      a1 = C5*x1 + x3*x5
      a1_rs = C5_rs*x1 + x10*x11 + x3*x8 + x5*x7
      a1_zeta = C3_zeta*x12 + C5_zeta*x1
      a1_rs_rs = C3_rs_rs*x12 + 16*C5_rs*x0*x9 + C5_rs_rs*x1 + 120*rs**4*x5 + &
      56*x11*x2 + 48*x6*x8
      a1_rs_zeta = C3_rs_zeta*x12 + C3_zeta*x4*x7 + C5_rs_zeta*x1 + C5_zeta* &
      x0*x10
      a1_zeta_zeta = C3_zeta_zeta*x12 + C5_zeta_zeta*x1
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = C2*x4
      x6 = rs**4
      x7 = 6*x6
      x8 = beta**4
      x9 = Ec*x8
      x10 = rs**5
      x11 = 24*x10
      x12 = C2_rs*x4
      x13 = rs**7
      x14 = 8*x13
      x15 = C4*x0
      x16 = rs**3
      x17 = 24*x16
      x18 = Ec_rs*x8
      x19 = 4*x2*x4
      x20 = 6*x6*x8
      a2 = C4*x1 + x3*x5 + x7*x9
      a2_rs = C4_rs*x1 + x11*x5 + x12*x3 + x14*x15 + x17*x9 + x18*x7
      a2_zeta = C2_zeta*x19 + C4_zeta*x1 + Ec_zeta*x20
      a2_rs_rs = C2_rs_rs*x19 + 16*C4_rs*x0*x13 + C4_rs_rs*x1 + Ec_rs_rs*x20 &
      + 72*rs**2*x9 + 48*x10*x12 + 56*x15*x2 + 48*x16*x18 + 120*x5*x6
      a2_rs_zeta = C2_rs_zeta*x19 + C2_zeta*x11*x4 + C4_rs_zeta*x1 + C4_zeta* &
      x0*x14 + Ec_rs_zeta*x20 + Ec_zeta*x17*x8
      a2_zeta_zeta = C2_zeta_zeta*x19 + C4_zeta_zeta*x1 + Ec_zeta_zeta*x20
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**7
      x3 = 8*x2
      x4 = C3*x0
      a3 = C3*x1
      a3_rs = C3_rs*x1 + x3*x4
      a3_zeta = C3_zeta*x1
      a3_rs_rs = 16*C3_rs*x0*x2 + C3_rs_rs*x1 + 56*rs**6*x4
      a3_rs_zeta = C3_rs_zeta*x1 + C3_zeta*x0*x3
      a3_zeta_zeta = C3_zeta_zeta*x1
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = Ec*x4
      x6 = rs**7
      x7 = 8*x6
      x8 = C2*x0
      x9 = rs**5
      x10 = 24*x9
      x11 = Ec_rs*x4
      x12 = 4*x2*x4
      a4 = C2*x1 + x3*x5
      a4_rs = C2_rs*x1 + x10*x5 + x11*x3 + x7*x8
      a4_zeta = C2_zeta*x1 + Ec_zeta*x12
      a4_rs_rs = 16*C2_rs*x0*x6 + C2_rs_rs*x1 + Ec_rs_rs*x12 + 120*rs**4*x5 + &
      48*x11*x9 + 56*x2*x8
      a4_rs_zeta = C2_rs_zeta*x1 + C2_zeta*x0*x7 + Ec_rs_zeta*x12 + Ec_zeta* &
      x10*x4
      a4_zeta_zeta = C2_zeta_zeta*x1 + Ec_zeta_zeta*x12

   END SUBROUTINE calc_a4

   SUBROUTINE calc_a5(rs, zeta, a5, a5_rs, a5_zeta, a5_rs_rs, a5_rs_zeta, a5_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      REAL(KIND=dp), INTENT(INOUT) :: a5, a5_rs, a5_zeta, a5_rs_rs, a5_rs_zeta, a5_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a5", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, beta, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, a1_, a1__rs, a1__zeta, a1__rs_rs, a1__rs_zeta, a1__zeta_zeta, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, a2_, a2__rs, a2__zeta, a2__rs_rs, a2__rs_zeta, a2__zeta_zeta, x0, x1, x2, x3, x4, a3_, a3__rs, a3__zeta, a3__rs_rs, a3__rs_zeta, a3__zeta_zeta, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, a4_, a4__rs, a4__zeta, a4__rs_rs, a4__rs_zeta, a4__zeta_zeta

      beta = 0.784949_dp
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = C3*x4
      x6 = rs**5
      x7 = 24*x6
      x8 = C3_rs*x4
      x9 = rs**7
      x10 = 8*x9
      x11 = C5*x0
      x12 = 4*x2*x4
      a1 = C5*x1 + x3*x5
      a1_rs = C5_rs*x1 + x10*x11 + x3*x8 + x5*x7
      a1_zeta = C3_zeta*x12 + C5_zeta*x1
      a1_rs_rs = C3_rs_rs*x12 + 16*C5_rs*x0*x9 + C5_rs_rs*x1 + 120*rs**4*x5 + &
      56*x11*x2 + 48*x6*x8
      a1_rs_zeta = C3_rs_zeta*x12 + C3_zeta*x4*x7 + C5_rs_zeta*x1 + C5_zeta* &
      x0*x10
      a1_zeta_zeta = C3_zeta_zeta*x12 + C5_zeta_zeta*x1
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = C2*x4
      x6 = rs**4
      x7 = 6*x6
      x8 = beta**4
      x9 = Ec*x8
      x10 = rs**5
      x11 = 24*x10
      x12 = C2_rs*x4
      x13 = rs**7
      x14 = 8*x13
      x15 = C4*x0
      x16 = rs**3
      x17 = 24*x16
      x18 = Ec_rs*x8
      x19 = 4*x2*x4
      x20 = 6*x6*x8
      a2 = C4*x1 + x3*x5 + x7*x9
      a2_rs = C4_rs*x1 + x11*x5 + x12*x3 + x14*x15 + x17*x9 + x18*x7
      a2_zeta = C2_zeta*x19 + C4_zeta*x1 + Ec_zeta*x20
      a2_rs_rs = C2_rs_rs*x19 + 16*C4_rs*x0*x13 + C4_rs_rs*x1 + Ec_rs_rs*x20 &
      + 72*rs**2*x9 + 48*x10*x12 + 56*x15*x2 + 48*x16*x18 + 120*x5*x6
      a2_rs_zeta = C2_rs_zeta*x19 + C2_zeta*x11*x4 + C4_rs_zeta*x1 + C4_zeta* &
      x0*x14 + Ec_rs_zeta*x20 + Ec_zeta*x17*x8
      a2_zeta_zeta = C2_zeta_zeta*x19 + C4_zeta_zeta*x1 + Ec_zeta_zeta*x20
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**7
      x3 = 8*x2
      x4 = C3*x0
      a3 = C3*x1
      a3_rs = C3_rs*x1 + x3*x4
      a3_zeta = C3_zeta*x1
      a3_rs_rs = 16*C3_rs*x0*x2 + C3_rs_rs*x1 + 56*rs**6*x4
      a3_rs_zeta = C3_rs_zeta*x1 + C3_zeta*x0*x3
      a3_zeta_zeta = C3_zeta_zeta*x1
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = Ec*x4
      x6 = rs**7
      x7 = 8*x6
      x8 = C2*x0
      x9 = rs**5
      x10 = 24*x9
      x11 = Ec_rs*x4
      x12 = 4*x2*x4
      a4 = C2*x1 + x3*x5
      a4_rs = C2_rs*x1 + x10*x5 + x11*x3 + x7*x8
      a4_zeta = C2_zeta*x1 + Ec_zeta*x12
      a4_rs_rs = 16*C2_rs*x0*x6 + C2_rs_rs*x1 + Ec_rs_rs*x12 + 120*rs**4*x5 + &
      48*x11*x9 + 56*x2*x8
      a4_rs_zeta = C2_rs_zeta*x1 + C2_zeta*x0*x7 + Ec_rs_zeta*x12 + Ec_zeta* &
      x10*x4
      a4_zeta_zeta = C2_zeta_zeta*x1 + Ec_zeta_zeta*x12
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**7
      x3 = 8*x2
      x4 = Ec*x0
      a5 = Ec*x1
      a5_rs = Ec_rs*x1 + x3*x4
      a5_zeta = Ec_zeta*x1
      a5_rs_rs = 16*Ec_rs*x0*x2 + Ec_rs_rs*x1 + 56*rs**6*x4
      a5_rs_zeta = Ec_rs_zeta*x1 + Ec_zeta*x0*x3
      a5_zeta_zeta = Ec_zeta_zeta*x1

   END SUBROUTINE calc_a5

   SUBROUTINE calc_Q(rs, zeta, mu, Q, Q_rs, Q_zeta, Q_rs_rs, Q_rs_zeta, Q_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta, mu
      REAL(KIND=dp), INTENT(INOUT) :: Q, Q_rs, Q_zeta, Q_rs_rs, Q_rs_zeta, Q_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_Q", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64, x65, x66, x67, a, c, d, q, b, alpha

      a = 5.84605_dp
      c = 3.91744_dp
      d = 3.44851_dp
      q = (-2 + 2*0.693147180559945_dp)/pi**2
      b = 3.44851_dp - 17.53815_dp*pi/(-4 + 4*0.693147180559945_dp)
      alpha = (1.0_dp/3.0_dp)*2**(2.0_dp/3.0_dp)*3**(1.0_dp/3.0_dp)/pi**(1.0_dp/3.0_dp &
      )
      x0 = phi2**3
      x1 = mu**2
      x2 = phi2**2
      x3 = 1.0_dp/x2
      x4 = x1*x3
      x5 = d*x4
      x6 = sqrt(rs)
      x7 = a*mu/phi2
      x8 = x6*x7 + 1
      x9 = rs*x5 + x8
      x10 = 1.0_dp/x9
      x11 = b*x4
      x12 = rs**(3.0_dp/2.0_dp)
      x13 = mu**3
      x14 = 1.0_dp/x0
      x15 = c*x13*x14
      x16 = rs*x11 + x12*x15 + x8
      x17 = log(x10*x16)
      x18 = q*x17
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
      x30 = q*x0*x19*x29
      x31 = 3*q*x17*x2
      x32 = a*mu*x3*x6
      x33 = phi2_zeta*x32
      x34 = -x33
      x35 = 2*phi2_zeta*x1*x14
      x36 = b*x35
      x37 = rs*x36
      x38 = phi2**(-4)
      x39 = c*phi2_zeta*x13*x38
      x40 = 3*x12*x39
      x41 = x34 - x37 - x40
      x42 = d*x35
      x43 = rs*x42
      x44 = x33 + x43
      x45 = x10*x41 + x28*x44
      x46 = x16**(-2)
      x47 = q*x0*x29*x46*x9
      x48 = q*x0*x19*x9
      x49 = (1.0_dp/4.0_dp)*x7/x12
      x50 = -x21 - 2*x5
      x51 = x24*x27
      x52 = x9**(-3)
      x53 = x16*x26*x52
      x54 = phi2_zeta*q*x19*x2*x9
      x55 = x34 - x43
      x56 = x33 + x37 + x40
      x57 = a*mu
      x58 = (1.0_dp/2.0_dp)*phi2_zeta*x20*x3*x57
      x59 = x27*x41
      x60 = 4*d*phi2_zeta*rs*x1*x14 + 2*x33
      x61 = phi2_zeta**2
      x62 = q*x0*x45
      x63 = phi2_zeta_zeta*x32
      x64 = 2*phi2_zeta_zeta*rs*x1*x14
      x65 = 2*x14*x57*x6*x61
      x66 = 6*rs*x1*x38*x61
      x67 = c*x13
      Q = x0*x18
      Q_rs = x30*x9
      Q_zeta = phi2_zeta*x31 + q*x0*x19*x45*x9
      Q_rs_rs = x30*(x22 + x5) + x47*(-x11 - x23 + x25) + x48*(x10*((3.0_dp/ &
      4.0_dp)*x15*x20 - x49) + x28*x49 + x50*x51 + x50*x53)
      Q_rs_zeta = x30*x55 + x47*x56 + x48*(x10*(-x36 - 9.0_dp/2.0_dp*x39*x6 - &
      x58) + x26*x59 + x28*(x42 + x58) + x44*x51 + x53*x60) + x54*(3* &
      x10*x24 + 3*x16*x26*x27)
      Q_zeta_zeta = 6*phi2*x18*x61 + phi2_zeta_zeta*x31 + x19*x55*x62 + x46* &
      x56*x62*x9 + x48*(x10*(-b*x64 + b*x66 - 3*phi2_zeta_zeta*x12*x38* &
      x67 - x63 + x65 + 12*x12*x61*x67/phi2**5) + x16*x44*x52*x60 + x28 &
      *(d*x64 - d*x66 + x63 - x65) + x59*x60) + x54*(6*x10*x41 + 6*x16* &
      x27*x44)

   END SUBROUTINE calc_Q

   SUBROUTINE ec_mu(rs, zeta, mu, ec_mu, ec_mu_rs, ec_mu_zeta, ec_mu_rs_rs, ec_mu_rs_zeta, ec_mu_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta, mu
      REAL(KIND=dp), INTENT(INOUT) :: ec_mu, ec_mu_rs, ec_mu_zeta, ec_mu_rs_rs, ec_mu_rs_zeta, ec_mu_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "ec_mu", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22

      x0 = rs**2
      x1 = beta**2
      x2 = mu**2
      x3 = x1*x2
      x4 = x0*x3 + 1
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
      x16 = x4**(-5)
      x17 = 8*rs*x1*x16*x2
      x18 = a1_zeta*x6
      x19 = a2_zeta*x8
      x20 = a3_zeta*x10
      x21 = a4_zeta*x12
      x22 = a5_zeta*x14
      ec_mu = Ec + x5*(-Q - x11 - x13 - x15 - x7 - x9)
      ec_mu_rs = Ec_rs + x17*(Q + x11 + x13 + x15 + x7 + x9) + x5*(-Q_rs - &
      a1_rs*x6 - a2_rs*x8 - a3_rs*x10 - a4_rs*x12 - a5_rs*x14)
      ec_mu_zeta = Ec_zeta + x5*(-Q_zeta - x18 - x19 - x20 - x21 - x22)
      ec_mu_rs_rs = Ec_rs_rs + beta**4*x0*x8*(-80*Q - 80*x11 - 80*x13 - 80*x15 &
      - 80*x7 - 80*x9)/x4**6 + rs*x1*x16*x2*(16*Q_rs + 16*a1_rs*x6 + 16 &
      *a2_rs*x8 + 16*a3_rs*x10 + 16*a4_rs*x12 + 16*a5_rs*x14) + x16*x3* &
      (8*Q + 8*x11 + 8*x13 + 8*x15 + 8*x7 + 8*x9) + x5*(-Q_rs_rs - &
      a1_rs_rs*x6 - a2_rs_rs*x8 - a3_rs_rs*x10 - a4_rs_rs*x12 - &
      a5_rs_rs*x14)
      ec_mu_rs_zeta = Ec_rs_zeta + x17*(Q_zeta + x18 + x19 + x20 + x21 + x22) &
      + x5*(-Q_rs_zeta - a1_rs_zeta*x6 - a2_rs_zeta*x8 - a3_rs_zeta*x10 &
      - a4_rs_zeta*x12 - a5_rs_zeta*x14)
      ec_mu_zeta_zeta = Ec_zeta_zeta + x5*(-Q_zeta_zeta - a1_zeta_zeta*x6 - &
      a2_zeta_zeta*x8 - a3_zeta_zeta*x10 - a4_zeta_zeta*x12 - &
      a5_zeta_zeta*x14)

   END SUBROUTINE ec_mu

END MODULE xc_sr_lda
