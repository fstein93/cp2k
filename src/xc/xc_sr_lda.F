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

! **************************************************************************************************
!> \brief ...
!> \param x ...
!> \param Q_ ...
!> \param order ...
! **************************************************************************************************
   PURE SUBROUTINE calc_Q(x, Q_, order)
      REAL(KIND=dp), INTENT(IN)                          :: x
      REAL(KIND=dp), INTENT(OUT)                         :: Q_(0:2)
      INTEGER, INTENT(IN)                                :: order

      CHARACTER(LEN=*), PARAMETER :: routineN = 'calc_Q', routineP = moduleN//':'//routineN

      REAL(KIND=dp)                                      :: x0, x1, x10, x11, x12, x2, x3, x4, x5, &
                                                            x6, x7, x8, x9

      x0 = x**2
      x1 = aa*x+1.0_dp
      x2 = dd*x0+x1
      x3 = 1.0_dp/x2
      x4 = bb*x0+cc*x**3+x1

      Q_(0) = qq*LOG(x3*x4)

      IF (order >= 1) THEN
         x5 = 2.0_dp*x
         x6 = aa+dd*x5
         x7 = x4*x6
         x8 = 3.0_dp*cc
         x9 = aa+bb*x5+x0*x8
         x10 = x2*x9
         x11 = -x10+x7

         Q_(1) = -qq*x11*x3/x4
      END IF

      IF (order >= 2) THEN
         x12 = x2**2

         Q_(2) = qq*(x10*x11-x11*x7-2.0_dp*x4*(-x12*(bb+x*x8)+x2*(dd*x4+x6*x9) &
                                               -x4*x6**2))/(x12*x4**2)
      END IF

   END SUBROUTINE calc_Q

! **************************************************************************************************
!> \brief ...
!> \param rs ...
!> \param g2 ...
!> \param order ...
! **************************************************************************************************
   PURE SUBROUTINE calc_g2(rs, g2, order)
      REAL(KIND=dp), INTENT(IN)                          :: rs
      REAL(KIND=dp), INTENT(OUT)                         :: g2(0:2)
      INTEGER, INTENT(IN)                                :: order

      CHARACTER(LEN=*), PARAMETER :: routineN = 'calc_g2', routineP = moduleN//':'//routineN

      REAL(KIND=dp)                                      :: x0, x1, x10, x2, x3, x4, x5, x6, x7, x8, &
                                                            x9

      g2 = 0.0_dp

      x0 = 2.0_dp**(2.0_dp/3.0_dp)
      x1 = 1.0_dp/alpha2
      x2 = (2.0_dp/5.0_dp)*x0*x1
      x3 = rs**2
      x4 = g2_1*rs
      x5 = -x4+1.0_dp
      x6 = g2_2*rs+g2_3*x3+1.0_dp

      g2(0) = x2*x5/(x3*x6)

      IF (order >= 1) THEN
         x7 = x6**2
         x8 = x4-1.0_dp
         x9 = g2_2+g2_3*2.0_dp*rs
         x10 = x8*x9

         g2(1) = x2*(rs*x10-x4*x6+2.0_dp*x6*x8)/(rs**3*x7)
      END IF

      IF (order >= 2) THEN
         g2(2) = (4.0_dp/5.0_dp)*x0*x1*(-2.0_dp*rs*x10*x6+x3*x6*(g2_1*x9+g2_3*x8) &
                                        -x3*x8*x9**2+2.0_dp*x4*x7+3.0_dp*x5*x7)/(rs**4*x6**3)
      END IF

   END SUBROUTINE calc_g2

! **************************************************************************************************
!> \brief ...
!> \param rs ...
!> \param g0 ...
!> \param order ...
! **************************************************************************************************
   PURE SUBROUTINE calc_g0(rs, g0, order)
      REAL(KIND=dp), INTENT(IN)                          :: rs
      REAL(KIND=dp), INTENT(INOUT)                       :: g0(0:2)
      INTEGER, INTENT(IN)                                :: order

      CHARACTER(LEN=*), PARAMETER :: routineN = 'calc_g0', routineP = moduleN//':'//routineN

      REAL(KIND=dp)                                      :: x0, x1, x2, x3, x4, x5, x6

      x0 = EXP(-g0_5*rs)
      x1 = rs**2
      x2 = rs*x1
      x3 = x1**2
      x4 = (1.0_dp/2.0_dp)*(g0_1*rs+g0_2*x1-g0_3*x2+g0_4*x3+1.0_dp)

      g0(0) = x0*x4

      IF (order >= 1) THEN
         x5 = 3.0_dp*g0_3
         x6 = g0_1+2.0_dp*g0_2*rs+4.0_dp*g0_4*x2-x1*x5

         g0(1) = (1.0_dp/2.0_dp)*x0*(-g0_5*(g0_1*rs+g0_2*x1-g0_3*x2+g0_4*x3+1.0_dp)+x6)
      END IF

      IF (order >= 2) THEN
         g0(2) = x0*(g0_2+6.0_dp*g0_4*x1+g0_5**2*x4-g0_5*x6-rs*x5)
      END IF

   END SUBROUTINE calc_g0

! **************************************************************************************************
!> \brief ...
!> \param n ...
!> \param zeta ...
!> \param phin ...
!> \param order ...
! **************************************************************************************************
   PURE SUBROUTINE calc_phin(n, zeta, phin, order)
      INTEGER, INTENT(IN)                                :: n
      REAL(KIND=dp), INTENT(IN)                          :: zeta
      REAL(KIND=dp), INTENT(OUT)                         :: phin(0:2)
      INTEGER, INTENT(IN)                                :: order

      CHARACTER(LEN=*), PARAMETER :: routineN = 'calc_phin', routineP = moduleN//':'//routineN

      REAL(KIND=dp)                                      :: zm3, zp3

      phin = 0.0_dp

      zp3 = (zeta+1.0_dp)**(1.0_dp/3.0_dp)
      zm3 = (-zeta+1.0_dp)**(1.0_dp/3.0_dp)

      phin(0) = (1.0_dp/2.0_dp)*(zp3+zm3)

      IF (order >= 1) THEN
         phin(1) = (1.0_dp/6.0_dp)*REAL(n, KIND=dp)*(zp3**(n-3)-zm3**(n-3))
      END IF

      IF (order >= 2) THEN
         phin(2) = (1.0_dp/18.0_dp)*REAL(n*(n-3), KIND=dp)*(zp3**(n-6)+zm3**(n-6))
      END IF

   END SUBROUTINE calc_phin

! **************************************************************************************************
!> \brief ...
!> \param rhoa ...
!> \param rhob ...
!> \param omega ...
!> \param ed ...
!> \param order ...
! **************************************************************************************************
   SUBROUTINE ec_lsd_sr(rhoa, rhob, omega, ed, order)
      REAL(KIND=dp), INTENT(IN)                          :: rhoa, rhob, omega
      REAL(KIND=dp), INTENT(OUT)                         :: ed(0:5)
      INTEGER, INTENT(IN)                                :: order

      CHARACTER(LEN=*), PARAMETER :: routineN = 'ec_lsd_sr', routineP = moduleN//':'//routineN

      REAL(KIND=dp) :: ed0(0:5), g0(0:2), g2m(0:2), g2p(0:2), phi2(0:2), phi8(0:2), Q(0:2), rho, &
         rs, x0, x1, x10, x100, x101, x102, x103, x104, x105, x106, x107, x108, x109, x11, x110, &
         x111, x112, x113, x114, x115, x116, x117, x118, x119, x12, x120, x121, x122, x123, x124, &
         x125, x126, x127, x128, x129, x13, x130, x131, x132, x133, x134, x135, x136, x137, x138, &
         x139, x14, x140, x141, x142, x143, x144, x145, x146, x147, x148, x149, x15, x150, x151, &
         x152, x153, x154, x155, x156, x157, x158, x159, x16, x160, x161, x162, x163, x164, x165, &
         x166, x17, x18, x19, x2, x20, x21, x22, x23, x24, x25, x26
      REAL(KIND=dp) :: x27, x28, x29, x3, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x4, &
         x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x5, x50, x51, x52, x53, x54, x55, x56, &
         x57, x58, x59, x6, x60, x61, x62, x63, x64, x65, x66, x67, x68, x69, x7, x70, x71, x72, &
         x73, x74, x75, x76, x77, x78, x79, x8, x80, x81, x82, x83, x84, x85, x86, x87, x88, x89, &
         x9, x90, x91, x92, x93, x94, x95, x96, x97, x98, x99, zeta

      rho = rhoa+rhob
      CALL calc_rs(rho, rs)
      zeta = MIN(MAX((rhoa-rhob)/(rhoa+rhob), -1.0_dp), 1.0_dp)

      ed0 = 0.0_dp
      CALL pw_lsd_ed_loc(rhoa, rhob, ed0, order)

      g0 = 0.0_dp
      CALL calc_g0(rs, g0, order)

      Q = 0.0_dp
      CALL calc_Q(omega*SQRT(rs), Q, order)

      g2p = 0.0_dp
      IF (zeta+1.0_dp > 0.0_dp) THEN
         CALL calc_g2((2.0_dp/(1.0_dp+zeta))**(1.0_dp/3.0_dp)*rs, g2p, order)
      END IF

      g2m = 0.0_dp
      IF (-zeta+1.0_dp > 0.0_dp) THEN
         CALL calc_g2((2.0_dp/(1.0_dp-zeta))**(1.0_dp/3.0_dp)*rs, g2m, order)
      END IF

      phi2 = 0.0_dp
      IF (ABS(zeta) >= 1.0_dp) THEN
         phi2(0) = 2.0_dp**(1.0_dp/3.0_dp)
      ELSE
         CALL calc_phin(2, zeta, phi2, order)
      END IF

      ! Here we do not need branches because there are no divergences up to second order
      phi8 = 0.0_dp
      CALL calc_phin(8, zeta, phi8, order)

      x0 = rs**2
      x1 = beta**2
      x2 = omega**2
      x3 = x0*x1*x2+1.0_dp
      x4 = x3**(-4)
      x5 = Q(0)*phi2(0)**3
      x6 = beta**8
      x7 = omega**8
      x8 = rs**8*x6*x7
      x9 = ed0(0)*x8
      x10 = sqrt2
      x11 = omega**5
      x12 = oorootpi
      x13 = rs**5
      x14 = zeta**2
      x15 = x14-1.0_dp
      x16 = g0(0)*x10*x11*x12*x13*x15*x6
      x17 = (1.0_dp/2.0_dp)*x16
      x18 = omega**6
      x19 = rs**6
      x20 = 4.0_dp*x19
      x21 = beta**6
      x22 = ed0(0)*x21
      x23 = g0(0)-1.0_dp/2.0_dp
      x24 = (3.0_dp/8.0_dp)*x14-3.0_dp/8.0_dp
      x25 = x13*x24*x6
      x26 = x18*(x20*x22+x23*x25)
      x27 = omega**3
      x28 = rs**3
      x29 = 2.0_dp*x21*x28
      x30 = g0(0)*x10*x12*x15
      x31 = x10*x12*x13*x6
      x32 = (1.0_dp/2.0_dp)*zeta
      x33 = (-x32+1.0_dp/2.0_dp)**2
      x34 = g2m(0)*x33
      x35 = x32+1.0_dp/2.0_dp
      x36 = x35**2
      x37 = g2p(0)*x36
      x38 = 1.0_dp/x0
      x39 = -d3_2+rs
      x40 = EXP(-d3_1*rs)
      x41 = -x14+1.0_dp
      x42 = x38*x39*x40*x41
      x43 = (9.0_dp/80.0_dp)*x42
      x44 = x27*(x29*x30-x31*(9.0_dp/80.0_dp*(x34+x37)+x43))
      x45 = omega**4
      x46 = rs**4
      x47 = 6.0_dp*x46
      x48 = beta**4
      x49 = ed0(0)*x48
      x50 = 4.0_dp*x21*x24*x28
      x51 = x13*x6
      x52 = alpha2**(-1)
      x53 = phi8(0)*x38*x52
      x54 = 9.0_dp/64.0_dp*(-d2_2+d2_3*rs)
      x55 = 1.0_dp/rs
      x56 = EXP(-d2_1*rs)
      x57 = x45*(x23*x50+x47*x49+x51*(-9.0_dp/64.0_dp*(x34+x37)-x41*x54*x55*x56+(9.0_dp/320.0_dp)*x53))

      ed(0) = ed0(0)+x4*(-x17-x26-x44-x5-x57-x9)

      IF (order >= 1) THEN
         x58 = x3**(-5)
         x59 = 8.0_dp*rs*x1*x2*x58
         x60 = phi2(0)**2
         x61 = Q(1)*omega/SQRT(rs)
         x62 = (1.0_dp/2.0_dp)*x60*x61
         x63 = rs**7
         x64 = ed0(0)*x6*x7
         x65 = 8.0_dp*x63*x64
         x66 = ed0(1)*x8
         x67 = g0(0)*x10*x11*x12*x15
         x68 = (5.0_dp/2.0_dp)*x46*x6*x67
         x69 = (1.0_dp/2.0_dp)*x11*x13*x6
         x70 = g0(1)*x10*x12*x15
         x71 = x69*x70
         x72 = 24.0_dp*x13
         x73 = ed0(1)*x21
         x74 = x23*x24
         x75 = x18*(g0(1)*x25+x20*x73+x22*x72+5*x46*x6*x74)
         x76 = x10*x12*x46*x6
         x77 = x38*x40
         x78 = 2.0_dp**(1.0_dp/3.0_dp)
         x79 = MIN(zeta+1.0_dp, 0.0_dp)
         IF (x79 <= 0.0_dp) THEN
            x80 = 0.0_dp
         ELSE
            x80 = x79**(-1.0_dp/3.0_dp)
         END IF
         x81 = g2m(1)*x33*x78*x80
         x82 = g2p(1)*x36*x78*x80
         x83 = -9.0_dp/40.0_dp*x14+9.0_dp/40.0_dp
         x84 = 1.0_dp/x28
         x85 = x39*x41
         x86 = x27*(6.0_dp*x0*x21*x30+x29*x70+x31*((9.0_dp/80.0_dp)*d3_1*x38*x40*x85+x39*x40*x83*x84+x77*((9.0_dp/80.0_dp)*x14 &
                                                                           -9.0_dp/80.0_dp)-9.0_dp/80.0_dp*x81-9.0_dp/80.0_dp*x82) &
                    -x76*9.0_dp/16.0_dp*(x34+x37+x42))
         x87 = 24.0_dp*x28
         x88 = ed0(1)*x48
         x89 = 4.0_dp*x21*x28
         x90 = g0(1)*x24
         x91 = 12.0_dp*x0*x21
         x92 = x46*x6
         x93 = d2_3*rs
         x94 = x41*x55*x56*(-d2_2+x93)
         x95 = phi8(0)*x52*x84
         x96 = d2_3*x41*x55*x56
         x97 = x38*x41*x56
         x98 = d2_1*x41*x55*x56
         x99 = x45*(x47*x88+x49*x87+x51*(x54*x97+x54*x98-9.0_dp/64.0_dp*(x81+x82+x96)-9.0_dp/160.0_dp*x95) &
                    +x74*x91+x89*x90+x92 &
                    *(-45.0_dp/64.0_dp*(x34+x37+x94)+(9.0_dp/64.0_dp)*x53))

         ed(1) = ed0(1)+x4*(-x62-x65-x66-x68-x71-x75-x86-x99)+x59*(x17+x26+x44+x5+x57+x9)
         ed(2) = ed0(2)+x4*(-x101-x102-x105-x109-x122-x130)
      END IF

      IF (order >= 2) THEN
         x100 = 3.0_dp*Q(0)*x60
         x101 = phi2(1)*x100
         x102 = ed0(2)*x8
         x103 = g0(0)*x10*x12*zeta
         x104 = x11*x13*x6
         x105 = x103*x104
         x106 = 4.0_dp*x19*x21
         x107 = x23*zeta
         x108 = (3.0_dp/4.0_dp)*x13*x6
         x109 = x18*(ed0(2)*x106+x107*x108)
         x110 = g0(0)*x10*x12
         x111 = x110*x89
         x112 = x32-1.0_dp/2.0_dp
         x113 = g2m(0)*x112
         x114 = g2p(0)*x35
         x115 = (9.0_dp/40.0_dp)*x77*zeta
         x116 = x115*x39
         x117 = x80**4
         x118 = g2m(1)*x117*x33*x78
         x119 = (3.0_dp/80.0_dp)*x118
         x120 = g2p(1)*x117*x36*x78
         x121 = (3.0_dp/80.0_dp)*x120
         x122 = x27*(x111*zeta+x31*(rs*x119+rs*x121-9.0_dp/80.0_dp*x113-9.0_dp/80.0_dp*x114+x116))
         x123 = 6.0_dp*x46*x48
         x124 = 3.0_dp*x21*x28
         x125 = phi8(1)*x38*x52
         x126 = -9.0_dp/32.0_dp*d2_2+(9.0_dp/32.0_dp)*x93
         x127 = x55*x56*zeta
         x128 = (3.0_dp/64.0_dp)*x118
         x129 = (3.0_dp/64.0_dp)*x120
         x130 = x45*(ed0(2)*x123+x107*x124+x51*(rs*x128+rs*x129-9.0_dp/64.0_dp*x113-9.0_dp/64.0_dp*x114+(9.0_dp/320.0_dp)*x125 &
                                                +x126*x127))
         x131 = x1*x2*x58
         x132 = x6*x7
         x133 = 5.0_dp*x11*x46*x6
         x134 = g0(2)*x10*x12*x15
         x135 = x28*x6
         x136 = (9.0_dp/4.0_dp)*x39*x41
         x137 = x40*x84
         x138 = (9.0_dp/8.0_dp)*x38*x40
         x139 = d3_1*x39*x41
         x140 = 2.0_dp**(2.0_dp/3.0_dp)
         x141 = x80**2
         x142 = g2m(2)*x140*x141*x33
         x143 = g2p(2)*x140*x141*x36
         x144 = 1.0_dp/x46
         x145 = (9.0_dp/20.0_dp)*x40*x84
         x146 = -45.0_dp/32.0_dp*d2_2+(45.0_dp/32.0_dp)*d2_3*rs
         x147 = (9.0_dp/32.0_dp)*d2_3*x41
         x148 = x126*x38*x56
         x149 = g0(1)*x10*x12*zeta
         x150 = g0(1)*zeta
         x151 = x39*zeta
         x152 = g2m(1)*rs*x117*x33*x78
         x153 = g2p(1)*rs*x117*x36*x78
         x154 = g2m(1)*x112*x78*x80
         x155 = g2p(1)*x35*x78*x80
         x156 = x80**5
         x157 = g2m(2)*rs*x140*x156*x33
         x158 = g2p(2)*rs*x140*x156*x36
         x159 = g2m(1)*rs*x112*x117*x78
         x160 = g2p(1)*rs*x117*x35*x78
         x161 = x80**7
         x162 = g2m(1)*rs*x161*x33*x78
         x163 = g2p(1)*rs*x161*x36*x78
         x164 = x80**8
         x165 = g2m(2)*x0*x140*x164*x33
         x166 = g2p(2)*x0*x140*x164*x36

         ed(3) = ed0(3)+16.0_dp*rs*x131*(x62+x65+x66+x68+x71+x75+x86+x99)+x0*x45*x48 &
                 *(-40.0_dp*x16-80.0_dp*(x26+x44+x5+x57+x9))/x3**6+x131*(4.0_dp*x16+8.0_dp*(x26+x44+x5+x57+x9)) &
                 +x4*((1.0_dp/4.0_dp)*Q(1)*omega*x60/rs**(3.0_dp/2.0_dp)-1.0_dp/4.0_dp*Q(2)*phi2(0)*x2*x55 &
                      -16.0_dp*ed0(1)*x132*x63-ed0(3)*x8-x133*x70-x134*x69-x18*(ed0(3)*x106+g0(2)*x25+48.0_dp*x13*x73+ &
                                                             20.0_dp*x135*x23*x24+120.0_dp*x22*x46+10.0_dp*x46*x6*x90)-56.0_dp*x19 &
                      *x64-x27*(12.0_dp*rs*x21*x30-x10*x12*x28*x6*(x136*x77+9.0_dp/4.0_dp*(x34+x37)) &
                                +x134*x29+x31*(-d3_1**2*x43+d3_1*x38*x40*x83+x137*(-x14+1.0_dp)*9.0_dp/20.0_dp-x139*x145 &
                                               -9.0_dp/80.0_dp*(x142+x143)-27.0_dp/40.0_dp*x144*x40*x85)+x70*x91 &
                                +x76*(x136*x137+x138*x139+x77*(9.0_dp/8.0_dp)*(x14-1.0_dp-x81+x82))) &
                      -10.0_dp*x28*x6*x67-x45*(ed0(3)*x123+g0(2)*x50+24.0_dp*(rs*x21*x23*x24+x0*x21*x90) &
                                               +72.0_dp*x0*x49+x135*(-45.0_dp/16.0_dp*(x34+x37+x94) &
                                                                     +(9.0_dp/16.0_dp)*x53) &
                                               +48.0_dp*x28*x88+x51 &
                                               *(-d2_1**2*x41*x54*x55*x56+d2_1*x147*x55*x56-d2_1*x148*x41 &
                                                 +(27.0_dp/160.0_dp)*phi8(0)*x144*x52-x126*x41*x56*x84-9.0_dp/64.0_dp* &
                                                 (x142+x143)+x147*x38*x56)+x92*(x146*x97 &
                                                                       +x146*x98-45.0_dp/32.0_dp*(x81+x82+x96)-9.0_dp/16.0_dp*x95)))
         ed(4) = ed0(4)+x4*(-ed0(4)*x8-8*ed0(2)*x132*x63-3.0_dp/2.0_dp*phi2(0)*phi2(1)*x61-x103*x133-x104*x149-x18* &
                            (ed0(4)*x106+ed0(2)*x21*x72+x108*x150+(15.0_dp/4.0_dp)*x23*x92*zeta) &
                            -x27*(x103*x91+x149*x89+x31*(-d3_1*x116+x115+x119+x121-x145*x151-9.0_dp/80.0_dp*(x154+x155) &
                                                         +(3.0_dp/80.0_dp)*(x157+x158)) &
                                  +x76*(-9.0_dp/16.0_dp*(x113-x114)+x138*x151+(3.0_dp/16.0_dp)*(x152+x153))) &
                            -x45*(ed0(4)*x123+ed0(2)*x48*x87+9.0_dp*x0*x21*x23*zeta+x124*x150 &
                                  +x51*(-d2_1*x126*x55*x56*zeta+(9.0_dp/32.0_dp)*d2_3*x127-9.0_dp/160.0_dp*phi8(1)*x52*x84+x128 &
                                        +x129-x148*zeta-9.0_dp/64.0_dp*(x154+x155)+(3.0_dp/64.0_dp)*(x157+x158)) &
                                  +x92*(-45.0_dp/64.0_dp*(x113+x114)+(9.0_dp/64.0_dp)*x125+x127*x146 &
                                        +(15.0_dp/64.0_dp)*(x152+x153))))+x59*(x101+x102+x105+x109+x122+x130)
         ed(5) = ed0(5)+x4*(-6.0_dp*Q(0)*phi2(0)*phi2(1)**2-ed0(5)*x8-phi2(2)*x100-x104*x110 &
                            -x18*(ed0(5)*x106+x51*((3.0_dp/4.0_dp)*g0(0)-3.0_dp/8.0_dp)) &
                            -x27*(x111+x31*(-9.0_dp/160.0_dp*(g2m(0)+g2p(0))+(3.0_dp/40.0_dp)*(x159+x160) &
                                            -1.0_dp/20.0_dp*(x162+x163)-1.0_dp/80.0_dp*(x165+x166)+x77*9.0_dp/40.0_dp*(-d3_2+rs))) &
                            -x45*(ed0(5)*x123+x21*x28*(3.0_dp*g0(0)-3.0_dp/2.0_dp)+x51*(-9.0_dp/128.0_dp*(g2m(0)+g2p(0)) &
                                                                                        +(9.0_dp/320.0_dp)*phi8(2)*x38*x52 &
                                                                                        +x126*x55*x56+(3.0_dp/32.0_dp) &
                                                                                        *(x159+x160)-1.0_dp/16.0_dp &
                                                                                        *(x162+x163)-1.0_dp/64.0_dp &
                                                                                        *(x165+x166))))
      END IF

   END SUBROUTINE ec_lsd_sr

! **************************************************************************************************
!> \brief ...
!> \param rho ...
!> \param omega ...
!> \param ed ...
!> \param order ...
! **************************************************************************************************
   SUBROUTINE ec_lda_sr(rho, omega, ed, order)
      REAL(KIND=dp), INTENT(IN)                          :: rho, omega
      REAL(KIND=dp), INTENT(OUT)                         :: ed(0:2)
      INTEGER, INTENT(IN)                                :: order

      CHARACTER(LEN=*), PARAMETER :: routineN = 'ec_lda_sr', routineP = moduleN//':'//routineN

      REAL(KIND=dp) :: ed0(0:2), g0(0:2), g2(0:2), Q(0:2), rs, x0, x1, x10, x11, x12, x13, x14, &
         x15, x16, x17, x18, x19, x2, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x3, x30, &
         x31, x32, x33, x34, x35, x36, x37, x38, x39, x4, x40, x41, x42, x43, x44, x45, x46, x47, &
         x48, x49, x5, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x6, x60, x61, x62, x63, &
         x64, x65, x66, x67, x68, x69, x7, x70, x71, x72, x73, x74, x75, x76, x77, x78, x79, x8, &
         x80, x81, x82, x83, x84, x85, x86, x87, x88, x89, x9

      rs = (3.0_dp/4.0_dp/pi/rho)**(1.0_dp/3.0_dp)

      ed = 0.0_dp

      ed0 = 0.0_dp
      CALL pw_lda_ed_loc(rho, ed0, order)

      g0 = 0.0_dp
      CALL calc_g0(rs, g0, order)

      Q = 0.0_dp
      CALL calc_Q(omega*SQRT(rs), Q, order)

      g2 = 0.0_dp
      CALL calc_g2(2.0_dp**(1.0_dp/3.0_dp)*rs, g2, order)

      x0 = rs**2
      x1 = beta**2
      x2 = omega**2
      x3 = x0*x1*x2+1.0_dp
      x4 = x3**(-4)
      x5 = beta**8
      x6 = omega**8
      x7 = rs**8
      x8 = x5*x6*x7
      x9 = ed0(0)*x8
      x10 = SQRT(2.0_dp)
      x11 = omega**5
      x12 = oorootpi
      x13 = rs**5
      x14 = g0(0)*x10*x11*x12*x13*x5
      x15 = omega**6
      x16 = rs**6
      x17 = 4.0_dp*x16
      x18 = beta**6
      x19 = ed0(0)*x18
      x20 = x13*x5
      x21 = x15*(x17*x19+x20*(-3.0_dp/8.0_dp*g0(0)+3.0_dp/16.0_dp))
      x22 = omega**3
      x23 = rs**3
      x24 = g0(0)*x10*x12
      x25 = x10*x12*x13*x5
      x26 = 2.0_dp**(1.0_dp/3.0_dp)
      x27 = g2(0)
      x28 = 1.0_dp/x0
      x29 = EXP(-d3_1*rs)
      x30 = x28*x29
      x31 = x30*((9.0_dp/80.0_dp)*d3_2-9.0_dp/80.0_dp*rs)
      x32 = x22*(-2*x18*x23*x24+x25*(-9.0_dp/160.0_dp*x27+x31))
      x33 = omega**4
      x34 = rs**4
      x35 = 6.0_dp*x34
      x36 = beta**4
      x37 = ed0(0)*x36
      x38 = alpha2**(-1)
      x39 = x28*x38
      x40 = (9.0_dp/64.0_dp)*d2_2
      x41 = d2_3*rs
      x42 = (9.0_dp/64.0_dp)*x41
      x43 = 1.0_dp/rs
      x44 = EXP(-d2_1*rs)
      x45 = x43*x44
      x46 = x45*(x40-x42)
      x47 = x33*(x18*x23*(-3.0_dp/2.0_dp*g0(0)+3.0_dp/4.0_dp)+x20*(-9.0_dp/128.0_dp*x27+(9.0_dp/320.0_dp)*x39+x46)+x35*x37)

      ed(0) = ed0(0)+x4*(-Q(0)+(1.0_dp/2.0_dp)*x14-x21-x32-x47-x9)
      IF (order >= 1) THEN
         x48 = x3**(-5)
         x49 = rs*x1*x2*x48
         x50 = 8.0_dp*Q(0)-4*x14+8*x21+8*x32+8*x47+8*x9
         x51 = Q(1)*omega
         x52 = 1.0_dp/SQRT(rs)
         x53 = rs**7
         x54 = ed0(0)*x5*x6
         x55 = g0(0)*x10*x11*x12
         x56 = g0(1)*x5
         x57 = (1.0_dp/2.0_dp)*x10*x11*x12*x13
         x58 = ed0(1)*x18
         x59 = (3.0_dp/8.0_dp)*x13
         x60 = x34*x5
         x61 = 24.0_dp*x13*x19+x17*x58-x56*x59+x60*(-15.0_dp/8.0_dp*g0(0)+15.0_dp/16.0_dp)
         x62 = g0(1)*x18
         x63 = 2.0_dp*x10*x12*x23
         x64 = x10*x12*x34*x5
         x65 = g2(1)*x26
         x66 = (9.0_dp/80.0_dp)*x30
         x67 = 1.0_dp/x23
         x68 = x29*x67
         x69 = d3_1*(-d3_2+rs)
         x70 = -6.0_dp*x0*x18*x24+x25*(-9.0_dp/160.0_dp*x65+x66*x69-x66+x68*(-9.0_dp/40.0_dp*d3_2+(9.0_dp/40.0_dp)*rs))-x62*x63 &
               +x64*(-9.0_dp/32.0_dp*x27+x30*((9.0_dp/16.0_dp)*d3_2-9.0_dp/16.0_dp*rs))
         x71 = ed0(1)*x36
         x72 = (3.0_dp/2.0_dp)*x23
         x73 = x38*x67
         x74 = d2_3*x43*x44
         x75 = x28*x44
         x76 = -d2_2+x41
         x77 = d2_1*x43*x44*x76
         x78 = x0*x18*(-9.0_dp/2.0_dp*g0(0)+9.0_dp/4.0_dp)+x20*(-9.0_dp/128.0_dp*x65-9.0_dp/160.0_dp*x73-9.0_dp/64.0_dp*x74 &
                                               +x75*(-x40+x42)+(9.0_dp/64.0_dp)*x77)+24*x23*x37+x35*x71+x60*(-45.0_dp/128.0_dp*x27 &
                                                     +(9.0_dp/64.0_dp)*x39+x45*((45.0_dp/64.0_dp)*d2_2-45.0_dp/64.0_dp*x41))-x62*x72

         ed(1) = ed0(1)+x4*(-ed0(1)*x8-x15*x61-x22*x70-x33*x78+(5.0_dp/2.0_dp)*x34*x5*x55-1.0_dp/2.0_dp*x51*x52 &
                            -8.0_dp*x53*x54+x56*x57)+x49*x50
      END IF

      IF (order >= 2) THEN
         x79 = g0(2)*x5
         x80 = x23*x5
         x81 = x10*x12
         x82 = g0(2)*x18
         x83 = (9.0_dp/4.0_dp)*d3_2
         x84 = (9.0_dp/4.0_dp)*rs
         x85 = (9.0_dp/8.0_dp)*x30
         x86 = 2.0_dp**(2.0_dp/3.0_dp)*g2(2)
         x87 = (9.0_dp/20.0_dp)*x68
         x88 = 1.0_dp/x34
         x89 = (9.0_dp/32.0_dp)*d2_3*x44

         ed(2) = ed0(2)+x0*x33*x36*(40*x14-80.0_dp*(Q(0)+x21+x32+x47+x9))/x3**6+x1*x2*x48*x50 &
                 +x4*(-1.0_dp/4.0_dp*Q(2)*x2*x43-16.0_dp*ed0(1)*x5*x53*x6-ed0(2)*x8+5.0_dp*g0(1)*x10*x11*x12*x34*x5 &
                      -x15*(ed0(2)*x17*x18-15.0_dp/4.0_dp*g0(1)*x34*x5+48.0_dp*x13*x58+120.0_dp*x19*x34-x59*x79 &
                            +x80*(-15.0_dp/2.0_dp*g0(0)+15.0_dp/4.0_dp))-56.0_dp*x16*x54 &
                      -x22*(-12.0_dp*g0(1)*x0*x18*x81-12.0_dp*rs*x18*x24+x23*x5*x81*(-9.0_dp/8.0_dp*x27+x30*(x83-x84)) &
                            +x25*(d3_1**2*x31+(9.0_dp/40.0_dp)*d3_1*x30+x29*x88*((27.0_dp/40.0_dp)*d3_2-27.0_dp/40.0_dp*rs) &
                                  -x69*x87-9.0_dp/160.0_dp*x86+x87)-x63*x82+x64*(-9.0_dp/16.0_dp*x65+x68*(-x83+x84)+x69*x85-x85))+ &
                      10.0_dp*x23*x5*x55-x33*(ed0(2)*x35*x36-9.0_dp*g0(1)*x0*x18-rs*x18*(9.0_dp*g0(0)-9.0_dp/2.0_dp) &
                                              +72.0_dp*x0*x37+x20*(d2_1**2*x46+d2_1*x43*x89-9.0_dp/32.0_dp*d2_1*x75*x76+x28*x89 &
                                                                   +(27.0_dp/160.0_dp)*x38*x88 &
                                                                   +x44*x67*((9.0_dp/32.0_dp)*d2_2-9.0_dp/32.0_dp*x41)-9.0_dp/ &
                                                                   128.0_dp*x86)+48.0_dp*x23*x71 &
                                              +x60*(-45.0_dp/64.0_dp*x65-9.0_dp/16.0_dp*x73-45.0_dp/32.0_dp*x74 &
                                                    +x75*(-45.0_dp/32.0_dp*d2_2+(45.0_dp/32.0_dp)*x41)+(45.0_dp/32.0_dp)*x77) &
                                              -x72*x82+x80*(-45.0_dp/32.0_dp*x27+(9.0_dp/16.0_dp)*x39 &
                                                            +x45*((45.0_dp/16.0_dp)*d2_2-45.0_dp/16.0_dp*x41))) &
                      +x57*x79+(1.0_dp/4.0_dp)*x51/rs**(3.0_dp/2.0_dp))+x49*(8.0_dp*Q(1)*omega*x52+128.0_dp*ed0(0)*x5*x53*x6 &
                                                                             +16.0_dp*ed0(1)*x5*x6*x7-40.0_dp*g0(0)*x10*x11*x12 &
                                                                             *x34*x5-8.0_dp*g0(1)*x10*x11*x12*x13*x5 &
                                                                             +16.0_dp*x15*x61+16.0_dp*x22*x70+16.0_dp*x33*x78)
      END IF

   END SUBROUTINE ec_lda_sr

! **************************************************************************************************
!> \brief ...
!> \param rho ...
!> \param omega ...
!> \param ed ...
!> \param order ...
! **************************************************************************************************
   PURE SUBROUTINE ex_lda_sr(rho, omega, ed, order)
      REAL(KIND=dp), INTENT(IN)                          :: rho, omega
      REAL(KIND=dp), INTENT(OUT)                         :: ed(0:2)
      INTEGER, INTENT(IN)                                :: order

      CHARACTER(LEN=*), PARAMETER :: routineN = 'ex_lda_sr', routineP = moduleN//':'//routineN

      REAL(KIND=dp) :: rs, x0, x1, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x2, x20, x21, &
         x22, x23, x24, x25, x26, x27, x28, x29, x3, x30, x31, x32, x33, x34, x35, x4, x5, x6, x7, &
         x8, x9

      ed = 0.0_dp

      rs = (3.0_dp/4.0_dp/pi/rho)**(1.0_dp/3.0_dp)

      x0 = 18.0_dp**(1.0_dp/3.0_dp)
      x1 = pi**(5.0_dp/3.0_dp)
      x2 = rs**4
      x3 = x0/(x1*x2)
      x4 = 12.0_dp**(1.0_dp/3.0_dp)
      x5 = pi**(1.0_dp/3.0_dp)
      x6 = 1.0_dp/x5
      x7 = omega*rs*x4*x6
      x8 = 1.0_dp/pi
      x9 = omega**3
      x10 = rs**3
      x11 = x10*x8*x9
      x12 = (2.0_dp/9.0_dp)*x11
      x13 = rootpi
      x14 = 1.0_dp/omega
      x15 = 1.0_dp/rs
      x16 = ERF(0.5_dp*x0*x14*x15*x5)
      x17 = omega**(-2)
      x18 = pi**(2.0_dp/3.0_dp)
      x19 = rs**2
      x20 = EXP(-0.75_dp*x17*x18*x4/x19)
      x21 = -x12+(1.0_dp/3.0_dp)*x7
      x22 = x12+x13*x16+x20*x21-1.0_dp/2.0_dp*x7
      x23 = omega*rs*x22*x4*x6

      ed(0) = x3*(1.0_dp/8.0_dp*x23-9.0_dp/32.0_dp)
      IF (order >= 1) THEN
         x24 = x0*x15/x18
         x25 = omega*x18*x2*x22*x4
         x26 = rs**6
         x27 = (8.0_dp/27.0_dp)*x26*x9
         x28 = omega*x18*x2*x4
         x29 = pi**(4.0_dp/3.0_dp)
         x30 = x27-4.0_dp/27.0_dp*x28
         x31 = omega*rs*x4*x6*(-2.0_dp/3.0_dp*rs*x1*x17*x20*x21*x4+(4.0_dp/9.0_dp)*x0*x14*x19*x20*x29+x20*x30-x27&
+(2.0_dp/9.0_dp)*x28)

         ed(1) = x24*(2.0_dp/9.0_dp*x23-1.0_dp/2.0_dp)+x3*(-(1.0_dp/18.0_dp)*x25+1.0_dp/8.0_dp*x31)
      END IF

      IF (order >= 2) THEN
         x32 = (8.0_dp/81.0_dp)*x7
         x33 = rs**7
         x34 = omega*x1*x33*x4
         x35 = rs**9

         ed(2) = x0*x19*x5*(x22*x32-2.0_dp/9.0_dp)+x24*(-(16.0_dp/81.0_dp)*x25+4.0_dp/9.0_dp*x31) &
                 -x3*(x28*((2.0_dp/81.0_dp)*omega*x18*x2*x4-2.0_dp/27.0_dp*rs*x1*x17*x20*x21*x4 &
                           +(4.0_dp/81.0_dp)*x0*x14*x19*x20*x29+(1.0_dp/9.0_dp)*x20*x30-8.0_dp/243.0_dp*x26*x9) &
                      -x34*(-4.0_dp/81.0_dp*omega*rs*x4*x6+(16.0_dp/729.0_dp)*x10*x8*x9+(8.0_dp/81.0_dp)*x13*x16 &
                            +(8.0_dp/81.0_dp)*x20*x21)-x7*(-4.0_dp/81.0_dp*omega*x1*x33*x4-4.0_dp/81.0_dp &
                                                           *pi**(7.0_dp/3.0_dp)*rs**5*x0*x14*x20-1.0_dp/6.0_dp*rs*x1*x17*x20 &
                                                           *x30*x4-2.0_dp/9.0_dp*pi**3*x10*x20/x9+(1.0_dp/8.0_dp) &
                                                           *pi**(8.0_dp/3.0_dp)*x17*x2*x20*x4 &
                                                           *(-16.0_dp/243.0_dp*x11+x32)+(1.0_dp/8.0_dp)*x20 &
                                                           *((64.0_dp/243.0_dp)*x34-64.0_dp/81.0_dp*pi*x35*x9) &
                                                           +(8.0_dp/81.0_dp)*pi*x35*x9+(1.0_dp/8.0_dp)*pi**(10.0_dp/3.0_dp) &
                                                           *x0*x19*x20*(-16.0_dp/81.0_dp*x11+(8.0_dp/27.0_dp)*x7)/omega**4))
      END IF

   END SUBROUTINE ex_lda_sr

END MODULE xc_sr_lda
