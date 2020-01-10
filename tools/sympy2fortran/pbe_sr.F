
   ELEMENTAL SUBROUTINE calc_s(rho, drho, max_deriv, s, s_rho, s_drho, s_rho_rho, s_rho_drho, s_drho_drho)
      REAL(KIND=dp), INTENT(IN) :: rho, drho
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: s, s_rho, s_drho, s_rho_rho, s_rho_drho, s_drho_drho

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_s", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2


      x0 = 3**(2.0_dp/3.0_dp)/pi**(2.0_dp/3.0_dp)
      x1 = (1.0_dp/6.0_dp)*x0/rho**(4.0_dp/3.0_dp)
      x2 = (2.0_dp/9.0_dp)*x0/rho**(7.0_dp/3.0_dp)
      s = drho*x1
      s_rho = -drho*x2
      s_drho = x1
      s_rho_rho = (14.0_dp/27.0_dp)*drho*x0/rho**(10.0_dp/3.0_dp)
      s_rho_drho = -x2
      s_drho_drho = 0

   END SUBROUTINE calc_s

   ELEMENTAL SUBROUTINE calc_b(rho, mu, max_deriv, b, b_rho, b_rho_rho)
      REAL(KIND=dp), INTENT(IN) :: rho, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: b, b_rho, b_rho_rho

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_b", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64, x65, x66, b_T0, b_T, b_T_mu, b_T_mu_mu

      CALL calc_b_T(rho, max_deriv, b_T, b_T_mu, b_T_mu_mu)

      b_T0 = 7.0_dp/81.0_dp
      x0 = 1.0_dp/b_T0
      x1 = 3**(1.0_dp/3.0_dp)
      x2 = pi**(4.0_dp/3.0_dp)
      x3 = 1.0_dp/x2
      x4 = mu**2
      x5 = rho**(2.0_dp/3.0_dp)
      x6 = 1.0_dp/x5
      x7 = x1*x3*x4*x6
      x8 = exp(-0.0833333333333333_dp*alpha_X*x7)
      x9 = c**2
      x10 = 1.0_dp/x9
      x11 = x10*x7
      x12 = rho**(-4.0_dp/3.0_dp)
      x13 = 3**(2.0_dp/3.0_dp)
      x14 = pi**(-8.0_dp/3.0_dp)
      x15 = c**(-4)
      x16 = mu**4
      x17 = x12*x13*x14*x15*x16
      x18 = 1.0_dp/x4
      x19 = x13*x18*x2*x5
      x20 = exp(x19)
      x21 = x1*x10*x20*x3*x4*x6
      x22 = -11.0_dp/6.0_dp*x11 - x17 + x21*(x11 - 7.0_dp/6.0_dp) - 1
      x23 = pi**(-1.0_dp/6.0_dp)
      x24 = 1.0_dp/c
      x25 = rho**(1.0_dp/3.0_dp)
      x26 = 1.0_dp/x25
      x27 = erf(2.14502939711103_dp*c*x1*x25/mu)
      x28 = -x17*(x11 - 6) + x21*(6*mu*x13*x23*x24*x26*x27 - 9*x1*x10*x3*x4*x6 &
      + 9437157.0_dp/2.0_dp)
      x29 = 1.0_dp/x28
      x30 = b_PBE*x0*x22*x29*x8
      x31 = rho**(-5.0_dp/3.0_dp)
      x32 = x1*x10*x3*x31*x4
      x33 = rho**(-7.0_dp/3.0_dp)
      x34 = x13*x14*x15*x16*x33
      x35 = 1.0_dp/rho
      x36 = x10*x20*x35
      x37 = x1*x10*x20*x3*x31*x4
      x38 = -2.0_dp/3.0_dp*x20*x34 + (11.0_dp/9.0_dp)*x32 + (4.0_dp/3.0_dp)*x34 + &
      x36*(2*x11 - 7.0_dp/3.0_dp) + x37*(-2.0_dp/3.0_dp*x11 + 7.0_dp/9.0_dp)
      x39 = b_PBE*x0*x29*x38*x8
      x40 = b_PBE*x0*x8
      x41 = x28**(-2)
      x42 = pi**(-4)
      x43 = c**(-6)
      x44 = mu**6
      x45 = x42*x43*x44/rho**3
      x46 = (4.0_dp/3.0_dp)*x11
      x47 = mu*x13*x23*x24*x26*x27
      x48 = exp(-x19*x9)
      x49 = x35*x48
      x50 = mu*x12*x13*x23*x24*x27
      x51 = x21*(-6*x32 - 12*x49 + 2*x50) + x34*(-x46 + 8) + x36*(18*x11 - 12* &
      x47 - 9437157) + x37*(-6*x11 + 4*x47 + 3145719) - 2*x45
      x52 = x22*x40*x41*x51
      x53 = rho**(-8.0_dp/3.0_dp)
      x54 = rho**(-10.0_dp/3.0_dp)
      x55 = (1.0_dp/9.0_dp)*alpha_X*x1*x3*x31*x4
      x56 = x1*x10*x3*x4*x53
      x57 = x13*x14*x15*x16*x54
      x58 = rho**(-2)
      x59 = x10*x20*x58
      x60 = x1*x10*x20*x3*x4*x53
      x61 = x10*x12*x13*x18*x2*x20
      x62 = b_PBE*x0*x51*x8
      x63 = 12*x11
      x64 = 8*x47
      x65 = 16*x48
      x66 = (8.0_dp/3.0_dp)*mu*x13*x23*x24*x27
      b = x30
      b_rho = (1.0_dp/18.0_dp)*alpha_X*b_PBE*x0*x1*x22*x29*x3*x31*x4*x8 + x39 + &
      x52
      b_rho_rho = (1.0_dp/324.0_dp)*alpha_X**2*x13*x14*x16*x30*x54 - 5.0_dp/ &
      54.0_dp*alpha_X*b_PBE*x0*x1*x22*x29*x3*x4*x53*x8 + x22*x40*x41*( &
      x21*(8*x12*x13*x18*x2*x48*x9 - x33*x66 + 10*x56 + x58*x65) + x36* &
      (-24*x32 - 48*x49 + 8*x50) + x37*(-x12*x66 + 8*x32 + x35*x65) + &
      x57*((28.0_dp/9.0_dp)*x11 - 56.0_dp/3.0_dp) + x59*(-30*x11 + 20*x47 + &
      15728595) + x60*(10*x11 - 20.0_dp/3.0_dp*x47 - 5242865) + x61*(x63 &
      - x64 - 6291438) + (26.0_dp/3.0_dp)*x42*x43*x44/rho**4) + x22*x62*( &
      x21*(-12*x32 - 24*x49 + 4*x50) + x34*(-8.0_dp/3.0_dp*x11 + 16) + &
      x36*(36*x11 - 24*x47 - 18874314) + x37*(-x63 + x64 + 6291438) - 4 &
      *x45)/x28**3 + x29*x40*(-8.0_dp/3.0_dp*x1*x15*x20*x3*x4*x53 + 2*x20 &
      *x57 - 55.0_dp/27.0_dp*x56 - 28.0_dp/9.0_dp*x57 + x59*(-10.0_dp/3.0_dp* &
      x11 + 35.0_dp/9.0_dp) + x60*((10.0_dp/9.0_dp)*x11 - 35.0_dp/27.0_dp) + &
      x61*(x46 - 14.0_dp/9.0_dp)) + 2*x38*x41*x62 + x39*x55 + x52*x55

   END SUBROUTINE calc_b

   ELEMENTAL SUBROUTINE calc_Fx(rho, drho, mu, max_deriv, Fx, Fx_rho, Fx_drho, Fx_rho_rho, Fx_rho_drho, Fx_drho_drho)
      REAL(KIND=dp), INTENT(IN) :: rho, drho, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: Fx, Fx_rho, Fx_drho, Fx_rho_rho, Fx_rho_drho, Fx_drho_drho

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_Fx", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, kappa, b, b_rho, b_rho_rho, s, s_rho, s_drho, s_rho_rho, s_rho_drho, s_drho_drho

      CALL calc_b(rho, drho, max_deriv, b, b_rho, b_rho_rho)
      CALL calc_s(rho, drho, max_deriv, s, s_rho, s_drho, s_rho_rho, s_rho_drho, s_drho_drho)

      kappa = 0.84_dp
      x0 = 1.0_dp/kappa
      x1 = s**2*x0
      x2 = 2*b*s*x0
      x3 = 2*b*s
      x4 = b_rho*s
      x5 = 2*b
      Fx = -kappa*(b*x1 + 1) + kappa + 1
      Fx_rho = -kappa*(b_rho*x1 + s_rho*x2)
      Fx_drho = -s_drho*x3
      Fx_rho_rho = -kappa*(b_rho_rho*x1 + s_rho**2*x0*x5 + 4*s_rho*x0*x4 + &
      s_rho_rho*x2)
      Fx_rho_drho = -kappa*(2*b*s_drho*s_rho*x0 + 2*s_drho*x0*x4 + s_rho_drho* &
      x2)
      Fx_drho_drho = -s_drho**2*x5 - s_drho_drho*x3

   END SUBROUTINE calc_Fx

   ELEMENTAL SUBROUTINE calc_phi(zeta, max_deriv, phi, phi_zeta, phi_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: phi, phi_zeta, phi_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_phi", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1


      x0 = zeta + 1
      x1 = -zeta + 1
      phi = (1.0_dp/2.0_dp)*x0**(2.0_dp/3.0_dp) + (1.0_dp/2.0_dp)*x1**(2.0_dp/3.0_dp)
      phi_zeta = -1.0_dp/3.0_dp/x1**(1.0_dp/3.0_dp) + (1.0_dp/3.0_dp)/x0**(1.0_dp/ &
      3.0_dp)
      phi_zeta_zeta = -1.0_dp/9.0_dp/x1**(4.0_dp/3.0_dp) - 1.0_dp/9.0_dp/x0**(4.0_dp/ &
      3.0_dp)

   END SUBROUTINE calc_phi

   ELEMENTAL SUBROUTINE calc_t_ab(rho, zeta, drho, max_deriv, t_ab, t_ab_rho, t_ab_zeta, t_ab_drho, t_ab_rho_rho, t_ab_rho_zeta, t_ab_rho_drho, t_ab_zeta_zeta, t_ab_zeta_drho, t_ab_drho_drho)
      REAL(KIND=dp), INTENT(IN) :: rho, zeta, drho
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: t_ab, t_ab_rho, t_ab_zeta, t_ab_drho, t_ab_rho_rho, t_ab_rho_zeta, t_ab_rho_drho, t_ab_zeta_zeta, t_ab_zeta_drho, t_ab_drho_drho

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_t_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, phi, phi_zeta, phi_zeta_zeta

      CALL calc_phi(rho, zeta, drho, max_deriv, phi, phi_zeta, phi_zeta_zeta)

      x0 = rho**(-7.0_dp/6.0_dp)
      x1 = (1.0_dp/12.0_dp)*x0
      x2 = 3**(5.0_dp/6.0_dp)
      x3 = pi**(1.0_dp/6.0_dp)
      x4 = x2*x3/phi
      x5 = x1*x4
      x6 = rho**(-13.0_dp/6.0_dp)
      x7 = (7.0_dp/72.0_dp)*x4*x6
      x8 = phi**(-2)
      x9 = phi_zeta*x2*x3*x8
      x10 = x1*x9
      x11 = drho*x0*x2*x3
      t_ab = drho*x5
      t_ab_rho = -drho*x7
      t_ab_zeta = -drho*x10
      t_ab_drho = x5
      t_ab_rho_rho = (91.0_dp/432.0_dp)*drho*x4/rho**(19.0_dp/6.0_dp)
      t_ab_rho_zeta = (7.0_dp/72.0_dp)*drho*x6*x9
      t_ab_rho_drho = -x7
      t_ab_zeta_zeta = -1.0_dp/12.0_dp*phi_zeta_zeta*x11*x8 + (1.0_dp/6.0_dp)* &
      phi_zeta**2*x11/phi**3
      t_ab_zeta_drho = -x10
      t_ab_drho_drho = 0

   END SUBROUTINE calc_t_ab

   ELEMENTAL SUBROUTINE calc_beta_ab(rho, zeta, mu, max_deriv, beta_ab, beta_ab_rho, beta_ab_zeta, beta_ab_rho_rho, beta_ab_rho_zeta, beta_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rho, zeta, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: beta_ab, beta_ab_rho, beta_ab_zeta, beta_ab_rho_rho, beta_ab_rho_zeta, beta_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_beta_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, beta_PBE, alpha_C, Ecmuab, Ecmuab_rho, Ecmuab_zeta, Ecmuab_rho_rho, Ecmuab_rho_zeta, Ecmuab_zeta_zeta, Ecab, Ecab_rho, Ecab_zeta, Ecab_rho_rho, Ecab_rho_zeta, Ecab_zeta_zeta

      CALL calc_Ecmuab(rho, zeta, max_deriv, Ecmuab, Ecmuab_rho, Ecmuab_zeta, Ecmuab_rho_rho, Ecmuab_rho_zeta, Ecmuab_zeta_zeta)
      CALL calc_Ecab(rho, zeta, max_deriv, Ecab, Ecab_rho, Ecab_zeta, Ecab_rho_rho, Ecab_rho_zeta, Ecab_zeta_zeta)

      beta_PBE = 0.066725_dp
      alpha_C = 2.78_dp
      x0 = 1.0_dp/Ecab
      x1 = (Ecmuab*x0)**alpha_C
      x2 = Ecab**2
      x3 = 1.0_dp/x2
      x4 = Ecab_rho*x3
      x5 = -Ecmuab*x4 + Ecmuab_rho*x0
      x6 = 1.0_dp/Ecmuab
      x7 = Ecab*alpha_C*beta_PBE*x1*x6
      x8 = Ecmuab*x3
      x9 = -Ecab_zeta*x8 + Ecmuab_zeta*x0
      x10 = alpha_C*beta_PBE*x1*x5*x6
      x11 = Ecmuab**(-2)
      x12 = Ecab*alpha_C*beta_PBE*x1*x11*x5
      x13 = alpha_C**2*beta_PBE*x1*x11*x2
      x14 = 2*Ecmuab/Ecab**3
      x15 = Ecab_zeta*x3
      x16 = alpha_C*beta_PBE*x1*x9
      beta_ab = beta_PBE*x1
      beta_ab_rho = x5*x7
      beta_ab_zeta = x7*x9
      beta_ab_rho_rho = Ecab_rho*x10 - Ecmuab_rho*x12 + x13*x5**2 + x7*( &
      Ecab_rho**2*x14 - Ecab_rho_rho*x8 - 2*Ecmuab_rho*x4 + &
      Ecmuab_rho_rho*x0)
      beta_ab_rho_zeta = Ecab_zeta*x10 - Ecmuab_zeta*x12 + x13*x5*x9 + x7*( &
      Ecab_rho*Ecab_zeta*x14 - Ecab_rho_zeta*x8 - Ecmuab_rho*x15 + &
      Ecmuab_rho_zeta*x0 - Ecmuab_zeta*x4)
      beta_ab_zeta_zeta = -Ecab*Ecmuab_zeta*x11*x16 + Ecab_zeta*x16*x6 + x13* &
      x9**2 + x7*(Ecab_zeta**2*x14 - Ecab_zeta_zeta*x8 - 2*Ecmuab_zeta* &
      x15 + Ecmuab_zeta_zeta*x0)

   END SUBROUTINE calc_beta_ab

   ELEMENTAL SUBROUTINE calc_A_ab(rho, zeta, mu, max_deriv, A_ab, A_ab_rho, A_ab_zeta, A_ab_rho_rho, A_ab_rho_zeta, A_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rho, zeta, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: A_ab, A_ab_rho, A_ab_zeta, A_ab_rho_rho, A_ab_rho_zeta, A_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_A_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, gam, Ecmuab, Ecmuab_rho, Ecmuab_zeta, Ecmuab_rho_rho, Ecmuab_rho_zeta, Ecmuab_zeta_zeta, beta_ab, beta_ab_rho, beta_ab_zeta, beta_ab_rho_rho, beta_ab_rho_zeta, beta_ab_zeta_zeta, phi, phi_zeta, phi_zeta_zeta

      CALL calc_Ecmuab(rho, zeta, max_deriv, Ecmuab, Ecmuab_rho, Ecmuab_zeta, Ecmuab_rho_rho, Ecmuab_rho_zeta, Ecmuab_zeta_zeta)
      CALL calc_beta_ab(rho, zeta, max_deriv, beta_ab, beta_ab_rho, beta_ab_zeta, beta_ab_rho_rho, beta_ab_rho_zeta, beta_ab_zeta_zeta)
      CALL calc_phi(rho, zeta, max_deriv, phi, phi_zeta, phi_zeta_zeta)

      gam = 0.031091_dp
      x0 = 1.0_dp/gam
      x1 = phi**(-3)
      x2 = x0*x1
      x3 = Ecmuab*x2
      x4 = exp(-x3)
      x5 = x4 - 1
      x6 = x0/x5
      x7 = gam**(-2)
      x8 = x5**(-2)
      x9 = Ecmuab_rho*x1*x4*x7*x8
      x10 = beta_ab*x9
      x11 = phi**(-4)
      x12 = 3*Ecmuab*x0*x11
      x13 = -Ecmuab_zeta*x2 + phi_zeta*x12
      x14 = beta_ab*x4*x8
      x15 = beta_ab*x1*x4*x7*x8
      x16 = Ecmuab_rho**2/(gam**3*phi**6)
      x17 = 2*beta_ab*exp(-2.0_dp*x3)/x5**3
      x18 = x0*x13*x4*x8
      x19 = x0*x13**2
      A_ab = beta_ab*x6
      A_ab_rho = beta_ab_rho*x6 + x10
      A_ab_zeta = beta_ab_zeta*x6 - x0*x13*x14
      A_ab_rho_rho = Ecmuab_rho_rho*x15 + 2*beta_ab_rho*x9 + beta_ab_rho_rho* &
      x6 - x14*x16 + x16*x17
      A_ab_rho_zeta = -3*Ecmuab_rho*beta_ab*phi_zeta*x11*x4*x7*x8 - Ecmuab_rho &
      *x1*x13*x17*x7 + Ecmuab_rho_zeta*x15 - beta_ab_rho*x18 + &
      beta_ab_rho_zeta*x6 + beta_ab_zeta*x9 + x10*x13
      A_ab_zeta_zeta = -2*beta_ab_zeta*x18 + beta_ab_zeta_zeta*x6 - x0*x14*( &
      -12*Ecmuab*phi_zeta**2*x0/phi**5 + 6*Ecmuab_zeta*phi_zeta*x0*x11 &
      - Ecmuab_zeta_zeta*x2 + phi_zeta_zeta*x12) - x14*x19 + x17*x19

   END SUBROUTINE calc_A_ab

   ELEMENTAL SUBROUTINE calc_H_ab(rho, zeta, drho, mu, max_deriv, H_ab, H_ab_rho, H_ab_zeta, H_ab_drho, H_ab_rho_rho, H_ab_rho_zeta, H_ab_rho_drho, H_ab_zeta_zeta, H_ab_zeta_drho, H_ab_drho_drho)
      REAL(KIND=dp), INTENT(IN) :: rho, zeta, drho, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: H_ab, H_ab_rho, H_ab_zeta, H_ab_drho, H_ab_rho_rho, H_ab_rho_zeta, H_ab_rho_drho, H_ab_zeta_zeta, H_ab_zeta_drho, H_ab_drho_drho

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_H_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64, x65, x66, x67, x68, x69, x70, x71, x72, x73, x74, x75, x76, x77, x78, x79, x80, x81, x82, x83, x84, x85, x86, x87, x88, x89, x90, x91, x92, x93, x94, x95, x96, x97, x98, x99, x100, x101, x102, x103, x104, x105, x106, x107, x108, x109, x110, x111, x112, x113, x114, x115, x116, x117, x118, x119, x120, x121, x122, x123, x124, x125, x126, x127, x128, x129, gam, beta_ab, beta_ab_rho, beta_ab_zeta, beta_ab_rho_rho, beta_ab_rho_zeta, beta_ab_zeta_zeta, tab, tab_rho, tab_zeta, tab_drho, tab_rho_rho, tab_rho_zeta, tab_rho_drho, tab_zeta_zeta, tab_zeta_drho, tab_drho_drho, A_ab, A_ab_rho, A_ab_zeta, A_ab_rho_rho, A_ab_rho_zeta, A_ab_zeta_zeta, phi, phi_zeta, phi_zeta_zeta

      CALL calc_beta_ab(rho, zeta, drho, max_deriv, beta_ab, beta_ab_rho, beta_ab_zeta, beta_ab_rho_rho, beta_ab_rho_zeta, beta_ab_zeta_zeta)
      CALL calc_t_ab(rho, zeta, drho, max_deriv, tab, tab_rho, tab_zeta, tab_drho, tab_rho_rho, tab_rho_zeta, tab_rho_drho, tab_zeta_zeta, tab_zeta_drho, tab_drho_drho)
      CALL calc_A_ab(rho, zeta, drho, max_deriv, A_ab, A_ab_rho, A_ab_zeta, A_ab_rho_rho, A_ab_rho_zeta, A_ab_zeta_zeta)
      CALL calc_phi(rho, zeta, drho, max_deriv, phi, phi_zeta, phi_zeta_zeta)

      gam = 0.031091_dp
      x0 = tab**2
      x1 = A_ab*x0 + 1
      x2 = x0*x1
      x3 = 1.0_dp/gam
      x4 = A_ab**2
      x5 = tab**4
      x6 = x1 + x4*x5
      x7 = 1.0_dp/x6
      x8 = beta_ab*x2*x3*x7 + 1
      x9 = gam*log(x8)
      x10 = 1.0_dp/x8
      x11 = gam*phi*x10
      x12 = x0*x1*x3*x7
      x13 = beta_ab_rho*x12
      x14 = 2*tab_rho
      x15 = beta_ab*tab*x1*x3*x7
      x16 = x14*x15
      x17 = A_ab_rho*x0
      x18 = 2*A_ab*tab
      x19 = tab_rho*x18
      x20 = x17 + x19
      x21 = beta_ab*x0*x3*x7
      x22 = x20*x21
      x23 = 2*A_ab
      x24 = A_ab_rho*x5
      x25 = 4*tab_rho
      x26 = tab**3
      x27 = x26*x4
      x28 = -x17 - x19 - x23*x24 - x25*x27
      x29 = x6**(-2)
      x30 = beta_ab*x0*x1*x29*x3
      x31 = x28*x30
      x32 = x13 + x16 + x22 + x31
      x33 = beta_ab_zeta*x12
      x34 = 2*tab_zeta
      x35 = x15*x34
      x36 = A_ab_zeta*x0
      x37 = tab_zeta*x18
      x38 = x36 + x37
      x39 = x21*x38
      x40 = 2*A_ab*x5
      x41 = 4*tab_zeta
      x42 = -A_ab_zeta*x40 - x27*x41 - x36 - x37
      x43 = x30*x42
      x44 = x33 + x35 + x39 + x43
      x45 = 2*A_ab*tab_drho
      x46 = beta_ab*x26*x3*x7
      x47 = x45*x46
      x48 = 2*tab_drho
      x49 = x15*x48
      x50 = A_ab*tab*tab_drho
      x51 = tab_drho*x26*x4
      x52 = -2*x50 - 4*x51
      x53 = beta_ab*x2*x29*x3*x52
      x54 = gam*x10*(x47 + x49 + x53)
      x55 = x8**(-2)
      x56 = gam*phi*x55
      x57 = 2*beta_ab*tab*x1*x3*x7
      x58 = beta_ab_rho*tab*x1*x3*x7
      x59 = tab_rho**2
      x60 = 2*beta_ab*x1*x3*x7
      x61 = beta_ab*tab*x20*x3*x7
      x62 = 4*A_ab*tab
      x63 = tab_rho*x62
      x64 = 2*x17
      x65 = x63 + x64
      x66 = beta_ab_rho*x0*x3*x7
      x67 = A_ab_rho_rho*x0
      x68 = tab_rho_rho*x18
      x69 = A_ab_rho*tab
      x70 = x25*x69
      x71 = x23*x59
      x72 = beta_ab*tab*x1*x28*x29*x3
      x73 = 4*A_ab
      x74 = -8*tab_rho*x27 - x24*x73 - x63 - x64
      x75 = beta_ab_rho*x0*x1*x29*x3
      x76 = beta_ab*x0*x28*x29*x3
      x77 = x6**(-3)
      x78 = beta_ab*x0*x1*x28*x3*x77
      x79 = 2*x5
      x80 = 4*x26*x4
      x81 = 12*x0*x4
      x82 = gam*phi_zeta*x10
      x83 = -x33 - x35 - x39 - x43
      x84 = beta_ab*tab_rho*x1*x3*x7
      x85 = beta_ab_zeta*tab*x1*x3*x7
      x86 = beta_ab_zeta*x0*x3*x7
      x87 = beta_ab*tab*x3*x38*x7
      x88 = A_ab_rho_zeta*x0
      x89 = tab_rho_zeta*x18
      x90 = A_ab*tab_rho
      x91 = x34*x90
      x92 = x34*x69
      x93 = A_ab_zeta*tab
      x94 = x14*x93
      x95 = beta_ab_zeta*x0*x1*x29*x3
      x96 = beta_ab*tab*x1*x29*x3*x42
      x97 = beta_ab*x0*x29*x3*x42
      x98 = tab_zeta*x62
      x99 = 2*x36
      x100 = tab_zeta*x26
      x101 = -A_ab_zeta*x5*x73 - 8*x100*x4 - x98 - x99
      x102 = 8*A_ab*A_ab_rho*x26
      x103 = 8*A_ab_zeta*x26
      x104 = -x47 - x49 - x53
      x105 = gam*phi*x104*x55
      x106 = 2*A_ab*tab_drho*x26*x3*x7
      x107 = A_ab*beta_ab*tab_drho*x0*x3*x7
      x108 = 2*beta_ab*tab*tab_drho*x3*x7
      x109 = tab_rho_drho*x18
      x110 = tab_rho*x45
      x111 = x48*x69
      x112 = x0*x1*x29*x3*x52
      x113 = beta_ab*tab*x1*x29*x3*x52
      x114 = beta_ab*x0*x29*x3*x52
      x115 = 2*A_ab*beta_ab*tab_drho*x26*x29*x3
      x116 = 2*beta_ab*tab*tab_drho*x1*x29*x3
      x117 = 12*tab_drho*x0*x4
      x118 = beta_ab*x0*x1*x3*x77*(-4*x50 - 8*x51)
      x119 = tab_zeta**2
      x120 = x98 + x99
      x121 = A_ab_zeta_zeta*x0
      x122 = tab_zeta_zeta*x18
      x123 = x41*x93
      x124 = x119*x23
      x125 = tab_zeta_drho*x18
      x126 = tab_zeta*x45
      x127 = x48*x93
      x128 = tab_drho**2
      x129 = beta_ab*x128*x3*x7
      H_ab = phi*x9
      H_ab_rho = x11*x32
      H_ab_zeta = phi_zeta*x9 + x11*x44
      H_ab_drho = phi*x54
      H_ab_rho_rho = x11*(beta_ab_rho_rho*x12 + tab_rho_rho*x57 + x21*(x67 + &
      x68 + x70 + x71) + x25*x58 + x25*x61 + x25*x72 + x30*(-16*A_ab* &
      A_ab_rho*tab_rho*x26 - A_ab_rho**2*x79 - A_ab_rho_rho*x40 - &
      tab_rho_rho*x80 - x59*x81 - x67 - x68 - x70 - x71) + x59*x60 + &
      x65*x66 + x65*x76 + x74*x75 + x74*x78) - x56*(-x13 - x16 - x22 - &
      x31)**2
      H_ab_rho_zeta = x11*(beta_ab_rho_zeta*x12 + tab_rho_zeta*x57 + x101*x78 &
      + x14*x85 + x14*x87 + x14*x96 + x20*x86 + x20*x97 + x21*(x88 + &
      x89 + x91 + x92 + x94) + x28*x95 + x30*(-A_ab_rho_zeta*x40 - 2* &
      A_ab_zeta*x24 - tab_rho*tab_zeta*x81 - tab_rho_zeta*x80 - &
      tab_zeta*x102 - x103*x90 - x88 - x89 - x91 - x92 - x94) + x34*x58 &
      + x34*x61 + x34*x72 + x34*x84 + x38*x66 + x38*x76 + x42*x75) + &
      x32*x56*x83 + x32*x82
      H_ab_rho_drho = x105*x32 + x11*(beta_ab_rho*x106 + beta_ab_rho*x112 + &
      tab_rho_drho*x57 + x107*x25 + x108*x20 + x113*x14 + x114*x20 + &
      x115*x28 + x116*x28 + x118*x28 + x21*(x109 + x110 + x111) + x30*( &
      -tab_drho*x102 - tab_rho*x117 - tab_rho_drho*x80 - x109 - x110 - &
      x111) + x48*x58 + x48*x84)
      H_ab_zeta_zeta = phi_zeta_zeta*x9 + x11*(beta_ab*x0*x1*x101*x3*x42*x77 + &
      beta_ab_zeta_zeta*x12 + tab_zeta_zeta*x57 + x101*x95 + x119*x60 + &
      x120*x86 + x120*x97 + x21*(x121 + x122 + x123 + x124) + x30*(-16* &
      A_ab*A_ab_zeta*x100 - A_ab_zeta**2*x79 - A_ab_zeta_zeta*x40 - &
      tab_zeta_zeta*x80 - x119*x81 - x121 - x122 - x123 - x124) + x41* &
      x85 + x41*x87 + x41*x96) - x56*x83**2 + x82*(4*beta_ab*tab* &
      tab_zeta*x1*x3*x7 + 2*beta_ab*x0*x1*x29*x3*x42 + 2*beta_ab*x0*x3* &
      x38*x7 + 2*beta_ab_zeta*x0*x1*x3*x7)
      H_ab_zeta_drho = phi_zeta*x54 + x105*x44 + x11*(2*beta_ab*tab_drho* &
      tab_zeta*x1*x3*x7 + beta_ab_zeta*x106 + beta_ab_zeta*x112 + &
      tab_zeta_drho*x57 + x107*x41 + x108*x38 + x113*x34 + x114*x38 + &
      x115*x42 + x116*x42 + x118*x42 + x21*(x125 + x126 + x127) + x30*( &
      -A_ab*tab_drho*x103 - tab_zeta*x117 - tab_zeta_drho*x80 - x125 - &
      x126 - x127) + x48*x85)
      H_ab_drho_drho = -x104**2*x56 + x11*(4*A_ab*beta_ab*tab_drho*x26*x29*x3* &
      x52 + 2*A_ab*tab_drho_drho*x46 + 10*A_ab*x0*x129 + 4*beta_ab*tab* &
      tab_drho*x1*x29*x3*x52 + tab_drho_drho*x57 + x118*x52 + x129*(2* &
      A_ab*x0 + 2) + x30*(-tab_drho_drho*x18 - tab_drho_drho*x80 - x128 &
      *x23 - x128*x81))

   END SUBROUTINE calc_H_ab

   ELEMENTAL SUBROUTINE calc_t(rho, drho, max_deriv, t, t_rho, t_drho, t_rho_rho, t_rho_drho, t_drho_drho)
      REAL(KIND=dp), INTENT(IN) :: rho, drho
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: t, t_rho, t_drho, t_rho_rho, t_rho_drho, t_drho_drho

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_t", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2


      x0 = 3**(5.0_dp/6.0_dp)*pi**(1.0_dp/6.0_dp)
      x1 = (1.0_dp/12.0_dp)*x0/rho**(7.0_dp/6.0_dp)
      x2 = (7.0_dp/72.0_dp)*x0/rho**(13.0_dp/6.0_dp)
      t = drho*x1
      t_rho = -drho*x2
      t_drho = x1
      t_rho_rho = (91.0_dp/432.0_dp)*drho*x0/rho**(19.0_dp/6.0_dp)
      t_rho_drho = -x2
      t_drho_drho = 0

   END SUBROUTINE calc_t

   ELEMENTAL SUBROUTINE calc_beta(rho, mu, max_deriv, beta, beta_rho, beta_rho_rho)
      REAL(KIND=dp), INTENT(IN) :: rho, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: beta, beta_rho, beta_rho_rho

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_beta", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, beta_PBE, alpha_C, Ecmu, Ecmu_rho, Ecmu_rho_rho, Ec, Ec_rho, Ec_rho_rho

      CALL calc_Ecmu(rho, max_deriv, Ecmu, Ecmu_rho, Ecmu_rho_rho)
      CALL calc_Ec(rho, max_deriv, Ec, Ec_rho, Ec_rho_rho)

      beta_PBE = 0.066725_dp
      alpha_C = 2.78_dp
      x0 = 1.0_dp/Ec
      x1 = (Ecmu*x0)**alpha_C
      x2 = 1.0_dp/Ecmu
      x3 = Ec**2
      x4 = 1.0_dp/x3
      x5 = Ecmu*x4
      x6 = -Ec_rho*x5 + Ecmu_rho*x0
      x7 = alpha_C*beta_PBE*x1*x2*x6
      x8 = Ec*alpha_C
      x9 = Ecmu**(-2)
      beta = beta_PBE*x1
      beta_rho = Ec*x7
      beta_rho_rho = Ec_rho*x7 - Ecmu_rho*beta_PBE*x1*x6*x8*x9 + alpha_C**2* &
      beta_PBE*x1*x3*x6**2*x9 + beta_PBE*x1*x2*x8*(-2*Ec_rho*Ecmu_rho* &
      x4 - Ec_rho_rho*x5 + Ecmu_rho_rho*x0 + 2*Ec_rho**2*Ecmu/Ec**3)

   END SUBROUTINE calc_beta

   ELEMENTAL SUBROUTINE calc_A(rho, mu, max_deriv, A, A_rho, A_rho_rho)
      REAL(KIND=dp), INTENT(IN) :: rho, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: A, A_rho, A_rho_rho

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_A", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, gam, Ecmu, Ecmu_rho, Ecmu_rho_rho, beta, beta_rho, beta_rho_rho

      CALL calc_Ecmu(rho, max_deriv, Ecmu, Ecmu_rho, Ecmu_rho_rho)
      CALL calc_beta(rho, max_deriv, beta, beta_rho, beta_rho_rho)

      gam = 0.031091_dp
      x0 = 1.0_dp/gam
      x1 = Ecmu*x0
      x2 = exp(-x1)
      x3 = x2 - 1
      x4 = x0/x3
      x5 = gam**(-2)
      x6 = x3**(-2)
      x7 = beta*x2*x5*x6
      x8 = x2*x6
      x9 = Ecmu_rho**2*beta/gam**3
      A = beta*x4
      A_rho = Ecmu_rho*x7 + beta_rho*x4
      A_rho_rho = 2*Ecmu_rho*beta_rho*x5*x8 + Ecmu_rho_rho*x7 + beta_rho_rho* &
      x4 - x8*x9 + 2*x9*exp(-2.0_dp*x1)/x3**3

   END SUBROUTINE calc_A

   ELEMENTAL SUBROUTINE calc_H(rho, drho, mu, max_deriv, H, H_rho, H_drho, H_rho_rho, H_rho_drho, H_drho_drho)
      REAL(KIND=dp), INTENT(IN) :: rho, drho, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: H, H_rho, H_drho, H_rho_rho, H_rho_drho, H_drho_drho

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_H", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64, x65, x66, x67, x68, gam, beta, beta_rho, beta_rho_rho, t, t_rho, t_drho, t_rho_rho, t_rho_drho, t_drho_drho, A, A_rho, A_rho_rho

      CALL calc_beta(rho, drho, max_deriv, beta, beta_rho, beta_rho_rho)
      CALL calc_t(rho, drho, max_deriv, t, t_rho, t_drho, t_rho_rho, t_rho_drho, t_drho_drho)
      CALL calc_A(rho, drho, max_deriv, A, A_rho, A_rho_rho)

      gam = 0.031091_dp
      x0 = 1.0_dp/gam
      x1 = t**2
      x2 = A*x1 + 1
      x3 = A**2
      x4 = t**4
      x5 = x2 + x3*x4
      x6 = 1.0_dp/x5
      x7 = x0*x1*x2*x6
      x8 = beta*x7 + 1
      x9 = gam/x8
      x10 = x1*x2
      x11 = beta_rho*x0*x6
      x12 = x10*x11
      x13 = 2*beta*t*x0*x2*x6
      x14 = t_rho*x13
      x15 = A_rho*x1
      x16 = 2*A*t
      x17 = t_rho*x16
      x18 = x15 + x17
      x19 = beta*x0*x18*x6
      x20 = x1*x19
      x21 = 2*A
      x22 = A_rho*x4
      x23 = t**3
      x24 = 4*x23*x3
      x25 = -t_rho*x24 - x15 - x17 - x21*x22
      x26 = beta*x25
      x27 = x5**(-2)
      x28 = x0*x1*x2*x26*x27
      x29 = x12 + x14 + x20 + x28
      x30 = beta*x0*x23*x6
      x31 = 2*A*t_drho*x30
      x32 = 2*t*t_drho
      x33 = beta*x0*x2*x6
      x34 = x32*x33
      x35 = A*t*t_drho
      x36 = -t_drho*x24 - 2*x35
      x37 = beta*x0*x27*x36
      x38 = x10*x37
      x39 = gam/x8**2
      x40 = 4*t*t_rho
      x41 = beta_rho*x0*x2*x6
      x42 = t_rho**2
      x43 = 2*beta*x0*x2*x6
      x44 = 2*x1*x18
      x45 = A_rho_rho*x1
      x46 = t_rho_rho*x16
      x47 = A_rho*x40
      x48 = x21*x42
      x49 = beta*x0*x1*x6
      x50 = beta*x0*x2*x25*x27
      x51 = A*t_rho
      x52 = x23*x3
      x53 = -4*A*x22 - 4*t*x51 - 8*t_rho*x52 - 2*x15
      x54 = beta_rho*x0*x1*x2*x27
      x55 = beta*x0*x25*x27
      x56 = x5**(-3)
      x57 = beta*x0*x1*x2*x27
      x58 = 12*x1*x3
      x59 = -x31 - x34 - x38
      x60 = 2*A*t_drho*x23
      x61 = A*beta*x0*x1*x6
      x62 = t_rho_drho*x16
      x63 = 2*t_drho*x51
      x64 = A_rho*x32
      x65 = 8*t_drho
      x66 = x0*x1*x2*x56*(-4*x35 - x52*x65)
      x67 = t_drho**2
      x68 = beta*t_drho*x0*x27*x36
      H = gam*log(x8)
      H_rho = x29*x9
      H_drho = x9*(x31 + x34 + x38)
      H_rho_rho = -x39*(-x12 - x14 - x20 - x28)**2 + x9*(beta_rho_rho*x7 + &
      t_rho_rho*x13 + x0*x1*x2*x26*x53*x56 + x11*x44 + x19*x40 + x40* &
      x41 + x40*x50 + x42*x43 + x44*x55 + x49*(x45 + x46 + x47 + x48) + &
      x53*x54 + x57*(-16*A*A_rho*t_rho*x23 - 2*A_rho**2*x4 - A_rho_rho* &
      x21*x4 - t_rho_rho*x24 - x42*x58 - x45 - x46 - x47 - x48))
      H_rho_drho = x29*x39*x59 + x9*(2*t*t_rho*x2*x37 + 2*t_drho*t_rho*x33 + 4 &
      *t_drho*t_rho*x61 + t_rho_drho*x13 + x1*x18*x37 + x11*x60 + x19* &
      x32 + x26*x66 + x32*x41 + x32*x50 + x36*x54 + x49*(x62 + x63 + &
      x64) + x55*x60 + x57*(-A*A_rho*x23*x65 - t_drho*t_rho*x58 - &
      t_rho_drho*x24 - x62 - x63 - x64))
      H_drho_drho = -x39*x59**2 + x9*(2*A*t_drho_drho*x30 + 4*A*x23*x68 + beta &
      *x36*x66 + 4*t*x2*x68 + t_drho_drho*x13 + x43*x67 + x57*( &
      -t_drho_drho*x16 - t_drho_drho*x24 - x21*x67 - x58*x67) + 10*x61* &
      x67)

   END SUBROUTINE calc_H

   ELEMENTAL SUBROUTINE lsd_pbe_sr_c(rho, zeta, drho, mu, max_deriv, lsd_pbe_sr_c, lsd_pbe_sr_c_rho, lsd_pbe_sr_c_zeta, lsd_pbe_sr_c_drho, lsd_pbe_sr_c_rho_rho, lsd_pbe_sr_c_rho_zeta, lsd_pbe_sr_c_rho_drho, lsd_pbe_sr_c_zeta_zeta, lsd_pbe_sr_c_zeta_drho, lsd_pbe_sr_c_drho_drho)
      REAL(KIND=dp), INTENT(IN) :: rho, zeta, drho, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: lsd_pbe_sr_c, lsd_pbe_sr_c_rho, lsd_pbe_sr_c_zeta, lsd_pbe_sr_c_drho, lsd_pbe_sr_c_rho_rho, lsd_pbe_sr_c_rho_zeta, lsd_pbe_sr_c_rho_drho, lsd_pbe_sr_c_zeta_zeta, lsd_pbe_sr_c_zeta_drho, lsd_pbe_sr_c_drho_drho

      CHARACTER(LEN=*), PARAMETER :: routineN = "lsd_pbe_sr_c", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: Ecmuab, Ecmuab_rho, Ecmuab_zeta, Ecmuab_rho_rho, Ecmuab_rho_zeta, Ecmuab_zeta_zeta, H_ab, H_ab_rho, H_ab_zeta, H_ab_drho, H_ab_rho_rho, H_ab_rho_zeta, H_ab_rho_drho, H_ab_zeta_zeta, H_ab_zeta_drho, H_ab_drho_drho

      CALL calc_Ecmuab(rho, zeta, drho, max_deriv, Ecmuab, Ecmuab_rho, Ecmuab_zeta, Ecmuab_rho_rho, Ecmuab_rho_zeta, Ecmuab_zeta_zeta)
      CALL calc_H_ab(rho, zeta, drho, max_deriv, H_ab, H_ab_rho, H_ab_zeta, H_ab_drho, H_ab_rho_rho, H_ab_rho_zeta, H_ab_rho_drho, H_ab_zeta_zeta, H_ab_zeta_drho, H_ab_drho_drho)

      lsd_pbe_sr_c = Ecmuab + H_ab
      lsd_pbe_sr_c_rho = Ecmuab_rho + H_ab_rho
      lsd_pbe_sr_c_zeta = Ecmuab_zeta + H_ab_zeta
      lsd_pbe_sr_c_drho = H_ab_drho
      lsd_pbe_sr_c_rho_rho = Ecmuab_rho_rho + H_ab_rho_rho
      lsd_pbe_sr_c_rho_zeta = Ecmuab_rho_zeta + H_ab_rho_zeta
      lsd_pbe_sr_c_rho_drho = H_ab_rho_drho
      lsd_pbe_sr_c_zeta_zeta = Ecmuab_zeta_zeta + H_ab_zeta_zeta
      lsd_pbe_sr_c_zeta_drho = H_ab_zeta_drho
      lsd_pbe_sr_c_drho_drho = H_ab_drho_drho

   END SUBROUTINE lsd_pbe_sr_c

   ELEMENTAL SUBROUTINE lda_pbe_sr_c(rho, drho, mu, max_deriv, lda_pbe_sr_c, lda_pbe_sr_c_rho, lda_pbe_sr_c_drho, lda_pbe_sr_c_rho_rho, lda_pbe_sr_c_rho_drho, lda_pbe_sr_c_drho_drho)
      REAL(KIND=dp), INTENT(IN) :: rho, drho, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: lda_pbe_sr_c, lda_pbe_sr_c_rho, lda_pbe_sr_c_drho, lda_pbe_sr_c_rho_rho, lda_pbe_sr_c_rho_drho, lda_pbe_sr_c_drho_drho

      CHARACTER(LEN=*), PARAMETER :: routineN = "lda_pbe_sr_c", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: Ecmu, Ecmu_rho, Ecmu_rho_rho, H, H_rho, H_drho, H_rho_rho, H_rho_drho, H_drho_drho

      CALL calc_Ecmu(rho, drho, max_deriv, Ecmu, Ecmu_rho, Ecmu_rho_rho)
      CALL calc_H(rho, drho, max_deriv, H, H_rho, H_drho, H_rho_rho, H_rho_drho, H_drho_drho)

      lda_pbe_sr_c = Ecmu + H
      lda_pbe_sr_c_rho = Ecmu_rho + H_rho
      lda_pbe_sr_c_drho = H_drho
      lda_pbe_sr_c_rho_rho = Ecmu_rho_rho + H_rho_rho
      lda_pbe_sr_c_rho_drho = H_rho_drho
      lda_pbe_sr_c_drho_drho = H_drho_drho

   END SUBROUTINE lda_pbe_sr_c

   ELEMENTAL SUBROUTINE pbe_sr_x(rho, drho, mu, max_deriv, pbe_sr_x, pbe_sr_x_rho, pbe_sr_x_drho, pbe_sr_x_rho_rho, pbe_sr_x_rho_drho, pbe_sr_x_drho_drho)
      REAL(KIND=dp), INTENT(IN) :: rho, drho, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: pbe_sr_x, pbe_sr_x_rho, pbe_sr_x_drho, pbe_sr_x_rho_rho, pbe_sr_x_rho_drho, pbe_sr_x_drho_drho

      CHARACTER(LEN=*), PARAMETER :: routineN = "pbe_sr_x", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: Ex, Ex_rho, Ex_rho_rho, Fx, Fx_rho, Fx_drho, Fx_rho_rho, Fx_rho_drho, Fx_drho_drho

      CALL calc_Ex(rho, drho, max_deriv, Ex, Ex_rho, Ex_rho_rho)
      CALL calc_Fx(rho, drho, max_deriv, Fx, Fx_rho, Fx_drho, Fx_rho_rho, Fx_rho_drho, Fx_drho_drho)

      pbe_sr_x = Ex*Fx
      pbe_sr_x_rho = Ex*Fx_rho + Ex_rho*Fx
      pbe_sr_x_drho = Ex*Fx_drho
      pbe_sr_x_rho_rho = Ex*Fx_rho_rho + 2*Ex_rho*Fx_rho + Ex_rho_rho*Fx
      pbe_sr_x_rho_drho = Ex*Fx_rho_drho + Ex_rho*Fx_drho
      pbe_sr_x_drho_drho = Ex*Fx_drho_drho

   END SUBROUTINE pbe_sr_x

   ELEMENTAL SUBROUTINE rz2ab(rhoa, rhob, max_deriv, rz2ab, rz2ab_rhoa, rz2ab_rhob, rz2ab_rhoa_rhoa, rz2ab_rhoa_rhob, rz2ab_rhob_rhob)
      REAL(KIND=dp), INTENT(IN) :: rhoa, rhob
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: rz2ab, rz2ab_rhoa, rz2ab_rhob, rz2ab_rhoa_rhoa, rz2ab_rhoa_rhob, rz2ab_rhob_rhob

      CHARACTER(LEN=*), PARAMETER :: routineN = "rz2ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, f, f_rho, f_zeta, f_rho_rho, f_rho_zeta, f_zeta_zeta

      CALL calc_f(rhoa, rhob, max_deriv, f, f_rho, f_zeta, f_rho_rho, f_rho_zeta, f_zeta_zeta)

      x0 = 1.0_dp/rho
      x1 = f_zeta*x0
      x2 = f_rho - x1*zeta
      x3 = 2*f_rho_zeta*x0
      x4 = rho**(-2)
      x5 = 2*f_zeta*x4
      x6 = f_zeta_zeta*x4
      x7 = 2*x6*zeta
      x8 = -x3*zeta
      x9 = x5*zeta
      x10 = x6*zeta**2
      x11 = f_rho_rho + x10 + x6 + x8 + x9
      rz2ab = f
      rz2ab_rhoa = x1 + x2
      rz2ab_rhob = -x1 + x2
      rz2ab_rhoa_rhoa = x11 + x3 - x5 - x7
      rz2ab_rhoa_rhob = f_rho_rho + x10 - x6 + x8 + x9
      rz2ab_rhob_rhob = x11 - x3 + x5 + x7

   END SUBROUTINE rz2ab
