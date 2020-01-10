
   PURE SUBROUTINE calc_s(rho, drho, s, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rho, drho
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: s(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_s", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2


      x0 = 3**(2.0_dp/3.0_dp)/pi**(2.0_dp/3.0_dp)
      x1 = (1.0_dp/6.0_dp)*x0/rho**(4.0_dp/3.0_dp)
      x2 = (2.0_dp/9.0_dp)*x0/rho**(7.0_dp/3.0_dp)
      s(0) = drho*x1
      s(1) = -drho*x2
      s(2) = x1
      s(3) = (14.0_dp/27.0_dp)*drho*x0/rho**(10.0_dp/3.0_dp)
      s(4) = -x2
      s(5) = 0

   END SUBROUTINE calc_s

   PURE SUBROUTINE calc_c_1(mu, c_1, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: c_1(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_c_1", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: 


      c_1(0) = 1 + 22*mu**2/c**2 + 144*mu**4/c**4

   END SUBROUTINE calc_c_1

   PURE SUBROUTINE calc_c_2(mu, c_2, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: c_2(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_c_2", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1


      x0 = c**(-2)
      x1 = mu**2
      c_2(0) = x0*x1*(144*x0*x1 - 14)

   END SUBROUTINE calc_c_2

   PURE SUBROUTINE calc_c_3(mu, c_3, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: c_3(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_c_3", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: 


      c_3(0) = -864*mu**4*(-1 + 2*mu**2/c**2)/c**4

   END SUBROUTINE calc_c_3

   PURE SUBROUTINE calc_c_4(mu, c_4, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: c_4(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_c_4", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0


      x0 = mu**2/c**2
      c_4(0) = x0*(-24*x0 + 1048573 + 8*sqrt(pi)*mu*erf(0.5_dp*c/mu)/c)

   END SUBROUTINE calc_c_4

   PURE SUBROUTINE calc_b_T(mu, b_T, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: b_T(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_b_T", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0


      x0 = exp(0.25_dp/mu**2)
      ! Not supported in Fortran:
! c_1
! c_2
! c_3
! c_4
b_T(0) = (x0*c_2(mu) - c_1(mu))/(54*x0*c_4(mu) + c_3(mu))

   END SUBROUTINE calc_b_T

   PURE SUBROUTINE calc_b(mu, b, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: b(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_b", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: 


      ! Not supported in Fortran:
! b_T
b(0) = (81.0_dp/7.0_dp)*b_PBE*b_T(mu)*exp(-alpha_X*mu**2)

   END SUBROUTINE calc_b

   PURE SUBROUTINE calc_Fx(rho, drho, mu, Fx, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rho, drho, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: Fx(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_Fx", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, b(0:2), s(0:5)

      CALL calc_b(mu, b, max_deriv)
      CALL calc_s(rho, drho, s, max_deriv)

      x0 = 2*b(0)*s(0)
      x1 = 2*b(0)
      Fx(0) = -kappa*(b(0)*s(0)**2/kappa + 1) + kappa + 1
      Fx(1) = -s(1)*x0
      Fx(2) = -s(2)*x0
      Fx(3) = -s(1)**2*x1 - s(3)*x0
      Fx(4) = -s(1)*s(2)*x1 - s(4)*x0
      Fx(5) = -s(2)**2*x1 - s(5)*x0

   END SUBROUTINE calc_Fx

   PURE SUBROUTINE calc_phi(zeta, phi, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: phi(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_phi", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1


      x0 = zeta + 1
      x1 = -zeta + 1
      phi(0) = (1.0_dp/2.0_dp)*x0**(2.0_dp/3.0_dp) + (1.0_dp/2.0_dp)*x1**(2.0_dp/ &
      3.0_dp)
      phi(1) = -1.0_dp/3.0_dp/x1**(1.0_dp/3.0_dp) + (1.0_dp/3.0_dp)/x0**(1.0_dp/ &
      3.0_dp)
      phi(2) = -1.0_dp/9.0_dp/x1**(4.0_dp/3.0_dp) - 1.0_dp/9.0_dp/x0**(4.0_dp/3.0_dp)

   END SUBROUTINE calc_phi

   PURE SUBROUTINE calc_t_ab(rho, zeta, drho, t_ab, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rho, zeta, drho
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: t_ab(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_t_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, phi(0:2)

      CALL calc_phi(zeta, phi, max_deriv)

      x0 = rho**(-7.0_dp/6.0_dp)
      x1 = (1.0_dp/12.0_dp)*x0
      x2 = 3**(5.0_dp/6.0_dp)
      x3 = pi**(1.0_dp/6.0_dp)
      x4 = x2*x3/phi(0)
      x5 = x1*x4
      x6 = rho**(-13.0_dp/6.0_dp)
      x7 = (7.0_dp/72.0_dp)*x4*x6
      x8 = phi(0)**(-2)
      x9 = phi(1)*x2*x3*x8
      x10 = x1*x9
      x11 = drho*x0*x2*x3
      t_ab(0) = drho*x5
      t_ab(1) = -drho*x7
      t_ab(2) = -drho*x10
      t_ab(3) = x5
      t_ab(4) = (91.0_dp/432.0_dp)*drho*x4/rho**(19.0_dp/6.0_dp)
      t_ab(5) = (7.0_dp/72.0_dp)*drho*x6*x9
      t_ab(6) = -x7
      t_ab(7) = -1.0_dp/12.0_dp*phi(2)*x11*x8 + (1.0_dp/6.0_dp)*phi(1)**2*x11/phi &
      (0)**3
      t_ab(8) = -x10
      t_ab(9) = 0

   END SUBROUTINE calc_t_ab

   PURE SUBROUTINE calc_beta_ab(rho, zeta, mu, beta_ab, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rho, zeta, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: beta_ab(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_beta_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, Ecmuab(0:5), Ecab(0:5)

      CALL calc_Ecmuab(rho, zeta, Ecmuab, max_deriv)
      CALL calc_Ecab(rho, zeta, Ecab, max_deriv)

      x0 = 1.0_dp/Ecab(0)
      x1 = (Ecmuab(0)*x0)**alpha_C
      x2 = Ecab(0)**2
      x3 = 1.0_dp/x2
      x4 = Ecab(1)*x3
      x5 = -Ecmuab(0)*x4 + Ecmuab(1)*x0
      x6 = 1.0_dp/Ecmuab(0)
      x7 = Ecab(0)*alpha_C*beta_PBE*x1*x6
      x8 = Ecab(2)*x3
      x9 = -Ecmuab(0)*x8 + Ecmuab(2)*x0
      x10 = alpha_C*beta_PBE*x1*x5*x6
      x11 = Ecmuab(0)**(-2)
      x12 = Ecab(0)*alpha_C*beta_PBE*x1*x11*x5
      x13 = alpha_C**2*beta_PBE*x1*x11*x2
      x14 = Ecmuab(0)*x3
      x15 = 2*Ecmuab(0)/Ecab(0)**3
      x16 = alpha_C*beta_PBE*x1*x9
      beta_ab(0) = beta_PBE*x1
      beta_ab(1) = x5*x7
      beta_ab(2) = x7*x9
      beta_ab(3) = Ecab(1)*x10 - Ecmuab(1)*x12 + x13*x5**2 + x7*(Ecab(1)**2* &
      x15 - Ecab(3)*x14 - 2*Ecmuab(1)*x4 + Ecmuab(3)*x0)
      beta_ab(4) = Ecab(2)*x10 - Ecmuab(2)*x12 + x13*x5*x9 + x7*(Ecab(1)*Ecab &
      (2)*x15 - Ecab(4)*x14 - Ecmuab(1)*x8 - Ecmuab(2)*x4 + Ecmuab(4)* &
      x0)
      beta_ab(5) = -Ecab(0)*Ecmuab(2)*x11*x16 + Ecab(2)*x16*x6 + x13*x9**2 + &
      x7*(Ecab(2)**2*x15 - Ecab(5)*x14 - 2*Ecmuab(2)*x8 + Ecmuab(5)*x0)

   END SUBROUTINE calc_beta_ab

   PURE SUBROUTINE calc_A_ab(rho, zeta, mu, A_ab, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rho, zeta, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: A_ab(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_A_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, Ecmuab(0:5), beta_ab(0:5), phi(0:2)

      CALL calc_Ecmuab(rho, zeta, Ecmuab, max_deriv)
      CALL calc_beta_ab(rho, zeta, beta_ab, max_deriv)
      CALL calc_phi(zeta, phi, max_deriv)

      x0 = 1.0_dp/gamma
      x1 = phi(0)**(-3)
      x2 = x0*x1
      x3 = Ecmuab(0)*x2
      x4 = exp(-x3)
      x5 = x4 - 1
      x6 = x0/x5
      x7 = gamma**(-2)
      x8 = x5**(-2)
      x9 = Ecmuab(1)*x1*x4*x7*x8
      x10 = beta_ab(0)*x9
      x11 = phi(0)**(-4)
      x12 = 3*Ecmuab(0)*x0*x11
      x13 = -Ecmuab(2)*x2 + phi(1)*x12
      x14 = beta_ab(0)*x4*x8
      x15 = beta_ab(0)*x1*x4*x7*x8
      x16 = Ecmuab(1)**2/(gamma**3*phi(0)**6)
      x17 = 2*beta_ab(0)*exp(-2.0_dp*x3)/x5**3
      x18 = x0*x13*x4*x8
      x19 = x0*x13**2
      A_ab(0) = beta_ab(0)*x6
      A_ab(1) = beta_ab(1)*x6 + x10
      A_ab(2) = beta_ab(2)*x6 - x0*x13*x14
      A_ab(3) = Ecmuab(3)*x15 + 2*beta_ab(1)*x9 + beta_ab(3)*x6 - x14*x16 + &
      x16*x17
      A_ab(4) = -3*Ecmuab(1)*beta_ab(0)*phi(1)*x11*x4*x7*x8 - Ecmuab(1)*x1* &
      x13*x17*x7 + Ecmuab(4)*x15 - beta_ab(1)*x18 + beta_ab(2)*x9 + &
      beta_ab(4)*x6 + x10*x13
      A_ab(5) = -2*beta_ab(2)*x18 + beta_ab(5)*x6 - x0*x14*(-12*Ecmuab(0)*phi &
      (1)**2*x0/phi(0)**5 + 6*Ecmuab(2)*phi(1)*x0*x11 - Ecmuab(5)*x2 + &
      phi(2)*x12) - x14*x19 + x17*x19

   END SUBROUTINE calc_A_ab

   PURE SUBROUTINE calc_H_ab(rho, zeta, drho, mu, H_ab, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rho, zeta, drho, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: H_ab(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_H_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64, x65, x66, x67, x68, x69, x70, x71, x72, x73, x74, x75, x76, x77, x78, x79, x80, x81, x82, x83, x84, x85, x86, x87, x88, x89, x90, x91, x92, x93, x94, x95, x96, x97, x98, x99, x100, x101, x102, x103, x104, x105, x106, x107, x108, x109, x110, x111, x112, x113, x114, x115, x116, x117, x118, x119, x120, x121, x122, x123, x124, x125, x126, x127, x128, x129, x130, x131, beta_ab(0:5), t_ab(0:9), A_ab(0:5), phi(0:2)

      CALL calc_beta_ab(rho, zeta, beta_ab, max_deriv)
      CALL calc_t_ab(rho, zeta, drho, t_ab, max_deriv)
      CALL calc_A_ab(rho, zeta, A_ab, max_deriv)
      CALL calc_phi(zeta, phi, max_deriv)

      x0 = t_ab(0)**2
      x1 = A_ab(0)*x0 + 1
      x2 = x0*x1
      x3 = 1.0_dp/gamma
      x4 = A_ab(0)**2
      x5 = t_ab(0)**4
      x6 = x1 + x4*x5
      x7 = 1.0_dp/x6
      x8 = beta_ab(0)*x2*x3*x7 + 1
      x9 = gamma*log(x8)
      x10 = 1.0_dp/x8
      x11 = gamma*phi(0)*x10
      x12 = x0*x1*x3*x7
      x13 = beta_ab(1)*x12
      x14 = 2*t_ab(1)
      x15 = beta_ab(0)*t_ab(0)*x1*x3*x7
      x16 = x14*x15
      x17 = A_ab(1)*x0
      x18 = A_ab(0)*t_ab(0)*t_ab(1)
      x19 = 2*x18
      x20 = x17 + x19
      x21 = beta_ab(0)*x0*x3*x7
      x22 = x20*x21
      x23 = 2*A_ab(0)
      x24 = A_ab(1)*x5
      x25 = 4*t_ab(1)
      x26 = t_ab(0)**3
      x27 = x26*x4
      x28 = -x17 - x19 - x23*x24 - x25*x27
      x29 = x6**(-2)
      x30 = beta_ab(0)*x0*x1*x29*x3
      x31 = x28*x30
      x32 = x13 + x16 + x22 + x31
      x33 = beta_ab(2)*x12
      x34 = 2*t_ab(2)
      x35 = x15*x34
      x36 = A_ab(2)*x0
      x37 = A_ab(0)*t_ab(0)*t_ab(2)
      x38 = 2*x37
      x39 = x36 + x38
      x40 = x21*x39
      x41 = A_ab(2)*x5
      x42 = 4*t_ab(2)
      x43 = -x23*x41 - x27*x42 - x36 - x38
      x44 = x30*x43
      x45 = x33 + x35 + x40 + x44
      x46 = beta_ab(0)*x26*x3*x7
      x47 = 2*A_ab(0)*t_ab(3)*x46
      x48 = 2*t_ab(3)
      x49 = x15*x48
      x50 = A_ab(0)*t_ab(0)*t_ab(3)
      x51 = 4*x26*x4
      x52 = -t_ab(3)*x51 - 2*x50
      x53 = beta_ab(0)*x2*x29*x3*x52
      x54 = gamma*x10*(x47 + x49 + x53)
      x55 = x8**(-2)
      x56 = gamma*phi(0)*x55
      x57 = 2*beta_ab(0)*t_ab(0)*x1*x3*x7
      x58 = beta_ab(1)*t_ab(0)*x1*x3*x7
      x59 = t_ab(1)**2
      x60 = 2*beta_ab(0)*x1*x3*x7
      x61 = beta_ab(0)*t_ab(0)*x20*x3*x7
      x62 = 4*x18
      x63 = 2*x17
      x64 = x62 + x63
      x65 = beta_ab(1)*x0*x3*x7
      x66 = A_ab(3)*x0
      x67 = 2*A_ab(0)*t_ab(0)
      x68 = t_ab(4)*x67
      x69 = A_ab(1)*t_ab(0)
      x70 = x25*x69
      x71 = x23*x59
      x72 = beta_ab(0)*t_ab(0)*x1*x28*x29*x3
      x73 = 4*A_ab(0)
      x74 = 8*t_ab(1)*x26
      x75 = -x24*x73 - x4*x74 - x62 - x63
      x76 = beta_ab(1)*x0*x1*x29*x3
      x77 = beta_ab(0)*x0*x28*x29*x3
      x78 = x6**(-3)
      x79 = beta_ab(0)*x0*x1*x28*x3*x78
      x80 = 2*A_ab(0)*x5
      x81 = 2*x5
      x82 = A_ab(0)*A_ab(1)*x26
      x83 = 12*x0*x4
      x84 = gamma*phi(1)*x10
      x85 = -x33 - x35 - x40 - x44
      x86 = beta_ab(0)*t_ab(1)*x1*x3*x7
      x87 = 2*t_ab(2)*x1
      x88 = beta_ab(2)*t_ab(0)*x1*x3*x7
      x89 = beta_ab(2)*x0*x3*x7
      x90 = beta_ab(0)*t_ab(0)*x3*x39*x7
      x91 = A_ab(4)*x0
      x92 = t_ab(5)*x67
      x93 = 2*A_ab(0)*t_ab(1)
      x94 = t_ab(2)*x93
      x95 = x34*x69
      x96 = A_ab(2)*t_ab(0)
      x97 = x14*x96
      x98 = beta_ab(2)*x0*x1*x29*x3
      x99 = beta_ab(0)*x0*x29*x3*x43
      x100 = 4*x37
      x101 = 2*x36
      x102 = 8*t_ab(2)
      x103 = -x100 - x101 - x102*x27 - x41*x73
      x104 = A_ab(0)*A_ab(2)
      x105 = 12*t_ab(1)*x0
      x106 = -x47 - x49 - x53
      x107 = gamma*phi(0)*x106*x55
      x108 = 2*A_ab(0)*t_ab(3)*x26*x3*x7
      x109 = A_ab(0)*beta_ab(0)*t_ab(3)*x0*x3*x7
      x110 = t_ab(6)*x67
      x111 = t_ab(3)*x93
      x112 = x48*x69
      x113 = x0*x1*x29*x3*x52
      x114 = beta_ab(0)*t_ab(0)*x29*x3*x52
      x115 = beta_ab(0)*x0*x29*x3*x52
      x116 = 2*A_ab(0)*beta_ab(0)*t_ab(3)*x26*x29*x3
      x117 = 8*A_ab(0)*t_ab(3)*x26
      x118 = t_ab(3)*x4
      x119 = beta_ab(0)*x0*x1*x3*x78*(-8*x118*x26 - 4*x50)
      x120 = t_ab(2)**2
      x121 = x100 + x101
      x122 = A_ab(5)*x0
      x123 = t_ab(7)*x67
      x124 = x42*x96
      x125 = x120*x23
      x126 = beta_ab(0)*t_ab(0)*x1*x29*x3*x43
      x127 = t_ab(8)*x67
      x128 = A_ab(0)*t_ab(3)*x34
      x129 = x48*x96
      x130 = t_ab(3)**2
      x131 = beta_ab(0)*x130*x3*x7
      H_ab(0) = phi(0)*x9
      H_ab(1) = x11*x32
      H_ab(2) = phi(1)*x9 + x11*x45
      H_ab(3) = phi(0)*x54
      H_ab(4) = x11*(beta_ab(3)*x12 + t_ab(4)*x57 + x21*(x66 + x68 + x70 + &
      x71) + x25*x58 + x25*x61 + x25*x72 + x30*(-A_ab(1)**2*x81 - A_ab( &
      3)*x80 - 16*t_ab(1)*x82 - t_ab(4)*x51 - x59*x83 - x66 - x68 - x70 &
      - x71) + x59*x60 + x64*x65 + x64*x77 + x75*x76 + x75*x79) - x56*( &
      -x13 - x16 - x22 - x31)**2
      H_ab(5) = x11*(2*beta_ab(0)*t_ab(0)*t_ab(1)*x1*x29*x3*x43 + 2*beta_ab(0 &
      )*t_ab(0)*t_ab(2)*x1*x28*x29*x3 + beta_ab(1)*t_ab(0)*x3*x7*x87 + &
      beta_ab(4)*x12 + t_ab(5)*x57 + x103*x79 + x14*x88 + x14*x90 + x20 &
      *x89 + x20*x99 + x21*(x91 + x92 + x94 + x95 + x97) + x28*x98 + &
      x30*(-2*A_ab(2)*x24 - A_ab(4)*x80 - t_ab(2)*x105*x4 - t_ab(5)*x51 &
      - x102*x82 - x104*x74 - x91 - x92 - x94 - x95 - x97) + x34*x61 + &
      x34*x86 + x39*x65 + x39*x77 + x43*x76) + x32*x56*x85 + x32*x84
      H_ab(6) = x107*x32 + x11*(beta_ab(1)*x108 + beta_ab(1)*x113 + 2*t_ab(1) &
      *x1*x114 + t_ab(6)*x57 + x109*x25 + x115*x20 + x116*x28 + x119* &
      x28 + x21*(x110 + x111 + x112) + x30*(-A_ab(1)*x117 - t_ab(6)*x51 &
      - x105*x118 - x110 - x111 - x112) + x48*x58 + x48*x61 + x48*x72 + &
      x48*x86)
      H_ab(7) = phi(2)*x9 + x11*(beta_ab(0)*x103*x2*x3*x43*x78 + beta_ab(5)* &
      x12 + t_ab(7)*x57 + x103*x98 + x120*x60 + x121*x89 + x121*x99 + &
      x126*x42 + x21*(x122 + x123 + x124 + x125) + x30*(-A_ab(2)**2*x81 &
      - A_ab(5)*x80 - 16*t_ab(2)*x104*x26 - t_ab(7)*x51 - x120*x83 - &
      x122 - x123 - x124 - x125) + x42*x88 + x42*x90) - x56*x85**2 + &
      x84*(4*beta_ab(0)*t_ab(0)*t_ab(2)*x1*x3*x7 + 2*beta_ab(0)*x0*x1* &
      x29*x3*x43 + 2*beta_ab(0)*x0*x3*x39*x7 + 2*beta_ab(2)*x0*x1*x3*x7 &
      )
      H_ab(8) = phi(1)*x54 + x107*x45 + x11*(beta_ab(0)*t_ab(3)*x3*x7*x87 + &
      beta_ab(2)*x108 + beta_ab(2)*x113 + t_ab(8)*x57 + x109*x42 + x114 &
      *x87 + x115*x39 + x116*x43 + x119*x43 + x126*x48 + x21*(x127 + &
      x128 + x129) + x30*(-A_ab(2)*x117 - 12*t_ab(2)*t_ab(3)*x0*x4 - &
      t_ab(8)*x51 - x127 - x128 - x129) + x48*x88 + x48*x90)
      H_ab(9) = -x106**2*x56 + x11*(4*A_ab(0)*beta_ab(0)*t_ab(3)*x26*x29*x3* &
      x52 + 2*A_ab(0)*t_ab(9)*x46 + 10*A_ab(0)*x0*x131 + 4*beta_ab(0)* &
      t_ab(0)*t_ab(3)*x1*x29*x3*x52 + t_ab(9)*x57 + x119*x52 + x131*(2* &
      A_ab(0)*x0 + 2) + x30*(-t_ab(9)*x51 - t_ab(9)*x67 - x130*x23 - &
      x130*x83))

   END SUBROUTINE calc_H_ab

   PURE SUBROUTINE calc_t(rho, drho, t, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rho, drho
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: t(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_t", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2


      x0 = 3**(5.0_dp/6.0_dp)*pi**(1.0_dp/6.0_dp)
      x1 = (1.0_dp/12.0_dp)*x0/rho**(7.0_dp/6.0_dp)
      x2 = (7.0_dp/72.0_dp)*x0/rho**(13.0_dp/6.0_dp)
      t(0) = drho*x1
      t(1) = -drho*x2
      t(2) = x1
      t(3) = (91.0_dp/432.0_dp)*drho*x0/rho**(19.0_dp/6.0_dp)
      t(4) = -x2
      t(5) = 0

   END SUBROUTINE calc_t

   PURE SUBROUTINE calc_beta(rho, mu, beta, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rho, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: beta(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_beta", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, Ecmu(0:2), Ec(0:2)

      CALL calc_Ecmu(rho, Ecmu, max_deriv)
      CALL calc_Ec(rho, Ec, max_deriv)

      x0 = 1.0_dp/Ec(0)
      x1 = (Ecmu(0)*x0)**alpha_C
      x2 = 1.0_dp/Ecmu(0)
      x3 = Ec(0)**2
      x4 = 1.0_dp/x3
      x5 = Ecmu(0)*x4
      x6 = -Ec(1)*x5 + Ecmu(1)*x0
      x7 = alpha_C*beta_PBE*x1*x2*x6
      x8 = Ec(0)*alpha_C
      x9 = Ecmu(0)**(-2)
      beta(0) = beta_PBE*x1
      beta(1) = Ec(0)*x7
      beta(2) = Ec(1)*x7 - Ecmu(1)*beta_PBE*x1*x6*x8*x9 + alpha_C**2*beta_PBE &
      *x1*x3*x6**2*x9 + beta_PBE*x1*x2*x8*(-2*Ec(1)*Ecmu(1)*x4 - Ec(2)* &
      x5 + Ecmu(2)*x0 + 2*Ec(1)**2*Ecmu(0)/Ec(0)**3)

   END SUBROUTINE calc_beta

   PURE SUBROUTINE calc_A(rho, mu, A, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rho, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: A(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_A", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, Ecmu(0:2), beta(0:2)

      CALL calc_Ecmu(rho, Ecmu, max_deriv)
      CALL calc_beta(rho, beta, max_deriv)

      x0 = 1.0_dp/gamma
      x1 = Ecmu(0)*x0
      x2 = exp(-x1)
      x3 = x2 - 1
      x4 = x0/x3
      x5 = gamma**(-2)
      x6 = x3**(-2)
      x7 = beta(0)*x2*x5*x6
      x8 = x2*x6
      x9 = Ecmu(1)**2*beta(0)/gamma**3
      A(0) = beta(0)*x4
      A(1) = Ecmu(1)*x7 + beta(1)*x4
      A(2) = 2*Ecmu(1)*beta(1)*x5*x8 + Ecmu(2)*x7 + beta(2)*x4 - x8*x9 + 2*x9 &
      *exp(-2.0_dp*x1)/x3**3

   END SUBROUTINE calc_A

   PURE SUBROUTINE calc_H(rho, drho, mu, H, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rho, drho, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: H(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_H", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64, x65, x66, x67, x68, x69, x70, beta(0:2), t(0:5), A(0:2)

      CALL calc_beta(rho, beta, max_deriv)
      CALL calc_t(rho, drho, t, max_deriv)
      CALL calc_A(rho, A, max_deriv)

      x0 = 1.0_dp/gamma
      x1 = t(0)**2
      x2 = A(0)*x1 + 1
      x3 = A(0)**2
      x4 = t(0)**4
      x5 = x2 + x3*x4
      x6 = 1.0_dp/x5
      x7 = x0*x1*x2*x6
      x8 = beta(0)*x7 + 1
      x9 = gamma/x8
      x10 = x1*x2
      x11 = beta(1)*x0*x6
      x12 = x10*x11
      x13 = 2*beta(0)*t(0)*x0*x2*x6
      x14 = t(1)*x13
      x15 = A(1)*x1
      x16 = A(0)*t(0)*t(1)
      x17 = 2*x16
      x18 = x15 + x17
      x19 = beta(0)*x0*x1*x18*x6
      x20 = 2*A(0)
      x21 = A(1)*x4
      x22 = 4*t(1)
      x23 = t(0)**3
      x24 = x23*x3
      x25 = -x15 - x17 - x20*x21 - x22*x24
      x26 = beta(0)*x25
      x27 = x5**(-2)
      x28 = x0*x1*x2*x26*x27
      x29 = x12 + x14 + x19 + x28
      x30 = beta(0)*x0*x23*x6
      x31 = 2*A(0)*t(2)*x30
      x32 = 2*t(2)
      x33 = beta(0)*t(0)*x0*x2*x32*x6
      x34 = A(0)*t(0)*t(2)
      x35 = 4*x23*x3
      x36 = -t(2)*x35 - 2*x34
      x37 = beta(0)*x0*x27*x36
      x38 = x10*x37
      x39 = gamma/x8**2
      x40 = 4*t(1)*x2
      x41 = beta(1)*t(0)*x0*x6
      x42 = t(1)**2
      x43 = 2*beta(0)*x0*x2*x6
      x44 = beta(0)*t(0)*x0*x18*x6
      x45 = 2*x1*x18
      x46 = A(2)*x1
      x47 = 2*A(0)*t(0)
      x48 = t(3)*x47
      x49 = A(1)*t(0)
      x50 = x22*x49
      x51 = x20*x42
      x52 = beta(0)*x0*x1*x6
      x53 = beta(0)*t(0)*x0*x25*x27
      x54 = -4*A(0)*x21 - 8*t(1)*x24 - 2*x15 - 4*x16
      x55 = beta(1)*x0*x1*x2*x27
      x56 = beta(0)*x0*x25*x27
      x57 = x5**(-3)
      x58 = beta(0)*x0*x1*x2*x27
      x59 = 12*x1*x3
      x60 = -x31 - x33 - x38
      x61 = 2*A(0)*t(2)*x23
      x62 = A(0)*beta(0)*x0*x1*x6
      x63 = 2*t(2)*x2
      x64 = t(4)*x47
      x65 = A(0)*t(1)*x32
      x66 = x32*x49
      x67 = t(0)*x2
      x68 = 8*t(2)
      x69 = x0*x1*x2*x57*(-x24*x68 - 4*x34)
      x70 = t(2)**2
      H(0) = gamma*log(x8)
      H(1) = x29*x9
      H(2) = x9*(x31 + x33 + x38)
      H(3) = -x39*(-x12 - x14 - x19 - x28)**2 + x9*(beta(2)*x7 + t(3)*x13 + &
      x0*x1*x2*x26*x54*x57 + x11*x45 + x22*x44 + x40*x41 + x40*x53 + &
      x42*x43 + x45*x56 + x52*(x46 + x48 + x50 + x51) + x54*x55 + x58*( &
      -16*A(0)*A(1)*t(1)*x23 - 2*A(1)**2*x4 - A(2)*x20*x4 - t(3)*x35 - &
      x42*x59 - x46 - x48 - x50 - x51))
      H(4) = x29*x39*x60 + x9*(2*beta(0)*t(1)*t(2)*x0*x2*x6 + 2*beta(0)*t(1)* &
      x0*x27*x36*x67 + 4*t(1)*t(2)*x62 + t(4)*x13 + x1*x18*x37 + x11* &
      x61 + x26*x69 + x32*x44 + x36*x55 + x41*x63 + x52*(x64 + x65 + &
      x66) + x53*x63 + x56*x61 + x58*(-A(0)*A(1)*x23*x68 - t(1)*t(2)* &
      x59 - t(4)*x35 - x64 - x65 - x66))
      H(5) = -x39*x60**2 + x9*(4*A(0)*beta(0)*t(2)*x0*x23*x27*x36 + 2*A(0)*t( &
      5)*x30 + 4*beta(0)*t(2)*x0*x27*x36*x67 + beta(0)*x36*x69 + t(5)* &
      x13 + x43*x70 + x58*(-t(5)*x35 - t(5)*x47 - x20*x70 - x59*x70) + &
      10*x62*x70)

   END SUBROUTINE calc_H

   PURE SUBROUTINE lsd_pbe_sr_c(rho, zeta, drho, mu, ec_pbe_mu_ab, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rho, zeta, drho, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: ec_pbe_mu_ab(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "lsd_pbe_sr_c", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: Ecmuab(0:5), H_ab(0:9)

      CALL calc_Ecmuab(rho, zeta, Ecmuab, max_deriv)
      CALL calc_H_ab(rho, zeta, drho, H_ab, max_deriv)

      ec_pbe_mu_ab(0) = Ecmuab(0) + H_ab(0)
      ec_pbe_mu_ab(1) = Ecmuab(1) + H_ab(1)
      ec_pbe_mu_ab(2) = Ecmuab(2) + H_ab(2)
      ec_pbe_mu_ab(3) = H_ab(3)
      ec_pbe_mu_ab(4) = Ecmuab(3) + H_ab(4)
      ec_pbe_mu_ab(5) = Ecmuab(4) + H_ab(5)
      ec_pbe_mu_ab(6) = H_ab(6)
      ec_pbe_mu_ab(7) = Ecmuab(5) + H_ab(7)
      ec_pbe_mu_ab(8) = H_ab(8)
      ec_pbe_mu_ab(9) = H_ab(9)

   END SUBROUTINE lsd_pbe_sr_c

   PURE SUBROUTINE lda_pbe_sr_c(rho, drho, mu, ec_pbe_mu, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rho, drho, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: ec_pbe_mu(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "lda_pbe_sr_c", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: Ecmu(0:2), H(0:5)

      CALL calc_Ecmu(rho, Ecmu, max_deriv)
      CALL calc_H(rho, drho, H, max_deriv)

      ec_pbe_mu(0) = Ecmu(0) + H(0)
      ec_pbe_mu(1) = Ecmu(1) + H(1)
      ec_pbe_mu(2) = H(2)
      ec_pbe_mu(3) = Ecmu(2) + H(3)
      ec_pbe_mu(4) = H(4)
      ec_pbe_mu(5) = H(5)

   END SUBROUTINE lda_pbe_sr_c

   PURE SUBROUTINE pbe_sr_x(rho, drho, mu, ex_pbe_mu, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rho, drho, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: ex_pbe_mu(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "pbe_sr_x", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: Ex(0:2), Fx(0:5)

      CALL calc_Ex(rho, Ex, max_deriv)
      CALL calc_Fx(rho, drho, Fx, max_deriv)

      ex_pbe_mu(0) = Ex(0)*Fx(0)
      ex_pbe_mu(1) = Ex(0)*Fx(1) + Ex(1)*Fx(0)
      ex_pbe_mu(2) = Ex(0)*Fx(2)
      ex_pbe_mu(3) = Ex(0)*Fx(3) + 2*Ex(1)*Fx(1) + Ex(2)*Fx(0)
      ex_pbe_mu(4) = Ex(0)*Fx(4) + Ex(1)*Fx(2)
      ex_pbe_mu(5) = Ex(0)*Fx(5)

   END SUBROUTINE pbe_sr_x
