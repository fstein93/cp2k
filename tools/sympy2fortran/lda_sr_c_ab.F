
   SUBROUTINE calc_D3(rs, D3, D3_rs, D3_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      REAL(KIND=dp), INTENT(INOUT) :: D3, D3_rs, D3_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_D3", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, d3_1, d3_2

      d3_1 = 0.31_dp
      d3_2 = 4.95_dp
      x0 = -d3_2 + rs
      x1 = rs**(-2)
      x2 = exp(-d3_1*rs)
      x3 = x1*x2
      x4 = x0*x3
      x5 = x2/rs**3
      x6 = d3_1*x1*x2
      x7 = 4*x5
      D3 = x4
      D3_rs = -x0*x6 + x3 - x5*(-2*d3_2 + 2*rs)
      D3_rs_rs = d3_1**2*x4 + d3_1*x0*x7 - 2*x6 - x7 + 6*x0*x2/rs**4

   END SUBROUTINE calc_D3

   SUBROUTINE calc_D2(rs, D2, D2_rs, D2_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      REAL(KIND=dp), INTENT(INOUT) :: D2, D2_rs, D2_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_D2", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, d2_1, d2_2, d2_3

      d2_1 = 0.547_dp
      d2_2 = 0.388_dp
      d2_3 = 0.676_dp
      x0 = 1.0_dp/rs
      x1 = exp(-d2_1*rs)
      x2 = x1*(-d2_2 + d2_3*rs)
      x3 = x0*x2
      x4 = d2_3*x1
      x5 = x0*x4
      x6 = rs**(-2)
      D2 = x3
      D2_rs = -d2_1*x3 - x2*x6 + x5
      D2_rs_rs = d2_1**2*x3 + 2*d2_1*x2*x6 - 2*d2_1*x5 - 2*x4*x6 + 2*x2/rs**3

   END SUBROUTINE calc_D2

   SUBROUTINE calc_g2z(rs, zeta, g2z, g2z_rs, g2z_zeta, g2z_rs_rs, g2z_rs_zeta, g2z_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      REAL(KIND=dp), INTENT(INOUT) :: g2z, g2z_rs, g2z_zeta, g2z_rs_rs, g2z_rs_zeta, g2z_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_g2z", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, g2_1, g2_2, g2_3

      g2_1 = 0.02267_dp
      g2_2 = 0.4319_dp
      g2_3 = 0.04_dp
      x0 = (1.0_dp/2.0_dp)*zeta + 1.0_dp/2.0_dp
      x1 = x0**2
      x2 = alpha**(-2)
      x3 = rs**2
      x4 = 1.0_dp/x3
      x5 = zeta + 1
      x6 = 1.0_dp/x5
      x7 = x6**(2.0_dp/3.0_dp)
      x8 = 1.0_dp/x7
      x9 = 2**(1.0_dp/3.0_dp)
      x10 = x6**(1.0_dp/3.0_dp)
      x11 = -g2_1*rs*x10*x9 + 1
      x12 = g2_2*x10*x9
      x13 = rs*x12
      x14 = 2**(2.0_dp/3.0_dp)
      x15 = g2_3*x14*x3*x7
      x16 = x13 + x15 + 1
      x17 = 1.0_dp/x16
      x18 = (2.0_dp/5.0_dp)*x11*x17*x2*x4*x8
      x19 = 1.0_dp/x10
      x20 = (2.0_dp/5.0_dp)*g2_1*x17*x19*x2*x4*x9
      x21 = x1*x20
      x22 = rs**(-3)
      x23 = x11*x17*x2*x22*x8
      x24 = x16**(-2)
      x25 = x1*x24
      x26 = g2_3*rs*x14*x7
      x27 = -x12 - 2*x26
      x28 = (2.0_dp/5.0_dp)*x11*x2*x27*x4*x8
      x29 = 1.0_dp/rs
      x30 = g2_1*x1*x17*x19*x2*x9
      x31 = x1*x11*x17*x2*x4*x8
      x32 = (1.0_dp/3.0_dp)*x6
      x33 = (2.0_dp/3.0_dp)*x6
      x34 = x13*x32 + x15*x33
      x35 = (2.0_dp/5.0_dp)*x1*x11*x2*x24*x4*x8
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
      g2z = x1*x18
      g2z_rs = -4.0_dp/5.0_dp*x1*x23 - x21 + x25*x28
      g2z_zeta = x0*x18 + (2.0_dp/15.0_dp)*x29*x30*x6 + (4.0_dp/15.0_dp)*x31*x6 + &
      x34*x35
      g2z_rs_rs = -4.0_dp/5.0_dp*g2_1*x1*x19*x2*x24*x27*x4*x9 - 4.0_dp/5.0_dp* &
      g2_3*x1*x11*x14*x2*x24*x4 - 8.0_dp/5.0_dp*x1*x11*x2*x22*x24*x27*x8 &
      + (8.0_dp/5.0_dp)*x22*x30 + x37*(-2*x12 - 4*x26) + (12.0_dp/5.0_dp)* &
      x1*x11*x17*x2*x8/rs**4
      g2z_rs_zeta = (2.0_dp/15.0_dp)*g2_1*x19*x2*x25*x27*x29*x6*x9 - x0*x20 + &
      x0*x24*x28 - 4.0_dp/5.0_dp*x1*x11*x2*x22*x24*x42 + (4.0_dp/15.0_dp)* &
      x1*x11*x2*x24*x27*x4*x6*x8 - x21*x6 - x23*x38 - x23*x39 + x35*( &
      x12*x32 + x26*x41) + x37*x43 - 2.0_dp/5.0_dp*x4*x40
      g2z_zeta_zeta = g2_1*x0*x17*x19*x2*x44*x9 + (8.0_dp/15.0_dp)*x0*x11*x17* &
      x2*x4*x6*x8 + (2.0_dp/5.0_dp)*x1*x11*x2*x36*x4*x42*x43 + (1.0_dp/ &
      5.0_dp)*x11*x17*x2*x4*x8 - 4.0_dp/45.0_dp*x31*x45 + x35*(-4.0_dp/ &
      9.0_dp*x13*x45 - 10.0_dp/9.0_dp*x15*x45) + x38*x46 + x39*x46 + x40* &
      x44

   END SUBROUTINE calc_g2z

   SUBROUTINE calc_g0(rs, g0, g0_rs, g0_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      REAL(KIND=dp), INTENT(INOUT) :: g0, g0_rs, g0_rs_rs

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
      x4 = x0*((1.0_dp/2.0_dp)*g0_2*x2 - 1.0_dp/2.0_dp*g0_3*x3 + (1.0_dp/2.0_dp)* &
      g0_4*rs**4 + rs*x1 + 1.0_dp/2.0_dp)
      x5 = x0*(g0_2*rs - 3.0_dp/2.0_dp*g0_3*x2 + 2*g0_4*x3 + x1)
      g0 = x4
      g0_rs = -g0_5*x4 + x5
      g0_rs_rs = g0_5**2*x4 - 2*g0_5*x5 + x0*(g0_2 - 3*g0_3*rs + 6*g0_4*x2)

   END SUBROUTINE calc_g0

   SUBROUTINE calc_phin(zeta, n, phin, phin_zeta, phin_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: zeta, n
      REAL(KIND=dp), INTENT(INOUT) :: phin, phin_zeta, phin_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_phin", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9

      x0 = zeta + 1
      x1 = (1.0_dp/3.0_dp)*n
      x2 = x0**x1
      x3 = -zeta + 1
      x4 = x3**x1
      x5 = (1.0_dp/6.0_dp)*n*x2
      x6 = (1.0_dp/6.0_dp)*n*x4
      x7 = x0**(-2)
      x8 = (1.0_dp/18.0_dp)*n**2
      x9 = x3**(-2)
      phin = (1.0_dp/2.0_dp)*x2 + (1.0_dp/2.0_dp)*x4
      phin_zeta = -x6/x3 + x5/x0
      phin_zeta_zeta = x2*x7*x8 + x4*x8*x9 - x5*x7 - x6*x9

   END SUBROUTINE calc_phin

   SUBROUTINE calc_C2(rs, zeta, C2, C2_rs, C2_zeta, C2_rs_rs, C2_rs_zeta, C2_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      REAL(KIND=dp), INTENT(INOUT) :: C2, C2_rs, C2_zeta, C2_rs_rs, C2_rs_zeta, C2_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C2", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7

      x0 = g0 - 1.0_dp/2.0_dp
      x1 = rs**(-3)
      x2 = zeta**2
      x3 = (3.0_dp/8.0_dp)*x2 - 3.0_dp/8.0_dp
      x4 = x1*x3
      x5 = rs**(-4)
      x6 = x0*x5
      x7 = (3.0_dp/4.0_dp)*x1*zeta
      C2 = x0*x4
      C2_rs = g0_rs*x4 - x6*((9.0_dp/8.0_dp)*x2 - 9.0_dp/8.0_dp)
      C2_zeta = x0*x7
      C2_rs_rs = -6*g0_rs*x3*x5 + g0_rs_rs*x4 + 12*x0*x3/rs**5
      C2_rs_zeta = g0_rs*x7 - 9.0_dp/4.0_dp*x6*zeta
      C2_zeta_zeta = x1*((3.0_dp/4.0_dp)*g0 - 3.0_dp/8.0_dp)

   END SUBROUTINE calc_C2

   SUBROUTINE calc_C3(rs, zeta, C3, C3_rs, C3_zeta, C3_rs_rs, C3_rs_zeta, C3_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      REAL(KIND=dp), INTENT(INOUT) :: C3, C3_rs, C3_zeta, C3_rs_rs, C3_rs_zeta, C3_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C3", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9

      x0 = zeta**2
      x1 = sqrt(2.0_dp)
      x2 =1.0_dp/sqrt(pi)
      x3 = rs**(-3)
      x4 = g0*x1*x2*x3
      x5 = rs**(-4)
      x6 = x0 - 1
      x7 = x1*x2*x6
      x8 = g0_rs*x3
      x9 = (1.0_dp/2.0_dp)*x1*x2*x6
      C3 = x4*((1.0_dp/2.0_dp)*x0 - 1.0_dp/2.0_dp)
      C3_rs = -3.0_dp/2.0_dp*g0*x5*x7 + x8*x9
      C3_zeta = x4*zeta
      C3_rs_rs = 6*g0*x7/rs**5 - 3*g0_rs*x5*x7 + g0_rs_rs*x3*x9
      C3_rs_zeta = -3*g0*x1*x2*x5*zeta + x1*x2*x8*zeta
      C3_zeta_zeta = x4

   END SUBROUTINE calc_C3

   SUBROUTINE calc_C4(rs, zeta, C4, C4_rs, C4_zeta, C4_rs_rs, C4_rs_zeta, C4_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      REAL(KIND=dp), INTENT(INOUT) :: C4, C4_rs, C4_zeta, C4_rs_rs, C4_rs_zeta, C4_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C4", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10

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
