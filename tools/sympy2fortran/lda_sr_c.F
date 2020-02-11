
   ELEMENTAL SUBROUTINE calc_D3(rs, max_deriv, D3, D3_rs, D3_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: D3, D3_rs, D3_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_D3", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, d3_1, d3_2

      d3_1 = 0.31_dp
      d3_2 = -4.95_dp
      x0 = d3_2 + rs
      x1 = rs**(-2)
      x2 = exp(-d3_1*rs)
      x3 = x1*x2
      x4 = x0*x3
      x5 = x2/rs**3
      x6 = d3_1*x1*x2
      x7 = 4*x5
      D3 = x4
      D3_rs = -x0*x6 + x3 - x5*(2*d3_2 + 2*rs)
      D3_rs_rs = d3_1**2*x4 + d3_1*x0*x7 - 2*x6 - x7 + 6*x0*x2/rs**4

   END SUBROUTINE calc_D3

   ELEMENTAL SUBROUTINE calc_D2(rs, max_deriv, D2, D2_rs, D2_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: D2, D2_rs, D2_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_D2", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6

      x0 = 1.0_dp/rs
      x1 = exp(-d2_1*rs)
      x2 = x1*(d2_2 + d2_3*rs)
      x3 = x0*x2
      x4 = d2_3*x1
      x5 = x0*x4
      x6 = rs**(-2)
      D2 = x3
      D2_rs = -d2_1*x3 - x2*x6 + x5
      D2_rs_rs = d2_1**2*x3 + 2*d2_1*x2*x6 - 2*d2_1*x5 - 2*x4*x6 + 2*x2/rs**3

   END SUBROUTINE calc_D2

   ELEMENTAL SUBROUTINE calc_g2(rs, max_deriv, g2, g2_rs, g2_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: g2, g2_rs, g2_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_g2", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, g2_1, g2_2, g2_3

      g2_1 = 0.02267_dp
      g2_2 = 0.4319_dp
      g2_3 = 0.04_dp
      x0 = rs**2
      x1 = 1.0_dp/x0
      x2 = (1.0_dp/5.0_dp)*x1
      x3 = alpha**(-2)
      x4 = 2**(1.0_dp/3.0_dp)
      x5 = -g2_1*rs*x4 + 1
      x6 = g2_2*x4
      x7 = 2**(2.0_dp/3.0_dp)
      x8 = g2_3*x0*x7 + rs*x6 + 1
      x9 = 1.0_dp/x8
      x10 = x3*x5*x9
      x11 = g2_1*x3*x4*x9
      x12 = rs**(-3)
      x13 = g2_3*rs*x7
      x14 = -2*x13 - x6
      x15 = x8**(-2)
      x16 = x14*x15*x3*x5
      x17 = (4.0_dp/5.0_dp)*x12
      x18 = (2.0_dp/5.0_dp)*x1*x15*x3
      g2 = x10*x2
      g2_rs = -2.0_dp/5.0_dp*x10*x12 - x11*x2 + x16*x2
      g2_rs_rs = -g2_1*x14*x18*x4 - g2_3*x18*x5*x7 + x11*x17 + x14*x2*x3*x5*( &
      -4*x13 - 2*x6)/x8**3 - x16*x17 + (6.0_dp/5.0_dp)*x10/rs**4

   END SUBROUTINE calc_g2

   ELEMENTAL SUBROUTINE calc_g0(rs, max_deriv, g0, g0_rs, g0_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: g0, g0_rs, g0_rs_rs

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

   ELEMENTAL SUBROUTINE calc_C2(rs, max_deriv, C2, C2_rs, C2_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: C2, C2_rs, C2_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C2", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, g0, g0_rs, g0_rs_rs

      CALL calc_g0(rs, max_deriv, g0, g0_rs, g0_rs_rs)

      x0 = rs**(-3)
      x1 = (3.0_dp/8.0_dp)*x0
      x2 = rs**(-4)
      C2 = x0*(-3.0_dp/8.0_dp*g0 + 3.0_dp/16.0_dp)
      C2_rs = -g0_rs*x1 + x2*((9.0_dp/8.0_dp)*g0 - 9.0_dp/16.0_dp)
      C2_rs_rs = (9.0_dp/4.0_dp)*g0_rs*x2 - g0_rs_rs*x1 + (-9.0_dp/2.0_dp*g0 + &
      9.0_dp/4.0_dp)/rs**5

   END SUBROUTINE calc_C2

   ELEMENTAL SUBROUTINE calc_C3(rs, max_deriv, C3, C3_rs, C3_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: C3, C3_rs, C3_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C3", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, g0, g0_rs, g0_rs_rs

      CALL calc_g0(rs, max_deriv, g0, g0_rs, g0_rs_rs)

      x0 = c3_1/rs**3
      x1 = c3_1*g0
      x2 = rs**(-4)
      C3 = -g0*x0
      C3_rs = -g0_rs*x0 + 3*x1*x2
      C3_rs_rs = 6*c3_1*g0_rs*x2 - g0_rs_rs*x0 - 12*x1/rs**5

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
      C4_rs = x0*(-9.0_dp/64.0_dp*D2_rs - 9.0_dp/64.0_dp*g2_rs - 9.0_dp/160.0_dp*x4 &
      ) + x3*((27.0_dp/64.0_dp)*D2 + (27.0_dp/64.0_dp)*g2 - 27.0_dp/320.0_dp* &
      x2)
      C4_rs_rs = x0*(-9.0_dp/64.0_dp*D2_rs_rs - 9.0_dp/64.0_dp*g2_rs_rs + (27.0_dp/ &
      160.0_dp)*x1*x3) + x3*((27.0_dp/32.0_dp)*D2_rs + (27.0_dp/32.0_dp)* &
      g2_rs + (27.0_dp/80.0_dp)*x4) + (-27.0_dp/16.0_dp*D2 - 27.0_dp/16.0_dp* &
      g2 + (27.0_dp/80.0_dp)*x2)/rs**5

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
      x3 = x0*x1/rs**4
      C5 = x2*(-9.0_dp/80.0_dp*D3 - 9.0_dp/80.0_dp*g2)
      C5_rs = x2*(-9.0_dp/80.0_dp*D3_rs - 9.0_dp/80.0_dp*g2_rs) + x3*((27.0_dp/ &
      80.0_dp)*D3 + (27.0_dp/80.0_dp)*g2)
      C5_rs_rs = x2*(-9.0_dp/80.0_dp*D3_rs_rs - 9.0_dp/80.0_dp*g2_rs_rs) + x3*(( &
      27.0_dp/40.0_dp)*D3_rs + (27.0_dp/40.0_dp)*g2_rs) + x0*x1*(-27.0_dp/ &
      20.0_dp*D3 - 27.0_dp/20.0_dp*g2)/rs**5

   END SUBROUTINE calc_C5

   ELEMENTAL SUBROUTINE calc_a1(rs, max_deriv, a1, a1_rs, a1_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a1, a1_rs, a1_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a1", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, C3, C3_rs, C3_rs_rs, C5, C5_rs, C5_rs_rs

      CALL calc_C3(rs, max_deriv, C3, C3_rs, C3_rs_rs)
      CALL calc_C5(rs, max_deriv, C5, C5_rs, C5_rs_rs)

      x0 = b0**8
      x1 = C5*x0
      x2 = b0**6
      x3 = 4*x2
      x4 = C5_rs*x0
      x5 = 1.0_dp/rs
      x6 = C3*x2
      x7 = rs**(-2)
      a1 = C3*x3 + x1
      a1_rs = C3_rs*x3 + 8*x1*x5 + x4 + 24*x5*x6
      a1_rs_rs = 48*C3_rs*x2*x5 + C3_rs_rs*x3 + C5_rs_rs*x0 + 56*x1*x7 + 16*x4 &
      *x5 + 120*x6*x7

   END SUBROUTINE calc_a1

   ELEMENTAL SUBROUTINE calc_a2(rs, max_deriv, a2, a2_rs, a2_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a2, a2_rs, a2_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a2", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, C2, C2_rs, C2_rs_rs, C4, C4_rs, C4_rs_rs, Ec, Ec_rs, Ec_rs_rs

      CALL calc_C2(rs, max_deriv, C2, C2_rs, C2_rs_rs)
      CALL calc_C4(rs, max_deriv, C4, C4_rs, C4_rs_rs)
      CALL calc_Ec(rs, max_deriv, Ec, Ec_rs, Ec_rs_rs)

      x0 = b0**8
      x1 = C4*x0
      x2 = b0**6
      x3 = 4*x2
      x4 = b0**4
      x5 = 6*x4
      x6 = C4_rs*x0
      x7 = 1.0_dp/rs
      x8 = 24*x7
      x9 = C2*x2
      x10 = Ec*x4
      x11 = rs**(-2)
      x12 = 48*x7
      a2 = C2*x3 + Ec*x5 + x1
      a2_rs = C2_rs*x3 + Ec_rs*x5 + 8*x1*x7 + x10*x8 + x6 + x8*x9
      a2_rs_rs = C2_rs*x12*x2 + C2_rs_rs*x3 + C4_rs_rs*x0 + Ec_rs*x12*x4 + &
      Ec_rs_rs*x5 + 56*x1*x11 + 72*x10*x11 + 120*x11*x9 + 16*x6*x7

   END SUBROUTINE calc_a2

   ELEMENTAL SUBROUTINE calc_a3(rs, max_deriv, a3, a3_rs, a3_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a3, a3_rs, a3_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a3", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, C3, C3_rs, C3_rs_rs

      CALL calc_C3(rs, max_deriv, C3, C3_rs, C3_rs_rs)

      x0 = b0**8
      x1 = C3*x0
      x2 = C3_rs*x0
      x3 = 1.0_dp/rs
      a3 = x1
      a3_rs = 8*x1*x3 + x2
      a3_rs_rs = C3_rs_rs*x0 + 16*x2*x3 + 56*x1/rs**2

   END SUBROUTINE calc_a3

   ELEMENTAL SUBROUTINE calc_a4(rs, max_deriv, a4, a4_rs, a4_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a4, a4_rs, a4_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a4", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, C2, C2_rs, C2_rs_rs, Ec, Ec_rs, Ec_rs_rs

      CALL calc_C2(rs, max_deriv, C2, C2_rs, C2_rs_rs)
      CALL calc_Ec(rs, max_deriv, Ec, Ec_rs, Ec_rs_rs)

      x0 = b0**8
      x1 = C2*x0
      x2 = b0**6
      x3 = 4*x2
      x4 = C2_rs*x0
      x5 = 1.0_dp/rs
      x6 = Ec*x2
      x7 = rs**(-2)
      a4 = Ec*x3 + x1
      a4_rs = Ec_rs*x3 + 8*x1*x5 + x4 + 24*x5*x6
      a4_rs_rs = C2_rs_rs*x0 + 48*Ec_rs*x2*x5 + Ec_rs_rs*x3 + 56*x1*x7 + 16*x4 &
      *x5 + 120*x6*x7

   END SUBROUTINE calc_a4

   ELEMENTAL SUBROUTINE calc_a5(rs, max_deriv, a5, a5_rs, a5_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a5, a5_rs, a5_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a5", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, Ec, Ec_rs, Ec_rs_rs

      CALL calc_Ec(rs, max_deriv, Ec, Ec_rs, Ec_rs_rs)

      x0 = b0**8
      x1 = Ec*x0
      x2 = Ec_rs*x0
      x3 = 1.0_dp/rs
      a5 = x1
      a5_rs = 8*x1*x3 + x2
      a5_rs_rs = Ec_rs_rs*x0 + 16*x2*x3 + 56*x1/rs**2

   END SUBROUTINE calc_a5

   ELEMENTAL SUBROUTINE calc_Q(rs, mu, max_deriv, Q, Q_rs, Q_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: Q, Q_rs, Q_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_Q", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, aa, cc, dd, qq, bb

      aa = 5.84605_dp
      cc = 3.91744_dp
      dd = 3.44851_dp
      qq = 2*(-1 + 0.693147180559945_dp)/pi**2
      bb = 3.44851_dp - 17.53815_dp*pi/(-4 + 4*0.693147180559945_dp)
      x0 = xx**2
      x1 = aa*xx + 1
      x2 = dd*x0 + x1
      x3 = 1.0_dp/x2
      x4 = bb*x0 + cc*xx**3 + x1
      x5 = 2*xx
      x6 = aa + dd*x5
      x7 = x4*x6
      x8 = 3*cc*x0
      x9 = aa + bb*x5 + x8
      x10 = x2*x9
      x11 = xx*(-x10 + x7)
      x12 = x2**2
      Q = qq*log(x3*x4)
      Q_rs = -1.0_dp/2.0_dp*qq*x11*x3/(rs*x4)
      Q_rs_rs = (1.0_dp/4.0_dp)*qq*xx*(x10*x11 - x11*x7 + x4*(x12*(-aa + x8) + &
      x2*(aa*x4 - x5*x6*x9) + x4*x5*x6**2))/(rs**2*x12*x4**2)

   END SUBROUTINE calc_Q

   ELEMENTAL SUBROUTINE calc_ec_mu(rs, mu, max_deriv, ec_mu, ec_mu_rs, ec_mu_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: ec_mu, ec_mu_rs, ec_mu_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_ec_mu", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, Ec, Ec_rs, Ec_rs_rs, Q, Q_rs, Q_rs_rs, a1, a1_rs, a1_rs_rs, a2, a2_rs, a2_rs_rs, a3, a3_rs, a3_rs_rs, a4, a4_rs, a4_rs_rs, a5, a5_rs, a5_rs_rs

      CALL calc_Ec(rs, max_deriv, Ec, Ec_rs, Ec_rs_rs)
      CALL calc_Q(rs, mu, max_deriv, Q, Q_rs, Q_rs_rs)
      CALL calc_a1(rs, max_deriv, a1, a1_rs, a1_rs_rs)
      CALL calc_a2(rs, max_deriv, a2, a2_rs, a2_rs_rs)
      CALL calc_a3(rs, max_deriv, a3, a3_rs, a3_rs_rs)
      CALL calc_a4(rs, max_deriv, a4, a4_rs, a4_rs_rs)
      CALL calc_a5(rs, max_deriv, a5, a5_rs, a5_rs_rs)

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
      x17 = rs*x1*x16*x2
      x18 = 8*Q + 8*x11 + 8*x13 + 8*x15 + 8*x7 + 8*x9
      ec_mu = Ec + x5*(-Q - x11 - x13 - x15 - x7 - x9)
      ec_mu_rs = Ec_rs + x17*x18 + x5*(-Q_rs - a1_rs*x6 - a2_rs*x8 - a3_rs*x10 &
      - a4_rs*x12 - a5_rs*x14)
      ec_mu_rs_rs = Ec_rs_rs + beta**4*x0*x8*(-80*Q - 80*x11 - 80*x13 - 80*x15 &
      - 80*x7 - 80*x9)/x4**6 + x16*x18*x3 + x17*(16*Q_rs + 16*a1_rs*x6 &
      + 16*a2_rs*x8 + 16*a3_rs*x10 + 16*a4_rs*x12 + 16*a5_rs*x14) + x5* &
      (-Q_rs_rs - a1_rs_rs*x6 - a2_rs_rs*x8 - a3_rs_rs*x10 - a4_rs_rs* &
      x12 - a5_rs_rs*x14)

   END SUBROUTINE calc_ec_mu
