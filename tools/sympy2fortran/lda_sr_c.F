
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

   SUBROUTINE calc_g2z(rs, g2z, g2z_rs, g2z_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      REAL(KIND=dp), INTENT(INOUT) :: g2z, g2z_rs, g2z_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_g2z", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, g2_1, g2_2, g2_3

      g2_1 = 0.02267_dp
      g2_2 = 0.4319_dp
      g2_3 = 0.04_dp
      x0 = rs**2
      x1 = 1.0_dp/x0
      x2 = (1.0_dp/10.0_dp)*x1
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
      x17 = (2.0_dp/5.0_dp)*x12
      x18 = (1.0_dp/5.0_dp)*x1*x15*x3
      g2z = x10*x2
      g2z_rs = -1.0_dp/5.0_dp*x10*x12 - x11*x2 + x16*x2
      g2z_rs_rs = -g2_1*x14*x18*x4 - g2_3*x18*x5*x7 + x11*x17 + x14*x2*x3*x5* &
      (-4*x13 - 2*x6)/x8**3 - x16*x17 + (3.0_dp/5.0_dp)*x10/rs**4

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

   SUBROUTINE calc_C2(rs, C2, C2_rs, C2_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      REAL(KIND=dp), INTENT(INOUT) :: C2, C2_rs, C2_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C2", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2

      x0 = rs**(-3)
      x1 = (3.0_dp/8.0_dp)*x0
      x2 = rs**(-4)
      C2 = x0*(-3.0_dp/8.0_dp*g0 + 3.0_dp/16.0_dp)
      C2_rs = -g0_rs*x1 + x2*((9.0_dp/8.0_dp)*g0 - 9.0_dp/16.0_dp)
      C2_rs_rs = (9.0_dp/4.0_dp)*g0_rs*x2 - g0_rs_rs*x1 + (-9.0_dp/2.0_dp*g0 + &
      9.0_dp/4.0_dp)/rs**5

   END SUBROUTINE calc_C2

   SUBROUTINE calc_C3(rs, C3, C3_rs, C3_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      REAL(KIND=dp), INTENT(INOUT) :: C3, C3_rs, C3_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C3", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5

      x0 = (1.0_dp/2.0_dp)/rs**3
      x1 = sqrt(2.0_dp)
      x2 =1.0_dp/sqrt(pi)
      x3 = g0*x1*x2
      x4 = rs**(-4)
      x5 = g0_rs*x1*x2
      C3 = -x0*x3
      C3_rs = -x0*x5 + (3.0_dp/2.0_dp)*x3*x4
      C3_rs_rs = -g0_rs_rs*x0*x1*x2 + 3*x4*x5 - 6*x3/rs**5

   END SUBROUTINE calc_C3

   SUBROUTINE calc_C4(rs, C4, C4_rs, C4_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      REAL(KIND=dp), INTENT(INOUT) :: C4, C4_rs, C4_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C4", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4

      x0 = rs**(-3)
      x1 = alpha**(-2)
      x2 = x1/rs**2
      x3 = rs**(-4)
      x4 = x0*x1
      C4 = x0*(-9.0_dp/64.0_dp*D2 - 9.0_dp/32.0_dp*g2z + (9.0_dp/320.0_dp)*x2)
      C4_rs = x0*(-9.0_dp/64.0_dp*D2_rs - 9.0_dp/32.0_dp*g2z_rs - 9.0_dp/160.0_dp* &
      x4) + x3*((27.0_dp/64.0_dp)*D2 + (27.0_dp/32.0_dp)*g2z - 27.0_dp/ &
      320.0_dp*x2)
      C4_rs_rs = x0*(-9.0_dp/64.0_dp*D2_rs_rs - 9.0_dp/32.0_dp*g2z_rs_rs + ( &
      27.0_dp/160.0_dp)*x1*x3) + x3*((27.0_dp/32.0_dp)*D2_rs + (27.0_dp/ &
      16.0_dp)*g2z_rs + (27.0_dp/80.0_dp)*x4) + (-27.0_dp/16.0_dp*D2 - &
      27.0_dp/8.0_dp*g2z + (27.0_dp/80.0_dp)*x2)/rs**5

   END SUBROUTINE calc_C4

   SUBROUTINE calc_C5(rs, C5, C5_rs, C5_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      REAL(KIND=dp), INTENT(INOUT) :: C5, C5_rs, C5_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C5", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3

      x0 = sqrt(2.0_dp)
      x1 =1.0_dp/sqrt(pi)
      x2 = x0*x1/rs**3
      x3 = x0*x1/rs**4
      C5 = x2*(-9.0_dp/80.0_dp*D3 - 9.0_dp/40.0_dp*g2z)
      C5_rs = x2*(-9.0_dp/80.0_dp*D3_rs - 9.0_dp/40.0_dp*g2z_rs) + x3*((27.0_dp/ &
      80.0_dp)*D3 + (27.0_dp/40.0_dp)*g2z)
      C5_rs_rs = x2*(-9.0_dp/80.0_dp*D3_rs_rs - 9.0_dp/40.0_dp*g2z_rs_rs) + x3*(( &
      27.0_dp/40.0_dp)*D3_rs + (27.0_dp/20.0_dp)*g2z_rs) + x0*x1*(-27.0_dp/ &
      20.0_dp*D3 - 27.0_dp/10.0_dp*g2z)/rs**5

   END SUBROUTINE calc_C5

   SUBROUTINE calc_a1(rs, a1, a1_rs, a1_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      REAL(KIND=dp), INTENT(INOUT) :: a1, a1_rs, a1_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a1", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, beta

      beta = 0.784949_dp
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = C3*x4
      x6 = rs**5
      x7 = C3_rs*x4
      x8 = rs**7
      x9 = C5*x0
      a1 = C5*x1 + x3*x5
      a1_rs = C5_rs*x1 + x3*x7 + 24*x5*x6 + 8*x8*x9
      a1_rs_rs = C3_rs_rs*x3*x4 + 16*C5_rs*x0*x8 + C5_rs_rs*x1 + 120*rs**4*x5 &
      + 56*x2*x9 + 48*x6*x7

   END SUBROUTINE calc_a1

   SUBROUTINE calc_a2(rs, a2, a2_rs, a2_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      REAL(KIND=dp), INTENT(INOUT) :: a2, a2_rs, a2_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a2", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, beta, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, a1_, a1__rs, a1__rs_rs

      beta = 0.784949_dp
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = C3*x4
      x6 = rs**5
      x7 = C3_rs*x4
      x8 = rs**7
      x9 = C5*x0
      a1 = C5*x1 + x3*x5
      a1_rs = C5_rs*x1 + x3*x7 + 24*x5*x6 + 8*x8*x9
      a1_rs_rs = C3_rs_rs*x3*x4 + 16*C5_rs*x0*x8 + C5_rs_rs*x1 + 120*rs**4*x5 &
      + 56*x2*x9 + 48*x6*x7
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
      x11 = C2_rs*x4
      x12 = rs**7
      x13 = C4*x0
      x14 = rs**3
      x15 = Ec_rs*x8
      a2 = C4*x1 + x3*x5 + x7*x9
      a2_rs = C4_rs*x1 + 24*x10*x5 + x11*x3 + 8*x12*x13 + 24*x14*x9 + x15*x7
      a2_rs_rs = C2_rs_rs*x3*x4 + 16*C4_rs*x0*x12 + C4_rs_rs*x1 + Ec_rs_rs*x7 &
      *x8 + 72*rs**2*x9 + 48*x10*x11 + 56*x13*x2 + 48*x14*x15 + 120*x5* &
      x6

   END SUBROUTINE calc_a2

   SUBROUTINE calc_a3(rs, a3, a3_rs, a3_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      REAL(KIND=dp), INTENT(INOUT) :: a3, a3_rs, a3_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a3", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, beta, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, a1_, a1__rs, a1__rs_rs, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, a2_, a2__rs, a2__rs_rs

      beta = 0.784949_dp
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = C3*x4
      x6 = rs**5
      x7 = C3_rs*x4
      x8 = rs**7
      x9 = C5*x0
      a1 = C5*x1 + x3*x5
      a1_rs = C5_rs*x1 + x3*x7 + 24*x5*x6 + 8*x8*x9
      a1_rs_rs = C3_rs_rs*x3*x4 + 16*C5_rs*x0*x8 + C5_rs_rs*x1 + 120*rs**4*x5 &
      + 56*x2*x9 + 48*x6*x7
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
      x11 = C2_rs*x4
      x12 = rs**7
      x13 = C4*x0
      x14 = rs**3
      x15 = Ec_rs*x8
      a2 = C4*x1 + x3*x5 + x7*x9
      a2_rs = C4_rs*x1 + 24*x10*x5 + x11*x3 + 8*x12*x13 + 24*x14*x9 + x15*x7
      a2_rs_rs = C2_rs_rs*x3*x4 + 16*C4_rs*x0*x12 + C4_rs_rs*x1 + Ec_rs_rs*x7 &
      *x8 + 72*rs**2*x9 + 48*x10*x11 + 56*x13*x2 + 48*x14*x15 + 120*x5* &
      x6
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**7
      x3 = C3*x0
      a3 = C3*x1
      a3_rs = C3_rs*x1 + 8*x2*x3
      a3_rs_rs = 16*C3_rs*x0*x2 + C3_rs_rs*x1 + 56*rs**6*x3

   END SUBROUTINE calc_a3

   SUBROUTINE calc_a4(rs, a4, a4_rs, a4_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      REAL(KIND=dp), INTENT(INOUT) :: a4, a4_rs, a4_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a4", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, beta, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, a1_, a1__rs, a1__rs_rs, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, a2_, a2__rs, a2__rs_rs, x0, x1, x2, x3, a3_, a3__rs, a3__rs_rs

      beta = 0.784949_dp
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = C3*x4
      x6 = rs**5
      x7 = C3_rs*x4
      x8 = rs**7
      x9 = C5*x0
      a1 = C5*x1 + x3*x5
      a1_rs = C5_rs*x1 + x3*x7 + 24*x5*x6 + 8*x8*x9
      a1_rs_rs = C3_rs_rs*x3*x4 + 16*C5_rs*x0*x8 + C5_rs_rs*x1 + 120*rs**4*x5 &
      + 56*x2*x9 + 48*x6*x7
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
      x11 = C2_rs*x4
      x12 = rs**7
      x13 = C4*x0
      x14 = rs**3
      x15 = Ec_rs*x8
      a2 = C4*x1 + x3*x5 + x7*x9
      a2_rs = C4_rs*x1 + 24*x10*x5 + x11*x3 + 8*x12*x13 + 24*x14*x9 + x15*x7
      a2_rs_rs = C2_rs_rs*x3*x4 + 16*C4_rs*x0*x12 + C4_rs_rs*x1 + Ec_rs_rs*x7 &
      *x8 + 72*rs**2*x9 + 48*x10*x11 + 56*x13*x2 + 48*x14*x15 + 120*x5* &
      x6
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**7
      x3 = C3*x0
      a3 = C3*x1
      a3_rs = C3_rs*x1 + 8*x2*x3
      a3_rs_rs = 16*C3_rs*x0*x2 + C3_rs_rs*x1 + 56*rs**6*x3
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = Ec*x4
      x6 = rs**7
      x7 = C2*x0
      x8 = rs**5
      x9 = Ec_rs*x4
      a4 = C2*x1 + x3*x5
      a4_rs = C2_rs*x1 + x3*x9 + 24*x5*x8 + 8*x6*x7
      a4_rs_rs = 16*C2_rs*x0*x6 + C2_rs_rs*x1 + Ec_rs_rs*x3*x4 + 120*rs**4*x5 &
      + 56*x2*x7 + 48*x8*x9

   END SUBROUTINE calc_a4

   SUBROUTINE calc_a5(rs, a5, a5_rs, a5_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      REAL(KIND=dp), INTENT(INOUT) :: a5, a5_rs, a5_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a5", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, beta, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, a1_, a1__rs, a1__rs_rs, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, a2_, a2__rs, a2__rs_rs, x0, x1, x2, x3, a3_, a3__rs, a3__rs_rs, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, a4_, a4__rs, a4__rs_rs

      beta = 0.784949_dp
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = C3*x4
      x6 = rs**5
      x7 = C3_rs*x4
      x8 = rs**7
      x9 = C5*x0
      a1 = C5*x1 + x3*x5
      a1_rs = C5_rs*x1 + x3*x7 + 24*x5*x6 + 8*x8*x9
      a1_rs_rs = C3_rs_rs*x3*x4 + 16*C5_rs*x0*x8 + C5_rs_rs*x1 + 120*rs**4*x5 &
      + 56*x2*x9 + 48*x6*x7
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
      x11 = C2_rs*x4
      x12 = rs**7
      x13 = C4*x0
      x14 = rs**3
      x15 = Ec_rs*x8
      a2 = C4*x1 + x3*x5 + x7*x9
      a2_rs = C4_rs*x1 + 24*x10*x5 + x11*x3 + 8*x12*x13 + 24*x14*x9 + x15*x7
      a2_rs_rs = C2_rs_rs*x3*x4 + 16*C4_rs*x0*x12 + C4_rs_rs*x1 + Ec_rs_rs*x7 &
      *x8 + 72*rs**2*x9 + 48*x10*x11 + 56*x13*x2 + 48*x14*x15 + 120*x5* &
      x6
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**7
      x3 = C3*x0
      a3 = C3*x1
      a3_rs = C3_rs*x1 + 8*x2*x3
      a3_rs_rs = 16*C3_rs*x0*x2 + C3_rs_rs*x1 + 56*rs**6*x3
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = Ec*x4
      x6 = rs**7
      x7 = C2*x0
      x8 = rs**5
      x9 = Ec_rs*x4
      a4 = C2*x1 + x3*x5
      a4_rs = C2_rs*x1 + x3*x9 + 24*x5*x8 + 8*x6*x7
      a4_rs_rs = 16*C2_rs*x0*x6 + C2_rs_rs*x1 + Ec_rs_rs*x3*x4 + 120*rs**4*x5 &
      + 56*x2*x7 + 48*x8*x9
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**7
      x3 = Ec*x0
      a5 = Ec*x1
      a5_rs = Ec_rs*x1 + 8*x2*x3
      a5_rs_rs = 16*Ec_rs*x0*x2 + Ec_rs_rs*x1 + 56*rs**6*x3

   END SUBROUTINE calc_a5

   SUBROUTINE calc_Q(rs, mu, Q, Q_rs, Q_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs, mu
      REAL(KIND=dp), INTENT(INOUT) :: Q, Q_rs, Q_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_Q", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, a, c, d, q, b, alpha

      a = 5.84605_dp
      c = 3.91744_dp
      d = 3.44851_dp
      q = (-2 + 2*0.693147180559945_dp)/pi**2
      b = 3.44851_dp - 17.53815_dp*pi/(-4 + 4*0.693147180559945_dp)
      alpha = (1.0_dp/3.0_dp)*2**(2.0_dp/3.0_dp)*3**(1.0_dp/3.0_dp)/pi**(1.0_dp/3.0_dp &
      )
      x0 = mu**2
      x1 = d*x0
      x2 = sqrt(rs)
      x3 = a*mu
      x4 = x2*x3 + 1
      x5 = rs*x1 + x4
      x6 = 1.0_dp/x5
      x7 = b*x0
      x8 = rs**(3.0_dp/2.0_dp)
      x9 = c*mu**3
      x10 = rs*x7 + x4 + x8*x9
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
      x22 = q*x11*x21
      x23 = q*x5
      x24 = (1.0_dp/4.0_dp)*x3/x8
      Q = q*log(x10*x6)
      Q_rs = x22*x5
      Q_rs_rs = x11*x23*(x10*x18*(-2*x1 - x13)/x5**3 + 2*x16*x18*x19 + x20* &
      x24 + x6*((3.0_dp/4.0_dp)*x12*x9 - x24)) + x22*(x1 + x14) + x21*x23 &
      *(-x15 + x17 - x7)/x10**2

   END SUBROUTINE calc_Q

   SUBROUTINE ec_mu(rs, mu, ec_mu, ec_mu_rs, ec_mu_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs, mu
      REAL(KIND=dp), INTENT(INOUT) :: ec_mu, ec_mu_rs, ec_mu_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "ec_mu", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18

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

   END SUBROUTINE ec_mu

