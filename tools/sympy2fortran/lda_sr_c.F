
   PURE SUBROUTINE calc_D3(rs, D3, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: D3(0:)

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
      D3(0) = x4
      D3(1) = -x0*x6 + x3 - x5*(-2*d3_2 + 2*rs)
      D3(2) = d3_1**2*x4 + d3_1*x0*x7 - 2*x6 - x7 + 6*x0*x2/rs**4

   END SUBROUTINE calc_D3

   PURE SUBROUTINE calc_D2(rs, D2, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: D2(0:)

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
      D2(0) = x3
      D2(1) = -d2_1*x3 - x2*x6 + x5
      D2(2) = d2_1**2*x3 + 2*d2_1*x2*x6 - 2*d2_1*x5 - 2*x4*x6 + 2*x2/rs**3

   END SUBROUTINE calc_D2

   PURE SUBROUTINE calc_g2z(rs, g2z, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: g2z(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_g2z", routineP = moduleN//":"//routineN

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
      g2z(0) = x10*x2
      g2z(1) = -2.0_dp/5.0_dp*x10*x12 - x11*x2 + x16*x2
      g2z(2) = -g2_1*x14*x18*x4 - g2_3*x18*x5*x7 + x11*x17 + x14*x2*x3*x5*(-4 &
      *x13 - 2*x6)/x8**3 - x16*x17 + (6.0_dp/5.0_dp)*x10/rs**4

   END SUBROUTINE calc_g2z

   PURE SUBROUTINE calc_g0(rs, g0, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: g0(0:)

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
      g0(0) = x4
      g0(1) = -g0_5*x4 + x5
      g0(2) = g0_5**2*x4 - 2*g0_5*x5 + x0*(g0_2 - 3*g0_3*rs + 6*g0_4*x2)

   END SUBROUTINE calc_g0

   PURE SUBROUTINE calc_C2(rs, C2, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: C2(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C2", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, g0(0:2)

      CALL calc_g0(rs, g0, max_deriv)

      x0 = rs**(-3)
      x1 = (3.0_dp/8.0_dp)*x0
      x2 = rs**(-4)
      C2(0) = x0*(-3.0_dp/8.0_dp*g0(0) + 3.0_dp/16.0_dp)
      C2(1) = -g0(1)*x1 + x2*((9.0_dp/8.0_dp)*g0(0) - 9.0_dp/16.0_dp)
      C2(2) = (9.0_dp/4.0_dp)*g0(1)*x2 - g0(2)*x1 + (-9.0_dp/2.0_dp*g0(0) + 9.0_dp &
      /4.0_dp)/rs**5

   END SUBROUTINE calc_C2

   PURE SUBROUTINE calc_C3(rs, C3, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: C3(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C3", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, c3_1, g0(0:2)

      CALL calc_g0(rs, g0, max_deriv)

      c3_1 = one/(sqrt(pi)*sqrt(two))
      x0 = c3_1/rs**3
      x1 = c3_1*g0(0)
      x2 = rs**(-4)
      C3(0) = -g0(0)*x0
      C3(1) = -g0(1)*x0 + 3*x1*x2
      C3(2) = 6*c3_1*g0(1)*x2 - g0(2)*x0 - 12*x1/rs**5

   END SUBROUTINE calc_C3

   PURE SUBROUTINE calc_C4(rs, C4, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: C4(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C4", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, g2z(0:2), D2(0:2)

      CALL calc_g2z(rs, g2z, max_deriv)
      CALL calc_D2(rs, D2, max_deriv)

      x0 = rs**(-3)
      x1 = alpha**(-2)
      x2 = x1/rs**2
      x3 = rs**(-4)
      x4 = x0*x1
      C4(0) = x0*(-9.0_dp/64.0_dp*D2(0) - 9.0_dp/64.0_dp*g2z(0) + (9.0_dp/320.0_dp) &
      *x2)
      C4(1) = x0*(-9.0_dp/64.0_dp*D2(1) - 9.0_dp/64.0_dp*g2z(1) - 9.0_dp/160.0_dp* &
      x4) + x3*((27.0_dp/64.0_dp)*D2(0) + (27.0_dp/64.0_dp)*g2z(0) - 27.0_dp &
      /320.0_dp*x2)
      C4(2) = x0*(-9.0_dp/64.0_dp*D2(2) - 9.0_dp/64.0_dp*g2z(2) + (27.0_dp/160.0_dp &
      )*x1*x3) + x3*((27.0_dp/32.0_dp)*D2(1) + (27.0_dp/32.0_dp)*g2z(1) + ( &
      27.0_dp/80.0_dp)*x4) + (-27.0_dp/16.0_dp*D2(0) - 27.0_dp/16.0_dp*g2z(0 &
      ) + (27.0_dp/80.0_dp)*x2)/rs**5

   END SUBROUTINE calc_C4

   PURE SUBROUTINE calc_C5(rs, C5, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: C5(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C5", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, g2z(0:2), D3(0:2)

      CALL calc_g2z(rs, g2z, max_deriv)
      CALL calc_D3(rs, D3, max_deriv)

      x0 = sqrt(2.0_dp)
      x1 = 1.0_dp/sqrt(pi)
      x2 = x0*x1/rs**3
      x3 = x0*x1/rs**4
      C5(0) = x2*(-9.0_dp/80.0_dp*D3(0) - 9.0_dp/80.0_dp*g2z(0))
      C5(1) = x2*(-9.0_dp/80.0_dp*D3(1) - 9.0_dp/80.0_dp*g2z(1)) + x3*((27.0_dp/ &
      80.0_dp)*D3(0) + (27.0_dp/80.0_dp)*g2z(0))
      C5(2) = x2*(-9.0_dp/80.0_dp*D3(2) - 9.0_dp/80.0_dp*g2z(2)) + x3*((27.0_dp/ &
      40.0_dp)*D3(1) + (27.0_dp/40.0_dp)*g2z(1)) + x0*x1*(-27.0_dp/20.0_dp* &
      D3(0) - 27.0_dp/20.0_dp*g2z(0))/rs**5

   END SUBROUTINE calc_C5

   PURE SUBROUTINE calc_a1(rs, a1, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a1(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a1", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, beta, C3(0:2), C5(0:2)

      CALL calc_C3(rs, C3, max_deriv)
      CALL calc_C5(rs, C5, max_deriv)

      beta = 0.784949_dp
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = C3(0)*x4
      x6 = rs**5
      x7 = C3(1)*x4
      x8 = rs**7
      x9 = C5(0)*x0
      a1(0) = C5(0)*x1 + x3*x5
      a1(1) = C5(1)*x1 + x3*x7 + 24*x5*x6 + 8*x8*x9
      a1(2) = C3(2)*x3*x4 + 16*C5(1)*x0*x8 + C5(2)*x1 + 120*rs**4*x5 + 56*x2* &
      x9 + 48*x6*x7

   END SUBROUTINE calc_a1

   PURE SUBROUTINE calc_a2(rs, a2, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a2(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a2", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, beta, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, a1_(0), a1_(1), a1_(2), Ec(0:2), C2(0:2), C4(0:2)

      CALL pw_lda_ed_loc(rs, Ec, max_deriv)
      CALL calc_C2(rs, C2, max_deriv)
      CALL calc_C4(rs, C4, max_deriv)

      beta = 0.784949_dp
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = C3(0)*x4
      x6 = rs**5
      x7 = C3(1)*x4
      x8 = rs**7
      x9 = C5(0)*x0
      a1(0) = C5(0)*x1 + x3*x5
      a1(1) = C5(1)*x1 + x3*x7 + 24*x5*x6 + 8*x8*x9
      a1(2) = C3(2)*x3*x4 + 16*C5(1)*x0*x8 + C5(2)*x1 + 120*rs**4*x5 + 56*x2* &
      x9 + 48*x6*x7
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = C2(0)*x4
      x6 = rs**4
      x7 = 6*x6
      x8 = beta**4
      x9 = Ec(0)*x8
      x10 = rs**5
      x11 = C2(1)*x4
      x12 = rs**7
      x13 = C4(0)*x0
      x14 = rs**3
      x15 = Ec(1)*x8
      a2(0) = C4(0)*x1 + x3*x5 + x7*x9
      a2(1) = C4(1)*x1 + 24*x10*x5 + x11*x3 + 8*x12*x13 + 24*x14*x9 + x15*x7
      a2(2) = C2(2)*x3*x4 + 16*C4(1)*x0*x12 + C4(2)*x1 + Ec(2)*x7*x8 + 72*rs &
      **2*x9 + 48*x10*x11 + 56*x13*x2 + 48*x14*x15 + 120*x5*x6

   END SUBROUTINE calc_a2

   PURE SUBROUTINE calc_a3(rs, a3, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a3(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a3", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, beta, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, a1_(0), a1_(1), a1_(2), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, a2_(0), a2_(1), a2_(2), C3(0:2)

      CALL calc_C3(rs, C3, max_deriv)

      beta = 0.784949_dp
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = C3(0)*x4
      x6 = rs**5
      x7 = C3(1)*x4
      x8 = rs**7
      x9 = C5(0)*x0
      a1(0) = C5(0)*x1 + x3*x5
      a1(1) = C5(1)*x1 + x3*x7 + 24*x5*x6 + 8*x8*x9
      a1(2) = C3(2)*x3*x4 + 16*C5(1)*x0*x8 + C5(2)*x1 + 120*rs**4*x5 + 56*x2* &
      x9 + 48*x6*x7
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = C2(0)*x4
      x6 = rs**4
      x7 = 6*x6
      x8 = beta**4
      x9 = Ec(0)*x8
      x10 = rs**5
      x11 = C2(1)*x4
      x12 = rs**7
      x13 = C4(0)*x0
      x14 = rs**3
      x15 = Ec(1)*x8
      a2(0) = C4(0)*x1 + x3*x5 + x7*x9
      a2(1) = C4(1)*x1 + 24*x10*x5 + x11*x3 + 8*x12*x13 + 24*x14*x9 + x15*x7
      a2(2) = C2(2)*x3*x4 + 16*C4(1)*x0*x12 + C4(2)*x1 + Ec(2)*x7*x8 + 72*rs &
      **2*x9 + 48*x10*x11 + 56*x13*x2 + 48*x14*x15 + 120*x5*x6
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**7
      x3 = C3(0)*x0
      a3(0) = C3(0)*x1
      a3(1) = C3(1)*x1 + 8*x2*x3
      a3(2) = 16*C3(1)*x0*x2 + C3(2)*x1 + 56*rs**6*x3

   END SUBROUTINE calc_a3

   PURE SUBROUTINE calc_a4(rs, a4, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a4(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a4", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, beta, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, a1_(0), a1_(1), a1_(2), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, a2_(0), a2_(1), a2_(2), x0, x1, x2, x3, a3_(0), a3_(1), a3_(2), C2(0:2), Ec(0:2)

      CALL calc_C2(rs, C2, max_deriv)
      CALL pw_lda_ed_loc(rs, Ec, max_deriv)

      beta = 0.784949_dp
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = C3(0)*x4
      x6 = rs**5
      x7 = C3(1)*x4
      x8 = rs**7
      x9 = C5(0)*x0
      a1(0) = C5(0)*x1 + x3*x5
      a1(1) = C5(1)*x1 + x3*x7 + 24*x5*x6 + 8*x8*x9
      a1(2) = C3(2)*x3*x4 + 16*C5(1)*x0*x8 + C5(2)*x1 + 120*rs**4*x5 + 56*x2* &
      x9 + 48*x6*x7
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = C2(0)*x4
      x6 = rs**4
      x7 = 6*x6
      x8 = beta**4
      x9 = Ec(0)*x8
      x10 = rs**5
      x11 = C2(1)*x4
      x12 = rs**7
      x13 = C4(0)*x0
      x14 = rs**3
      x15 = Ec(1)*x8
      a2(0) = C4(0)*x1 + x3*x5 + x7*x9
      a2(1) = C4(1)*x1 + 24*x10*x5 + x11*x3 + 8*x12*x13 + 24*x14*x9 + x15*x7
      a2(2) = C2(2)*x3*x4 + 16*C4(1)*x0*x12 + C4(2)*x1 + Ec(2)*x7*x8 + 72*rs &
      **2*x9 + 48*x10*x11 + 56*x13*x2 + 48*x14*x15 + 120*x5*x6
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**7
      x3 = C3(0)*x0
      a3(0) = C3(0)*x1
      a3(1) = C3(1)*x1 + 8*x2*x3
      a3(2) = 16*C3(1)*x0*x2 + C3(2)*x1 + 56*rs**6*x3
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = Ec(0)*x4
      x6 = rs**7
      x7 = C2(0)*x0
      x8 = rs**5
      x9 = Ec(1)*x4
      a4(0) = C2(0)*x1 + x3*x5
      a4(1) = C2(1)*x1 + x3*x9 + 24*x5*x8 + 8*x6*x7
      a4(2) = 16*C2(1)*x0*x6 + C2(2)*x1 + Ec(2)*x3*x4 + 120*rs**4*x5 + 56*x2* &
      x7 + 48*x8*x9

   END SUBROUTINE calc_a4

   PURE SUBROUTINE calc_a5(rs, a5, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a5(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a5", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, beta, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, a1_(0), a1_(1), a1_(2), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, a2_(0), a2_(1), a2_(2), x0, x1, x2, x3, a3_(0), a3_(1), a3_(2), x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, a4_(0), a4_(1), a4_(2), Ec(0:2)

      CALL pw_lda_ed_loc(rs, Ec, max_deriv)

      beta = 0.784949_dp
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = C3(0)*x4
      x6 = rs**5
      x7 = C3(1)*x4
      x8 = rs**7
      x9 = C5(0)*x0
      a1(0) = C5(0)*x1 + x3*x5
      a1(1) = C5(1)*x1 + x3*x7 + 24*x5*x6 + 8*x8*x9
      a1(2) = C3(2)*x3*x4 + 16*C5(1)*x0*x8 + C5(2)*x1 + 120*rs**4*x5 + 56*x2* &
      x9 + 48*x6*x7
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = C2(0)*x4
      x6 = rs**4
      x7 = 6*x6
      x8 = beta**4
      x9 = Ec(0)*x8
      x10 = rs**5
      x11 = C2(1)*x4
      x12 = rs**7
      x13 = C4(0)*x0
      x14 = rs**3
      x15 = Ec(1)*x8
      a2(0) = C4(0)*x1 + x3*x5 + x7*x9
      a2(1) = C4(1)*x1 + 24*x10*x5 + x11*x3 + 8*x12*x13 + 24*x14*x9 + x15*x7
      a2(2) = C2(2)*x3*x4 + 16*C4(1)*x0*x12 + C4(2)*x1 + Ec(2)*x7*x8 + 72*rs &
      **2*x9 + 48*x10*x11 + 56*x13*x2 + 48*x14*x15 + 120*x5*x6
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**7
      x3 = C3(0)*x0
      a3(0) = C3(0)*x1
      a3(1) = C3(1)*x1 + 8*x2*x3
      a3(2) = 16*C3(1)*x0*x2 + C3(2)*x1 + 56*rs**6*x3
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = Ec(0)*x4
      x6 = rs**7
      x7 = C2(0)*x0
      x8 = rs**5
      x9 = Ec(1)*x4
      a4(0) = C2(0)*x1 + x3*x5
      a4(1) = C2(1)*x1 + x3*x9 + 24*x5*x8 + 8*x6*x7
      a4(2) = 16*C2(1)*x0*x6 + C2(2)*x1 + Ec(2)*x3*x4 + 120*rs**4*x5 + 56*x2* &
      x7 + 48*x8*x9
      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**7
      x3 = Ec(0)*x0
      a5(0) = Ec(0)*x1
      a5(1) = Ec(1)*x1 + 8*x2*x3
      a5(2) = 16*Ec(1)*x0*x2 + Ec(2)*x1 + 56*rs**6*x3

   END SUBROUTINE calc_a5

   PURE SUBROUTINE calc_Q(xx, mu, Q, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: xx, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: Q(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_Q", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, aa, cc, dd, qq, bb, alpha


      aa = 5.84605_dp
      cc = 3.91744_dp
      dd = 3.44851_dp
      qq = (-2 + 2*0.693147180559945_dp)/pi**2
      bb = 3.44851_dp - 17.53815_dp*pi/(-4 + 4*0.693147180559945_dp)
      alpha = (1.0_dp/3.0_dp)*2**(2.0_dp/3.0_dp)*3**(1.0_dp/3.0_dp)/pi**(1.0_dp/3.0_dp &
      )
      x0 = xx**2
      x1 = aa*xx + 1
      x2 = dd*x0 + x1
      x3 = 1.0_dp/x2
      x4 = bb*x0 + cc*xx**3 + x1
      x5 = 1.0_dp/x4
      x6 = 2*bb
      x7 = x6*xx
      x8 = 3*cc*x0
      x9 = aa + x7 + x8
      x10 = x2**(-2)
      x11 = -aa
      x12 = dd*xx
      x13 = 2*x12
      x14 = x11 - x13
      x15 = x14*x4
      x16 = x10*x15 + x3*x9
      x17 = qq*x16*x5
      x18 = qq*x2
      x19 = 2*x10
      Q(0) = qq*log(x3*x4)
      Q(1) = x17*x2
      Q(2) = x16*x18*(x11 - x7 - x8)/x4**2 + x17*(aa + x13) + x18*x5*(-dd*x19 &
      *x4 + x14*x19*x9 + x15*(-2*aa - 4*x12)/x2**3 + x3*(6*cc*xx + x6))

   END SUBROUTINE calc_Q

   PURE SUBROUTINE ec_mu(rs, mu, ec_mu, max_deriv)
      REAL(KIND=dp), INTENT(IN) :: rs, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: ec_mu(0:)

      CHARACTER(LEN=*), PARAMETER :: routineN = "ec_mu", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, Ec(0:2), Q(0:2), a1(0:2), a2(0:2), a3(0:2), a4(0:2), a5(0:2)

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
      x4 = x0*x3 + 1
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
      x16 = x4**(-5)
      x17 = rs*x1*x16*x2
      x18 = 8*Q(0) + 8*x11 + 8*x13 + 8*x15 + 8*x7 + 8*x9
      ec_mu(0) = Ec(0) + x5*(-Q(0) - x11 - x13 - x15 - x7 - x9)
      ec_mu(1) = Ec(1) + x17*x18 + x5*(-Q(1) - a1(1)*x6 - a2(1)*x8 - a3(1)*x10 &
      - a4(1)*x12 - a5(1)*x14)
      ec_mu(2) = Ec(2) + beta**4*x0*x8*(-80*Q(0) - 80*x11 - 80*x13 - 80*x15 - &
      80*x7 - 80*x9)/x4**6 + x16*x18*x3 + x17*(16*Q(1) + 16*a1(1)*x6 + &
      16*a2(1)*x8 + 16*a3(1)*x10 + 16*a4(1)*x12 + 16*a5(1)*x14) + x5*( &
      -Q(2) - a1(2)*x6 - a2(2)*x8 - a3(2)*x10 - a4(2)*x12 - a5(2)*x14)

   END SUBROUTINE ec_mu
