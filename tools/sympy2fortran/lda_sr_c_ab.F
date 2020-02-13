
   ELEMENTAL SUBROUTINE calc_D3(rs, max_deriv, D3, D3_rs, D3_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: D3, D3_rs, D3_rs_rs

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

   ELEMENTAL SUBROUTINE calc_D2(rs, max_deriv, D2, D2_rs, D2_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: D2, D2_rs, D2_rs_rs

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

   ELEMENTAL SUBROUTINE calc_g2z_ab(rs, zeta, max_deriv, g2z_ab, g2z_ab_rs, g2z_ab_zeta, g2z_ab_rs_rs, g2z_ab_rs_zeta, g2z_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: g2z_ab, g2z_ab_rs, g2z_ab_zeta, g2z_ab_rs_rs, g2z_ab_rs_zeta, g2z_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_g2z_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, x61, x62, x63, x64, x65, x66, x67, x68, x69, x70, x71, x72, x73, x74, x75, x76, x77, x78, x79, x80, x81, x82, x83, x84, x85

      x0 = (1.0_dp/2.0_dp)*zeta
      x1 = x0 + 1.0_dp/2.0_dp
      x2 = x1**2
      x3 = alpha**(-2)
      x4 = rs**2
      x5 = 1.0_dp/x4
      x6 = zeta + 1
      x7 = 1.0_dp/x6
      x8 = x7**(2.0_dp/3.0_dp)
      x9 = 1.0_dp/x8
      x10 = 2**(1.0_dp/3.0_dp)
      x11 = g2_1*rs*x10
      x12 = x7**(1.0_dp/3.0_dp)
      x13 = -x11*x12 + 1
      x14 = g2_2*x10*x12
      x15 = rs*x14
      x16 = 2**(2.0_dp/3.0_dp)
      x17 = g2_3*x16*x4
      x18 = x17*x8
      x19 = x15 + x18 + 1
      x20 = 1.0_dp/x19
      x21 = (2.0_dp/5.0_dp)*x13*x20*x3*x5*x9
      x22 = (-x0 + 1.0_dp/2.0_dp)**2
      x23 = -zeta + 1
      x24 = 1.0_dp/x23
      x25 = x24**(2.0_dp/3.0_dp)
      x26 = 1.0_dp/x25
      x27 = x24**(1.0_dp/3.0_dp)
      x28 = -x11*x27 + 1
      x29 = g2_2*x10*x27
      x30 = rs*x29
      x31 = x17*x25
      x32 = x30 + x31 + 1
      x33 = 1.0_dp/x32
      x34 = (2.0_dp/5.0_dp)*x26*x28*x3*x33*x5
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
      x45 = -x14 - 2*x44
      x46 = x19**(-2)
      x47 = (2.0_dp/5.0_dp)*x13*x3*x45*x46*x5*x9
      x48 = g2_3*rs*x16*x25
      x49 = -x29 - 2*x48
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
      g2z_ab = x2*x21 + x22*x34
      g2z_ab_rs = -4.0_dp/5.0_dp*x2*x42 + x2*x47 - 4.0_dp/5.0_dp*x22*x43 + x22*x51 &
      - x37 - x40
      g2z_ab_zeta = x1*x21 - 2.0_dp/15.0_dp*x24*x52*x54 - 4.0_dp/15.0_dp*x24*x57 + &
      x34*x56 + (2.0_dp/15.0_dp)*x52*x53*x7 + (4.0_dp/15.0_dp)*x55*x7 + x60 &
      *x61 + x64*x65
      g2z_ab_rs_rs = -4.0_dp/5.0_dp*g2_1*x10*x2*x3*x35*x45*x46*x5 - 4.0_dp/5.0_dp* &
      g2_1*x10*x22*x3*x38*x49*x5*x50 - 4.0_dp/5.0_dp*g2_3*x13*x16*x2*x3* &
      x46*x5 - 4.0_dp/5.0_dp*g2_3*x16*x22*x28*x3*x5*x50 + (12.0_dp/5.0_dp)* &
      x13*x2*x20*x3*x67*x9 - 8.0_dp/5.0_dp*x13*x2*x3*x41*x45*x46*x9 + ( &
      12.0_dp/5.0_dp)*x22*x26*x28*x3*x33*x67 - 8.0_dp/5.0_dp*x22*x26*x28*x3 &
      *x41*x49*x50 + x53*x66 + x54*x66 + x69*(-2*x14 - 4*x44) + x71*(-2 &
      *x29 - 4*x48)
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

   END SUBROUTINE calc_g2z_ab

   ELEMENTAL SUBROUTINE calc_g2z_p1_ab(rs, zeta, max_deriv, g2z_p1_ab, g2z_p1_ab_rs, g2z_p1_ab_zeta, g2z_p1_ab_rs_rs, g2z_p1_ab_rs_zeta, g2z_p1_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: g2z_p1_ab, g2z_p1_ab_rs, g2z_p1_ab_zeta, g2z_p1_ab_rs_rs, g2z_p1_ab_rs_zeta, g2z_p1_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_g2z_p1_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29

      x0 = 2**(2.0_dp/3.0_dp)
      x1 = alpha**(-2)
      x2 = rs**2
      x3 = 1.0_dp/x2
      x4 = -g2_1*rs + 1
      x5 = g2_2*rs
      x6 = g2_3*x2
      x7 = x5 + x6 + 1
      x8 = 1.0_dp/x7
      x9 = x0*x1*x3*x4*x8
      x10 = g2_1*x0*x1*x3*x8
      x11 = rs**(-3)
      x12 = x0*x1*x11*x4*x8
      x13 = g2_3*rs
      x14 = -g2_2 - 2*x13
      x15 = x7**(-2)
      x16 = (2.0_dp/5.0_dp)*x0*x1*x15*x3
      x17 = 1.0_dp/rs
      x18 = g2_1*x0*x1*x17*x8
      x19 = (1.0_dp/6.0_dp)*g2_2
      x20 = rs*x19 + (1.0_dp/3.0_dp)*x6
      x21 = (2.0_dp/5.0_dp)*x0*x1*x15*x3*x4
      x22 = (8.0_dp/5.0_dp)*x11
      x23 = (4.0_dp/5.0_dp)*x0*x1*x15*x3
      x24 = x0*x1*x14*x15*x4
      x25 = x7**(-3)
      x26 = (2.0_dp/5.0_dp)*x0*x1*x14*x25*x3*x4
      x27 = g2_1*x20
      x28 = x0*x1*x15*x20*x4
      x29 = (1.0_dp/3.0_dp)*x5 + (2.0_dp/3.0_dp)*x6
      g2z_p1_ab = (2.0_dp/5.0_dp)*x9
      g2z_p1_ab_rs = -2.0_dp/5.0_dp*x10 - 4.0_dp/5.0_dp*x12 + x14*x16*x4
      g2z_p1_ab_zeta = (1.0_dp/15.0_dp)*x18 + x20*x21 + (8.0_dp/15.0_dp)*x9
      g2z_p1_ab_rs_rs = g2_1*x0*x1*x22*x8 - g2_1*x14*x23 - g2_3*x23*x4 - x22* &
      x24 + x26*(-2*g2_2 - 4*x13) + (12.0_dp/5.0_dp)*x0*x1*x4*x8/rs**4
      g2z_p1_ab_rs_zeta = (1.0_dp/15.0_dp)*g2_1*x0*x1*x14*x15*x17 - 3.0_dp/5.0_dp* &
      x10 - 4.0_dp/5.0_dp*x11*x28 - 16.0_dp/15.0_dp*x12 - x16*x27 + x21*(( &
      2.0_dp/3.0_dp)*x13 + x19) + (8.0_dp/15.0_dp)*x24*x3 + x26*x29
      g2z_p1_ab_zeta_zeta = (2.0_dp/15.0_dp)*x0*x1*x15*x17*x27 + (2.0_dp/5.0_dp)* &
      x0*x1*x20*x25*x29*x3*x4 + (2.0_dp/15.0_dp)*x18 + x21*(-1.0_dp/9.0_dp* &
      x5 - 5.0_dp/18.0_dp*x6) + (16.0_dp/15.0_dp)*x28*x3 + (4.0_dp/9.0_dp)*x9

   END SUBROUTINE calc_g2z_p1_ab

   ELEMENTAL SUBROUTINE calc_g2z_m1_ab(rs, zeta, max_deriv, g2z_m1_ab, g2z_m1_ab_rs, g2z_m1_ab_zeta, g2z_m1_ab_rs_rs, g2z_m1_ab_rs_zeta, g2z_m1_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: g2z_m1_ab, g2z_m1_ab_rs, g2z_m1_ab_zeta, g2z_m1_ab_rs_rs, g2z_m1_ab_rs_zeta, g2z_m1_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_g2z_m1_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28

      x0 = 2**(2.0_dp/3.0_dp)
      x1 = alpha**(-2)
      x2 = (2.0_dp/5.0_dp)*x0*x1
      x3 = rs**2
      x4 = 1.0_dp/x3
      x5 = g2_1*rs
      x6 = -x5 + 1
      x7 = g2_2*rs + g2_3*x3 + 1
      x8 = rs**(-3)
      x9 = x7**2
      x10 = 1.0_dp/x9
      x11 = x5 - 1
      x12 = x11*x7
      x13 = g2_3*rs
      x14 = g2_2 + 2*x13
      x15 = x11*x14
      x16 = rs*x15
      x17 = x16 - x5*x7
      x18 = x0*x1*x4
      x19 = x7**(-3)
      x20 = x0*x1*x19
      x21 = g2_1*rs*x9
      x22 = x14**2*x3
      x23 = x11*x22
      x24 = x6*x9
      x25 = -x16*(2*g2_2*rs + 2*g2_3*x3 + 2)
      x26 = x3*x7
      x27 = g2_1*x14
      x28 = x26*x27
      g2z_m1_ab = x2*x4*x6/x7
      g2z_m1_ab_rs = x10*x2*x8*(2*x12 + x17)
      g2z_m1_ab_zeta = (1.0_dp/15.0_dp)*x10*x18*(8*x12 + x17)
      g2z_m1_ab_rs_rs = (4.0_dp/5.0_dp)*x20*(2*x21 - x23 + 3*x24 + x25 + x26*( &
      g2_3*x11 + x27))/rs**4
      g2z_m1_ab_rs_zeta = (1.0_dp/15.0_dp)*x20*x8*(rs*x7*(x11*(g2_2 + 4*x13) + &
      x14*x5 - 8*x15) + 9*x21 - 2*x23 + 16*x24 + x25 + x28)
      g2z_m1_ab_zeta_zeta = (1.0_dp/45.0_dp)*x18*x19*(-rs*x12*(6*g2_2 + 11*x13) &
      + 6*x21 + x22*x6 + 20*x24 + x28)

   END SUBROUTINE calc_g2z_m1_ab

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

   ELEMENTAL SUBROUTINE calc_phin(zeta, max_deriv, phin, phin_zeta, phin_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: phin, phin_zeta, phin_zeta_zeta

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

   ELEMENTAL SUBROUTINE calc_C2_ab(rs, zeta, max_deriv, C2_ab, C2_ab_rs, C2_ab_zeta, C2_ab_rs_rs, C2_ab_rs_zeta, C2_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: C2_ab, C2_ab_rs, C2_ab_zeta, C2_ab_rs_rs, C2_ab_rs_zeta, C2_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C2_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, g0, g0_rs, g0_rs_rs

      CALL calc_g0(rs, max_deriv, g0, g0_rs, g0_rs_rs)

      x0 = g0 - 1.0_dp/2.0_dp
      x1 = rs**(-3)
      x2 = zeta**2
      x3 = (3.0_dp/8.0_dp)*x2 - 3.0_dp/8.0_dp
      x4 = x1*x3
      x5 = rs**(-4)
      x6 = x0*x5
      x7 = (3.0_dp/4.0_dp)*x1*zeta
      C2_ab = x0*x4
      C2_ab_rs = g0_rs*x4 - x6*((9.0_dp/8.0_dp)*x2 - 9.0_dp/8.0_dp)
      C2_ab_zeta = x0*x7
      C2_ab_rs_rs = -6*g0_rs*x3*x5 + g0_rs_rs*x4 + 12*x0*x3/rs**5
      C2_ab_rs_zeta = g0_rs*x7 - 9.0_dp/4.0_dp*x6*zeta
      C2_ab_zeta_zeta = x1*((3.0_dp/4.0_dp)*g0 - 3.0_dp/8.0_dp)

   END SUBROUTINE calc_C2_ab

   ELEMENTAL SUBROUTINE calc_C3_ab(rs, zeta, max_deriv, C3_ab, C3_ab_rs, C3_ab_zeta, C3_ab_rs_rs, C3_ab_rs_zeta, C3_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: C3_ab, C3_ab_rs, C3_ab_zeta, C3_ab_rs_rs, C3_ab_rs_zeta, C3_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C3_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, c3_1, g0, g0_rs, g0_rs_rs

      CALL calc_g0(rs, max_deriv, g0, g0_rs, g0_rs_rs)

      c3_1 = one/(sqrt(pi)*sqrt(two))
      x0 = c3_1*g0
      x1 = rs**(-3)
      x2 = zeta**2
      x3 = x2 - 1
      x4 = x1*x3
      x5 = c3_1*g0_rs*x1
      x6 = rs**(-4)
      x7 = c3_1*g0*x6
      x8 = 2*x0*x1
      C3_ab = x0*x4
      C3_ab_rs = x3*x5 - x7*(3*x2 - 3)
      C3_ab_zeta = x8*zeta
      C3_ab_rs_rs = -6*c3_1*g0_rs*x3*x6 + c3_1*g0_rs_rs*x4 + 12*x0*x3/rs**5
      C3_ab_rs_zeta = 2*x5*zeta - 6*x7*zeta
      C3_ab_zeta_zeta = x8

   END SUBROUTINE calc_C3_ab

   ELEMENTAL SUBROUTINE calc_C4_ab(rs, zeta, max_deriv, C4_ab, C4_ab_rs, C4_ab_zeta, C4_ab_rs_rs, C4_ab_rs_zeta, C4_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: C4_ab, C4_ab_rs, C4_ab_zeta, C4_ab_rs_rs, C4_ab_rs_zeta, C4_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C4_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, g2z_ab, g2z_ab_rs, g2z_ab_zeta, g2z_ab_rs_rs, g2z_ab_rs_zeta, g2z_ab_zeta_zeta, D2, D2_rs, D2_rs_rs, phi8, phi8_zeta, phi8_zeta_zeta

      CALL calc_g2z_ab(rs, zeta, max_deriv, g2z_ab, g2z_ab_rs, g2z_ab_zeta, g2z_ab_rs_rs, g2z_ab_rs_zeta, g2z_ab_zeta_zeta)
      CALL calc_D2(rs, max_deriv, D2, D2_rs, D2_rs_rs)
      CALL calc_phi8(zeta, max_deriv, phi8, phi8_zeta, phi8_zeta_zeta)

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
      C4_ab = x0*(-9.0_dp/64.0_dp*g2z_ab + (9.0_dp/320.0_dp)*x3 - 9.0_dp/64.0_dp*x5)
      C4_ab_rs = x0*(-9.0_dp/64.0_dp*g2z_ab_rs - 9.0_dp/160.0_dp*x7 - 9.0_dp/64.0_dp &
      *x8) + x6*((27.0_dp/64.0_dp)*g2z_ab - 27.0_dp/320.0_dp*x3 + (27.0_dp/ &
      64.0_dp)*x5)
      C4_ab_zeta = x0*(-9.0_dp/64.0_dp*g2z_ab_zeta + (9.0_dp/320.0_dp)*x10 + x9* &
      zeta)
      C4_ab_rs_rs = x0*(-9.0_dp/64.0_dp*D2_rs_rs*x4 - 9.0_dp/64.0_dp*g2z_ab_rs_rs &
      + (27.0_dp/160.0_dp)*phi8*x1*x6) + x6*((27.0_dp/32.0_dp)*g2z_ab_rs + &
      (27.0_dp/80.0_dp)*x7 + (27.0_dp/32.0_dp)*x8) + (-27.0_dp/16.0_dp*g2z_ab &
      + (27.0_dp/80.0_dp)*x3 - 27.0_dp/16.0_dp*x5)/rs**5
      C4_ab_rs_zeta = x0*((9.0_dp/32.0_dp)*D2_rs*zeta - 9.0_dp/64.0_dp* &
      g2z_ab_rs_zeta - 9.0_dp/160.0_dp*phi8_zeta*x0*x1) + x6*(-27.0_dp/ &
      32.0_dp*D2*zeta + (27.0_dp/64.0_dp)*g2z_ab_zeta - 27.0_dp/320.0_dp*x10 &
      )
      C4_ab_zeta_zeta = x0*(-9.0_dp/64.0_dp*g2z_ab_zeta_zeta + (9.0_dp/320.0_dp)* &
      phi8_zeta_zeta*x1*x2 + x9)

   END SUBROUTINE calc_C4_ab

   ELEMENTAL SUBROUTINE calc_C5_ab(rs, zeta, max_deriv, C5_ab, C5_ab_rs, C5_ab_zeta, C5_ab_rs_rs, C5_ab_rs_zeta, C5_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: C5_ab, C5_ab_rs, C5_ab_zeta, C5_ab_rs_rs, C5_ab_rs_zeta, C5_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_C5_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, g2z_ab, g2z_ab_rs, g2z_ab_zeta, g2z_ab_rs_rs, g2z_ab_rs_zeta, g2z_ab_zeta_zeta, D3, D3_rs, D3_rs_rs

      CALL calc_g2z_ab(rs, zeta, max_deriv, g2z_ab, g2z_ab_rs, g2z_ab_zeta, g2z_ab_rs_rs, g2z_ab_rs_zeta, g2z_ab_zeta_zeta)
      CALL calc_D3(rs, max_deriv, D3, D3_rs, D3_rs_rs)

      x0 = sqrt(2.0_dp)
      x1 = 1.0_dp/sqrt(pi)
      x2 = x0*x1/rs**3
      x3 = -zeta**2 + 1
      x4 = D3*x3
      x5 = x0*x1/rs**4
      x6 = D3_rs*x3
      x7 = (9.0_dp/40.0_dp)*D3
      C5_ab = x2*(-9.0_dp/80.0_dp*g2z_ab - 9.0_dp/80.0_dp*x4)
      C5_ab_rs = x2*(-9.0_dp/80.0_dp*g2z_ab_rs - 9.0_dp/80.0_dp*x6) + x5*((27.0_dp/ &
      80.0_dp)*g2z_ab + (27.0_dp/80.0_dp)*x4)
      C5_ab_zeta = x2*(-9.0_dp/80.0_dp*g2z_ab_zeta + x7*zeta)
      C5_ab_rs_rs = x2*(-9.0_dp/80.0_dp*D3_rs_rs*x3 - 9.0_dp/80.0_dp*g2z_ab_rs_rs &
      ) + x5*((27.0_dp/40.0_dp)*g2z_ab_rs + (27.0_dp/40.0_dp)*x6) + x0*x1*( &
      -27.0_dp/20.0_dp*g2z_ab - 27.0_dp/20.0_dp*x4)/rs**5
      C5_ab_rs_zeta = x2*((9.0_dp/40.0_dp)*D3_rs*zeta - 9.0_dp/80.0_dp* &
      g2z_ab_rs_zeta) + x5*(-27.0_dp/40.0_dp*D3*zeta + (27.0_dp/80.0_dp)* &
      g2z_ab_zeta)
      C5_ab_zeta_zeta = x2*(-9.0_dp/80.0_dp*g2z_ab_zeta_zeta + x7)

   END SUBROUTINE calc_C5_ab

   ELEMENTAL SUBROUTINE calc_a1_ab(rs, zeta, max_deriv, a1_ab, a1_ab_rs, a1_ab_zeta, a1_ab_rs_rs, a1_ab_rs_zeta, a1_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a1_ab, a1_ab_rs, a1_ab_zeta, a1_ab_rs_rs, a1_ab_rs_zeta, a1_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a1_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, C3_ab, C3_ab_rs, C3_ab_zeta, C3_ab_rs_rs, C3_ab_rs_zeta, C3_ab_zeta_zeta, C5_ab, C5_ab_rs, C5_ab_zeta, C5_ab_rs_rs, C5_ab_rs_zeta, C5_ab_zeta_zeta

      CALL calc_C3_ab(rs, zeta, max_deriv, C3_ab, C3_ab_rs, C3_ab_zeta, C3_ab_rs_rs, C3_ab_rs_zeta, C3_ab_zeta_zeta)
      CALL calc_C5_ab(rs, zeta, max_deriv, C5_ab, C5_ab_rs, C5_ab_zeta, C5_ab_rs_rs, C5_ab_rs_zeta, C5_ab_zeta_zeta)

      x0 = b0**8
      x1 = C5_ab*x0
      x2 = b0**6
      x3 = 4*x2
      x4 = C5_ab_rs*x0
      x5 = 1.0_dp/rs
      x6 = 24*x5
      x7 = C3_ab*x2
      x8 = 8*x5
      x9 = C5_ab_zeta*x0
      x10 = rs**(-2)
      a1_ab = C3_ab*x3 + x1
      a1_ab_rs = C3_ab_rs*x3 + x1*x8 + x4 + x6*x7
      a1_ab_zeta = C3_ab_zeta*x3 + x9
      a1_ab_rs_rs = 48*C3_ab_rs*x2*x5 + C3_ab_rs_rs*x3 + C5_ab_rs_rs*x0 + 56* &
      x1*x10 + 120*x10*x7 + 16*x4*x5
      a1_ab_rs_zeta = C3_ab_rs_zeta*x3 + C3_ab_zeta*x2*x6 + C5_ab_rs_zeta*x0 + &
      x8*x9
      a1_ab_zeta_zeta = C3_ab_zeta_zeta*x3 + C5_ab_zeta_zeta*x0

   END SUBROUTINE calc_a1_ab

   ELEMENTAL SUBROUTINE calc_a2_ab(rs, zeta, max_deriv, a2_ab, a2_ab_rs, a2_ab_zeta, a2_ab_rs_rs, a2_ab_rs_zeta, a2_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a2_ab, a2_ab_rs, a2_ab_zeta, a2_ab_rs_rs, a2_ab_rs_zeta, a2_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a2_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, C2_ab, C2_ab_rs, C2_ab_zeta, C2_ab_rs_rs, C2_ab_rs_zeta, C2_ab_zeta_zeta, C4_ab, C4_ab_rs, C4_ab_zeta, C4_ab_rs_rs, C4_ab_rs_zeta, C4_ab_zeta_zeta, Ec_ab, Ec_ab_rs, Ec_ab_zeta, Ec_ab_rs_rs, Ec_ab_rs_zeta, Ec_ab_zeta_zeta

      CALL calc_C2_ab(rs, zeta, max_deriv, C2_ab, C2_ab_rs, C2_ab_zeta, C2_ab_rs_rs, C2_ab_rs_zeta, C2_ab_zeta_zeta)
      CALL calc_C4_ab(rs, zeta, max_deriv, C4_ab, C4_ab_rs, C4_ab_zeta, C4_ab_rs_rs, C4_ab_rs_zeta, C4_ab_zeta_zeta)
      CALL calc_Ec_ab(rs, zeta, max_deriv, Ec_ab, Ec_ab_rs, Ec_ab_zeta, Ec_ab_rs_rs, Ec_ab_rs_zeta, Ec_ab_zeta_zeta)

      x0 = b0**8
      x1 = C4_ab*x0
      x2 = b0**6
      x3 = 4*x2
      x4 = b0**4
      x5 = 6*x4
      x6 = C4_ab_rs*x0
      x7 = 1.0_dp/rs
      x8 = 24*x7
      x9 = C2_ab*x2
      x10 = 8*x7
      x11 = Ec_ab*x4
      x12 = C4_ab_zeta*x0
      x13 = rs**(-2)
      x14 = 48*x7
      a2_ab = C2_ab*x3 + Ec_ab*x5 + x1
      a2_ab_rs = C2_ab_rs*x3 + Ec_ab_rs*x5 + x1*x10 + x11*x8 + x6 + x8*x9
      a2_ab_zeta = C2_ab_zeta*x3 + Ec_ab_zeta*x5 + x12
      a2_ab_rs_rs = C2_ab_rs*x14*x2 + C2_ab_rs_rs*x3 + C4_ab_rs_rs*x0 + &
      Ec_ab_rs*x14*x4 + Ec_ab_rs_rs*x5 + 56*x1*x13 + 72*x11*x13 + 120* &
      x13*x9 + 16*x6*x7
      a2_ab_rs_zeta = C2_ab_rs_zeta*x3 + C2_ab_zeta*x2*x8 + C4_ab_rs_zeta*x0 + &
      Ec_ab_rs_zeta*x5 + Ec_ab_zeta*x4*x8 + x10*x12
      a2_ab_zeta_zeta = C2_ab_zeta_zeta*x3 + C4_ab_zeta_zeta*x0 + &
      Ec_ab_zeta_zeta*x5

   END SUBROUTINE calc_a2_ab

   ELEMENTAL SUBROUTINE calc_a3_ab(rs, zeta, max_deriv, a3_ab, a3_ab_rs, a3_ab_zeta, a3_ab_rs_rs, a3_ab_rs_zeta, a3_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a3_ab, a3_ab_rs, a3_ab_zeta, a3_ab_rs_rs, a3_ab_rs_zeta, a3_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a3_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, C3_ab, C3_ab_rs, C3_ab_zeta, C3_ab_rs_rs, C3_ab_rs_zeta, C3_ab_zeta_zeta, Ec_ab, Ec_ab_rs, Ec_ab_zeta, Ec_ab_rs_rs, Ec_ab_rs_zeta, Ec_ab_zeta_zeta

      CALL calc_C3_ab(rs, zeta, max_deriv, C3_ab, C3_ab_rs, C3_ab_zeta, C3_ab_rs_rs, C3_ab_rs_zeta, C3_ab_zeta_zeta)
      CALL calc_Ec_ab(rs, zeta, max_deriv, Ec_ab, Ec_ab_rs, Ec_ab_zeta, Ec_ab_rs_rs, Ec_ab_rs_zeta, Ec_ab_zeta_zeta)

      x0 = b0**8
      x1 = C3_ab*x0
      x2 = C3_ab_rs*x0
      x3 = 1.0_dp/rs
      x4 = 8*x3
      x5 = C3_ab_zeta*x0
      a3_ab = x1
      a3_ab_rs = x1*x4 + x2
      a3_ab_zeta = x5
      a3_ab_rs_rs = C3_ab_rs_rs*x0 + 16*x2*x3 + 56*x1/rs**2
      a3_ab_rs_zeta = C3_ab_rs_zeta*x0 + x4*x5
      a3_ab_zeta_zeta = C3_ab_zeta_zeta*x0

   END SUBROUTINE calc_a3_ab

   ELEMENTAL SUBROUTINE calc_a4_ab(rs, zeta, max_deriv, a4_ab, a4_ab_rs, a4_ab_zeta, a4_ab_rs_rs, a4_ab_rs_zeta, a4_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a4_ab, a4_ab_rs, a4_ab_zeta, a4_ab_rs_rs, a4_ab_rs_zeta, a4_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a4_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, C2_ab, C2_ab_rs, C2_ab_zeta, C2_ab_rs_rs, C2_ab_rs_zeta, C2_ab_zeta_zeta, Ec_ab, Ec_ab_rs, Ec_ab_zeta, Ec_ab_rs_rs, Ec_ab_rs_zeta, Ec_ab_zeta_zeta

      CALL calc_C2_ab(rs, zeta, max_deriv, C2_ab, C2_ab_rs, C2_ab_zeta, C2_ab_rs_rs, C2_ab_rs_zeta, C2_ab_zeta_zeta)
      CALL calc_Ec_ab(rs, zeta, max_deriv, Ec_ab, Ec_ab_rs, Ec_ab_zeta, Ec_ab_rs_rs, Ec_ab_rs_zeta, Ec_ab_zeta_zeta)

      x0 = beta**8
      x1 = rs**8*x0
      x2 = rs**6
      x3 = 4*x2
      x4 = beta**6
      x5 = Ec_ab*x4
      x6 = rs**7
      x7 = 8*x6
      x8 = C2_ab*x0
      x9 = rs**5
      x10 = 24*x9
      x11 = Ec_ab_rs*x4
      x12 = 4*x2*x4
      a4_ab = C2_ab*x1 + x3*x5
      a4_ab_rs = C2_ab_rs*x1 + x10*x5 + x11*x3 + x7*x8
      a4_ab_zeta = C2_ab_zeta*x1 + Ec_ab_zeta*x12
      a4_ab_rs_rs = 16*C2_ab_rs*x0*x6 + C2_ab_rs_rs*x1 + Ec_ab_rs_rs*x12 + 120 &
      *rs**4*x5 + 48*x11*x9 + 56*x2*x8
      a4_ab_rs_zeta = C2_ab_rs_zeta*x1 + C2_ab_zeta*x0*x7 + Ec_ab_rs_zeta*x12 &
      + Ec_ab_zeta*x10*x4
      a4_ab_zeta_zeta = C2_ab_zeta_zeta*x1 + Ec_ab_zeta_zeta*x12

   END SUBROUTINE calc_a4_ab

   ELEMENTAL SUBROUTINE calc_a5_ab(rs, zeta, max_deriv, a5_ab, a5_ab_rs, a5_ab_zeta, a5_ab_rs_rs, a5_ab_rs_zeta, a5_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: a5_ab, a5_ab_rs, a5_ab_zeta, a5_ab_rs_rs, a5_ab_rs_zeta, a5_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_a5_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, Ec_ab, Ec_ab_rs, Ec_ab_zeta, Ec_ab_rs_rs, Ec_ab_rs_zeta, Ec_ab_zeta_zeta

      CALL calc_Ec_ab(rs, zeta, max_deriv, Ec_ab, Ec_ab_rs, Ec_ab_zeta, Ec_ab_rs_rs, Ec_ab_rs_zeta, Ec_ab_zeta_zeta)

      x0 = b0**8
      x1 = Ec_ab*x0
      x2 = Ec_ab_rs*x0
      x3 = 1.0_dp/rs
      x4 = 8*x3
      x5 = Ec_ab_zeta*x0
      a5_ab = x1
      a5_ab_rs = x1*x4 + x2
      a5_ab_zeta = x5
      a5_ab_rs_rs = Ec_ab_rs_rs*x0 + 16*x2*x3 + 56*x1/rs**2
      a5_ab_rs_zeta = Ec_ab_rs_zeta*x0 + x4*x5
      a5_ab_zeta_zeta = Ec_ab_zeta_zeta*x0

   END SUBROUTINE calc_a5_ab

   ELEMENTAL SUBROUTINE calc_Q_ab(rs, zeta, mu, max_deriv, Q_ab, Q_ab_rs, Q_ab_zeta, Q_ab_rs_rs, Q_ab_rs_zeta, Q_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: Q_ab, Q_ab_rs, Q_ab_zeta, Q_ab_rs_rs, Q_ab_rs_zeta, Q_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_Q_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, x60, aa, cc, dd, qq, bb, phi2, phi2_zeta, phi2_zeta_zeta

      CALL calc_phi2(zeta, max_deriv, phi2, phi2_zeta, phi2_zeta_zeta)

      aa = 5.84605_dp
      cc = 3.91744_dp
      dd = 3.44851_dp
      qq = 2*(-1 + 0.693147180559945_dp)/pi**2
      bb = -3*pi*a/(-4 + 4*0.693147180559945_dp) + 3.44851_dp
      x0 = phi2**(-4)
      x1 = xx**2
      x2 = x0*x1
      x3 = dd*x2
      x4 = aa*xx
      x5 = x4/phi2**2
      x6 = x5 + 1
      x7 = x3 + x6
      x8 = 1.0_dp/x7
      x9 = bb*x2
      x10 = phi2**(-6)
      x11 = xx**3
      x12 = cc*x11
      x13 = x10*x12
      x14 = x13 + x6 + x9
      x15 = 1.0_dp/x14
      x16 = 1.0_dp/rs
      x17 = x16*x5
      x18 = (1.0_dp/2.0_dp)*x17
      x19 = x16*x9
      x20 = (3.0_dp/2.0_dp)*x13*x16
      x21 = x18 + x19 + x20
      x22 = -x18
      x23 = x16*x3
      x24 = x22 - x23
      x25 = x7**(-2)
      x26 = x14*x25
      x27 = x21*x8 + x24*x26
      x28 = qq*x15*x27
      x29 = phi2**(-3)
      x30 = aa*x29*xx
      x31 = phi2_zeta*x30
      x32 = -x31
      x33 = phi2**(-5)
      x34 = 2*bb*x1*x33
      x35 = phi2_zeta*x34
      x36 = phi2**(-7)
      x37 = 3*cc*x11*x36
      x38 = phi2_zeta*x37
      x39 = x32 - x35 - x38
      x40 = dd*phi2_zeta*x1*x33
      x41 = 2*x40
      x42 = x31 + x41
      x43 = x26*x42 + x39*x8
      x44 = qq*x15*x43
      x45 = x14**(-2)
      x46 = qq*x15*x7
      x47 = rs**(-2)
      x48 = (1.0_dp/4.0_dp)*x47*x5
      x49 = x24*x25
      x50 = x7**(-3)
      x51 = x32 - x41
      x52 = qq*x45*x7*(x31 + x35 + x38)
      x53 = (1.0_dp/2.0_dp)*x16*x31
      x54 = x25*x42
      x55 = x14*x50*(2*x31 + 4*x40)
      x56 = phi2_zeta_zeta*x30
      x57 = phi2_zeta**2
      x58 = 2*x0*x4*x57
      x59 = 6*x10*x57
      x60 = dd*x1
      Q_ab = qq*log(x14*x8)
      Q_ab_rs = x28*x7
      Q_ab_zeta = x44*x7
      Q_ab_rs_rs = qq*x27*x45*x7*(-x19 - x20 + x22) + x28*(x18 + x23) + x46*( &
      x14*x24*x50*(-x17 - 2*x23) + x26*x48 + x49*(2*bb*x0*x1*x16 + 3*cc &
      *x10*x11*x16 + x17) + x8*((3.0_dp/4.0_dp)*x13*x47 - x48))
      Q_ab_rs_zeta = x27*x52 + x28*x51 + x46*(x21*x54 + x24*x55 + x26*(x16*x41 &
      + x53) + x39*x49 + x8*(-9.0_dp/2.0_dp*phi2_zeta*x12*x16*x36 - x16* &
      x35 - x53))
      Q_ab_zeta_zeta = x43*x52 + x44*x51 + x46*(x26*(2*phi2_zeta_zeta*x33*x60 &
      + x56 - x58 - x59*x60) + x42*x55 + x54*(-2*aa*phi2_zeta*x29*xx - &
      4*bb*phi2_zeta*x1*x33 - 6*cc*phi2_zeta*x11*x36) + x8*(bb*x1*x59 - &
      phi2_zeta_zeta*x34 - phi2_zeta_zeta*x37 - x56 + x58 + 12*x12*x57/ &
      phi2**8))

   END SUBROUTINE calc_Q_ab

   ELEMENTAL SUBROUTINE calc_ec_mu_ab(rs, zeta, mu, max_deriv, ec_mu_ab, ec_mu_ab_rs, ec_mu_ab_zeta, ec_mu_ab_rs_rs, ec_mu_ab_rs_zeta, ec_mu_ab_zeta_zeta)
      REAL(KIND=dp), INTENT(IN) :: rs, zeta, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: ec_mu_ab, ec_mu_ab_rs, ec_mu_ab_zeta, ec_mu_ab_rs_rs, ec_mu_ab_rs_zeta, ec_mu_ab_zeta_zeta

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_ec_mu_ab", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, Q_ab, Q_ab_rs, Q_ab_zeta, Q_ab_rs_rs, Q_ab_rs_zeta, Q_ab_zeta_zeta, phi2, phi2_zeta, phi2_zeta_zeta, a1_ab, a1_ab_rs, a1_ab_zeta, a1_ab_rs_rs, a1_ab_rs_zeta, a1_ab_zeta_zeta, a2_ab, a2_ab_rs, a2_ab_zeta, a2_ab_rs_rs, a2_ab_rs_zeta, a2_ab_zeta_zeta, a3_ab, a3_ab_rs, a3_ab_zeta, a3_ab_rs_rs, a3_ab_rs_zeta, a3_ab_zeta_zeta, a4_ab, a4_ab_rs, a4_ab_zeta, a4_ab_rs_rs, a4_ab_rs_zeta, a4_ab_zeta_zeta, a5_ab, a5_ab_rs, a5_ab_zeta, a5_ab_rs_rs, a5_ab_rs_zeta, a5_ab_zeta_zeta, Ec_ab, Ec_ab_rs, Ec_ab_zeta, Ec_ab_rs_rs, Ec_ab_rs_zeta, Ec_ab_zeta_zeta

      CALL calc_Q_ab(rs, zeta, mu, max_deriv, Q_ab, Q_ab_rs, Q_ab_zeta, Q_ab_rs_rs, Q_ab_rs_zeta, Q_ab_zeta_zeta)
      CALL calc_phi2(zeta, max_deriv, phi2, phi2_zeta, phi2_zeta_zeta)
      CALL calc_a1_ab(rs, zeta, max_deriv, a1_ab, a1_ab_rs, a1_ab_zeta, a1_ab_rs_rs, a1_ab_rs_zeta, a1_ab_zeta_zeta)
      CALL calc_a2_ab(rs, zeta, max_deriv, a2_ab, a2_ab_rs, a2_ab_zeta, a2_ab_rs_rs, a2_ab_rs_zeta, a2_ab_zeta_zeta)
      CALL calc_a3_ab(rs, zeta, max_deriv, a3_ab, a3_ab_rs, a3_ab_zeta, a3_ab_rs_rs, a3_ab_rs_zeta, a3_ab_zeta_zeta)
      CALL calc_a4_ab(rs, zeta, max_deriv, a4_ab, a4_ab_rs, a4_ab_zeta, a4_ab_rs_rs, a4_ab_rs_zeta, a4_ab_zeta_zeta)
      CALL calc_a5_ab(rs, zeta, max_deriv, a5_ab, a5_ab_rs, a5_ab_zeta, a5_ab_rs_rs, a5_ab_rs_zeta, a5_ab_zeta_zeta)
      CALL calc_Ec_ab(rs, zeta, max_deriv, Ec_ab, Ec_ab_rs, Ec_ab_zeta, Ec_ab_rs_rs, Ec_ab_rs_zeta, Ec_ab_zeta_zeta)

      x0 = b0**2
      x1 = x0 + 1
      x2 = x1**(-4)
      x3 = phi2**3
      x4 = Q_ab*x3
      x5 = mu**3
      x6 = a1_ab*x5
      x7 = mu**4
      x8 = a2_ab*x7
      x9 = mu**5
      x10 = a3_ab*x9
      x11 = mu**6
      x12 = a4_ab*x11
      x13 = mu**8
      x14 = a5_ab*x13
      x15 = Q_ab_rs*x3
      x16 = a1_ab_rs*x5
      x17 = a2_ab_rs*x7
      x18 = a3_ab_rs*x9
      x19 = a4_ab_rs*x11
      x20 = a5_ab_rs*x13
      x21 = 1.0_dp/rs
      x22 = x1**(-5)
      x23 = x0*x22*(8*x10 + 8*x12 + 8*x14 + 8*x4 + 8*x6 + 8*x8)
      x24 = Q_ab_zeta*x3
      x25 = a1_ab_zeta*x5
      x26 = a2_ab_zeta*x7
      x27 = a3_ab_zeta*x9
      x28 = a4_ab_zeta*x11
      x29 = a5_ab_zeta*x13
      x30 = phi2**2
      x31 = Q_ab*phi2_zeta*x30
      x32 = rs**(-2)
      x33 = x0*x21*x22
      x34 = 3*x30
      ec_mu_ab = Ec_ab + x2*(-x10 - x12 - x14 - x4 - x6 - x8)
      ec_mu_ab_rs = Ec_ab_rs + x2*(-x15 - x16 - x17 - x18 - x19 - x20) + x21* &
      x23
      ec_mu_ab_zeta = Ec_ab_zeta + x2*(-x24 - x25 - x26 - x27 - x28 - x29 - 3* &
      x31)
      ec_mu_ab_rs_rs = Ec_ab_rs_rs + b0**4*x32*(-80*x10 - 80*x12 - 80*x14 - 80 &
      *x4 - 80*x6 - 80*x8)/x1**6 + x2*(-Q_ab_rs_rs*x3 - a1_ab_rs_rs*x5 &
      - a2_ab_rs_rs*x7 - a3_ab_rs_rs*x9 - a4_ab_rs_rs*x11 - a5_ab_rs_rs &
      *x13) + x23*x32 + x33*(16*x15 + 16*x16 + 16*x17 + 16*x18 + 16*x19 &
      + 16*x20)
      ec_mu_ab_rs_zeta = Ec_ab_rs_zeta + x2*(-Q_ab_rs*phi2_zeta*x34 - &
      Q_ab_rs_zeta*x3 - a1_ab_rs_zeta*x5 - a2_ab_rs_zeta*x7 - &
      a3_ab_rs_zeta*x9 - a4_ab_rs_zeta*x11 - a5_ab_rs_zeta*x13) + x33*( &
      8*x24 + 8*x25 + 8*x26 + 8*x27 + 8*x28 + 8*x29 + 24*x31)
      ec_mu_ab_zeta_zeta = Ec_ab_zeta_zeta + x2*(-6*Q_ab*phi2*phi2_zeta**2 - &
      Q_ab*phi2_zeta_zeta*x34 - 6*Q_ab_zeta*phi2_zeta*x30 - &
      Q_ab_zeta_zeta*x3 - a1_ab_zeta_zeta*x5 - a2_ab_zeta_zeta*x7 - &
      a3_ab_zeta_zeta*x9 - a4_ab_zeta_zeta*x11 - a5_ab_zeta_zeta*x13)

   END SUBROUTINE calc_ec_mu_ab
