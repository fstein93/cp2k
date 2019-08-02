
   SUBROUTINE calc_A(rs, mu, A, A_rs, A_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs, mu
      REAL(KIND=dp), INTENT(INOUT) :: A, A_rs, A_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_A", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0

      x0 = (1.0_dp/6.0_dp)*12**(1.0_dp/3.0_dp)*mu/pi**(1.0_dp/3.0_dp)
      A = rs*x0
      A_rs = x0
      A_rs_rs = 0

   END SUBROUTINE calc_A

   SUBROUTINE ex_mu(rs, mu, ex_mu, ex_mu_rs, ex_mu_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs, mu
      REAL(KIND=dp), INTENT(INOUT) :: ex_mu, ex_mu_rs, ex_mu_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "ex_mu", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, x40

      x0 = 18**(1.0_dp/3.0_dp)
      x1 = pi**(2.0_dp/3.0_dp)
      x2 = 1.0_dp/x1
      x3 = 1.0_dp/rs
      x4 = x0*x2*x3
      x5 = 12**(1.0_dp/3.0_dp)
      x6 = pi**(1.0_dp/3.0_dp)
      x7 = 1.0_dp/x6
      x8 = mu*rs*x5*x7
      x9 = 1.0_dp/pi
      x10 = mu**3
      x11 = rs**3
      x12 = sqrt(pi)
      x13 = 1.0_dp/mu
      x14 = erf(0.5_dp*x0*x13*x3*x6)
      x15 = rs**2
      x16 = 1.0_dp/x15
      x17 = mu**(-2)
      x18 = exp(-0.75_dp*x1*x16*x17*x5)
      x19 = mu*x5*x7
      x20 = (1.0_dp/3.0_dp)*x19
      x21 = rs*x20
      x22 = x10*x9
      x23 = x11*x22
      x24 = (2.0_dp/9.0_dp)*x23
      x25 = x21 - x24
      x26 = -1.0_dp/12.0_dp*mu*rs*x5*x7 + (1.0_dp/27.0_dp)*x10*x11*x9 + (1.0_dp/ &
      6.0_dp)*x12*x14 + (1.0_dp/6.0_dp)*x18*x25
      x27 = x26*x8
      x28 = x0*x16*x2
      x29 = (1.0_dp/6.0_dp)*mu*rs*x5*x7
      x30 = (1.0_dp/2.0_dp)*x19
      x31 = x10*x15*x9
      x32 = (2.0_dp/3.0_dp)*x31
      x33 = x0*x13*x18*x6
      x34 = rs*x30
      x35 = 1.0_dp/x11
      x36 = x1*x17*x18*x35*x5
      x37 = -x16*x33 + x18*(x20 - x32) - x30 + x32 + x36*(-1.0_dp/3.0_dp*x23 + &
      x34)
      x38 = x12*x14 + x18*x25 + x24 - x34
      x39 = (4.0_dp/3.0_dp)*rs*x22
      x40 = (3.0_dp/2.0_dp)*x8
      ex_mu = x4*(x27 - 3.0_dp/8.0_dp)
      ex_mu_rs = x28*(-x27 + 3.0_dp/8.0_dp) + x4*(x19*x26 + x29*x37)
      ex_mu_rs_rs = x0*x2*x35*(x21*x38 - 3.0_dp/4.0_dp) + x28*(-x20*x38 - x21* &
      x37) + x4*(x20*x37 + x29*(-x18*x39 + 2*x33*x35 + x36*(x19 - 2*x31 &
      ) + x39 + x1*x17*x18*x5*(x23 - x40)/rs**4 - 9*pi*x18/(rs**5*x10) &
      + pi**(4.0_dp/3.0_dp)*x0*x18*(-x23 + x40)/(mu**4*rs**6)))

   END SUBROUTINE ex_mu
