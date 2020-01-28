
   ELEMENTAL SUBROUTINE calc_ex_mu(rs, mu, max_deriv, ex_mu, ex_mu_rs, ex_mu_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: ex_mu, ex_mu_rs, ex_mu_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_ex_mu", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, facx_A, facx

      facx_A = (1.0_dp/6.0_dp)*12**(1.0_dp/3.0_dp)/pi**(1.0_dp/3.0_dp)
      facx = 18**(1.0_dp/3.0_dp)/pi**(2.0_dp/3.0_dp)
      x0 = 1.0_dp/rs
      x1 = facx_A*mu
      x2 = 3*x1
      x3 = rs**3
      x4 = facx_A**3
      x5 = mu**3
      x6 = x4*x5
      x7 = 4*x3*x6
      x8 = 1.0_dp/facx_A
      x9 = 1.0_dp/mu
      x10 = rs**2
      x11 = 1.0_dp/x10
      x12 = facx_A**(-2)
      x13 = mu**(-2)
      x14 = exp(-0.25_dp*x11*x12*x13)
      x15 = 2*x1
      x16 = rs*x15 - x7
      x17 = x14*x16
      x18 = x1*(-rs*x2 + x17 + x7 + sqrt(pi)*erf(0.5_dp*x0*x8*x9))
      x19 = facx*(-rs*x18 + 3.0_dp/8.0_dp)
      x20 = facx_A*mu*rs
      x21 = 12*x10*x6
      x22 = x14*x8*x9
      x23 = x14*(x15 - x21)
      x24 = 1.0_dp/x3
      x25 = x12*x13*x24
      x26 = -x11*x22 + (1.0_dp/2.0_dp)*x14*x16*x25 - x2 + x21 + x23
      x27 = facx*(-x18 - x20*x26)
      x28 = 2*x24
      x29 = 24*rs*x6
      ex_mu = -x0*x19
      ex_mu_rs = -x0*x27 + x11*x19
      ex_mu_rs_rs = -facx*x0*(-x15*x26 - x20*(-x14*x29 + x22*x28 + x23*x25 + &
      x29 - 3.0_dp/2.0_dp*x12*x13*x17/rs**4 - 1.0_dp/2.0_dp*x14/(rs**5*x4* &
      x5) + (1.0_dp/4.0_dp)*x17/(facx_A**4*mu**4*rs**6))) + 2*x11*x27 - &
      x19*x28

   END SUBROUTINE calc_ex_mu
