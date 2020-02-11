
   ELEMENTAL SUBROUTINE calc_ex_mu(rs, mu, max_deriv, ex_mu, ex_mu_rs, ex_mu_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: ex_mu, ex_mu_rs, ex_mu_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_ex_mu", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, facx_A, facx

      facx_A = (1.0_dp/6.0_dp)*12**(1.0_dp/3.0_dp)/pi**(1.0_dp/3.0_dp)
      facx = 18**(1.0_dp/3.0_dp)/pi**(2.0_dp/3.0_dp)
      x0 = Ax**2
      x1 = (1.0_dp/4.0_dp)/x0
      x2 = exp(-x1)
      x3 = facx*x2
      x4 = exp(x1)
      x5 = 4*Ax**3
      x6 = facx/rs**2
      x7 = Ax**4
      x8 = 12*x6*x7
      x9 = 32*x7
      ex_mu = (1.0_dp/8.0_dp)*x3*(8*Ax*(2*Ax + x4*(-3*Ax + x5 + sqrt(pi)*erf( &
      0.5_dp/Ax)) - x5) - 3*x4)/rs
      ex_mu_rs = -3*x0*x6 - x2*x8 + (3.0_dp/8.0_dp)*x6 + x8
      ex_mu_rs_rs = (3.0_dp/4.0_dp)*x3*(-8*x0 + x4*x9 - x4 - x9)/rs**3

   END SUBROUTINE calc_ex_mu
