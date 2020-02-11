
   ELEMENTAL SUBROUTINE calc_ex_mu(rs, mu, max_deriv, ex_mu, ex_mu_rs, ex_mu_rs_rs)
      REAL(KIND=dp), INTENT(IN) :: rs, mu
      INTEGER, INTENT(IN) :: max_deriv
      REAL(KIND=dp), INTENT(OUT) :: ex_mu, ex_mu_rs, ex_mu_rs_rs

      CHARACTER(LEN=*), PARAMETER :: routineN = "calc_ex_mu", routineP = moduleN//":"//routineN

      REAL(KIND=dp) :: x0, x1, x2, x3, facx_A, facx

      facx_A = (1.0_dp/6.0_dp)*12**(1.0_dp/3.0_dp)/pi**(1.0_dp/3.0_dp)
      facx = 18**(1.0_dp/3.0_dp)/pi**(2.0_dp/3.0_dp)
      x0 = Ax**2
      x1 = 8*x0
      x2 = 32*Ax**4
      x3 = exp_Ax*x2 - x2
      ex_mu = (1.0_dp/8.0_dp)*facx*(8*Ax*(4*Ax**3 - 2*Ax*exp_Ax*(2*x0 - 1) - 3* &
      Ax + sqrt(pi)*erf(0.5_dp/Ax)) - 3)/rs
      ex_mu_rs = -3.0_dp/8.0_dp*facx*(x1 + x3 - 1)/rs**2
      ex_mu_rs_rs = -3.0_dp/4.0_dp*facx*(exp_Ax*x1 + x3 + 1)/rs**3

   END SUBROUTINE calc_ex_mu
